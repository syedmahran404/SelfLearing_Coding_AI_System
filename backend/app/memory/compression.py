"""Summarization / compression.

Two jobs:

1. **Session summarization** — when a session ends or its short-term window
   exceeds a budget, compress old turns into a `Session.summary` and drop
   the corresponding short-term entries. This prevents short-term memory
   from leaking into context unboundedly.

2. **Memory consolidation** — when several memories with the same `kind`
   and similar content accumulate, fold them into a single canonical row.
   Run by the lifecycle worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.llm.provider import LLMProvider
from app.llm.types import ChatMessage, CompletionRequest, Role
from app.observability import get_logger

logger = get_logger("memory.compression")


SUMMARIZE_SYSTEM = (
    "You are a precise summarizer. Compress the following conversation into "
    "a short, factual summary that preserves: user goals, decisions made, "
    "code or commands produced, and unresolved issues. Do not add new "
    "information. Use 4-8 bullet lines."
)

CONSOLIDATE_SYSTEM = (
    "You merge multiple short notes about the same topic into a single, "
    "non-redundant note. Preserve all factual content. Drop duplicates. "
    "Output the merged note only — no preamble."
)


@dataclass(slots=True)
class SummaryResult:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


class Summarizer:
    """Wraps the LLM provider with the two summarization prompts."""

    def __init__(self, llm: LLMProvider, *, model: str | None = None) -> None:
        self._llm = llm
        self._model = model or llm.settings.llm_planner_model

    async def summarize_turns(
        self, turns: Iterable[dict], *, prior_summary: str | None = None
    ) -> SummaryResult:
        """Compress a chronological list of turn dicts (role, content, ...)."""
        body_lines: list[str] = []
        if prior_summary:
            body_lines.append(f"[PRIOR SUMMARY]\n{prior_summary}\n")
        body_lines.append("[TURNS]")
        for t in turns:
            role = t.get("role", "user")
            content = t.get("content", "")
            agent = t.get("agent")
            tag = f"{role}" + (f"/{agent}" if agent else "")
            body_lines.append(f"- {tag}: {content}")

        req = CompletionRequest(
            model=self._model,
            messages=[
                ChatMessage(role=Role.SYSTEM, content=SUMMARIZE_SYSTEM),
                ChatMessage(role=Role.USER, content="\n".join(body_lines)),
            ],
            temperature=0.1,
            max_tokens=500,
            purpose="memory.summarize_turns",
        )
        resp = await self._llm.complete(req)
        return SummaryResult(
            text=resp.content.strip(),
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
            cost_usd=resp.usage.cost_usd,
        )

    async def consolidate(self, contents: list[str]) -> SummaryResult:
        body = "\n\n---\n\n".join(contents)
        req = CompletionRequest(
            model=self._model,
            messages=[
                ChatMessage(role=Role.SYSTEM, content=CONSOLIDATE_SYSTEM),
                ChatMessage(role=Role.USER, content=body),
            ],
            temperature=0.1,
            max_tokens=600,
            purpose="memory.consolidate",
        )
        resp = await self._llm.complete(req)
        return SummaryResult(
            text=resp.content.strip(),
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
            cost_usd=resp.usage.cost_usd,
        )
