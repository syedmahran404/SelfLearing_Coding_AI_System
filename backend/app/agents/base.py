"""BaseAgent contract + shared LLM helpers.

Each subclass implements `responsibility` (a constant) and `run(input)`.
The base class provides:
- `_llm_json` : structured-JSON helper that handles schema, retries on
                parse failure, and accumulates token/cost into a span.
- `_llm_text` : plain-text helper.
- `_span`     : open a span attributed to this agent.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Mapping

from app.config import Settings
from app.llm.provider import LLMProvider
from app.llm.types import ChatMessage, CompletionRequest, Role
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput


class BaseAgent(ABC):
    """Abstract specialist agent."""

    name: str = "abstract"          # short, registry-stable identifier
    responsibility: str = ""         # one-line description
    default_temperature: float = 0.2

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._tracer = tracer
        self._settings = settings
        self._logger = get_logger(f"agent.{self.name}")

    # ── external entrypoint ──
    @abstractmethod
    async def run(self, ai: AgentInput) -> AgentOutput: ...

    # ── helpers ──
    def _model_for(self, kind: str = "default") -> str:
        s = self._settings
        if s is None:
            return "gpt-4o-mini"
        if kind == "planner":
            return s.llm_planner_model
        if kind == "coder":
            return s.llm_coder_model
        return s.llm_default_model

    async def _llm_text(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        purpose: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return (text, usage_dict). Raises if the LLM is unavailable."""
        assert self._llm is not None, f"agent {self.name} requires an LLM"
        req = CompletionRequest(
            model=model or self._model_for("default"),
            messages=[
                ChatMessage(role=Role.SYSTEM, content=system),
                ChatMessage(role=Role.USER, content=user),
            ],
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens,
            purpose=purpose or f"agent.{self.name}",
        )
        resp = await self._llm.complete(req)
        return resp.content.strip(), {
            "tokens_in": resp.usage.prompt_tokens,
            "tokens_out": resp.usage.completion_tokens,
            "cost_usd": resp.usage.cost_usd,
        }

    async def _llm_json(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        purpose: str | None = None,
        repair_attempts: int = 1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ask the LLM to emit JSON matching ``schema``.

        We pass ``response_schema`` to providers that support it (OpenAI), and
        always also include the schema in the user prompt as a fallback for
        providers that don't honor it. On parse failure we retry once with a
        repair instruction.
        """
        assert self._llm is not None, f"agent {self.name} requires an LLM"

        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
        user_prompt = (
            f"{user}\n\n"
            "Reply ONLY with a JSON value matching this schema. No prose:\n"
            f"```json\n{schema_text}\n```"
        )

        last_err: str | None = None
        usage_total = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        attempts = repair_attempts + 1

        for attempt in range(attempts):
            req = CompletionRequest(
                model=model or self._model_for("default"),
                messages=[
                    ChatMessage(role=Role.SYSTEM, content=system),
                    ChatMessage(role=Role.USER, content=user_prompt if attempt == 0 else (
                        f"{user_prompt}\n\nYour previous reply could not be parsed: {last_err}\n"
                        "Reply with valid JSON only."
                    )),
                ],
                temperature=(
                    temperature if temperature is not None else self.default_temperature
                ),
                max_tokens=max_tokens,
                response_schema=dict(schema),
                purpose=purpose or f"agent.{self.name}",
            )
            resp = await self._llm.complete(req)
            usage_total["tokens_in"] += resp.usage.prompt_tokens
            usage_total["tokens_out"] += resp.usage.completion_tokens
            usage_total["cost_usd"] += resp.usage.cost_usd
            try:
                parsed = _extract_json(resp.content)
                return parsed, usage_total
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = str(exc)
                self._logger.warning(
                    "agent_json_parse_failed",
                    attempt=attempt + 1,
                    error=last_err,
                    raw=resp.content[:240],
                )

        # All attempts exhausted: return empty dict so callers can degrade.
        return {}, usage_total

    async def _span(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        payload: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open a trace span for this agent."""
        assert self._tracer is not None
        async with trace_span(self._tracer, name, kind, payload=payload) as span:
            yield span


# ── JSON extraction ─────────────────────────────────────────────────────────


_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a text blob.

    LLMs sometimes wrap JSON in ``` fences or add a single sentence of
    preamble; we try the strict parse first, then fenced extraction, then
    a brace-balanced extractor as a last resort.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")

    # 1) fast path
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) fenced block
    m = _FENCED.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3) brace-balanced scan
    start = s.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"could not extract JSON from response: {s[:120]!r}")


# ── registry ────────────────────────────────────────────────────────────────


class AgentRegistry:
    """Tiny dict-wrapper. Distinct from the tool registry to keep concerns
    obvious in stack traces."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._agents:
            raise ValueError(f"agent already registered: {agent.name!r}")
        self._agents[agent.name] = agent

    def get(self, name: str) -> BaseAgent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"unknown agent: {name!r}") from exc

    def list(self) -> list[str]:
        return list(self._agents.keys())

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": a.name, "responsibility": a.responsibility} for a in self._agents.values()
        ]
