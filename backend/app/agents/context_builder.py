"""ContextBuilder — pulls and budgets retrieval results.

This agent does NOT call the LLM. It assembles a `BudgetedContext` for a
specialist agent by:
1. Pulling top-K from each layer (long-term memory, similar episodes,
   lessons, RAG, project chunks).
2. Truncating each section to its token budget.
3. Returning a single `BudgetedContext` with `estimated_tokens` filled in.

Token estimates use the LLM provider's `estimate_tokens` (tiktoken if
available; heuristic otherwise) so we don't pay any LLM call cost just to
plan context.
"""
from __future__ import annotations

from typing import Iterable
from uuid import UUID

from app.agents.base import BaseAgent
from app.config import Settings
from app.llm.token_meter import estimate_tokens
from app.memory.service import MemoryService
from app.observability import Tracer
from app.observability.tracing import SpanKind, trace_span
from app.rag.service import RagService
from app.schemas.agent_io import (
    AgentInput,
    AgentOutput,
    BudgetedContext,
    ContextChunk,
    TaskIntent,
)


# Per-section token budgets. Tunable; orchestrator can override if needed.
SECTION_BUDGETS: dict[str, int] = {
    "short_term": 1_500,
    "long_term": 1_500,
    "episodic": 1_000,
    "lessons": 800,
    "rag": 4_000,
    "project": 2_500,
}


class ContextBuilderAgent(BaseAgent):
    name = "context_builder"
    responsibility = "Assemble a budgeted context from memory + RAG for a subtask."

    def __init__(
        self,
        *,
        memory: MemoryService,
        rag: RagService | None,
        tracer: Tracer,
        settings: Settings,
    ) -> None:
        super().__init__(llm=None, tracer=tracer, settings=settings)
        self._memory = memory
        self._rag = rag

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.context_builder.run",
            SpanKind.AGENT,
            payload={"intent": ai.intent.value},
        ) as span:
            ctx = await self._build(ai)
            span["payload"].update(
                {
                    "long_term": len(ctx.long_term),
                    "episodic": len(ctx.episodic),
                    "lessons": len(ctx.lessons),
                    "rag": len(ctx.rag),
                    "estimated_tokens": ctx.estimated_tokens,
                }
            )
            return AgentOutput(
                agent=self.name,
                summary=f"Built context: {ctx.estimated_tokens} tokens across sections.",
                confidence=0.9,
                metadata={"context": ctx.model_dump(mode="json")},
            )

    async def _build(self, ai: AgentInput) -> BudgetedContext:
        # Note: this method needs a DB session. The orchestrator passes one
        # via `extras["db_session"]`. If absent (rare path / tests), we skip
        # DB-bound retrieval and return an empty context.
        db = ai.extras.get("db_session")
        text = (ai.subtask.description if ai.subtask else None) or ai.request

        long_term: list[ContextChunk] = []
        episodic: list[ContextChunk] = []
        lessons: list[ContextChunk] = []

        if db is not None and ai.user_id is not None:
            mem_hits = await self._memory.recall(
                db,
                user_id=UUID(str(ai.user_id)),
                text=text,
                top_k=8,
                project_id=UUID(str(ai.project_id)) if ai.project_id else None,
            )
            for h in mem_hits:
                kind = h.memory.kind
                cc = ContextChunk(
                    source="memory",
                    id=str(h.memory.id),
                    text=h.memory.content,
                    score=h.blended_score,
                    metadata={"kind": kind, "tags": list(h.memory.tags or [])},
                )
                if kind in {"failure_rule", "success_rule", "convention", "preference"}:
                    lessons.append(cc)
                else:
                    long_term.append(cc)

            ep_hits = await self._memory.similar_episodes(
                db,
                user_id=UUID(str(ai.user_id)),
                text=text,
                top_k=4,
                intent=ai.intent.value if ai.intent != TaskIntent.QA else None,
                project_id=UUID(str(ai.project_id)) if ai.project_id else None,
            )
            for e in ep_hits:
                summary = e.episode.summary or e.episode.title
                episodic.append(
                    ContextChunk(
                        source="episode",
                        id=str(e.episode.id),
                        text=f"[{e.episode.outcome}] {summary}",
                        score=e.score,
                        metadata={"intent": e.episode.intent, "outcome": e.episode.outcome},
                    )
                )

        # RAG
        rag_chunks: list[ContextChunk] = []
        if self._rag is not None and ai.intent != TaskIntent.QA:
            try:
                hits = await self._rag.search(
                    text,
                    top_k=8,
                    user_id=UUID(str(ai.user_id)) if ai.user_id else None,
                    project_id=UUID(str(ai.project_id)) if ai.project_id else None,
                )
                for h in hits:
                    rag_chunks.append(
                        ContextChunk(
                            source="rag",
                            id=h.id,
                            text=h.text,
                            score=h.blended_score,
                            metadata={
                                "source_uri": h.source_uri,
                                "language": h.language,
                                "chunk_kind": h.chunk_kind,
                                "name": h.name,
                                "lines": [h.line_start, h.line_end],
                            },
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("rag_search_failed", error=str(exc))

        # Short-term: most recent N turns from Redis.
        short_term: list[ContextChunk] = []
        if ai.session_id is not None:
            turns = await self._memory.get_short_term(ai.session_id)
            for i, t in enumerate(turns[-8:]):
                short_term.append(
                    ContextChunk(
                        source="shortterm",
                        id=f"st_{i}",
                        text=f"{t.get('role')}: {t.get('content', '')[:600]}",
                        score=1.0,
                    )
                )

        # Project chunks come from the indexer; left empty here — added when the
        # project understanding subsystem is wired (task #12 hooks into here).
        project: list[ContextChunk] = []

        # Apply per-section budgets.
        budgeted = BudgetedContext(
            short_term=_budget(short_term, SECTION_BUDGETS["short_term"]),
            long_term=_budget(long_term, SECTION_BUDGETS["long_term"]),
            episodic=_budget(episodic, SECTION_BUDGETS["episodic"]),
            project=_budget(project, SECTION_BUDGETS["project"]),
            lessons=_budget(lessons, SECTION_BUDGETS["lessons"]),
            rag=_budget(rag_chunks, SECTION_BUDGETS["rag"]),
        )
        budgeted.estimated_tokens = _sum_tokens(budgeted)
        return budgeted


def _budget(chunks: Iterable[ContextChunk], max_tokens: int) -> list[ContextChunk]:
    """Greedy budget: keep highest-scored chunks until the token cap is hit."""
    items = sorted(chunks, key=lambda c: c.score, reverse=True)
    out: list[ContextChunk] = []
    used = 0
    for c in items:
        n = estimate_tokens(c.text)
        if used + n > max_tokens and out:
            break
        out.append(c)
        used += n
        if used >= max_tokens:
            break
    return out


def _sum_tokens(ctx: BudgetedContext) -> int:
    total = 0
    for section in (ctx.short_term, ctx.long_term, ctx.episodic, ctx.project, ctx.lessons, ctx.rag):
        for c in section:
            total += estimate_tokens(c.text)
    return total
