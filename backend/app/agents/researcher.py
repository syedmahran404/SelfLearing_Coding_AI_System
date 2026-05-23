"""Researcher — answers factual / API / design questions using retrieved context.

May independently issue a RAG query (`rag.search`) when the orchestrator
didn't pre-stage one. Output is a plain answer with optional citations
embedded in the text — agents downstream can extract them by regex.
"""
from __future__ import annotations

from uuid import UUID

from app.agents.base import BaseAgent
from app.agents.prompts import RESEARCHER
from app.observability.tracing import SpanKind, trace_span
from app.rag.service import RagService
from app.schemas.agent_io import AgentInput, AgentOutput


class ResearcherAgent(BaseAgent):
    name = "researcher"
    responsibility = "Answer factual / API / design questions using retrieved context."

    def __init__(self, *, llm, rag: RagService | None, tracer, settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(llm=llm, tracer=tracer, settings=settings)
        self._rag = rag

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.researcher.run",
            SpanKind.AGENT,
            payload={"intent": ai.intent.value},
        ) as span:
            extra_rag_text = ""
            if self._rag is not None and not ai.context.rag:
                # No RAG was pre-staged. Try one direct fetch.
                try:
                    hits = await self._rag.search(
                        (ai.subtask.description if ai.subtask else None) or ai.request,
                        top_k=6,
                        user_id=UUID(str(ai.user_id)) if ai.user_id else None,
                        project_id=UUID(str(ai.project_id)) if ai.project_id else None,
                    )
                    if hits:
                        extra_rag_text = "\n--- additional rag ---\n" + "\n".join(
                            f"- ({h.blended_score:.2f}) {h.source_uri}#{h.line_start}-{h.line_end}: {h.text[:400]}"
                            for h in hits
                        )
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning("rag_inline_failed", error=str(exc))

            user = (
                f"USER_REQUEST:\n{ai.request}\n\n"
                f"SUBTASK:\n{(ai.subtask.description if ai.subtask else '(none)')}\n\n"
                f"[CONTEXT]\n{_format_context(ai)}{extra_rag_text}\n"
            )
            text, usage = await self._llm_text(
                system=RESEARCHER,
                user=user,
                temperature=0.1,
                max_tokens=900,
                purpose="agent.researcher",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            confidence = _confidence_from_context(ai)
            return AgentOutput(
                agent=self.name,
                summary="Research answer produced.",
                answer=text,
                confidence=confidence,
            )


def _format_context(ai: AgentInput) -> str:
    parts: list[str] = []
    if ai.context.lessons:
        parts.append("--- lessons ---")
        for c in ai.context.lessons[:6]:
            parts.append(f"- ({c.score:.2f}) {c.text[:300]}")
    if ai.context.long_term:
        parts.append("--- memory ---")
        for c in ai.context.long_term[:6]:
            parts.append(f"- ({c.score:.2f}) {c.text[:300]}")
    if ai.context.rag:
        parts.append("--- rag ---")
        for c in ai.context.rag[:8]:
            uri = (c.metadata or {}).get("source_uri", "?")
            lines = (c.metadata or {}).get("lines", [0, 0])
            parts.append(f"- ({c.score:.2f}) {uri}:{lines[0]}-{lines[1]} {c.text[:400]}")
    if ai.context.episodic:
        parts.append("--- similar past episodes ---")
        for c in ai.context.episodic[:4]:
            parts.append(f"- ({c.score:.2f}) {c.text[:240]}")
    return "\n".join(parts)


def _confidence_from_context(ai: AgentInput) -> float:
    """Less retrieved evidence → lower confidence."""
    score = 0.4
    score += 0.05 * min(len(ai.context.rag), 6)
    score += 0.03 * min(len(ai.context.long_term), 6)
    score += 0.04 * min(len(ai.context.lessons), 4)
    return max(0.1, min(0.95, score))
