"""MemoryAgent — decides what is worth remembering from this run.

Called at the end of a run with the original request, the final answer,
any reflection lesson, and the outcome. Asks the LLM for a short list of
`MemoryWriteRequest`s. The orchestrator persists them through MemoryService
(which dedups + embeds + records).
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import MEMORY_AGENT
from app.memory.service import MemoryService
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, MemoryWriteRequest

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["writes"],
    "properties": {
        "writes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "content"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["preference", "convention", "failure_rule", "success_rule", "fact"],
                    },
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "project_scoped": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
            },
        }
    },
}


class MemoryAgent(BaseAgent):
    name = "memory_agent"
    responsibility = "Decide what is worth writing to long-term memory."
    default_temperature = 0.1

    def __init__(self, *, llm, memory: MemoryService, tracer, settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(llm=llm, tracer=tracer, settings=settings)
        self._memory = memory

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.memory_agent.run",
            SpanKind.AGENT,
            payload={"intent": ai.intent.value},
        ) as span:
            user = (
                f"USER_REQUEST:\n{ai.request}\n\n"
                f"OUTCOME: {ai.extras.get('outcome', 'unknown')}\n"
                f"FINAL_ANSWER:\n{(ai.extras.get('final_answer') or '')[:1500]}\n\n"
                f"REFLECTION_LESSON: {ai.extras.get('reflection_lesson') or '(none)'}\n\n"
                "EXISTING_MEMORIES (recent):\n" + _format_existing(ai)
            )
            data, usage = await self._llm_json(
                system=MEMORY_AGENT,
                user=user,
                schema=_SCHEMA,
                temperature=0.1,
                max_tokens=600,
                purpose="agent.memory_agent",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            writes: list[MemoryWriteRequest] = []
            for w in (data.get("writes") or [])[:6]:  # cap noisy outputs
                try:
                    writes.append(
                        MemoryWriteRequest(
                            kind=str(w["kind"]),
                            content=str(w["content"]).strip(),
                            tags=list(w.get("tags") or []),
                            project_scoped=bool(w.get("project_scoped", False)),
                            confidence=float(w.get("confidence", 0.6)),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue

            span["payload"]["writes"] = len(writes)
            return AgentOutput(
                agent=self.name,
                summary=f"Proposed {len(writes)} memory write(s).",
                memory_writes=writes,
                confidence=0.8 if writes else 0.5,
            )


def _format_existing(ai: AgentInput) -> str:
    lines: list[str] = []
    for c in ai.context.long_term[:8]:
        lines.append(f"- ({(c.metadata or {}).get('kind','?')}) {c.text[:240]}")
    return "\n".join(lines) if lines else "(none retrieved)"
