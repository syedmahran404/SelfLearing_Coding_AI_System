"""TaskDecomposer — splits an oversized subtask into atomic ones."""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import DECOMPOSER
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, Subtask

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["subtasks"],
    "properties": {
        "subtasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["title", "description", "agent"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "agent": {
                        "type": "string",
                        "enum": ["researcher", "coder", "debugger", "evaluator"],
                    },
                    "success_predicate": {"type": "string"},
                    "required_tools": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


class TaskDecomposerAgent(BaseAgent):
    name = "decomposer"
    responsibility = "Split an oversized subtask into atomic, ordered subtasks."

    async def run(self, ai: AgentInput) -> AgentOutput:
        if ai.subtask is None:
            return AgentOutput(
                agent=self.name,
                summary="No subtask provided to decompose.",
                confidence=0.0,
                needs_more_info=True,
            )
        async with trace_span(
            self._tracer,
            "agent.decomposer.run",
            SpanKind.AGENT,
            payload={"subtask_title": ai.subtask.title},
        ) as span:
            user = (
                f"PARENT_SUBTASK:\n"
                f"title: {ai.subtask.title}\n"
                f"description: {ai.subtask.description}\n"
                f"success_predicate: {ai.subtask.success_predicate or '(none)'}\n"
            )
            data, usage = await self._llm_json(
                system=DECOMPOSER,
                user=user,
                schema=_SCHEMA,
                model=self._model_for("planner"),
                temperature=0.15,
                max_tokens=900,
                purpose="agent.decomposer",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            new_subs: list[Subtask] = []
            for s in (data.get("subtasks") or []):
                try:
                    new_subs.append(
                        Subtask(
                            title=str(s["title"]),
                            description=str(s["description"]),
                            agent=str(s.get("agent", "coder")),
                            success_predicate=s.get("success_predicate"),
                            required_tools=list(s.get("required_tools") or []),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue

            return AgentOutput(
                agent=self.name,
                summary=f"Decomposed into {len(new_subs)} atomic subtasks.",
                confidence=0.6 if new_subs else 0.1,
                metadata={"new_subtasks": [s.model_dump(mode="json") for s in new_subs]},
            )
