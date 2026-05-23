"""Planner — turns a request into a TaskGraph.

Pulls similar past episodes (`memory.similar_episodes`) and a few relevant
lessons from long-term memory before asking the LLM to produce a plan.
That keeps planning *grounded* in what has worked before.
"""
from __future__ import annotations

import json
from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import PLANNER
from app.config import Settings
from app.llm.provider import LLMProvider
from app.memory.service import MemoryService
from app.observability import Tracer
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, Subtask, TaskGraph, TaskIntent

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["intent", "title", "rationale", "subtasks"],
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in TaskIntent]},
        "title": {"type": "string"},
        "rationale": {"type": "string"},
        "subtasks": {
            "type": "array",
            "minItems": 1,
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
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "expected_outputs": {"type": "array", "items": {"type": "string"}},
                    "success_predicate": {"type": "string"},
                    "required_tools": {"type": "array", "items": {"type": "string"}},
                    "max_attempts": {"type": "integer"},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


class PlannerAgent(BaseAgent):
    name = "planner"
    responsibility = "Turn a user request into an ordered, verifiable TaskGraph."

    def __init__(
        self,
        *,
        llm: LLMProvider,
        memory: MemoryService,
        tracer: Tracer,
        settings: Settings,
    ) -> None:
        super().__init__(llm=llm, tracer=tracer, settings=settings)
        self._memory = memory

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.planner.run",
            SpanKind.AGENT,
            payload={"intent_hint": ai.intent.value, "request_len": len(ai.request)},
        ) as span:
            ctx_blob = _format_context(ai)
            user = (
                f"REQUEST:\n{ai.request}\n\n"
                f"INITIAL_INTENT_HINT: {ai.intent.value}\n\n"
                f"{ctx_blob}"
            )
            data, usage = await self._llm_json(
                system=PLANNER,
                user=user,
                schema=_PLAN_SCHEMA,
                model=self._model_for("planner"),
                temperature=0.15,
                max_tokens=1500,
                purpose="agent.planner",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            graph = self._build_graph(ai, data)
            span["payload"]["subtasks"] = len(graph.subtasks)

            return AgentOutput(
                agent=self.name,
                summary=f"Plan with {len(graph.subtasks)} subtask(s): {graph.title}",
                metadata={"task_graph": graph.model_dump(mode="json")},
                confidence=_planner_confidence(graph),
            )

    def _build_graph(self, ai: AgentInput, data: dict[str, Any]) -> TaskGraph:
        if not data:
            return _fallback_plan(ai)
        try:
            intent = TaskIntent(data.get("intent", ai.intent.value))
        except ValueError:
            intent = ai.intent

        raw_subtasks = data.get("subtasks") or []
        subtasks: list[Subtask] = []
        for s in raw_subtasks:
            try:
                subtasks.append(
                    Subtask(
                        title=str(s.get("title", "")),
                        description=str(s.get("description", "")),
                        agent=str(s.get("agent", "coder")),
                        depends_on=list(s.get("depends_on") or []),
                        expected_outputs=list(s.get("expected_outputs") or []),
                        success_predicate=s.get("success_predicate"),
                        required_tools=list(s.get("required_tools") or []),
                        max_attempts=int(s.get("max_attempts") or 2),
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        if not subtasks:
            return _fallback_plan(ai)

        return TaskGraph(
            intent=intent,
            title=str(data.get("title") or ai.request[:80]),
            rationale=str(data.get("rationale") or ""),
            subtasks=subtasks,
            notes=list(data.get("notes") or []),
        )


def _format_context(ai: AgentInput) -> str:
    """Inline a compact view of any retrieved context."""
    if ai.context.estimated_tokens == 0 and not (
        ai.context.long_term or ai.context.episodic or ai.context.lessons or ai.context.rag
    ):
        return ""
    parts: list[str] = ["[CONTEXT]"]
    for label, items in (
        ("lessons", ai.context.lessons),
        ("similar_episodes", ai.context.episodic),
        ("memory", ai.context.long_term),
        ("rag", ai.context.rag),
    ):
        if items:
            parts.append(f"--- {label} ---")
            for c in items[:5]:
                parts.append(f"- ({c.score:.2f}) {c.text[:240]}")
    return "\n".join(parts)


def _planner_confidence(graph: TaskGraph) -> float:
    """Cheap heuristic: more granular plans with predicates score higher."""
    if not graph.subtasks:
        return 0.1
    with_pred = sum(1 for s in graph.subtasks if s.success_predicate)
    base = 0.4 + 0.1 * min(len(graph.subtasks), 5)
    bonus = 0.05 * with_pred
    return max(0.0, min(1.0, base + bonus))


def _fallback_plan(ai: AgentInput) -> TaskGraph:
    """If the LLM didn't return a parseable plan, produce a one-step plan."""
    agent = "researcher" if ai.intent in (TaskIntent.QA, TaskIntent.RESEARCH, TaskIntent.EXPLAIN) else "coder"
    sub = Subtask(
        title="Address the request",
        description=ai.request[:400],
        agent=agent,
        success_predicate="The user's request is addressed in the answer.",
    )
    return TaskGraph(
        intent=ai.intent,
        title=ai.request[:80],
        rationale="Fallback plan (planner did not return a structured plan).",
        subtasks=[sub],
        notes=["fallback"],
    )


def graph_from_metadata(meta: dict[str, Any]) -> TaskGraph | None:
    """Helper: orchestrator pulls the graph back out of an AgentOutput."""
    raw = meta.get("task_graph")
    if not raw:
        return None
    try:
        return TaskGraph.model_validate(raw if isinstance(raw, dict) else json.loads(raw))
    except Exception:  # noqa: BLE001
        return None
