"""Reflector — root-cause + strategy delta + lesson on failure.

The orchestrator invokes the Reflector when:
- a subtask's evaluation fails, OR
- confidence is below the safety floor.

Output is consumed by the orchestrator (strategy delta + new subtasks) and
by the MemoryAgent (lesson → long-term memory).
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import REFLECTOR
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, Reflection, Subtask

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["root_cause", "strategy_delta", "confidence"],
    "properties": {
        "root_cause": {"type": "string"},
        "contributing_factors": {"type": "array", "items": {"type": "string"}},
        "strategy_delta": {"type": "string"},
        "lesson": {"type": "string"},
        "new_subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "description", "agent"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "agent": {"type": "string", "enum": ["researcher", "coder", "debugger", "evaluator"]},
                    "success_predicate": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "number"},
    },
}


class ReflectorAgent(BaseAgent):
    name = "reflector"
    responsibility = "Diagnose failures, propose strategy delta, extract a lesson."
    default_temperature = 0.15

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.reflector.run",
            SpanKind.REFLECT,
            payload={"subtask": (ai.subtask.title if ai.subtask else None)},
        ) as span:
            failures = ai.extras.get("failures") or []
            agent_summary = ai.extras.get("agent_output_summary") or ""
            tool_runs = ai.extras.get("tool_runs") or []
            user = (
                f"USER_REQUEST:\n{ai.request}\n\n"
                f"SUBTASK:\n  title: {(ai.subtask.title if ai.subtask else '?')}\n"
                f"  description: {(ai.subtask.description if ai.subtask else '?')}\n\n"
                f"AGENT_SUMMARY:\n{agent_summary}\n\n"
                f"FAILURES:\n- " + "\n- ".join(failures or ["(no explicit failures recorded)"]) + "\n\n"
                f"TOOL_RUNS:\n{_format_tool_runs(tool_runs)}\n\n"
                f"PRIOR_LESSONS:\n{_format_lessons(ai)}\n"
            )
            data, usage = await self._llm_json(
                system=REFLECTOR,
                user=user,
                schema=_SCHEMA,
                temperature=0.15,
                max_tokens=900,
                purpose="agent.reflector",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            new_subs = []
            for s in (data.get("new_subtasks") or []):
                try:
                    new_subs.append(
                        Subtask(
                            title=str(s["title"]),
                            description=str(s["description"]),
                            agent=str(s.get("agent", "coder")),
                            success_predicate=s.get("success_predicate"),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue

            reflection = Reflection(
                subtask_id=(ai.subtask.id if ai.subtask else "?"),
                root_cause=str(data.get("root_cause") or "unknown"),
                contributing_factors=list(data.get("contributing_factors") or []),
                strategy_delta=str(data.get("strategy_delta") or ""),
                new_subtasks=new_subs,
                lesson=data.get("lesson"),
                confidence=float(data.get("confidence", 0.4)),
            )

            return AgentOutput(
                agent=self.name,
                summary=f"Reflection: {reflection.root_cause[:160]}",
                confidence=reflection.confidence,
                metadata={"reflection": reflection.model_dump(mode="json")},
            )


def _format_tool_runs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "(none)"
    out: list[str] = []
    for r in runs[-6:]:
        out.append(
            f"- {r.get('tool')}: ok={r.get('ok')} exit={r.get('exit_code')} "
            f"err={(r.get('error') or '-')[:200]}"
        )
        stderr = (r.get("stderr") or "")[-400:]
        if stderr:
            out.append(f"    stderr: {stderr}")
    return "\n".join(out)


def _format_lessons(ai: AgentInput) -> str:
    if not ai.context.lessons:
        return "(none)"
    return "\n".join(f"- {c.text[:300]}" for c in ai.context.lessons[:6])
