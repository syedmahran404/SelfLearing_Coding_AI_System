"""Debugger — root-cause + minimal patch."""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import DEBUGGER
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, CodeChange, ToolInvocation

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["root_cause", "patch", "confidence"],
    "properties": {
        "root_cause": {"type": "string"},
        "patch": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "operation"],
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["create", "modify", "delete"]},
                    "new_content": {"type": "string"},
                    "diff": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["tool"],
                "properties": {
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
    },
}


class DebuggerAgent(BaseAgent):
    name = "debugger"
    responsibility = "Diagnose failures and propose minimal validated fixes."
    default_temperature = 0.1

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.debugger.run",
            SpanKind.AGENT,
            payload={"intent": ai.intent.value},
        ) as span:
            error_blob = ai.extras.get("error") or ""
            user = (
                f"USER_REQUEST:\n{ai.request}\n\n"
                f"SUBTASK:\n"
                f"  title: {(ai.subtask.title if ai.subtask else '(none)')}\n"
                f"  description: {(ai.subtask.description if ai.subtask else '(none)')}\n\n"
                f"FAILURE_OBSERVED:\n{error_blob[:4000] if error_blob else '(no stack trace supplied)'}\n\n"
                f"[CONTEXT]\n{_format_ctx(ai)}\n"
            )
            data, usage = await self._llm_json(
                system=DEBUGGER,
                user=user,
                schema=_SCHEMA,
                model=self._model_for("coder"),
                temperature=0.1,
                max_tokens=1600,
                purpose="agent.debugger",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            patches = [
                _to_change(c) for c in (data.get("patch") or [])
            ]
            tools = [_to_tool(t) for t in (data.get("tool_calls") or [])]

            return AgentOutput(
                agent=self.name,
                summary=str(data.get("summary") or data.get("root_cause") or "Debugger produced a patch."),
                code_changes=[c for c in patches if c is not None],
                tool_calls=[t for t in tools if t is not None],
                confidence=float(data.get("confidence", 0.4)),
                metadata={"root_cause": data.get("root_cause", "")},
            )


def _format_ctx(ai: AgentInput) -> str:
    parts: list[str] = []
    if ai.context.lessons:
        parts.append("--- failure_rules / lessons ---")
        for c in ai.context.lessons[:8]:
            parts.append(f"- {c.text[:400]}")
    if ai.context.episodic:
        parts.append("--- past similar episodes ---")
        for c in ai.context.episodic[:4]:
            parts.append(f"- {c.text[:240]}")
    if ai.context.rag:
        parts.append("--- code / docs ---")
        for c in ai.context.rag[:6]:
            uri = (c.metadata or {}).get("source_uri", "?")
            parts.append(f"- {uri}: {c.text[:400]}")
    return "\n".join(parts) if parts else "(no retrieved context)"


def _to_change(d: dict[str, Any]) -> CodeChange | None:
    try:
        return CodeChange(
            path=str(d["path"]),
            operation=str(d["operation"]),  # type: ignore[arg-type]
            new_content=d.get("new_content"),
            diff=d.get("diff"),
            rationale=d.get("rationale"),
        )
    except Exception:  # noqa: BLE001
        return None


def _to_tool(d: dict[str, Any]) -> ToolInvocation | None:
    try:
        return ToolInvocation(
            tool=str(d["tool"]),
            args=dict(d.get("args") or {}),
            rationale=d.get("rationale"),
        )
    except Exception:  # noqa: BLE001
        return None
