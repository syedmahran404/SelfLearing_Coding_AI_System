"""Coder — writes / edits code structured as CodeChange + ToolInvocation."""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import CODER
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import (
    AgentInput,
    AgentOutput,
    CodeChange,
    ToolInvocation,
)


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "code_changes", "confidence"],
    "properties": {
        "summary": {"type": "string"},
        "code_changes": {
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
        "needs_more_info": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
}


class CoderAgent(BaseAgent):
    name = "coder"
    responsibility = "Write or edit code that satisfies the subtask."
    default_temperature = 0.2

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.coder.run",
            SpanKind.AGENT,
            payload={"intent": ai.intent.value},
        ) as span:
            user = (
                f"USER_REQUEST:\n{ai.request}\n\n"
                f"SUBTASK:\n"
                f"  title: {(ai.subtask.title if ai.subtask else '(none)')}\n"
                f"  description: {(ai.subtask.description if ai.subtask else '(none)')}\n"
                f"  success_predicate: {(ai.subtask.success_predicate if ai.subtask else '(none)')}\n"
                f"  required_tools: {(ai.subtask.required_tools if ai.subtask else [])}\n\n"
                f"[CONTEXT]\n{_format_code_context(ai)}\n"
            )
            data, usage = await self._llm_json(
                system=CODER,
                user=user,
                schema=_SCHEMA,
                model=self._model_for("coder"),
                temperature=0.2,
                max_tokens=2400,
                purpose="agent.coder",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]

            changes = [_to_change(c) for c in (data.get("code_changes") or [])]
            tools = [_to_tool(t) for t in (data.get("tool_calls") or [])]

            return AgentOutput(
                agent=self.name,
                summary=str(data.get("summary") or "Coder produced changes."),
                code_changes=[c for c in changes if c is not None],
                tool_calls=[t for t in tools if t is not None],
                needs_more_info=bool(data.get("needs_more_info", False)),
                confidence=float(data.get("confidence", 0.5)),
            )


def _format_code_context(ai: AgentInput) -> str:
    parts: list[str] = []
    if ai.context.lessons:
        parts.append("--- lessons ---")
        for c in ai.context.lessons[:6]:
            parts.append(f"- {c.text[:300]}")
    if ai.context.rag:
        parts.append("--- code chunks ---")
        for c in ai.context.rag[:8]:
            uri = (c.metadata or {}).get("source_uri", "?")
            lang = (c.metadata or {}).get("language", "?")
            parts.append(f"- {uri} [{lang}]\n{c.text[:600]}")
    if ai.context.long_term:
        parts.append("--- conventions / memory ---")
        for c in ai.context.long_term[:6]:
            parts.append(f"- {c.text[:240]}")
    if ai.context.short_term:
        parts.append("--- recent turns ---")
        for c in ai.context.short_term[-6:]:
            parts.append(f"- {c.text[:300]}")
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
