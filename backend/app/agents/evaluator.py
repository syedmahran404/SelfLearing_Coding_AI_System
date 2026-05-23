"""Evaluator — verdict on a subtask given its outputs and tool runs.

Two paths:
1. *Mechanical*: if recent tool runs include a `pytest_run` / `python_exec`
   with a clean exit, we can score with high confidence without an LLM.
2. *Judgment*: otherwise we ask the LLM, providing the success_predicate
   and the available evidence.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.agents.prompts import EVALUATOR
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, EvaluationResult
from app.tools.registry import ToolRegistry

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["passed", "score", "confidence"],
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number"},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "failures": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "object"},
    },
}


class EvaluatorAgent(BaseAgent):
    name = "evaluator"
    responsibility = "Decide whether a subtask passed; produce score + confidence."
    default_temperature = 0.0

    def __init__(self, *, llm, tools: ToolRegistry | None, tracer, settings) -> None:  # type: ignore[no-untyped-def]
        super().__init__(llm=llm, tracer=tracer, settings=settings)
        self._tools = tools  # currently unused; kept so we can run pytest if asked

    async def run(self, ai: AgentInput) -> AgentOutput:
        async with trace_span(
            self._tracer,
            "agent.evaluator.run",
            SpanKind.AGENT,
            payload={"subtask": (ai.subtask.title if ai.subtask else None)},
        ) as span:
            tool_runs = ai.extras.get("tool_runs") or []
            mechanical = _mechanical_verdict(tool_runs)
            if mechanical is not None:
                span["payload"]["path"] = "mechanical"
                return AgentOutput(
                    agent=self.name,
                    summary=("Subtask passed" if mechanical.passed else "Subtask failed"),
                    confidence=mechanical.confidence,
                    metadata={"evaluation": mechanical.model_dump(mode="json")},
                )

            # Fall back to LLM judgment.
            user = (
                f"USER_REQUEST:\n{ai.request}\n\n"
                f"SUBTASK:\n  title: {(ai.subtask.title if ai.subtask else '?')}\n"
                f"  description: {(ai.subtask.description if ai.subtask else '?')}\n"
                f"  success_predicate: {(ai.subtask.success_predicate if ai.subtask else '(none)')}\n\n"
                f"AGENT_OUTPUT_SUMMARY: {ai.extras.get('agent_output_summary', '')}\n\n"
                f"AGENT_ANSWER: {ai.extras.get('agent_answer', '')[:1500]}\n\n"
                f"TOOL_RUNS:\n{_format_tool_runs(tool_runs)}\n"
            )
            data, usage = await self._llm_json(
                system=EVALUATOR,
                user=user,
                schema=_SCHEMA,
                temperature=0.0,
                max_tokens=600,
                purpose="agent.evaluator",
            )
            span["tokens_in"] = usage["tokens_in"]
            span["tokens_out"] = usage["tokens_out"]
            span["cost_usd"] = usage["cost_usd"]
            span["payload"]["path"] = "llm"

            evald = EvaluationResult(
                subtask_id=(ai.subtask.id if ai.subtask else "?"),
                passed=bool(data.get("passed", False)),
                score=float(data.get("score", 0.0)),
                confidence=float(data.get("confidence", 0.4)),
                reasons=list(data.get("reasons") or []),
                failures=list(data.get("failures") or []),
                metrics={k: float(v) for k, v in (data.get("metrics") or {}).items() if isinstance(v, (int, float))},
            )
            return AgentOutput(
                agent=self.name,
                summary=("Subtask passed" if evald.passed else "Subtask failed"),
                confidence=evald.confidence,
                metadata={"evaluation": evald.model_dump(mode="json")},
            )


def _mechanical_verdict(tool_runs: list[dict[str, Any]]) -> EvaluationResult | None:
    """Return a verdict purely from tool output, when possible."""
    if not tool_runs:
        return None
    last_pytest = next(
        (r for r in reversed(tool_runs) if r.get("tool") == "pytest_run"), None
    )
    if last_pytest is not None:
        out = last_pytest.get("output") or {}
        passed_n = int(out.get("passed", 0))
        failed_n = int(out.get("failed", 0)) + int(out.get("errors", 0))
        ok = bool(last_pytest.get("ok")) and failed_n == 0 and passed_n > 0
        score = 1.0 if ok else (0.5 if passed_n and failed_n else 0.0)
        return EvaluationResult(
            subtask_id="?",
            passed=ok,
            score=score,
            confidence=0.92 if passed_n + failed_n > 0 else 0.5,
            reasons=[f"pytest: {passed_n} passed, {failed_n} failed/errored"],
            failures=[]
            if ok
            else [f"pytest reported {failed_n} failure(s)/error(s)"],
            metrics={"pytest_passed": float(passed_n), "pytest_failed": float(failed_n)},
        )

    last_pyexec = next(
        (r for r in reversed(tool_runs) if r.get("tool") == "python_exec"), None
    )
    if last_pyexec is not None:
        ok = bool(last_pyexec.get("ok"))
        return EvaluationResult(
            subtask_id="?",
            passed=ok,
            score=1.0 if ok else 0.0,
            confidence=0.85 if ok else 0.7,
            reasons=[f"python_exec exit={last_pyexec.get('exit_code')}"],
            failures=[]
            if ok
            else [(last_pyexec.get("error") or "non-zero exit")],
        )

    return None


def _format_tool_runs(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "(none)"
    lines: list[str] = []
    for r in runs[-6:]:
        ok = r.get("ok")
        tool = r.get("tool")
        lines.append(f"- {tool}: ok={ok} exit={r.get('exit_code')} err={r.get('error') or '-'}")
        out = r.get("output")
        if isinstance(out, dict) and out:
            for k in ("passed", "failed", "errors", "exit_code"):
                if k in out:
                    lines.append(f"    {k}: {out[k]}")
    return "\n".join(lines)
