"""ToolExecutor — runs ToolInvocations through the registry.

Pure dispatcher: no LLM call. Takes the upstream agent's `tool_calls`
(already provided in `ai.extras["tool_calls"]`) and runs them sequentially
inside a shared workdir. Records every run as a `tool_run` payload that
the orchestrator can persist as a `ToolRun` row.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.config import Settings
from app.observability import Tracer
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import AgentInput, AgentOutput, ToolInvocation
from app.tools.registry import ToolRegistry


class ToolExecutorAgent(BaseAgent):
    name = "tool_executor"
    responsibility = "Run validated tool invocations inside the sandbox."

    def __init__(self, *, tools: ToolRegistry, tracer: Tracer, settings: Settings) -> None:
        super().__init__(llm=None, tracer=tracer, settings=settings)
        self._tools = tools

    async def run(self, ai: AgentInput) -> AgentOutput:
        invocations: list[ToolInvocation] = []
        raw = ai.extras.get("tool_calls") or []
        for r in raw:
            try:
                if isinstance(r, ToolInvocation):
                    invocations.append(r)
                else:
                    invocations.append(ToolInvocation.model_validate(r))
            except Exception:  # noqa: BLE001
                continue

        if not invocations:
            return AgentOutput(
                agent=self.name,
                summary="No tool invocations supplied.",
                confidence=1.0,
            )

        async with trace_span(
            self._tracer,
            "agent.tool_executor.run",
            SpanKind.AGENT,
            payload={"count": len(invocations)},
        ) as span:
            results: list[dict] = []
            ok_count = 0
            # Single workdir for the whole batch so consecutive tools (e.g.
            # write file → run pytest) see each other's output.
            async with self._tools.sandbox.workdir(prefix="exec_") as wd:
                for inv in invocations:
                    res = await self._tools.invoke(
                        inv.tool,
                        inv.args,
                        workdir=wd,
                        trace_id=ai.trace_id,
                    )
                    if res.ok:
                        ok_count += 1
                    results.append(
                        {
                            "tool": inv.tool,
                            "args": inv.args,
                            "ok": res.ok,
                            "exit_code": res.exit_code,
                            "duration_ms": res.duration_ms,
                            "output": res.output,
                            "error": res.error,
                            "stdout": res.stdout[-1000:],
                            "stderr": res.stderr[-1000:],
                        }
                    )
            span["payload"]["ok_count"] = ok_count
            span["payload"]["fail_count"] = len(invocations) - ok_count

            confidence = ok_count / max(1, len(invocations))
            return AgentOutput(
                agent=self.name,
                summary=f"{ok_count}/{len(invocations)} tool runs succeeded.",
                confidence=confidence,
                metadata={"tool_runs": results},
            )
