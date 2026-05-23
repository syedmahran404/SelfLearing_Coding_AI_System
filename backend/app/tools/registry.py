"""Tool registry + default-set builder.

The orchestrator's ToolExecutor receives a `ToolRegistry` and is the only
caller that passes invocations through to tool `run()` methods. The
registry exposes:

- `register(tool)` — add a tool
- `get(name)`     — fetch a tool (raises if absent)
- `list()`        — names + schemas (used by the Planner prompt)
- `invoke(name, args, ...)` — full invocation pipeline:
       schema validation → permission check → workdir → run → result
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span
from app.tools.base import BaseTool, Permission, ToolError, ToolInput, ToolResult, ToolSchema
from app.tools.sandbox import Sandbox

logger = get_logger("tools.registry")


class ToolRegistry:
    """Maintains the canonical set of tools and runs invocations through
    a uniform pipeline (validate → sandbox → run → trace)."""

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        tracer: Tracer | None = None,
        allowed_permissions: list[Permission] | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._sandbox = sandbox
        self._tracer = tracer
        # Default policy: everything except network.
        self._allowed = set(
            allowed_permissions
            or [Permission.READ, Permission.WRITE, Permission.EXEC]
        )

    # ── registration ──
    def register(self, tool: BaseTool) -> None:
        name = tool.schema.name
        if name in self._tools:
            raise ValueError(f"tool already registered: {name!r}")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name!r}") from exc

    def list(self) -> list[ToolSchema]:
        return [t.schema for t in self._tools.values()]

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    # ── invocation pipeline ──
    async def invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        workdir: Path | None = None,
        timeout_s: int | None = None,
        trace_id: str | None = None,
    ) -> ToolResult:
        """Run a tool through the full safety pipeline.

        Always returns a ToolResult; never raises. (Exceptions from inside
        the tool are caught and converted.)
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name!r}")

        # Permission gate.
        for p in tool.schema.permissions:
            if p not in self._allowed:
                return ToolResult(
                    ok=False,
                    error=f"permission {p.value!r} not allowed by registry policy",
                )

        # Args validation.
        try:
            coerced = BaseTool.coerce_args(args, tool.schema.input_schema)
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))

        # Workdir scope.
        own_workdir = False
        if workdir is None:
            wd_ctx = self._sandbox.workdir(prefix=f"{name}_")
            wd: Path = await wd_ctx.__aenter__()
            own_workdir = True
        else:
            wd = workdir

        ti = ToolInput(
            args=coerced,
            workdir=wd,
            timeout_s=timeout_s or tool.schema.default_timeout_s,
            trace_id=trace_id,
        )

        started = time.perf_counter()
        try:
            if self._tracer is not None:
                async with trace_span(
                    self._tracer,
                    f"tool.{name}",
                    SpanKind.TOOL,
                    payload={"args_preview": _arg_preview(coerced)},
                ) as span:
                    res = await tool.run(ti)
                    span["payload"].update(
                        {"ok": res.ok, "exit_code": res.exit_code, "error": res.error}
                    )
            else:
                res = await tool.run(ti)
        except Exception as exc:  # noqa: BLE001
            res = ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        finally:
            if own_workdir:
                try:
                    await wd_ctx.__aexit__(None, None, None)  # type: ignore[unreachable]
                except Exception:  # noqa: BLE001
                    pass

        if res.duration_ms == 0:
            res.duration_ms = int((time.perf_counter() - started) * 1000)
        return res


def _arg_preview(args: dict[str, Any]) -> dict[str, Any]:
    """Don't shove large args into traces; truncate."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 120:
            out[k] = v[:120] + "…"
        else:
            out[k] = v
    return out


# ── default-set builder ─────────────────────────────────────────────────────


def build_default_registry(settings: Settings, *, tracer: Tracer | None = None) -> ToolRegistry:
    """Construct the default registry: file_ops, shell, code_exec, pytest_runner,
    repo_analyzer, docs_lookup, web_search."""
    sandbox = Sandbox.from_settings(settings)
    allowed = [Permission.READ, Permission.WRITE, Permission.EXEC]
    if settings.sandbox_network_default == "allow":
        allowed.append(Permission.NETWORK)
    reg = ToolRegistry(sandbox=sandbox, tracer=tracer, allowed_permissions=allowed)

    # Late imports to avoid cycles at package init.
    from app.tools.code_exec import PythonCodeTool
    from app.tools.file_ops import FileReadTool, FileWriteTool, FileListTool
    from app.tools.pytest_runner import PytestTool
    from app.tools.repo_analyzer import RepoAnalyzerTool
    from app.tools.shell import ShellTool

    reg.register(FileReadTool())
    reg.register(FileWriteTool(dryrun=settings.safety_dryrun_file_writes))
    reg.register(FileListTool())
    reg.register(ShellTool(sandbox))
    reg.register(PythonCodeTool(sandbox))
    reg.register(PytestTool(sandbox))
    reg.register(RepoAnalyzerTool())

    # web_search is registered only if network is enabled by policy.
    if Permission.NETWORK in allowed:
        from app.tools.web_search import WebSearchTool

        reg.register(WebSearchTool())

    return reg
