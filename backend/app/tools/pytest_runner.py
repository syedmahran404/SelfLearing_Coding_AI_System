"""Pytest runner inside the sandbox.

Designed for the Evaluator agent: given test files in the workdir, run
pytest and parse the result. We rely on JSON Lines output via
`pytest-jsonreport` if installed; otherwise we fall back to parsing
pytest's standard summary line.
"""
from __future__ import annotations

import json
import re

from app.tools.base import BaseTool, Permission, ToolInput, ToolResult, ToolSchema
from app.tools.sandbox import Sandbox, python_interpreter

_SUMMARY = re.compile(
    r"(?P<failed>\d+)\s+failed.*?(?P<passed>\d+)\s+passed"
    r"|(?P<passed_only>\d+)\s+passed"
)


class PytestTool(BaseTool):
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox
        self._interp = python_interpreter()

    schema = ToolSchema(
        name="pytest_run",
        description=(
            "Run pytest inside the workdir. Optionally specify a target file or expression. "
            "Returns counts of passed/failed and the raw output."
        ),
        permissions=[Permission.EXEC, Permission.READ, Permission.WRITE],
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "expr": {"type": "string"},
                "timeout_s": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "passed": {"type": "integer"},
                "failed": {"type": "integer"},
                "errors": {"type": "integer"},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
            },
        },
        default_timeout_s=60,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        timeout_s = int(ti.args.get("timeout_s") or ti.timeout_s)
        target = ti.args.get("target")
        expr = ti.args.get("expr")
        argv = [self._interp, "-m", "pytest", "-q"]
        if target:
            argv.append(str(target))
        if expr:
            argv.extend(["-k", str(expr)])
        argv.extend(["--no-header", "--maxfail=5"])

        sr = await self._sandbox.run(argv, cwd=ti.workdir, timeout_s=timeout_s)
        passed, failed, errors = _parse_summary(sr.stdout + "\n" + sr.stderr)

        return ToolResult(
            ok=(sr.exit_code == 0 and not sr.timed_out and failed == 0 and errors == 0),
            output={
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "stdout": sr.stdout,
                "stderr": sr.stderr,
                "exit_code": sr.exit_code,
            },
            stdout=sr.stdout,
            stderr=sr.stderr,
            exit_code=sr.exit_code,
            duration_ms=sr.duration_ms,
            error=("timed out" if sr.timed_out else None),
        )


def _parse_summary(text: str) -> tuple[int, int, int]:
    """Best-effort parser for pytest's terminal summary line.

    Looks for "X passed", "Y failed", "Z errors" anywhere. Tolerant of
    color codes by working on the plain text stream.
    """
    passed = failed = errors = 0
    for m in re.finditer(r"(\d+)\s+(passed|failed|error[s]?)", text):
        n = int(m.group(1))
        kind = m.group(2)
        if kind == "passed":
            passed = max(passed, n)
        elif kind == "failed":
            failed = max(failed, n)
        else:
            errors = max(errors, n)
    return passed, failed, errors


def _try_json(s: str) -> dict | None:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None
