"""Code execution tools (Python today; structured to add others).

`python_exec` writes the supplied source to `workdir/_run.py` and runs it
with the system's interpreter inside the sandbox. The executed program
sees only the workdir; rlimits and wallclock apply.

Stdin is supported (small payloads only — not stream-friendly inside
asyncio for v1).

Future: `js_exec` (node), `ts_exec` (tsx), `bash_exec` already covered by
`shell`.
"""
from __future__ import annotations

from app.tools.base import BaseTool, Permission, ToolInput, ToolResult, ToolSchema
from app.tools.sandbox import Sandbox, python_interpreter

_PYTHON_DENYLIST_TOKENS = (
    "subprocess.Popen(['rm",
    "os.system('rm",
    "shutil.rmtree('/'",
)


class PythonCodeTool(BaseTool):
    """Run a small Python program in the sandbox."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox
        self._interp = python_interpreter()

    schema = ToolSchema(
        name="python_exec",
        description=(
            "Run a small Python script inside the sandbox workdir. "
            "Provide source as a string. Optional stdin input. "
            "Resource limits and wallclock timeout apply."
        ),
        permissions=[Permission.EXEC, Permission.READ, Permission.WRITE],
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "stdin": {"type": "string"},
                "timeout_s": {"type": "integer"},
                "args": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["source"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
                "timed_out": {"type": "boolean"},
            },
        },
        default_timeout_s=20,
        safe_default=False,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        source = str(ti.args["source"])
        # Cheap heuristic guard against the obvious unsafe tropes.
        for tok in _PYTHON_DENYLIST_TOKENS:
            if tok in source:
                return ToolResult(
                    ok=False, error=f"refused: source contains denylisted pattern: {tok!r}"
                )
        timeout_s = int(ti.args.get("timeout_s") or ti.timeout_s)
        argv_extra = list(ti.args.get("args") or [])
        stdin_data = (ti.args.get("stdin") or "").encode("utf-8") if "stdin" in ti.args else None

        # Write the source to disk so tracebacks reference a real file.
        source_path = ti.workdir / "_run.py"
        try:
            source_path.write_text(source, encoding="utf-8")
        except OSError as exc:
            return ToolResult(ok=False, error=f"could not stage source: {exc}")

        sr = await self._sandbox.run(
            [self._interp, "-I", str(source_path), *argv_extra],
            cwd=ti.workdir,
            timeout_s=timeout_s,
            input_data=stdin_data,
        )
        return ToolResult(
            ok=sr.exit_code == 0 and not sr.timed_out,
            output={
                "stdout": sr.stdout,
                "stderr": sr.stderr,
                "exit_code": sr.exit_code,
                "timed_out": sr.timed_out,
            },
            stdout=sr.stdout,
            stderr=sr.stderr,
            exit_code=sr.exit_code,
            duration_ms=sr.duration_ms,
            error=("timed out" if sr.timed_out else None),
        )
