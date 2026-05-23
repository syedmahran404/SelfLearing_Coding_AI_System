"""Bounded shell tool.

`shell` runs a command via the sandbox. Notable constraints:
- argv form, not `bash -c "..."` — the LLM must commit to an argv list.
- denylist on argv[0] (e.g. `rm`, `dd`, `mkfs`, `chmod`, `sudo`, `curl`,
  `wget`) — defense in depth; sandbox already restricts FS to workdir but
  reckless commands inside the workdir can still ruin a task.
- forwarded stdout/stderr are bounded by the sandbox's standard truncation.
"""
from __future__ import annotations

from app.tools.base import BaseTool, Permission, ToolInput, ToolResult, ToolSchema
from app.tools.sandbox import Sandbox


_DENYLIST: frozenset[str] = frozenset(
    {
        "sudo", "su", "doas",
        "rm", "rmdir", "mv", "shred", "dd", "mkfs", "fdisk", "wipefs",
        "chmod", "chown", "chgrp",
        "curl", "wget", "ssh", "scp", "rsync",
        "kill", "killall", "pkill",
        "shutdown", "reboot", "halt", "poweroff",
    }
)


class ShellTool(BaseTool):
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    schema = ToolSchema(
        name="shell",
        description=(
            "Run a command (argv list) inside the workdir, with CPU/memory/wallclock caps. "
            "Network is denied. Common destructive commands are blocked."
        ),
        permissions=[Permission.EXEC, Permission.READ, Permission.WRITE],
        input_schema={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "timeout_s": {"type": "integer"},
            },
            "required": ["argv"],
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
        argv = list(ti.args.get("argv") or [])
        if not argv or not isinstance(argv[0], str):
            return ToolResult(ok=False, error="argv must be a non-empty list of strings")
        head = argv[0].strip().split("/")[-1].lower()
        if head in _DENYLIST:
            return ToolResult(ok=False, error=f"command not allowed: {head!r}")
        timeout_s = int(ti.args.get("timeout_s") or ti.timeout_s)

        sr = await self._sandbox.run(argv, cwd=ti.workdir, timeout_s=timeout_s)
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
