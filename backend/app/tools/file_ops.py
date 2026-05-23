"""File operations bounded to a workdir.

Three tools:
- `file_read`   : read a text file (size-capped, utf-8)
- `file_write`  : create/overwrite a text file (path-checked)
- `file_list`   : list paths under a directory (depth-capped)

`safe_path` enforces that the resolved path stays inside the workdir;
attempts to traverse out raise `ToolError` and return `ok=False`.

`file_write` honors `--dryrun`: when `safety_dryrun_file_writes=true`, it
records the requested write but does not touch disk. Useful when
exercising autonomous flows without persistence.
"""
from __future__ import annotations

from pathlib import Path

from app.tools.base import BaseTool, Permission, ToolError, ToolInput, ToolResult, ToolSchema

MAX_READ_BYTES = 256_000


class FileReadTool(BaseTool):
    schema = ToolSchema(
        name="file_read",
        description="Read a UTF-8 text file inside the workdir.",
        permissions=[Permission.READ],
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        output_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
        default_timeout_s=5,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        try:
            target = self.safe_path(ti.workdir, ti.args["path"])
            if not target.exists():
                return ToolResult(ok=False, error=f"not found: {ti.args['path']!r}")
            if not target.is_file():
                return ToolResult(ok=False, error=f"not a file: {ti.args['path']!r}")
            size = target.stat().st_size
            if size > MAX_READ_BYTES:
                return ToolResult(
                    ok=False,
                    error=f"file too large: {size} bytes (max {MAX_READ_BYTES})",
                )
            text = target.read_text(encoding="utf-8", errors="replace")
            return ToolResult(
                ok=True,
                output={"path": str(target.relative_to(ti.workdir)), "content": text},
            )
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))


class FileWriteTool(BaseTool):
    def __init__(self, *, dryrun: bool = False) -> None:
        self._dryrun = dryrun

    schema = ToolSchema(
        name="file_write",
        description="Create or overwrite a UTF-8 text file inside the workdir.",
        permissions=[Permission.WRITE],
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "create_dirs": {"type": "boolean"},
            },
            "required": ["path", "content"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "bytes": {"type": "integer"},
                "dryrun": {"type": "boolean"},
            },
        },
        default_timeout_s=5,
        safe_default=False,  # writes deserve a confidence check
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        try:
            target = self.safe_path(ti.workdir, ti.args["path"])
            content = ti.args["content"]
            create_dirs = bool(ti.args.get("create_dirs", True))
            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            n = len(content.encode("utf-8"))
            if self._dryrun:
                return ToolResult(
                    ok=True,
                    output={"path": ti.args["path"], "bytes": n, "dryrun": True},
                )
            target.write_text(content, encoding="utf-8")
            return ToolResult(
                ok=True,
                output={"path": ti.args["path"], "bytes": n, "dryrun": False},
            )
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
        except OSError as exc:
            return ToolResult(ok=False, error=f"OSError: {exc}")


class FileListTool(BaseTool):
    schema = ToolSchema(
        name="file_list",
        description="List files inside the workdir (depth-capped).",
        permissions=[Permission.READ],
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {"entries": {"type": "array", "items": {"type": "string"}}},
        },
        default_timeout_s=5,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        try:
            sub = ti.args.get("path", ".")
            target = self.safe_path(ti.workdir, sub)
            if not target.exists():
                return ToolResult(ok=True, output={"entries": []})
            max_depth = int(ti.args.get("max_depth", 4))
            entries: list[str] = []
            base = target.resolve()
            for p in _walk(base, max_depth=max_depth):
                rel = p.relative_to(ti.workdir.resolve()).as_posix()
                entries.append(rel + ("/" if p.is_dir() else ""))
            entries.sort()
            return ToolResult(ok=True, output={"entries": entries})
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))


def _walk(base: Path, *, max_depth: int) -> list[Path]:
    """Bounded recursive listing without symlink chasing."""
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        path, depth = stack.pop()
        try:
            for child in path.iterdir():
                if child.is_symlink():
                    continue
                out.append(child)
                if child.is_dir() and depth < max_depth:
                    stack.append((child, depth + 1))
        except (OSError, PermissionError):
            continue
    return out
