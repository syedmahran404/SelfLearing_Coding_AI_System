"""Repo analyzer tool — a fast, dependency-free overview of a directory.

Returns:
- file count, byte total
- per-language file/byte breakdown
- top-level directory layout (depth 1)
- a short list of "anchor" files we always surface (README, pyproject,
  package.json, Cargo.toml, etc.)

This tool is intentionally cheap; the deeper, AST-driven understanding
lives in `app.project.indexer`.
"""
from __future__ import annotations

import fnmatch
from collections import defaultdict
from pathlib import Path

from app.rag.chunking import _detect_language as detect_language  # type: ignore[attr-defined]
from app.tools.base import BaseTool, Permission, ToolError, ToolInput, ToolResult, ToolSchema


_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next"}
_ANCHOR_NAMES: tuple[str, ...] = (
    "README.md", "README.rst", "README", "readme.md",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "package-lock.json", "yarn.lock",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "Dockerfile", "docker-compose.yml", "Makefile",
    ".env.example", "ARCHITECTURE.md",
)
_IGNORE_PATTERNS = ("*.lock", "*.min.*", "*.map", "*.png", "*.jpg", "*.pdf", "*.zip")


class RepoAnalyzerTool(BaseTool):
    schema = ToolSchema(
        name="repo_analyzer",
        description=(
            "Quickly summarize a directory: file counts by language, top-level layout, "
            "and contents of well-known anchor files."
        ),
        permissions=[Permission.READ],
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "anchor_max_chars": {"type": "integer"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "root": {"type": "string"},
                "totals": {"type": "object"},
                "languages": {"type": "object"},
                "top_level": {"type": "array", "items": {"type": "string"}},
                "anchors": {"type": "object"},
            },
        },
        default_timeout_s=10,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        try:
            sub = ti.args.get("path", ".")
            target = self.safe_path(ti.workdir, sub)
            if not target.exists() or not target.is_dir():
                return ToolResult(ok=False, error=f"not a directory: {sub!r}")
            anchor_max = int(ti.args.get("anchor_max_chars", 4000))

            file_count = 0
            byte_total = 0
            by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})

            for p in target.rglob("*"):
                if p.is_dir():
                    if p.name in _SKIP_DIRS:
                        # Skip subtree by clearing iter via filter — rglob doesn't
                        # support that; we just continue (files inside still get
                        # examined; cost is negligible for typical repos).
                        continue
                    continue
                if not p.is_file():
                    continue
                rel = p.relative_to(target).as_posix()
                if any(part in _SKIP_DIRS for part in p.parts):
                    continue
                if any(fnmatch.fnmatch(rel, pat) for pat in _IGNORE_PATTERNS):
                    continue
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                lang = detect_language(rel)
                file_count += 1
                byte_total += sz
                by_lang[lang]["files"] += 1
                by_lang[lang]["bytes"] += sz

            top_level: list[str] = []
            for child in sorted(target.iterdir(), key=lambda x: x.name.lower()):
                if child.name.startswith(".") and child.name != ".env.example":
                    continue
                top_level.append(child.name + ("/" if child.is_dir() else ""))

            anchors: dict[str, str] = {}
            for name in _ANCHOR_NAMES:
                ap = target / name
                if ap.exists() and ap.is_file():
                    try:
                        text = ap.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    if len(text) > anchor_max:
                        text = text[:anchor_max] + "\n…[truncated]"
                    anchors[name] = text

            return ToolResult(
                ok=True,
                output={
                    "root": str(target),
                    "totals": {"files": file_count, "bytes": byte_total},
                    "languages": dict(by_lang),
                    "top_level": top_level,
                    "anchors": anchors,
                },
            )
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc))
