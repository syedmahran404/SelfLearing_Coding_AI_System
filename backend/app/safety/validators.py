"""Output validators — structural checks before the system acts.

Cheap, deterministic, no LLM calls. Each function raises ValidationError on
failure or returns the cleaned input. Called by the orchestrator and route
handlers right after an agent produces output.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Iterable

from app.observability import get_logger
from app.schemas.agent_io import AgentOutput, CodeChange, ToolInvocation

logger = get_logger("safety.validators")


class ValidationError(ValueError):
    """Raised when an agent output fails structural validation."""


# ── path ─────────────────────────────────────────────────────────────────


_DENY_PATH_FRAGMENTS: tuple[str, ...] = ("..", "/etc/", "/root/", "/proc/", "/sys/")
_DENY_PATH_PREFIX: tuple[str, ...] = ("/", "~", "\\")


def validate_path(path: str) -> str:
    """Reject path-traversal and absolute paths. Returns the normalized path."""
    if not isinstance(path, str) or not path.strip():
        raise ValidationError("path must be a non-empty string")
    if any(path.startswith(p) for p in _DENY_PATH_PREFIX):
        raise ValidationError(f"absolute or home-relative paths are not allowed: {path!r}")
    if any(frag in path for frag in _DENY_PATH_FRAGMENTS):
        raise ValidationError(f"path contains a denied fragment: {path!r}")
    # Collapse to posix form to compare across OSes.
    normalized = str(PurePosixPath(path))
    if normalized.startswith("/") or normalized.startswith(".."):
        raise ValidationError(f"normalized path escapes workdir: {normalized!r}")
    return normalized


# ── code change ──────────────────────────────────────────────────────────


_VALID_OPS = {"create", "modify", "delete"}


def validate_code_change(c: CodeChange) -> CodeChange:
    if c.operation not in _VALID_OPS:
        raise ValidationError(f"invalid operation: {c.operation!r}")
    c.path = validate_path(c.path)
    if c.operation in {"create", "modify"} and not c.new_content and not c.diff:
        raise ValidationError(f"{c.operation} change requires new_content or diff: {c.path!r}")
    if c.new_content is not None and len(c.new_content) > 1_000_000:
        raise ValidationError(f"new_content too large: {len(c.new_content)} bytes")
    return c


# ── tool invocation ──────────────────────────────────────────────────────


def validate_tool_invocation(inv: ToolInvocation, *, allowed_tools: Iterable[str]) -> ToolInvocation:
    allowed = set(allowed_tools)
    if inv.tool not in allowed:
        raise ValidationError(f"tool {inv.tool!r} is not in the allowed set")
    if not isinstance(inv.args, dict):
        raise ValidationError("tool args must be an object")
    return inv


# ── agent output ─────────────────────────────────────────────────────────


def validate_agent_output(
    out: AgentOutput,
    *,
    allowed_tools: Iterable[str] = (),
    require_confidence: bool = True,
) -> AgentOutput:
    """Run all relevant checks on an agent's output. Returns it (mutated in
    place where helpful — paths normalized) if valid; raises otherwise."""
    if require_confidence:
        if not (0.0 <= out.confidence <= 1.0):
            raise ValidationError(f"confidence out of range: {out.confidence}")

    cleaned_changes: list[CodeChange] = []
    for c in out.code_changes:
        try:
            cleaned_changes.append(validate_code_change(c))
        except ValidationError as exc:
            logger.warning("code_change_rejected", path=c.path, error=str(exc))
            continue
    out.code_changes = cleaned_changes

    if allowed_tools:
        cleaned_calls: list[ToolInvocation] = []
        for tc in out.tool_calls:
            try:
                cleaned_calls.append(validate_tool_invocation(tc, allowed_tools=allowed_tools))
            except ValidationError as exc:
                logger.warning("tool_call_rejected", tool=tc.tool, error=str(exc))
                continue
        out.tool_calls = cleaned_calls
    return out


# ── memory write ────────────────────────────────────────────────────────


_VALID_MEMORY_KINDS = {
    "preference",
    "convention",
    "failure_rule",
    "success_rule",
    "fact",
    "lesson",
    "episode",
}


def validate_memory_write(content: str, kind: str) -> tuple[str, str]:
    if kind not in _VALID_MEMORY_KINDS:
        raise ValidationError(f"invalid memory kind: {kind!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValidationError("memory content must be a non-empty string")
    if len(content) > 8_000:
        raise ValidationError(f"memory content too large: {len(content)} bytes")
    return content.strip(), kind


# ── light static checks for generated code ──────────────────────────────


_TODO_RE = re.compile(r"\bTODO\(.*?\)|\bTODO\b|\bFIXME\b", re.IGNORECASE)


def code_health_score(text: str) -> float:
    """Heuristic health score in [0,1] for a code blob.

    Uses a tiny set of cheap signals:
    - trailing whitespace
    - too many TODO/FIXME markers
    - extreme line length
    """
    if not text:
        return 0.0
    score = 1.0
    todos = len(_TODO_RE.findall(text))
    if todos > 5:
        score -= 0.2
    elif todos > 0:
        score -= 0.05
    long_lines = sum(1 for line in text.splitlines() if len(line) > 200)
    if long_lines > 3:
        score -= 0.1
    trailing = sum(1 for line in text.splitlines() if line != line.rstrip())
    if trailing > 5:
        score -= 0.05
    return max(0.0, min(1.0, score))
