"""Tool primitives: schema, permissions, base class, errors.

Design constraints
------------------
- Every tool declares a static `schema` (JSON-schema-ish) so the Planner
  knows *exactly* what arguments are valid. The orchestrator validates
  against this schema before calling.
- Every tool declares its permissions. The sandbox enforces them.
- Tools NEVER receive the application's settings/db. They receive only
  what's in `ToolInput.args` plus a per-run `workdir` they can write to.
- Tools should not raise; they should return a `ToolResult` with `ok=False`
  and a clean `error`. Surprises become failures, not crashes.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Permission(str, Enum):
    """Capabilities a tool may request."""

    READ = "read"      # may read inside workdir
    WRITE = "write"    # may write inside workdir
    EXEC = "exec"      # may spawn a subprocess
    NETWORK = "network"  # may make outbound network calls


class ToolError(Exception):
    """Raised internally; tools should *not* let this escape — convert to
    `ToolResult(ok=False, error=...)` at their boundary."""


@dataclass(slots=True)
class ToolSchema:
    """Static metadata for one tool."""

    name: str
    description: str
    permissions: list[Permission]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    default_timeout_s: int = 30
    safe_default: bool = True  # may be invoked without explicit user approval


@dataclass(slots=True)
class ToolInput:
    """Validated argument bundle for one tool invocation."""

    args: dict[str, Any]
    workdir: Path
    timeout_s: int
    trace_id: str | None = None


@dataclass(slots=True)
class ToolResult:
    """Uniform tool output — easy to log, store, and trace."""

    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "stdout": self.stdout[-2000:],  # keep payloads bounded
            "stderr": self.stderr[-2000:],
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class BaseTool(ABC):
    """All tools inherit from this."""

    schema: ToolSchema  # set on subclass

    @abstractmethod
    async def run(self, ti: ToolInput) -> ToolResult: ...

    # ── helpers all tools may use ──
    @staticmethod
    def safe_path(workdir: Path, candidate: str) -> Path:
        """Resolve `candidate` inside `workdir`. Reject path traversal."""
        wd = workdir.resolve()
        target = (wd / candidate).resolve()
        try:
            target.relative_to(wd)
        except ValueError as exc:  # not inside workdir
            raise ToolError(f"path escapes workdir: {candidate!r}") from exc
        return target

    @staticmethod
    def coerce_args(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        """Lightweight argument check: required fields exist, primitive types match.

        We avoid pulling in jsonschema to keep deps small; this catches the
        common errors that LLMs make (missing field, string-where-number).
        """
        out = dict(args)
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for r in required:
            if r not in out:
                raise ToolError(f"missing required arg: {r!r}")
        for k, sub in props.items():
            if k not in out:
                continue
            t = sub.get("type")
            v = out[k]
            if t == "string" and not isinstance(v, str):
                out[k] = str(v)
            elif t == "integer" and not isinstance(v, int):
                try:
                    out[k] = int(v)
                except (TypeError, ValueError) as exc:
                    raise ToolError(f"arg {k!r} not coercible to int") from exc
            elif t == "number" and not isinstance(v, (int, float)):
                try:
                    out[k] = float(v)
                except (TypeError, ValueError) as exc:
                    raise ToolError(f"arg {k!r} not coercible to number") from exc
            elif t == "boolean" and not isinstance(v, bool):
                if isinstance(v, str):
                    out[k] = v.lower() in {"true", "1", "yes", "y"}
                else:
                    out[k] = bool(v)
            elif t == "array" and not isinstance(v, list):
                raise ToolError(f"arg {k!r} must be an array")
            elif t == "object" and not isinstance(v, dict):
                raise ToolError(f"arg {k!r} must be an object")
        return out

    @staticmethod
    def safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            return str(value)
