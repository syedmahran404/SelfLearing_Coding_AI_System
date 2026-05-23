"""Tool framework — the only path the system has to side effects.

A *Tool* is a permissioned operation with:
- a JSON-schema input contract
- a JSON-schema output contract
- a declared permission set (read | write | exec | network)
- a strict timeout

Tools are registered into a `ToolRegistry`. The orchestrator's
`ToolExecutor` agent is the *only* consumer of this registry — agents
emit `ToolInvocation`s, the executor validates+permission-checks before
running anything.
"""

from app.tools.base import (
    BaseTool,
    Permission,
    ToolError,
    ToolInput,
    ToolResult,
    ToolSchema,
)
from app.tools.registry import ToolRegistry, build_default_registry

__all__ = [
    "BaseTool",
    "Permission",
    "ToolError",
    "ToolInput",
    "ToolResult",
    "ToolSchema",
    "ToolRegistry",
    "build_default_registry",
]
