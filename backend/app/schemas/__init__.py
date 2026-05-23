"""Pydantic schemas — boundary types for HTTP + agent I/O.

ORM models (`app.db.models`) are not exposed directly to the network or to
agents. We map them through these schemas so the public contract is stable
even if the storage layer changes.
"""

from app.schemas.agent_io import (
    AgentInput,
    AgentOutput,
    BudgetedContext,
    EvaluationResult,
    Reflection,
    Subtask,
    TaskGraph,
    TaskIntent,
)
from app.schemas.api import (
    ChatRequest,
    ChatResponseChunk,
    MemoryQuery,
    MemoryWrite,
    ProjectCreate,
    ProjectOut,
    SessionCreate,
    SessionOut,
)

__all__ = [
    "AgentInput",
    "AgentOutput",
    "BudgetedContext",
    "EvaluationResult",
    "Reflection",
    "Subtask",
    "TaskGraph",
    "TaskIntent",
    "ChatRequest",
    "ChatResponseChunk",
    "MemoryQuery",
    "MemoryWrite",
    "ProjectCreate",
    "ProjectOut",
    "SessionCreate",
    "SessionOut",
]
