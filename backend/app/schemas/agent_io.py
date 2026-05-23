"""Typed I/O for agents and the orchestrator.

Why pydantic at this seam? Agents must produce machine-checkable output —
free-form strings turn into bugs. Each agent declares the output schema it
will produce, the LLM is asked to emit JSON that matches it, and the
orchestrator validates before acting.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── intent classification ──


class TaskIntent(str, Enum):
    QA = "qa"
    CODE = "code"
    DEBUG = "debug"
    REFACTOR = "refactor"
    RESEARCH = "research"
    EXPLAIN = "explain"


# ── plan / subtask ──


class Subtask(BaseModel):
    """An atomic, verifiable step in a plan."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    description: str
    agent: str  # which specialist owns this step
    depends_on: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    success_predicate: str | None = None  # human description of "done"
    required_tools: list[str] = Field(default_factory=list)
    max_attempts: int = 2


class TaskGraph(BaseModel):
    """A plan: a DAG of subtasks plus the original intent."""

    intent: TaskIntent
    title: str
    rationale: str
    subtasks: list[Subtask]
    notes: list[str] = Field(default_factory=list)


# ── budgeted context ──


class ContextChunk(BaseModel):
    """A single retrieved chunk: who, what, where, score."""

    source: str  # "memory" | "rag" | "episode" | "project" | "lesson" | "shortterm"
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BudgetedContext(BaseModel):
    """The exact context an agent receives.

    Each section has a token budget enforced upstream. The agent never has
    to think about prompt length itself.
    """

    short_term: list[ContextChunk] = Field(default_factory=list)
    long_term: list[ContextChunk] = Field(default_factory=list)
    episodic: list[ContextChunk] = Field(default_factory=list)
    project: list[ContextChunk] = Field(default_factory=list)
    lessons: list[ContextChunk] = Field(default_factory=list)
    rag: list[ContextChunk] = Field(default_factory=list)
    estimated_tokens: int = 0


# ── agent I/O envelope ──


class AgentInput(BaseModel):
    """What an agent receives from the orchestrator."""

    run_id: str
    trace_id: str
    user_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    session_id: uuid.UUID | None = None
    intent: TaskIntent
    request: str
    subtask: Subtask | None = None
    context: BudgetedContext = Field(default_factory=BudgetedContext)
    extras: dict[str, Any] = Field(default_factory=dict)


class CodeChange(BaseModel):
    """A single code change the agent wants to apply."""

    path: str
    operation: Literal["create", "modify", "delete"]
    new_content: str | None = None
    diff: str | None = None
    rationale: str | None = None


class ToolInvocation(BaseModel):
    """A pending tool call the agent wants the ToolExecutor to run."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class MemoryWriteRequest(BaseModel):
    """A memory write proposal emitted by the MemoryAgent."""

    kind: str
    content: str
    tags: list[str] = Field(default_factory=list)
    project_scoped: bool = False
    confidence: float = 0.5


class AgentOutput(BaseModel):
    """The structured envelope every agent returns."""

    agent: str
    summary: str = ""
    answer: str | None = None
    code_changes: list[CodeChange] = Field(default_factory=list)
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    memory_writes: list[MemoryWriteRequest] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_more_info: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── evaluation ──


class EvaluationResult(BaseModel):
    """Verdict on a subtask. The Evaluator is the only emitter."""

    subtask_id: str
    passed: bool
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class Reflection(BaseModel):
    """Root-cause analysis + strategy delta produced by the Reflector."""

    subtask_id: str
    root_cause: str
    contributing_factors: list[str] = Field(default_factory=list)
    strategy_delta: str
    new_subtasks: list[Subtask] = Field(default_factory=list)
    lesson: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
