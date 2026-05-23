"""Run state types.

The orchestrator's mutable working set. Captured at structured points so
on a process restart we could resume from a checkpoint (future work — the
schema is here, the persistence hook is a TODO).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from app.schemas.agent_io import AgentOutput, Subtask, TaskGraph, TaskIntent


class RunStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    REFLECTING = "reflecting"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class SubtaskState:
    subtask: Subtask
    status: SubtaskStatus = SubtaskStatus.PENDING
    attempts: int = 0
    last_output: AgentOutput | None = None
    last_evaluation: dict[str, Any] | None = None
    last_reflection: dict[str, Any] | None = None
    last_tool_runs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RunState:
    run_id: str
    trace_id: str
    request: str
    intent: TaskIntent
    user_id: UUID | None = None
    project_id: UUID | None = None
    session_id: UUID | None = None
    episode_id: UUID | None = None
    status: RunStatus = RunStatus.PENDING
    plan: TaskGraph | None = None
    subtasks: list[SubtaskState] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    final_answer: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    confidence: float = 0.0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ── recursion / runaway-loop guards (see ARCHITECTURE.md §11) ──
    # Number of subtasks the orchestrator has *processed* so far.
    iterations: int = 0
    # Hard ceiling on `iterations`. Set by the orchestrator at run-start from
    # `safety_max_recursion_depth` and the initial plan size.
    max_iterations: int = 32
    # Remaining budget for reflector-driven `_insert_subtasks` calls. When
    # zero, further reflections still run but new_subtasks are dropped (a
    # warning is appended to `notes`).
    insertions_remaining: int = 0
    # Streak of *finally-failed* subtasks. Resets on any pass.
    consecutive_subtask_failures: int = 0
    # Trip threshold for the streak. Reaching it aborts the run as PARTIAL
    # rather than draining the token budget.
    max_consecutive_failures: int = 4

    @classmethod
    def new(
        cls,
        request: str,
        *,
        intent: TaskIntent,
        trace_id: str,
        user_id: UUID | None = None,
        project_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> RunState:
        return cls(
            run_id=uuid.uuid4().hex,
            trace_id=trace_id,
            request=request,
            intent=intent,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
        )

    @property
    def duration_ms(self) -> int:
        end = self.finished_at if self.finished_at is not None else time.time()
        return int((end - self.started_at) * 1000)

    def record_action(self, kind: str, payload: dict[str, Any]) -> None:
        self.actions.append({"ts": time.time(), "kind": kind, "payload": payload})

    def add_tokens(self, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self.tokens_used += int(tokens_in) + int(tokens_out)
        self.cost_usd += float(cost_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "request": self.request,
            "intent": self.intent.value,
            "status": self.status.value,
            "user_id": str(self.user_id) if self.user_id else None,
            "project_id": str(self.project_id) if self.project_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "episode_id": str(self.episode_id) if self.episode_id else None,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "subtasks": [
                {
                    "id": s.subtask.id,
                    "title": s.subtask.title,
                    "agent": s.subtask.agent,
                    "status": s.status.value,
                    "attempts": s.attempts,
                }
                for s in self.subtasks
            ],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
            "confidence": self.confidence,
            "final_answer": self.final_answer,
            "failures": self.failures,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "insertions_remaining": self.insertions_remaining,
            "consecutive_subtask_failures": self.consecutive_subtask_failures,
            "notes": self.notes,
        }
