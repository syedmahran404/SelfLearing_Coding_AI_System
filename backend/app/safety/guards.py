"""Execution guards.

`ExecutionGuard` is a per-run object the orchestrator holds: it tracks
recursion depth, accumulated tokens, wallclock, and consecutive failures.
Each tick the orchestrator calls `check()` before recursing/retrying; the
guard returns a `GuardDecision` (allow | abort | back_off).

`CircuitBreaker` is global to the process and trips when the same
agent/intent fails repeatedly in a row. Tripped breakers refuse
invocations until they cool down — a brake against runaway loops.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from app.config import Settings
from app.observability import get_logger

logger = get_logger("safety.guards")


class GuardDecision(str, Enum):
    ALLOW = "allow"
    BACKOFF = "backoff"
    ABORT = "abort"


class GuardError(RuntimeError):
    """Raised by guards.assert_allow when the decision is ABORT."""


@dataclass(slots=True)
class ExecutionGuard:
    """Per-run execution caps."""

    max_depth: int
    max_tokens: int
    max_wallclock_s: int
    started_at: float = field(default_factory=time.time)
    depth: int = 0
    tokens_used: int = 0
    consecutive_failures: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> ExecutionGuard:
        return cls(
            max_depth=settings.safety_max_recursion_depth,
            max_tokens=settings.llm_max_tokens_per_task,
            # Hard ceiling: 30 minutes per task. Below this, the LLM/tool timeouts
            # are usually the actual binding limit.
            max_wallclock_s=30 * 60,
        )

    # ── mutators ──
    def add_tokens(self, n: int) -> None:
        self.tokens_used += max(0, int(n))

    def push(self) -> None:
        self.depth += 1

    def pop(self) -> None:
        self.depth = max(0, self.depth - 1)

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0

    # ── checks ──
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def check(self) -> tuple[GuardDecision, str]:
        if self.tokens_used > self.max_tokens:
            return GuardDecision.ABORT, f"token budget exceeded ({self.tokens_used}/{self.max_tokens})"
        if self.depth > self.max_depth:
            return GuardDecision.ABORT, f"recursion depth exceeded ({self.depth}/{self.max_depth})"
        if self.elapsed_s() > self.max_wallclock_s:
            return GuardDecision.ABORT, f"wallclock exceeded ({self.elapsed_s():.0f}s)"
        if self.consecutive_failures >= 4:
            return GuardDecision.BACKOFF, f"{self.consecutive_failures} consecutive failures"
        return GuardDecision.ALLOW, "ok"

    def assert_allow(self) -> None:
        decision, reason = self.check()
        if decision == GuardDecision.ABORT:
            raise GuardError(reason)


# ── circuit breaker ─────────────────────────────────────────────────────


@dataclass(slots=True)
class _BreakerState:
    open_until: float = 0.0
    consecutive: int = 0


class CircuitBreaker:
    """Process-wide breaker keyed by `(agent, intent)`.

    Trips after `threshold` consecutive failures; stays open for
    `cooldown_s` seconds. While open, `is_open(...)` returns True and the
    caller should refuse the action. On success, the breaker resets.
    """

    def __init__(self, *, threshold: int = 4, cooldown_s: int = 60) -> None:
        self._threshold = max(1, int(threshold))
        self._cooldown = max(1, int(cooldown_s))
        self._states: dict[tuple[str, str], _BreakerState] = defaultdict(_BreakerState)

    def _key(self, agent: str, intent: str) -> tuple[str, str]:
        return (agent, intent)

    def is_open(self, agent: str, intent: str) -> bool:
        s = self._states[self._key(agent, intent)]
        return s.open_until > time.time()

    def record_failure(self, agent: str, intent: str) -> bool:
        """Returns True if the breaker just tripped."""
        s = self._states[self._key(agent, intent)]
        s.consecutive += 1
        if s.consecutive >= self._threshold:
            s.open_until = time.time() + self._cooldown
            s.consecutive = 0
            logger.warning("circuit_breaker_tripped", agent=agent, intent=intent, cooldown_s=self._cooldown)
            return True
        return False

    def record_success(self, agent: str, intent: str) -> None:
        s = self._states[self._key(agent, intent)]
        s.consecutive = 0
        s.open_until = 0.0

    def snapshot(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        now = time.time()
        for (agent, intent), s in self._states.items():
            out[f"{agent}:{intent}"] = {
                "consecutive": s.consecutive,
                "is_open": s.open_until > now,
                "open_until": s.open_until,
            }
        return out
