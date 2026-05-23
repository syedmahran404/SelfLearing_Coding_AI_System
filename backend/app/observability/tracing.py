"""Tracing primitives.

A *trace* is a single user-driven run through the system (one chat turn).
A trace is composed of *spans*, each representing an operation: agent call,
tool run, LLM invocation, memory retrieval, etc.

Spans are emitted as `TraceEvent`s on a single async bus (`Tracer`), which
fans them out to:
- Postgres (the `traces` table) — for queryable history
- the SSE stream of the originating request — for live frontend rendering
- stdout — for local dev

The tracer is intentionally lightweight (no opentelemetry dependency); we
control the schema completely and keep it tightly coupled to our domain
(agent, tool, memory, llm, eval, reflect).
"""
from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable

# Context variables — propagate trace/span automatically across async tasks.
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "slcai_trace_id", default=None
)
_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "slcai_span_id", default=None
)


def new_trace_id() -> str:
    """Generate a new trace id (URL-safe)."""
    return uuid.uuid4().hex


def current_trace_id() -> str | None:
    return _trace_id_var.get()


def current_span_id() -> str | None:
    return _span_id_var.get()


class SpanKind(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    MEMORY = "memory"
    EVAL = "eval"
    REFLECT = "reflect"
    HTTP = "http"
    SYSTEM = "system"


class SpanPhase(str, Enum):
    START = "start"
    END = "end"
    ERROR = "error"
    EVENT = "event"  # discrete annotation within a span


@dataclass(slots=True)
class TraceEvent:
    """A single trace event (start / end / error / annotation of a span)."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    kind: SpanKind
    name: str
    phase: SpanPhase
    ts: float = field(default_factory=time.time)
    duration_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "name": self.name,
            "phase": self.phase.value,
            "ts": self.ts,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "payload": self.payload,
            "error": self.error,
        }


# A subscriber is any async callable that consumes events.
TraceSubscriber = Callable[[TraceEvent], Awaitable[None]]


class Tracer:
    """Async pub/sub for trace events.

    Subscribers are awaited concurrently. Slow subscribers do not block the
    hot path — `emit` schedules delivery on a background task and returns
    immediately.

    Per-trace queues are exposed via `subscribe_trace(trace_id)` so that the
    SSE endpoint of an in-flight request can stream only that trace's events.
    """

    def __init__(self) -> None:
        self._global_subs: list[TraceSubscriber] = []
        self._per_trace_queues: dict[str, list[asyncio.Queue[TraceEvent]]] = {}
        self._lock = asyncio.Lock()

    # ── subscription ──
    def add_global_subscriber(self, sub: TraceSubscriber) -> None:
        self._global_subs.append(sub)

    @asynccontextmanager
    async def subscribe_trace(self, trace_id: str) -> AsyncIterator[asyncio.Queue[TraceEvent]]:
        """Subscribe to events for a single trace. Use as an async context manager.

        Yields an unbounded queue. The caller must `get()` events. On exit
        the queue is unregistered automatically.
        """
        q: asyncio.Queue[TraceEvent] = asyncio.Queue()
        async with self._lock:
            self._per_trace_queues.setdefault(trace_id, []).append(q)
        try:
            yield q
        finally:
            async with self._lock:
                qs = self._per_trace_queues.get(trace_id)
                if qs is not None:
                    if q in qs:
                        qs.remove(q)
                    if not qs:
                        self._per_trace_queues.pop(trace_id, None)

    # ── emission ──
    async def emit(self, event: TraceEvent) -> None:
        """Fire-and-forget event emission. Never raises."""
        # per-trace queues
        for q in self._per_trace_queues.get(event.trace_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover — unbounded queue
                pass

        # global subscribers — schedule, don't await, so emit() stays fast.
        for sub in self._global_subs:
            asyncio.create_task(_safe_invoke(sub, event))


async def _safe_invoke(sub: TraceSubscriber, event: TraceEvent) -> None:
    """Run a subscriber, swallowing exceptions (logged but never propagated)."""
    try:
        await sub(event)
    except Exception as exc:  # noqa: BLE001
        # Avoid importing logger at module level (could create cycles).
        from app.observability.logger import get_logger

        get_logger("observability.tracer").warning(
            "trace_subscriber_failed",
            error=str(exc),
            event=event.to_dict(),
        )


# ── span helpers ──


@asynccontextmanager
async def trace_span(
    tracer: Tracer,
    name: str,
    kind: SpanKind,
    *,
    payload: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Open a trace span scoped to an `async with` block.

    Yields a mutable dict that the caller can populate before close — common
    fields it understands when finishing the span: `tokens_in`, `tokens_out`,
    `cost_usd`, `payload` (merged), `error`.

    Sets the span_id contextvar so nested spans inherit the parent.
    """
    trace_id = current_trace_id() or new_trace_id()
    parent_span_id = current_span_id()
    span_id = uuid.uuid4().hex

    # Bind for the duration of this span.
    trace_token = _trace_id_var.set(trace_id)
    span_token = _span_id_var.set(span_id)

    started = time.perf_counter()
    span_data: dict[str, Any] = {"payload": dict(payload or {})}

    await tracer.emit(
        TraceEvent(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            phase=SpanPhase.START,
            payload=dict(payload or {}),
        )
    )
    try:
        yield span_data
    except Exception as exc:  # noqa: BLE001
        await tracer.emit(
            TraceEvent(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                kind=kind,
                name=name,
                phase=SpanPhase.ERROR,
                duration_ms=(time.perf_counter() - started) * 1000,
                payload=span_data.get("payload") or {},
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        raise
    else:
        await tracer.emit(
            TraceEvent(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                kind=kind,
                name=name,
                phase=SpanPhase.END,
                duration_ms=(time.perf_counter() - started) * 1000,
                tokens_in=span_data.get("tokens_in"),
                tokens_out=span_data.get("tokens_out"),
                cost_usd=span_data.get("cost_usd"),
                payload=span_data.get("payload") or {},
            )
        )
    finally:
        _span_id_var.reset(span_token)
        _trace_id_var.reset(trace_token)


def bind_trace(trace_id: str) -> contextvars.Token[str | None]:
    """Bind a trace_id to the current async context. Returns a token to reset.

    Used by the HTTP layer at request entry: bind once, every nested call
    inherits it.
    """
    return _trace_id_var.set(trace_id)


def reset_trace(token: contextvars.Token[str | None]) -> None:
    _trace_id_var.reset(token)
