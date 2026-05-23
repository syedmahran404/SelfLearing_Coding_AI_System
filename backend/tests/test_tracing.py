"""Tracing pub/sub + per-trace subscription."""
from __future__ import annotations

import asyncio

import pytest

from app.observability.tracing import SpanKind, Tracer, new_trace_id, trace_span


@pytest.mark.asyncio
async def test_global_subscriber_receives_events(tracer: Tracer):
    received: list = []

    async def sub(ev):
        received.append(ev)

    tracer.add_global_subscriber(sub)
    async with trace_span(tracer, "x", SpanKind.SYSTEM):
        pass
    # Subscribers run as tasks — give them a tick to drain.
    await asyncio.sleep(0.05)
    kinds = {ev.phase.value for ev in received}
    assert "start" in kinds and "end" in kinds


@pytest.mark.asyncio
async def test_per_trace_subscription_filters_by_id():
    t = Tracer()
    target_trace = new_trace_id()
    other_trace = new_trace_id()
    received: list = []

    async with t.subscribe_trace(target_trace) as q:
        # Emit two events: one matching, one not.
        from app.observability.tracing import SpanPhase, TraceEvent

        await t.emit(
            TraceEvent(
                trace_id=target_trace,
                span_id="a",
                parent_span_id=None,
                kind=SpanKind.SYSTEM,
                name="hit",
                phase=SpanPhase.START,
            )
        )
        await t.emit(
            TraceEvent(
                trace_id=other_trace,
                span_id="b",
                parent_span_id=None,
                kind=SpanKind.SYSTEM,
                name="miss",
                phase=SpanPhase.START,
            )
        )
        await asyncio.sleep(0.05)
        # Drain queue.
        while not q.empty():
            received.append(q.get_nowait())

    assert all(ev.trace_id == target_trace for ev in received)
    assert any(ev.name == "hit" for ev in received)
