"""/traces — list past traces, fetch one, and stream a live trace."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.deps import db_session, get_runtime
from app.bootstrap import Runtime
from app.db.models import TraceRecord
from app.observability import get_logger

logger = get_logger("api.traces")

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("/recent")
async def recent_traces(
    limit: int = 50,
    db: AsyncSession = Depends(db_session),
) -> dict[str, Any]:
    """Return distinct recent trace_ids and a small summary per trace.

    Cheap aggregate built on top of `traces` rows. For real production this
    becomes a materialized view; the row-shape is already friendly.
    """
    rows = (
        (
            await db.execute(
                select(TraceRecord)
                .order_by(desc(TraceRecord.ts))
                .limit(limit * 16)
            )
        )
        .scalars()
        .all()
    )
    by_trace: dict[str, dict[str, Any]] = {}
    for r in rows:
        agg = by_trace.setdefault(
            r.trace_id,
            {
                "trace_id": r.trace_id,
                "ts_min": r.ts,
                "ts_max": r.ts,
                "spans": 0,
                "errors": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
                "kinds": set(),
            },
        )
        agg["spans"] += 1
        agg["ts_min"] = min(agg["ts_min"], r.ts)
        agg["ts_max"] = max(agg["ts_max"], r.ts)
        agg["errors"] += 1 if r.phase == "error" else 0
        agg["tokens_in"] += int(r.tokens_in or 0)
        agg["tokens_out"] += int(r.tokens_out or 0)
        agg["cost_usd"] += float(r.cost_usd or 0.0)
        agg["kinds"].add(r.kind)
        if len(by_trace) >= limit:
            break
    items = [
        {**v, "kinds": sorted(v["kinds"]), "duration_ms": int((v["ts_max"] - v["ts_min"]) * 1000)}
        for v in by_trace.values()
    ]
    items.sort(key=lambda x: x["ts_max"], reverse=True)
    return {"traces": items}


@router.get("/{trace_id}")
async def get_trace(
    trace_id: str, db: AsyncSession = Depends(db_session)
) -> dict[str, Any]:
    rows = (
        (
            await db.execute(
                select(TraceRecord)
                .where(TraceRecord.trace_id == trace_id)
                .order_by(asc(TraceRecord.ts))
            )
        )
        .scalars()
        .all()
    )
    return {
        "trace_id": trace_id,
        "events": [
            {
                "span_id": r.span_id,
                "parent_span_id": r.parent_span_id,
                "kind": r.kind,
                "name": r.name,
                "phase": r.phase,
                "ts": r.ts,
                "duration_ms": r.duration_ms,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "payload": r.payload,
                "error": r.error,
            }
            for r in rows
        ],
    }


@router.get("/{trace_id}/stream")
async def stream_trace(
    trace_id: str,
    request: Request,
    rt: Runtime = Depends(get_runtime),
) -> EventSourceResponse:
    """Live SSE feed of trace events for `trace_id` while the run is in flight."""

    async def gen():
        # Subscribe immediately; sender must have used this trace_id.
        async with rt.tracer.subscribe_trace(trace_id) as q:
            yield {"event": "open", "data": json.dumps({"trace_id": trace_id})}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {
                    "event": ev.kind.value,
                    "data": json.dumps(ev.to_dict(), ensure_ascii=False, default=str),
                }
                # End of run heuristic: a top-level "orchestrator.run" END span.
                if ev.name == "orchestrator.run" and ev.phase.value == "end":
                    yield {"event": "close", "data": "{}"}
                    return

    return EventSourceResponse(gen())
