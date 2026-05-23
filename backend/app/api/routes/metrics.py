"""/metrics — process-wide observability snapshot."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/metrics", tags=["observability"])


@router.get("")
async def metrics(request: Request) -> dict[str, Any]:
    """Return the in-memory MetricsCollector snapshot.

    Includes per-kind and per-name counters (starts/ends/errors,
    tokens, cost, duration ms avg/max). Cheap to read; safe to expose
    behind your own auth in production.
    """
    collector = getattr(request.app.state, "metrics", None)
    if collector is None:
        return {"status": "metrics not initialized"}
    snap = collector.snapshot()
    return {
        "uptime_s": snap.uptime_s,
        "totals": snap.totals,
        "by_kind": snap.by_kind,
        "by_name": snap.by_name,
    }


@router.post("/reset")
async def reset_metrics(request: Request) -> dict[str, str]:
    """Admin-only: reset the in-memory aggregator. (Trace DB rows are kept.)"""
    collector = getattr(request.app.state, "metrics", None)
    if collector is not None:
        collector.reset()
    return {"status": "reset"}
