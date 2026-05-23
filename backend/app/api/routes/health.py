"""/health — liveness + readiness probes.

`liveness` always returns 200 if the process is alive.
`readiness` checks Postgres, Redis, Qdrant — used by docker compose's
healthcheck and orchestrators (k8s probes etc.).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.db.qdrant import get_qdrant
from app.db.redis_client import get_redis
from app.db.session import session_factory

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness — always 200 while the process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    """Readiness — verify each external dependency."""
    checks: dict[str, Any] = {}

    # Postgres
    try:
        async with session_factory()() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"fail: {exc}"

    # Redis
    try:
        checks["redis"] = "ok" if await get_redis().ping() else "fail"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"fail: {exc}"

    # Qdrant
    try:
        checks["qdrant"] = "ok" if await get_qdrant().healthcheck() else "fail"
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = f"fail: {exc}"

    # Runtime subsystems (informational, not gating).
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        checks["llm"] = "ok" if runtime.llm is not None else "absent"
        checks["agents"] = "ok" if runtime.agents is not None else "absent"
        checks["orchestrator"] = "ok" if runtime.orchestrator is not None else "absent"

    overall = "ok" if all(v == "ok" for k, v in checks.items() if k in {"postgres", "redis", "qdrant"}) else "degraded"
    return {"status": overall, "checks": checks}
