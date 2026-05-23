"""Common FastAPI dependencies (the things every route needs)."""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.bootstrap import Runtime
from app.db.session import get_db_session


def get_runtime(request: Request) -> Runtime:
    rt: Runtime | None = getattr(request.app.state, "runtime", None)
    if rt is None:
        raise HTTPException(status_code=503, detail="runtime not ready")
    return rt


async def db_session() -> AsyncIterator[AsyncSession]:
    async for s in get_db_session():
        yield s


def require_orchestrator(rt: Runtime = Depends(get_runtime)) -> Runtime:
    if rt.orchestrator is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialized")
    return rt


def require_memory(rt: Runtime = Depends(get_runtime)) -> Runtime:
    if rt.memory is None:
        raise HTTPException(status_code=503, detail="memory subsystem not initialized")
    return rt


def require_tools(rt: Runtime = Depends(get_runtime)) -> Runtime:
    if rt.tools is None:
        raise HTTPException(status_code=503, detail="tools registry not initialized")
    return rt


def require_rag(rt: Runtime = Depends(get_runtime)) -> Runtime:
    if rt.rag is None:
        raise HTTPException(status_code=503, detail="rag service not initialized")
    return rt
