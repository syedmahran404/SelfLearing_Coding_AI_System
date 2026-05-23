"""/memory — search + write + admin (lifecycle pass)."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import db_session, require_memory
from app.api.routes.projects import _resolve_user_id  # reuse demo-user helper
from app.bootstrap import Runtime
from app.db.models import Memory
from app.observability import get_logger
from app.safety.validators import validate_memory_write
from app.schemas.api import MemoryOut, MemoryQuery, MemoryWrite

logger = get_logger("api.memory")

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=list[MemoryOut])
async def search_memory(
    body: MemoryQuery,
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(require_memory),
) -> list[MemoryOut]:
    user_id = await _resolve_user_id(db)
    hits = await rt.memory.recall(  # type: ignore[union-attr]
        db,
        user_id=user_id,
        text=body.query,
        top_k=body.top_k,
        kind=body.kind,
        project_id=body.project_id,
    )
    out: list[MemoryOut] = []
    for h in hits:
        m = MemoryOut.model_validate(h.memory, from_attributes=True)
        m.score = h.blended_score
        out.append(m)
    return out


@router.post("", response_model=MemoryOut, status_code=201)
async def write_memory(
    body: MemoryWrite,
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(require_memory),
) -> MemoryOut:
    try:
        content, kind = validate_memory_write(body.content, body.kind)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc))

    user_id = await _resolve_user_id(db)
    mem, _created = await rt.memory.remember(  # type: ignore[union-attr]
        db,
        user_id=user_id,
        kind=kind,
        content=content,
        tags=body.tags,
        project_id=body.project_id,
        confidence=body.confidence,
    )
    return MemoryOut.model_validate(mem, from_attributes=True)


@router.get("", response_model=list[MemoryOut])
async def list_memory(
    project_id: uuid.UUID | None = None,
    kind: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(db_session),
) -> list[MemoryOut]:
    user_id = await _resolve_user_id(db)
    q = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.is_archived.is_(False))
        .order_by(desc(Memory.last_accessed_at))
        .limit(limit)
    )
    if kind is not None:
        q = q.where(Memory.kind == kind)
    if project_id is not None:
        q = q.where(Memory.project_id == project_id)
    rows = (await db.execute(q)).scalars().all()
    return [MemoryOut.model_validate(r, from_attributes=True) for r in rows]


@router.delete("/{memory_id}", status_code=204)
async def archive_memory(
    memory_id: uuid.UUID,
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(require_memory),
) -> None:
    await rt.memory.archive(db, memory_id)  # type: ignore[union-attr]


@router.post("/lifecycle/run")
async def run_lifecycle(
    rt: Runtime = Depends(require_memory),
) -> dict[str, Any]:
    """Admin: kick off a single dedup/decay/utility pass synchronously."""
    counts = await rt.memory.run_lifecycle_once()  # type: ignore[union-attr]
    return counts
