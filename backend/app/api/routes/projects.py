"""/projects — CRUD + indexing."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import db_session, get_runtime
from app.bootstrap import Runtime
from app.db.models import Project, User
from app.observability import get_logger
from app.schemas.api import ProjectCreate, ProjectOut

logger = get_logger("api.projects")

router = APIRouter(prefix="/projects", tags=["projects"])


# ── ensure-or-create demo user (until auth lands) ──
async def _resolve_user_id(db: AsyncSession) -> uuid.UUID:
    """Single-tenant demo path: get-or-create a user with handle='demo'."""
    row = (
        await db.execute(select(User).where(User.handle == "demo"))
    ).scalar_one_or_none()
    if row is None:
        row = User(handle="demo", display_name="Demo User")
        db.add(row)
        await db.flush()
    return row.id


# ── routes ──
@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate, db: AsyncSession = Depends(db_session)
) -> Project:
    user_id = await _resolve_user_id(db)
    p = Project(
        owner_id=user_id,
        name=body.name,
        slug=body.slug,
        description=body.description,
        languages=body.languages,
        repo_root=body.repo_root,
    )
    db.add(p)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(409, f"project slug already exists: {body.slug!r}")
    return p


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(db_session),
) -> list[Project]:
    rows = (
        (
            await db.execute(
                select(Project).order_by(Project.updated_at.desc()).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: uuid.UUID, db: AsyncSession = Depends(db_session)
) -> Project:
    row = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "project not found")
    return row


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID, db: AsyncSession = Depends(db_session)
) -> None:
    row = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "project not found")
    await db.delete(row)


# ── indexing ──
@router.post("/{project_id}/index", response_model=dict)
async def index_project(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    row = (
        await db.execute(select(Project).where(Project.id == project_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "project not found")
    if rt.project_indexer is None:
        raise HTTPException(503, "project indexer not initialized")
    try:
        stats = await rt.project_indexer.index(db, project=row)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    out = {
        "files_seen": stats.files_seen,
        "files_indexed": stats.files_indexed,
        "symbols": stats.symbols,
        "edges_total": stats.edges_total,
        "edges_resolved": stats.edges_resolved,
        "parse_errors": stats.parse_errors,
        "duration_ms": stats.duration_ms,
    }
    # Best-effort RAG ingestion of the same root for non-Python languages.
    if rt.rag is not None and row.repo_root:
        try:
            r = await rt.rag.ingest_directory(
                Path(row.repo_root),
                user_id=row.owner_id,
                project_id=row.id,
            )
            out["rag_chunks"] = r.chunks_created
            out["rag_files_indexed"] = r.files_indexed
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_ingest_skipped", error=str(exc))
    return out


@router.get("/{project_id}/symbols")
async def search_symbols(
    project_id: uuid.UUID,
    q: str = Query(..., min_length=1),
    kind: str | None = None,
    top_k: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    if rt.project_indexer is None:
        raise HTTPException(503, "project indexer not initialized")
    rows = await rt.project_indexer.query(
        db, project_id=project_id, term=q, kind=kind, top_k=top_k
    )
    return {
        "results": [
            {
                "id": str(r.id),
                "name": r.name,
                "qualified_name": r.qualified_name,
                "kind": r.kind,
                "file_path": r.file_path,
                "lines": [r.line_start, r.line_end],
                "signature": r.signature,
            }
            for r in rows
        ]
    }


@router.get("/{project_id}/symbols/{symbol_id}/neighbors")
async def symbol_neighbors(
    project_id: uuid.UUID,
    symbol_id: uuid.UUID,
    depth: int = Query(default=1, ge=0, le=3),
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    if rt.project_indexer is None:
        raise HTTPException(503, "project indexer not initialized")
    return await rt.project_indexer.neighbors(db, symbol_id=symbol_id, depth=depth)


@router.get("/{project_id}/architecture")
async def project_architecture(
    project_id: uuid.UUID,
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    if rt.project_indexer is None:
        raise HTTPException(503, "project indexer not initialized")
    clusters = await rt.project_indexer.architecture_map(db, project_id=project_id)
    return {"clusters": clusters}
