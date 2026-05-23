"""Long-term semantic memory.

A Memory row is *durable* knowledge: distilled facts, user preferences,
project conventions, learned coding patterns, failure rules. It always has:

- a relational row in Postgres (canonical content + lifecycle counters)
- a vector point in Qdrant under `vector_id`

Operations:
- `write`        : create (with dedup on content_sha), embed, upsert
- `query`        : vector search → re-rank by utility, return rows
- `record_use`   : bump access_count, optionally success/failure
- `archive`      : soft-delete (kept for export/debug)

Retrieval re-ranking
--------------------
Vector similarity alone is misleading. We blend it with `utility_score`
(see `memory/utils.py`) — recency * frequency * success-ratio — so a stale
high-similarity memory loses ground to a fresh, frequently-helpful one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import Memory
from app.db.qdrant import QdrantStore, VectorPoint
from app.llm.provider import LLMProvider
from app.memory.utils import content_sha, normalize_content, utility_score
from app.observability import get_logger

logger = get_logger("memory.long_term")


@dataclass(slots=True)
class MemoryHit:
    """A retrieval result — the ORM row + vector score + blended score."""

    memory: Memory
    vector_score: float
    blended_score: float


class LongTermMemory:
    """Durable semantic-memory store on Postgres + Qdrant."""

    def __init__(
        self,
        *,
        settings: Settings,
        qdrant: QdrantStore,
        llm: LLMProvider,
    ) -> None:
        self._settings = settings
        self._qdrant = qdrant
        self._llm = llm
        self._collection = qdrant.memories

    # ── write ────────────────────────────────────────────────────────────
    async def write(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        kind: str,
        content: str,
        project_id: UUID | None = None,
        summary: str | None = None,
        tags: list[str] | None = None,
        source_uri: str | None = None,
        source_version: str | None = None,
        confidence: float = 0.5,
        extra: dict[str, Any] | None = None,
    ) -> tuple[Memory, bool]:
        """Insert a memory or return the existing one if a duplicate exists.

        Returns `(memory, created)`. Dedup is by `(user_id, content_sha)`.
        """
        sha = content_sha(content)
        existing = (
            await session.execute(
                select(Memory).where(Memory.user_id == user_id, Memory.content_sha == sha)
            )
        ).scalar_one_or_none()

        if existing is not None:
            # Dedup: bump utility a little and update tags/metadata softly.
            existing.access_count += 1
            existing.last_accessed_at = _utcnow()
            if tags:
                merged = list({*existing.tags, *tags})
                existing.tags = merged
            if confidence > existing.confidence:
                existing.confidence = confidence
            await session.flush()
            return existing, False

        # Embed + upsert vector. Failure to embed is non-fatal — we keep the
        # row, just without a vector_id (it will be retried by lifecycle).
        vector_id: str | None = None
        try:
            emb = await self._llm.embed([content])
            if emb.vectors:
                point_id = sha[:32]  # stable per content
                await self._qdrant.upsert(
                    self._collection,
                    [
                        VectorPoint(
                            id=point_id,
                            vector=emb.vectors[0],
                            payload={
                                "user_id": str(user_id),
                                "project_id": str(project_id) if project_id else None,
                                "kind": kind,
                                "tags": tags or [],
                                "source_uri": source_uri,
                                "created_at": time.time(),
                                "content_sha": sha,
                            },
                        )
                    ],
                )
                vector_id = point_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_embed_failed", error=str(exc))

        mem = Memory(
            user_id=user_id,
            project_id=project_id,
            kind=kind,
            content=normalize_content(content),
            summary=summary,
            tags=tags or [],
            source_uri=source_uri,
            source_version=source_version,
            vector_id=vector_id,
            content_sha=sha,
            confidence=confidence,
            utility=0.5,
            extra=extra or {},
        )
        session.add(mem)
        await session.flush()
        return mem, True

    # ── retrieval ───────────────────────────────────────────────────────
    async def query(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        text: str,
        top_k: int = 10,
        kind: str | None = None,
        project_id: UUID | None = None,
        recall_multiplier: int = 4,
        min_blended: float = 0.0,
    ) -> list[MemoryHit]:
        """Vector recall → blend with utility → return top_k.

        We over-recall by `recall_multiplier` to give the re-ranker headroom.
        """
        # Fast-path: empty store / no query.
        if not text:
            return []

        try:
            emb = await self._llm.embed([text])
            qvec = emb.vectors[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_query_embed_failed", error=str(exc))
            return []

        flt: dict[str, Any] = {"user_id": str(user_id)}
        if kind is not None:
            flt["kind"] = kind
        if project_id is not None:
            flt["project_id"] = str(project_id)

        hits = await self._qdrant.search(
            self._collection,
            qvec,
            top_k=top_k * recall_multiplier,
            filter_must=flt,
        )
        if not hits:
            return []

        # Resolve ORM rows by content_sha (stable across reindexes).
        shas = [h.payload.get("content_sha") for h in hits if h.payload.get("content_sha")]
        if not shas:
            return []
        rows = (
            (
                await session.execute(
                    select(Memory).where(
                        Memory.user_id == user_id,
                        Memory.content_sha.in_(shas),
                        Memory.is_archived.is_(False),
                    )
                )
            )
            .scalars()
            .all()
        )
        by_sha = {r.content_sha: r for r in rows}

        results: list[MemoryHit] = []
        for h in hits:
            sha = h.payload.get("content_sha")
            row = by_sha.get(sha)
            if row is None:
                continue
            util = utility_score(
                base=max(0.4, row.confidence),
                last_accessed_at=row.last_accessed_at,
                access_count=row.access_count,
                success_count=row.success_count,
                failure_count=row.failure_count,
                halflife_days=self._settings.memory_decay_halflife_days,
            )
            # Blend: 70% similarity, 30% utility. Calibrated empirically;
            # tweakable via env later if needed.
            blended = 0.70 * h.score + 0.30 * util
            if blended < min_blended:
                continue
            results.append(MemoryHit(memory=row, vector_score=h.score, blended_score=blended))

        results.sort(key=lambda m: m.blended_score, reverse=True)
        return results[:top_k]

    # ── usage tracking ──────────────────────────────────────────────────
    async def record_use(
        self,
        session: AsyncSession,
        *,
        memory_ids: list[UUID],
        success: bool | None = None,
    ) -> None:
        """Bump access_count and optionally success/failure for each id."""
        if not memory_ids:
            return
        rows = (
            (
                await session.execute(
                    select(Memory).where(Memory.id.in_(memory_ids))
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            row.access_count += 1
            row.last_accessed_at = _utcnow()
            if success is True:
                row.success_count += 1
                row.confidence = min(1.0, row.confidence + 0.02)
            elif success is False:
                row.failure_count += 1
                row.confidence = max(0.0, row.confidence - 0.05)
            row.utility = utility_score(
                base=max(0.4, row.confidence),
                last_accessed_at=row.last_accessed_at,
                access_count=row.access_count,
                success_count=row.success_count,
                failure_count=row.failure_count,
                halflife_days=self._settings.memory_decay_halflife_days,
            )
        await session.flush()

    # ── archive (soft delete) ───────────────────────────────────────────
    async def archive(self, session: AsyncSession, memory_id: UUID) -> None:
        row = (
            await session.execute(select(Memory).where(Memory.id == memory_id))
        ).scalar_one_or_none()
        if row is None:
            return
        row.is_archived = True
        if row.vector_id:
            try:
                await self._qdrant.delete(self._collection, [row.vector_id])
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory_vector_delete_failed", error=str(exc))
        await session.flush()


def _utcnow():  # type: ignore[no-untyped-def]
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
