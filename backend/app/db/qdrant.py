"""Qdrant vector store wrapper.

Two collections are managed:

- `slcai_memories` — long-term semantic memories written by the MemoryAgent.
- `slcai_rag`      — ingested documentation and code for grounding.

Each collection is created on startup with the configured vector size. We
use cosine distance (the default for OpenAI / most sentence-transformer
embeddings).

The wrapper is intentionally narrow — only the operations our memory and
RAG layers need. Direct qdrant-client access is discouraged so swapping
backends (e.g. to pgvector) is a single-class change.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings
from app.observability.logger import get_logger

logger = get_logger("db.qdrant")


@dataclass(slots=True)
class VectorPoint:
    """A point we store in Qdrant: id, vector, and arbitrary JSON payload."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(slots=True)
class VectorHit:
    """A search result with score and payload."""

    id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    """Async Qdrant wrapper with two managed collections."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        memories_collection: str,
        rag_collection: str,
        vector_size: int,
    ) -> None:
        self._c = client
        self.memories = memories_collection
        self.rag = rag_collection
        self.vector_size = vector_size

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantStore:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False,
            timeout=30,
        )
        return cls(
            client,
            memories_collection=settings.qdrant_collection_memories,
            rag_collection=settings.qdrant_collection_rag,
            vector_size=settings.qdrant_vector_size,
        )

    async def healthcheck(self) -> bool:
        try:
            await self._c.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant_healthcheck_failed", error=str(exc))
            return False

    async def ensure_collections(self) -> None:
        """Create both collections if they don't exist. Idempotent."""
        for name in (self.memories, self.rag):
            await self._ensure_collection(name)

    async def _ensure_collection(self, name: str) -> None:
        try:
            existing = await self._c.get_collection(name)
            # Validate vector size matches; if not, log loudly.
            current = existing.config.params.vectors  # type: ignore[union-attr]
            size = current.size if hasattr(current, "size") else None
            if size is not None and size != self.vector_size:
                logger.warning(
                    "qdrant_collection_size_mismatch",
                    collection=name,
                    expected=self.vector_size,
                    actual=size,
                )
            return
        except (UnexpectedResponse, ValueError):
            pass
        except Exception as exc:  # noqa: BLE001
            # Treat "not found" specifically; rethrow others on actual ops.
            logger.info("qdrant_collection_check_miss", collection=name, error=str(exc))

        logger.info("qdrant_collection_create", collection=name, size=self.vector_size)
        await self._c.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=qm.Distance.COSINE,
            ),
        )
        # Index the most-filtered payload fields. Cheap and pays back fast.
        for field, schema in (
            ("user_id", qm.PayloadSchemaType.KEYWORD),
            ("project_id", qm.PayloadSchemaType.KEYWORD),
            ("kind", qm.PayloadSchemaType.KEYWORD),
            ("source_uri", qm.PayloadSchemaType.KEYWORD),
            ("created_at", qm.PayloadSchemaType.FLOAT),
        ):
            try:
                await self._c.create_payload_index(name, field_name=field, field_schema=schema)
            except Exception as exc:  # noqa: BLE001
                logger.debug("qdrant_index_create_skipped", field=field, error=str(exc))

    # ── upsert / delete ──
    async def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        qpoints = [
            qm.PointStruct(id=_to_qid(p.id), vector=p.vector, payload=p.payload)
            for p in points
        ]
        await self._c.upsert(collection_name=collection, points=qpoints, wait=True)

    async def delete(self, collection: str, ids: Sequence[str]) -> None:
        if not ids:
            return
        await self._c.delete(
            collection_name=collection,
            points_selector=qm.PointIdsList(points=[_to_qid(i) for i in ids]),
            wait=True,
        )

    # ── search ──
    async def search(
        self,
        collection: str,
        vector: list[float],
        *,
        top_k: int = 10,
        filter_must: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        flt = _build_filter(filter_must) if filter_must else None
        result = await self._c.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            query_filter=flt,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            VectorHit(id=str(r.id), score=float(r.score), payload=dict(r.payload or {}))
            for r in result
        ]

    async def count(self, collection: str, filter_must: dict[str, Any] | None = None) -> int:
        flt = _build_filter(filter_must) if filter_must else None
        res = await self._c.count(collection_name=collection, count_filter=flt, exact=True)
        return int(res.count)

    async def close(self) -> None:
        await self._c.close()


def _to_qid(s: str) -> str | int:
    """Qdrant accepts UUIDs or unsigned ints. We normalize string ids by hashing
    when they aren't already valid UUIDs."""
    try:
        uuid.UUID(s)
        return s
    except (ValueError, AttributeError):
        # Stable digest → UUID hex
        return uuid.uuid5(uuid.NAMESPACE_URL, s).hex


def _build_filter(must: dict[str, Any]) -> qm.Filter:
    conditions: list[qm.FieldCondition] = []
    for key, value in must.items():
        if isinstance(value, (list, tuple, set)):
            conditions.append(
                qm.FieldCondition(key=key, match=qm.MatchAny(any=list(value)))
            )
        else:
            conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
    return qm.Filter(must=conditions)


_qdrant_singleton: QdrantStore | None = None


def get_qdrant() -> QdrantStore:
    if _qdrant_singleton is None:
        raise RuntimeError("Qdrant store not initialized; call init_qdrant() first")
    return _qdrant_singleton


async def init_qdrant(settings: Settings) -> QdrantStore:
    global _qdrant_singleton
    if _qdrant_singleton is None:
        _qdrant_singleton = QdrantStore.from_settings(settings)
        await _qdrant_singleton.ensure_collections()
        logger.info("qdrant_init", url=settings.qdrant_url)
    return _qdrant_singleton


async def shutdown_qdrant() -> None:
    global _qdrant_singleton
    if _qdrant_singleton is not None:
        await _qdrant_singleton.close()
        _qdrant_singleton = None
        logger.info("qdrant_shutdown")
