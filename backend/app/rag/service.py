"""RagService — public RAG facade.

Agents and the orchestrator only see this class. It owns:
- an `EmbeddingService` (cached, batched)
- an `Ingestor` (write path)
- a `HybridRetriever` (read path)

All ops are trace-spanned for observability.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.config import Settings
from app.db.qdrant import QdrantStore
from app.llm.provider import LLMProvider
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span
from app.rag.embeddings import EmbeddingService
from app.rag.ingestion import IngestStats, Ingestor
from app.rag.retriever import HybridRetriever, RagHit

logger = get_logger("rag.service")


class RagService:
    """Public RAG facade."""

    def __init__(
        self,
        *,
        settings: Settings,
        qdrant: QdrantStore,
        llm: LLMProvider,
        tracer: Tracer,
    ) -> None:
        self._settings = settings
        self._tracer = tracer
        self.embeddings = EmbeddingService(llm)
        self.ingestor = Ingestor(qdrant=qdrant, embeddings=self.embeddings)
        self.retriever = HybridRetriever(qdrant=qdrant, embeddings=self.embeddings)

    # ── ingestion ────────────────────────────────────────────────────────
    async def ingest_text(
        self,
        *,
        text: str,
        source_uri: str,
        language: str | None = None,
        user_id: UUID | None = None,
        project_id: UUID | None = None,
        source_version: str | None = None,
        tags: list[str] | None = None,
    ) -> IngestStats:
        async with trace_span(
            self._tracer,
            "rag.ingest_text",
            SpanKind.MEMORY,
            payload={"source_uri": source_uri, "len": len(text)},
        ) as span:
            stats = await self.ingestor.ingest_text(
                text=text,
                source_uri=source_uri,
                language=language,
                user_id=str(user_id) if user_id else None,
                project_id=str(project_id) if project_id else None,
                source_version=source_version,
                tags=tags,
            )
            span["payload"].update(
                {"chunks": stats.chunks_created, "failed": stats.chunks_failed}
            )
            return stats

    async def ingest_directory(
        self,
        root: Path | str,
        *,
        user_id: UUID | None = None,
        project_id: UUID | None = None,
        source_version: str | None = None,
    ) -> IngestStats:
        async with trace_span(
            self._tracer,
            "rag.ingest_directory",
            SpanKind.MEMORY,
            payload={"root": str(root)},
        ) as span:
            stats = await self.ingestor.ingest_directory(
                root,
                user_id=str(user_id) if user_id else None,
                project_id=str(project_id) if project_id else None,
                source_version=source_version,
            )
            span["payload"].update(
                {
                    "files_seen": stats.files_seen,
                    "files_indexed": stats.files_indexed,
                    "chunks": stats.chunks_created,
                }
            )
            return stats

    # ── retrieval ────────────────────────────────────────────────────────
    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        user_id: UUID | None = None,
        project_id: UUID | None = None,
        languages: list[str] | None = None,
        kind: str | None = None,
    ) -> list[RagHit]:
        async with trace_span(
            self._tracer,
            "rag.search",
            SpanKind.MEMORY,
            payload={"top_k": top_k, "kind": kind, "languages": languages},
        ) as span:
            hits = await self.retriever.search(
                query,
                top_k=top_k,
                user_id=str(user_id) if user_id else None,
                project_id=str(project_id) if project_id else None,
                languages=languages,
                kind=kind,
            )
            span["payload"]["hits"] = len(hits)
            return hits

    # ── lifecycle ────────────────────────────────────────────────────────
    async def shutdown(self) -> None:
        # Embedding cache is in-process; nothing to flush.
        return None
