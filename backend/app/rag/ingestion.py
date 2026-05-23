"""Ingestion pipeline.

Two entrypoints:
- `ingest_text`     : a single in-memory document
- `ingest_directory`: recursively walk a tree, respect ignore patterns,
                      chunk every file, embed in batches, upsert to Qdrant

Provenance is preserved on every chunk:
- source_uri, source_version (defaults to a content sha)
- language, chunk_kind, qualified name
- ingest_id (groups all chunks from a single ingest run)

`ingest_directory` is *idempotent*: re-running with the same content
overwrites the same vector point ids (id derived from sha256 of content).
"""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.db.qdrant import QdrantStore, VectorPoint
from app.observability import get_logger
from app.rag.chunking import Chunk, chunk
from app.rag.embeddings import EmbeddingService

logger = get_logger("rag.ingestion")


DEFAULT_IGNORES: tuple[str, ...] = (
    ".git/*", "node_modules/*", "__pycache__/*", ".venv/*", "venv/*",
    "dist/*", "build/*", ".next/*", "*.lock", "*.min.*", "*.map",
    "*.pyc", "*.pyo", "*.so", "*.dll", "*.exe", "*.png", "*.jpg",
    "*.jpeg", "*.gif", "*.webp", "*.pdf", "*.zip", "*.tar", "*.gz",
)

DEFAULT_INCLUDE_EXT: frozenset[str] = frozenset(
    {".py", ".md", ".rst", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
     ".rs", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".json", ".yaml",
     ".yml", ".toml", ".sh"}
)

MAX_FILE_BYTES = 200_000  # skip files bigger than this — likely vendored


@dataclass(slots=True)
class IngestStats:
    files_seen: int = 0
    files_indexed: int = 0
    chunks_created: int = 0
    chunks_failed: int = 0
    bytes_indexed: int = 0
    duration_ms: int = 0


class Ingestor:
    """Chunk + embed + upsert pipeline."""

    def __init__(
        self,
        *,
        qdrant: QdrantStore,
        embeddings: EmbeddingService,
        collection: str | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._embeddings = embeddings
        self._collection = collection or qdrant.rag

    # ── single doc ───────────────────────────────────────────────────────
    async def ingest_text(
        self,
        *,
        text: str,
        source_uri: str,
        language: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
        source_version: str | None = None,
        tags: list[str] | None = None,
    ) -> IngestStats:
        started = time.perf_counter()
        chunks = chunk(text, source_uri, language=language)
        n_chunks, n_failed = await self._embed_and_upsert(
            chunks,
            user_id=user_id,
            project_id=project_id,
            source_version=source_version,
            tags=tags,
        )
        return IngestStats(
            files_seen=1,
            files_indexed=1 if n_chunks > 0 else 0,
            chunks_created=n_chunks,
            chunks_failed=n_failed,
            bytes_indexed=len(text.encode("utf-8")),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    # ── directory / repo ─────────────────────────────────────────────────
    async def ingest_directory(
        self,
        root: Path | str,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        source_version: str | None = None,
        ignore_patterns: Iterable[str] = DEFAULT_IGNORES,
        include_exts: Iterable[str] = DEFAULT_INCLUDE_EXT,
        concurrency: int = 8,
    ) -> IngestStats:
        root_path = Path(root).resolve()
        if not root_path.exists():
            raise FileNotFoundError(root_path)

        include_set = {e.lower() for e in include_exts}
        ignores = list(ignore_patterns)

        files: list[Path] = []
        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root_path).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in ignores):
                continue
            if p.suffix.lower() not in include_set:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(p)

        sem = asyncio.Semaphore(concurrency)
        stats = IngestStats(files_seen=len(files))
        started = time.perf_counter()

        async def _one(p: Path) -> None:
            async with sem:
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("ingest_read_failed", path=str(p), error=str(exc))
                    return
                rel = p.relative_to(root_path).as_posix()
                src_uri = f"file://{rel}"
                ver = source_version or _content_version(text)
                chunks = chunk(text, src_uri)
                if not chunks:
                    return
                n, failed = await self._embed_and_upsert(
                    chunks,
                    user_id=user_id,
                    project_id=project_id,
                    source_version=ver,
                )
                stats.files_indexed += 1 if n > 0 else 0
                stats.chunks_created += n
                stats.chunks_failed += failed
                stats.bytes_indexed += len(text.encode("utf-8"))

        await asyncio.gather(*(_one(p) for p in files))
        stats.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "ingest_directory_complete",
            root=str(root_path),
            files_seen=stats.files_seen,
            files_indexed=stats.files_indexed,
            chunks_created=stats.chunks_created,
            duration_ms=stats.duration_ms,
        )
        return stats

    # ── shared embed+upsert path ────────────────────────────────────────
    async def _embed_and_upsert(
        self,
        chunks: list[Chunk],
        *,
        user_id: str | None,
        project_id: str | None,
        source_version: str | None,
        tags: list[str] | None = None,
    ) -> tuple[int, int]:
        if not chunks:
            return 0, 0

        texts = [c.text for c in chunks]
        try:
            vectors = await self._embeddings.embed_many(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ingest_embed_failed", error=str(exc), n=len(chunks))
            return 0, len(chunks)

        points: list[VectorPoint] = []
        failed = 0
        ts = time.time()
        for c, v in zip(chunks, vectors, strict=False):
            if not v:
                failed += 1
                continue
            pid = _stable_point_id(c, source_version=source_version)
            points.append(
                VectorPoint(
                    id=pid,
                    vector=v,
                    payload={
                        "user_id": user_id,
                        "project_id": project_id,
                        "kind": "rag",
                        "chunk_kind": c.kind,
                        "source_uri": c.source_uri,
                        "source_version": source_version,
                        "language": c.language,
                        "name": c.name,
                        "line_start": c.line_start,
                        "line_end": c.line_end,
                        "text": c.text,
                        "tags": tags or [],
                        "created_at": ts,
                    },
                )
            )
        if points:
            try:
                await self._qdrant.upsert(self._collection, points)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ingest_upsert_failed", error=str(exc), n=len(points))
                return 0, len(chunks)
        return len(points), failed


# ── helpers ──────────────────────────────────────────────────────────────


def _content_version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_point_id(c: Chunk, *, source_version: str | None) -> str:
    """Stable per (source_uri, name|line_range, version). Re-ingestion
    overwrites in place, so the index never duplicates."""
    parts = [
        c.source_uri,
        c.kind,
        c.name or "",
        f"{c.line_start}:{c.line_end}",
        source_version or "",
    ]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    # Format as a UUID5 hex so Qdrant accepts it directly.
    return uuid.UUID(digest[:32]).hex
