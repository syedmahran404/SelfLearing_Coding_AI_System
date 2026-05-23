"""MemoryService — the *only* memory entrypoint agents and the orchestrator use.

This is a thin facade that:
- Owns one short-term, one long-term, one episodic store.
- Provides a `Summarizer` and a `LifecycleWorker` (started lazily).
- Exposes per-operation methods that *do not* require callers to know the
  underlying storage split.

Sessions are passed in by callers (so the request-scoped DB session is used
for all writes), but background lifecycle gets its own session via the
factory.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.qdrant import QdrantStore
from app.db.redis_client import RedisClient
from app.db.session import session_factory
from app.llm.provider import LLMProvider
from app.memory.compression import Summarizer
from app.memory.episodic import EpisodeHit, EpisodicMemory
from app.memory.lifecycle import LifecycleWorker
from app.memory.long_term import LongTermMemory, MemoryHit
from app.memory.short_term import ShortTermMemory
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span

logger = get_logger("memory.service")


class MemoryService:
    """Public memory facade."""

    def __init__(
        self,
        *,
        settings: Settings,
        redis: RedisClient,
        qdrant: QdrantStore,
        llm: LLMProvider,
        tracer: Tracer,
    ) -> None:
        self._settings = settings
        self._tracer = tracer
        self.short = ShortTermMemory(redis)
        self.long = LongTermMemory(settings=settings, qdrant=qdrant, llm=llm)
        self.episodic = EpisodicMemory(qdrant=qdrant, llm=llm)
        self.summarizer = Summarizer(llm)
        self._worker: LifecycleWorker | None = None
        self._llm = llm
        self._qdrant = qdrant

    # ── lifecycle worker ──
    def start_worker(self) -> None:
        if self._worker is None:
            self._worker = LifecycleWorker(
                settings=self._settings,
                session_factory=session_factory(),
                qdrant=self._qdrant,
                llm=self._llm,
                summarizer=self.summarizer,
            )
        self._worker.start()

    async def shutdown(self) -> None:
        if self._worker is not None:
            await self._worker.shutdown()

    async def run_lifecycle_once(self) -> dict[str, int]:
        """Manual lifecycle pass — used by tests and admin endpoints."""
        if self._worker is None:
            self._worker = LifecycleWorker(
                settings=self._settings,
                session_factory=session_factory(),
                qdrant=self._qdrant,
                llm=self._llm,
                summarizer=self.summarizer,
            )
        return await self._worker.run_once()

    # ── short-term wrappers ──
    async def push_short_term(
        self,
        session_id: UUID | str,
        *,
        role: str,
        content: str,
        agent: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        await self.short.push(
            session_id,
            role=role,
            content=content,
            agent=agent,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def get_short_term(self, session_id: UUID | str) -> list[dict[str, Any]]:
        return await self.short.get(session_id)

    # ── long-term wrappers (each emits a trace span) ──
    async def remember(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        kind: str,
        content: str,
        project_id: UUID | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.5,
        source_uri: str | None = None,
        summary: str | None = None,
    ):
        async with trace_span(
            self._tracer,
            "memory.write",
            SpanKind.MEMORY,
            payload={"kind": kind, "len": len(content)},
        ) as span:
            mem, created = await self.long.write(
                db,
                user_id=user_id,
                kind=kind,
                content=content,
                project_id=project_id,
                tags=tags,
                confidence=confidence,
                source_uri=source_uri,
                summary=summary,
            )
            span["payload"].update({"id": str(mem.id), "created": created})
            return mem, created

    async def recall(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        text: str,
        top_k: int = 10,
        kind: str | None = None,
        project_id: UUID | None = None,
    ) -> list[MemoryHit]:
        async with trace_span(
            self._tracer,
            "memory.recall",
            SpanKind.MEMORY,
            payload={"top_k": top_k, "kind": kind},
        ) as span:
            hits = await self.long.query(
                db,
                user_id=user_id,
                text=text,
                top_k=top_k,
                kind=kind,
                project_id=project_id,
            )
            span["payload"]["hits"] = len(hits)
            return hits

    async def feedback(
        self, db: AsyncSession, *, memory_ids: list[UUID], success: bool | None
    ) -> None:
        await self.long.record_use(db, memory_ids=memory_ids, success=success)

    async def archive(self, db: AsyncSession, memory_id: UUID) -> None:
        await self.long.archive(db, memory_id)

    # ── episodic wrappers ──
    async def start_episode(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        intent: str,
        title: str,
        input_text: str,
        project_id: UUID | None = None,
        session_id: UUID | None = None,
        trace_id: str | None = None,
        plan: dict | None = None,
    ):
        async with trace_span(
            self._tracer, "memory.episode_start", SpanKind.MEMORY, payload={"intent": intent}
        ):
            return await self.episodic.start(
                db,
                user_id=user_id,
                intent=intent,
                title=title,
                input_text=input_text,
                project_id=project_id,
                session_id=session_id,
                trace_id=trace_id,
                plan=plan,
            )

    async def finish_episode(self, db: AsyncSession, **kwargs: Any):
        async with trace_span(
            self._tracer,
            "memory.episode_finish",
            SpanKind.MEMORY,
            payload={"outcome": kwargs.get("outcome")},
        ):
            return await self.episodic.finish(db, **kwargs)

    async def similar_episodes(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        text: str,
        top_k: int = 5,
        intent: str | None = None,
        project_id: UUID | None = None,
        only_outcomes: list[str] | None = None,
    ) -> list[EpisodeHit]:
        async with trace_span(
            self._tracer,
            "memory.episode_recall",
            SpanKind.MEMORY,
            payload={"top_k": top_k, "intent": intent},
        ) as span:
            res = await self.episodic.query_similar(
                db,
                user_id=user_id,
                text=text,
                top_k=top_k,
                intent=intent,
                project_id=project_id,
                only_outcomes=only_outcomes,
            )
            span["payload"]["hits"] = len(res)
            return res

    # ── compression ──
    async def summarize_session(
        self, db: AsyncSession, *, session_id: UUID, prior_summary: str | None = None
    ) -> str | None:
        """Compress the short-term window into prose. Caller persists to DB."""
        turns = await self.short.get(session_id)
        if not turns:
            return prior_summary
        result = await self.summarizer.summarize_turns(turns, prior_summary=prior_summary)
        return result.text
