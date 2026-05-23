"""Background lifecycle worker.

Three passes (each idempotent), invoked on a timer:

1. **Recompute utility** — refresh `Memory.utility` based on age and access
   counters so retrieval scoring stays meaningful.
2. **Dedup / consolidate** — find groups of near-duplicate memories
   (cosine similarity ≥ threshold and identical kind+user+project), pick a
   canonical, copy access counters into it, archive the rest.
3. **Decay / archive** — archive memories whose utility falls below the
   archive threshold and whose age exceeds twice the halflife. Their
   vectors are removed from Qdrant; the row stays for export/forensics.

The worker is started by `MemoryService.start_worker()` and stopped via
`shutdown()`. It runs as an `asyncio.Task` and is aware of cancellation.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Iterable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.db.models import Memory
from app.db.qdrant import QdrantStore
from app.llm.provider import LLMProvider
from app.memory.compression import Summarizer
from app.memory.utils import cosine, content_sha, utility_score
from app.observability import get_logger

logger = get_logger("memory.lifecycle")


class LifecycleWorker:
    """Background task: dedup, decay, recompute utility."""

    ARCHIVE_UTILITY_THRESHOLD = 0.05
    ARCHIVE_AGE_HALFLIVES = 2.0  # only archive after 2 halflives have passed
    DEDUP_BATCH = 200

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker,
        qdrant: QdrantStore,
        llm: LLMProvider,
        summarizer: Summarizer,
    ) -> None:
        self._settings = settings
        self._sf = session_factory
        self._qdrant = qdrant
        self._llm = llm
        self._summarizer = summarizer
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping.clear()
            self._task = asyncio.create_task(self._loop(), name="memory-lifecycle")
            logger.info("memory_lifecycle_started", interval_s=self._settings.memory_lifecycle_interval_s)

    async def shutdown(self) -> None:
        self._stopping.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("memory_lifecycle_stopped")

    # ── main loop ──
    async def _loop(self) -> None:
        interval = max(60, self._settings.memory_lifecycle_interval_s)
        # Stagger first run so every restart doesn't thunder.
        await self._sleep_or_stop(min(interval, 30))
        while not self._stopping.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory_lifecycle_iter_failed", error=str(exc))
            await self._sleep_or_stop(interval)

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    # ── one pass ──
    async def run_once(self) -> dict[str, int]:
        """Run dedup + decay + utility recompute. Returns counters for tests/observability."""
        async with self._sf() as session:
            n_util = await self._recompute_utility(session)
            n_dedup = await self._dedup_pass(session)
            n_decay = await self._decay_pass(session)
            await session.commit()

        out = {"utility_updated": n_util, "deduped": n_dedup, "decayed": n_decay}
        logger.info("memory_lifecycle_pass", **out)
        return out

    async def _recompute_utility(self, session) -> int:  # type: ignore[no-untyped-def]
        rows = (
            (
                await session.execute(
                    select(Memory).where(Memory.is_archived.is_(False)).limit(self.DEDUP_BATCH * 4)
                )
            )
            .scalars()
            .all()
        )
        n = 0
        now = time.time()
        for r in rows:
            new_u = utility_score(
                base=max(0.4, r.confidence),
                last_accessed_at=r.last_accessed_at,
                access_count=r.access_count,
                success_count=r.success_count,
                failure_count=r.failure_count,
                halflife_days=self._settings.memory_decay_halflife_days,
                now=now,
            )
            if abs(new_u - r.utility) > 0.01:
                r.utility = new_u
                n += 1
        if n:
            await session.flush()
        return n

    async def _dedup_pass(self, session) -> int:  # type: ignore[no-untyped-def]
        """Find near-duplicates per (user, project, kind), merge counters,
        archive the loser. Cosine threshold from settings."""
        threshold = self._settings.memory_dedup_sim_threshold
        rows = (
            (
                await session.execute(
                    select(Memory)
                    .where(Memory.is_archived.is_(False), Memory.vector_id.is_not(None))
                    .order_by(Memory.created_at.asc())
                    .limit(self.DEDUP_BATCH)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return 0

        # Group by (user, project, kind).
        groups: dict[tuple, list[Memory]] = defaultdict(list)
        for r in rows:
            groups[(r.user_id, r.project_id, r.kind)].append(r)

        # We need vectors. Re-embed content (cheap with the local provider;
        # ok with real provider since groups are small).
        deduped = 0
        for _, group in groups.items():
            if len(group) < 2:
                continue
            try:
                emb = await self._llm.embed([m.content for m in group])
                vectors = emb.vectors
            except Exception as exc:  # noqa: BLE001
                logger.warning("dedup_embed_failed", error=str(exc))
                continue

            archived: set = set()
            for i in range(len(group)):
                if group[i].id in archived:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].id in archived:
                        continue
                    sim = cosine(vectors[i], vectors[j])
                    if sim < threshold:
                        continue
                    keep, drop = _pick_canonical(group[i], group[j])
                    keep.access_count += drop.access_count
                    keep.success_count += drop.success_count
                    keep.failure_count += drop.failure_count
                    if drop.utility > keep.utility:
                        keep.utility = drop.utility
                    drop.is_archived = True
                    archived.add(drop.id)
                    if drop.vector_id:
                        try:
                            await self._qdrant.delete(self._qdrant.memories, [drop.vector_id])
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("dedup_vector_delete_skipped", error=str(exc))
                    deduped += 1

        if deduped:
            await session.flush()
        return deduped

    async def _decay_pass(self, session) -> int:  # type: ignore[no-untyped-def]
        """Archive memories whose utility fell below the threshold and which
        are sufficiently old."""
        rows = (
            (
                await session.execute(
                    select(Memory).where(
                        Memory.is_archived.is_(False),
                        Memory.utility < self.ARCHIVE_UTILITY_THRESHOLD,
                    )
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return 0
        cutoff_age_days = self._settings.memory_decay_halflife_days * self.ARCHIVE_AGE_HALFLIVES
        cutoff_ts = time.time() - cutoff_age_days * 86400
        n = 0
        for r in rows:
            ts = r.created_at.timestamp() if r.created_at else 0.0
            if ts > cutoff_ts:
                continue  # too young to archive
            r.is_archived = True
            if r.vector_id:
                try:
                    await self._qdrant.delete(self._qdrant.memories, [r.vector_id])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("decay_vector_delete_skipped", error=str(exc))
            n += 1
        if n:
            await session.flush()
        return n


def _pick_canonical(a: Memory, b: Memory) -> tuple[Memory, Memory]:
    """Prefer higher utility, then more access, then older row."""
    if a.utility != b.utility:
        return (a, b) if a.utility > b.utility else (b, a)
    if a.access_count != b.access_count:
        return (a, b) if a.access_count > b.access_count else (b, a)
    return (a, b) if a.created_at <= b.created_at else (b, a)


def _content_shas(rows: Iterable[Memory]) -> list[str]:
    """Helper for tests: re-derive content_sha from current content."""
    return [content_sha(r.content) for r in rows]
