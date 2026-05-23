"""Persist trace events to Postgres.

A `Tracer` subscriber that batches `TraceEvent`s and flushes them to the
`traces` table. Designed to never block the hot path:

- Events are buffered in an `asyncio.Queue`.
- A single background task drains the queue in chunks, writing with one
  bulk insert per flush.
- On shutdown the buffer is drained best-effort.

If the DB is unavailable we drop events (with a warning); the in-process
fan-out to per-trace queues continues to work for SSE.
"""
from __future__ import annotations
import asyncio
import contextlib
from dataclasses import dataclass

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.models import TraceRecord
from app.observability import Tracer
from app.observability.logger import get_logger
from app.observability.tracing import TraceEvent

logger = get_logger("observability.persistence")


@dataclass(slots=True)
class _PersisterConfig:
    flush_interval_s: float = 0.5
    flush_batch: int = 64
    max_queue: int = 4096


class TracePersister:
    """Buffers trace events and bulk-inserts to Postgres."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        flush_interval_s: float = 0.5,
        flush_batch: int = 64,
        max_queue: int = 4096,
    ) -> None:
        self._sf = session_factory
        self._cfg = _PersisterConfig(
            flush_interval_s=flush_interval_s,
            flush_batch=flush_batch,
            max_queue=max_queue,
        )
        self._queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._dropped = 0

    async def __call__(self, event: TraceEvent) -> None:
        """Subscriber callable — `Tracer.add_global_subscriber(persister)`."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("trace_persister_drops", count=self._dropped)

    def attach(self, tracer: Tracer) -> None:
        tracer.add_global_subscriber(self)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="trace-persister")
            logger.info(
                "trace_persister_started",
                flush_interval_s=self._cfg.flush_interval_s,
                flush_batch=self._cfg.flush_batch,
            )

    async def shutdown(self) -> None:
        self._stopping.set()
        if self._task and not self._task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=5.0)
        # Final drain.
        await self._flush(final=True)
        logger.info("trace_persister_stopped", dropped=self._dropped)

    # ── internals ──
    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=self._cfg.flush_interval_s,
                )
            except asyncio.TimeoutError:
                pass
            await self._flush()

    async def _flush(self, *, final: bool = False) -> None:
        if self._queue.empty():
            return
        batch: list[TraceEvent] = []
        # Drain up to flush_batch (or all if final).
        max_take = (
            self._queue.qsize() if final else min(self._queue.qsize(), self._cfg.flush_batch)
        )
        for _ in range(max_take):
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not batch:
            return

        rows = [
            {
                "trace_id": e.trace_id,
                "span_id": e.span_id,
                "parent_span_id": e.parent_span_id,
                "kind": e.kind.value,
                "name": e.name,
                "phase": e.phase.value,
                "ts": e.ts,
                "duration_ms": e.duration_ms,
                "tokens_in": e.tokens_in,
                "tokens_out": e.tokens_out,
                "cost_usd": e.cost_usd,
                "payload": e.payload or {},
                "error": e.error,
            }
            for e in batch
        ]

        try:
            async with self._sf() as session:
                await session.execute(pg_insert(TraceRecord).values(rows))
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("trace_persister_flush_failed", error=str(exc), n=len(rows))
