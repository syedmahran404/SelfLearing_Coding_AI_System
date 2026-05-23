"""Short-term memory — rolling window per session, in Redis.

Why Redis:
- Microsecond LPUSH+LTRIM keeps the hot path cheap.
- TTL guarantees stale sessions disappear without manual GC.
- Atomic INCRBY for the per-session token budget lives next door.

The window holds a *light* snapshot of each turn (role, content,
agent, tokens) — not the full Message ORM row. Anything we want forever
goes through long-term memory; this layer is volatile by design.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from app.db.redis_client import RedisClient
from app.observability import get_logger

logger = get_logger("memory.short_term")


class ShortTermMemory:
    """Per-session rolling window."""

    DEFAULT_WINDOW = 30

    def __init__(self, redis: RedisClient, *, window: int = DEFAULT_WINDOW) -> None:
        self._r = redis
        self._window = window

    async def push(
        self,
        session_id: UUID | str,
        *,
        role: str,
        content: str,
        agent: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        item = {
            "ts": time.time(),
            "role": role,
            "content": content,
            "agent": agent,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "extra": extra or {},
        }
        await self._r.push_short_term(str(session_id), item, max_len=self._window)

    async def get(
        self, session_id: UUID | str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        return await self._r.get_short_term(str(session_id), limit=limit or self._window)

    async def clear(self, session_id: UUID | str) -> None:
        await self._r.clear_short_term(str(session_id))

    # ── token budget per session (atomic) ──
    async def add_tokens(self, session_id: UUID | str, n: int) -> int:
        return await self._r.incr_tokens("session", str(session_id), n)

    async def get_tokens(self, session_id: UUID | str) -> int:
        return await self._r.get_tokens("session", str(session_id))
