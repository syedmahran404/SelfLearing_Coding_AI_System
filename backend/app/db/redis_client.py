"""Async Redis client wrapper.

Used for:
- short-term memory (rolling chat window per session)
- per-session token-budget counters (atomic INCRBY with TTL)
- light-weight caches (e.g. recently retrieved memory ids)

We wrap `redis.asyncio` rather than expose it directly so that:
1. The application code uses our domain methods (`get_short_term`, `incr_tokens`)
   instead of arbitrary Redis commands — easier to reason about and to mock.
2. We can swap implementations (e.g. in-memory for tests) by satisfying the
   same surface area.
"""
from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis

from app.config import Settings
from app.observability.logger import get_logger

logger = get_logger("db.redis")


class RedisClient:
    """Thin async wrapper around redis-py with domain methods."""

    def __init__(self, client: redis.Redis, *, short_term_ttl: int, token_budget_ttl: int) -> None:
        self._r = client
        self._short_term_ttl = short_term_ttl
        self._token_budget_ttl = token_budget_ttl

    @classmethod
    def from_settings(cls, settings: Settings) -> RedisClient:
        client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            health_check_interval=30,
        )
        return cls(
            client,
            short_term_ttl=settings.redis_short_term_ttl_s,
            token_budget_ttl=settings.redis_token_budget_ttl_s,
        )

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_ping_failed", error=str(exc))
            return False

    async def close(self) -> None:
        await self._r.aclose()

    # ── short-term memory: rolling chat window ──
    @staticmethod
    def _stm_key(session_id: str) -> str:
        return f"slcai:stm:{session_id}"

    async def push_short_term(self, session_id: str, item: dict[str, Any], *, max_len: int = 50) -> None:
        """Append an item (e.g. a message) to the short-term window. LPUSH + LTRIM."""
        key = self._stm_key(session_id)
        pipe = self._r.pipeline()
        pipe.lpush(key, json.dumps(item, ensure_ascii=False, default=str))
        pipe.ltrim(key, 0, max_len - 1)
        pipe.expire(key, self._short_term_ttl)
        await pipe.execute()

    async def get_short_term(self, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        raw = await self._r.lrange(self._stm_key(session_id), 0, limit - 1)
        # Items were LPUSHed so newest is at index 0; return chronological.
        return [json.loads(x) for x in reversed(raw)]

    async def clear_short_term(self, session_id: str) -> None:
        await self._r.delete(self._stm_key(session_id))

    # ── token budget counters ──
    @staticmethod
    def _budget_key(scope: str, ident: str) -> str:
        return f"slcai:tok:{scope}:{ident}"

    async def incr_tokens(self, scope: str, ident: str, n: int) -> int:
        """Atomically increment the token counter. Returns the new value.

        Sets a TTL on first creation so stale counters expire automatically.
        """
        key = self._budget_key(scope, ident)
        new_value = await self._r.incrby(key, n)
        if new_value == n:
            await self._r.expire(key, self._token_budget_ttl)
        return int(new_value)

    async def get_tokens(self, scope: str, ident: str) -> int:
        v = await self._r.get(self._budget_key(scope, ident))
        return int(v) if v else 0

    # ── generic key-value (used carefully) ──
    async def setex_json(self, key: str, value: Any, ttl_s: int) -> None:
        await self._r.setex(key, ttl_s, json.dumps(value, ensure_ascii=False, default=str))

    async def get_json(self, key: str) -> Any | None:
        v = await self._r.get(key)
        return json.loads(v) if v else None

    async def delete(self, key: str) -> None:
        await self._r.delete(key)


_redis_singleton: RedisClient | None = None


def get_redis() -> RedisClient:
    """Module-level accessor; initialized via `init_redis`."""
    if _redis_singleton is None:
        raise RuntimeError("Redis client not initialized; call init_redis() first")
    return _redis_singleton


def init_redis(settings: Settings) -> RedisClient:
    """Initialize the global redis client. Idempotent."""
    global _redis_singleton
    if _redis_singleton is None:
        _redis_singleton = RedisClient.from_settings(settings)
        logger.info("redis_client_init", url=settings.redis_url)
    return _redis_singleton


async def shutdown_redis() -> None:
    global _redis_singleton
    if _redis_singleton is not None:
        await _redis_singleton.close()
        _redis_singleton = None
        logger.info("redis_client_shutdown")
