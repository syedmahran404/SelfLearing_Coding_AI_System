"""Embedding helpers: batching + cache.

Calling the LLM provider for a single embedding per chunk is wasteful.
This wrapper batches and applies a tiny per-process LRU cache so repeated
ingestions of identical chunks short-circuit.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Iterable

from app.llm.provider import LLMProvider
from app.observability import get_logger

logger = get_logger("rag.embeddings")


class EmbeddingService:
    """Batches embed calls and caches results in an in-process LRU.

    The cache is keyed by `(model, sha256(text))` so swapping models
    invalidates entries automatically. Cache is bounded — `cache_size`
    entries — so memory stays predictable.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        batch_size: int = 64,
        cache_size: int = 4096,
    ) -> None:
        self._llm = llm
        self._batch_size = max(1, batch_size)
        self._cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._cache_max = cache_size

    def _key(self, model: str, text: str) -> tuple[str, str]:
        return (model, hashlib.sha256((text or "").encode("utf-8")).hexdigest())

    def _cache_get(self, key: tuple[str, str]) -> list[float] | None:
        v = self._cache.get(key)
        if v is not None:
            self._cache.move_to_end(key)
        return v

    def _cache_put(self, key: tuple[str, str], value: list[float]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    async def embed_many(
        self, texts: Iterable[str], model: str | None = None
    ) -> list[list[float]]:
        m = model or self._llm.settings.llm_embedding_model
        ts = list(texts)
        out: list[list[float] | None] = [None] * len(ts)

        # Cache lookups.
        misses_idx: list[int] = []
        for i, t in enumerate(ts):
            v = self._cache_get(self._key(m, t))
            if v is not None:
                out[i] = v
            else:
                misses_idx.append(i)

        # Batch the misses.
        for batch_start in range(0, len(misses_idx), self._batch_size):
            batch_idx = misses_idx[batch_start : batch_start + self._batch_size]
            batch_texts = [ts[i] for i in batch_idx]
            try:
                resp = await self._llm.embed(batch_texts, model=m)
            except Exception as exc:  # noqa: BLE001
                logger.warning("embed_batch_failed", error=str(exc), n=len(batch_texts))
                # Fill with zero vectors so callers can keep going; they should
                # check `score`/length when retrieving.
                for i in batch_idx:
                    out[i] = []
                continue
            for j, i in enumerate(batch_idx):
                vec = resp.vectors[j] if j < len(resp.vectors) else []
                out[i] = vec
                if vec:
                    self._cache_put(self._key(m, ts[i]), vec)

        return [v if v is not None else [] for v in out]

    async def embed_one(self, text: str, model: str | None = None) -> list[float]:
        vecs = await self.embed_many([text], model=model)
        return vecs[0] if vecs else []
