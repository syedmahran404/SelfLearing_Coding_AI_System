"""Hybrid retrieval + reranking.

Pipeline per query:

    1. Vector search over Qdrant (over-recall by 2x).
    2. BM25 score over the same candidate set (text from each candidate's
       payload). Vector recall produces the candidate pool, so BM25 stays
       cheap and bounded.
    3. Rerank: blend `vector_score`, normalized `bm25_score`, and small
       provenance bonuses/penalties (chunk kind weight, language match,
       freshness via `source_version`).
    4. Cut to top_k.

We deliberately *don't* maintain a global BM25 index — that would mean a
second consistent corpus to keep in sync. Scoring against a Qdrant
candidate pool is enough for the precision we need (and BM25 only beats
vectors decisively on exact symbol matches, which the over-recall already
surfaces because Qdrant scores them well too).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from app.db.qdrant import QdrantStore
from app.observability import get_logger
from app.rag.embeddings import EmbeddingService

logger = get_logger("rag.retriever")


@dataclass(slots=True)
class RagHit:
    """A retrieved chunk with all the scores that produced its rank."""

    id: str
    text: str
    source_uri: str | None
    chunk_kind: str | None
    language: str | None
    name: str | None
    line_start: int | None
    line_end: int | None
    vector_score: float
    bm25_score: float
    blended_score: float
    payload: dict[str, Any]


# ── tokenization for BM25 ─────────────────────────────────────────────────
_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text or "")]


# ── chunk-kind weights (tunable) ──────────────────────────────────────────
_CHUNK_KIND_BIAS: dict[str, float] = {
    "function": 0.05,
    "class": 0.04,
    "section": 0.03,
    "doc": 0.02,
    "toplevel": 0.02,
    "window": 0.0,
}


class HybridRetriever:
    """Vector + BM25 + provenance-aware reranking."""

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

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        recall_multiplier: int = 4,
        user_id: str | None = None,
        project_id: str | None = None,
        languages: list[str] | None = None,
        kind: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RagHit]:
        if not query:
            return []

        # 1) vector recall.
        try:
            qvec = await self._embeddings.embed_one(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag_query_embed_failed", error=str(exc))
            return []
        if not qvec:
            return []

        flt: dict[str, Any] = {}
        if user_id is not None:
            flt["user_id"] = user_id
        if project_id is not None:
            flt["project_id"] = project_id
        if kind is not None:
            flt["chunk_kind"] = kind

        recall = await self._qdrant.search(
            self._collection,
            qvec,
            top_k=max(top_k * recall_multiplier, top_k),
            filter_must=flt or None,
            score_threshold=score_threshold,
        )
        if not recall:
            return []

        # 2) BM25 over candidate pool.
        candidate_texts = [(h.payload.get("text") or "") for h in recall]
        bm25_scores: list[float] = self._bm25_score(query, candidate_texts)

        # Normalize BM25 to [0,1] for blending.
        max_bm = max(bm25_scores) if bm25_scores else 0.0
        norm_bm = [s / max_bm if max_bm > 0 else 0.0 for s in bm25_scores]

        # 3) blend + provenance bonuses.
        out: list[RagHit] = []
        languages_set = {l.lower() for l in (languages or [])}
        for h, b in zip(recall, norm_bm, strict=False):
            payload = h.payload or {}
            base = 0.6 * h.score + 0.35 * b
            base += _CHUNK_KIND_BIAS.get(payload.get("chunk_kind", "window"), 0.0)
            if languages_set and (payload.get("language") or "").lower() in languages_set:
                base += 0.03
            blended = max(0.0, min(1.5, base))  # leave headroom; not strictly [0,1]
            out.append(
                RagHit(
                    id=str(h.id),
                    text=str(payload.get("text") or ""),
                    source_uri=payload.get("source_uri"),
                    chunk_kind=payload.get("chunk_kind"),
                    language=payload.get("language"),
                    name=payload.get("name"),
                    line_start=payload.get("line_start"),
                    line_end=payload.get("line_end"),
                    vector_score=float(h.score),
                    bm25_score=float(b),
                    blended_score=blended,
                    payload=payload,
                )
            )

        out.sort(key=lambda x: x.blended_score, reverse=True)
        return out[:top_k]

    # ── BM25 helper ──
    @staticmethod
    def _bm25_score(query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        tokenized = [_tokenize(d) for d in docs]
        # Guard against empty corpora to avoid div-by-zero in rank-bm25.
        if not any(tokenized):
            return [0.0] * len(docs)
        try:
            bm = BM25Okapi(tokenized)
            return [float(s) for s in bm.get_scores(_tokenize(query))]
        except Exception as exc:  # noqa: BLE001
            logger.debug("bm25_failed", error=str(exc))
            # fallback: very crude TF count of query terms
            qtoks = set(_tokenize(query))
            scores: list[float] = []
            for d in tokenized:
                hits = sum(1 for t in d if t in qtoks)
                scores.append(math.log1p(hits))
            return scores
