"""PatternEvolver — distill repeated successful episodes into CodingPatterns.

Run periodically (admin endpoint or background worker). Algorithm:

1. Pull the last N successful episodes (`outcome="success"`).
2. Group by `(intent, language?)`. Languages are heuristically derived
   from the request text + tags.
3. Within each group, cluster by summary-embedding cosine similarity (≥
   threshold). Each cluster of >= MIN_CLUSTER_SIZE produces one
   `CodingPattern`:
       - trigger      : minimal subset of features that distinguishes the
                        cluster from the global pool (intent + keywords)
       - template     : a concise "what to do" derived from the most-recent
                        successful summary
       - validation   : the success_predicate of the most-recent attempt
       - languages    : intersection of detected languages
4. Patterns whose `confidence` (cluster cohesion) exceeds the threshold are
   marked `is_active=True`.

The Planner consults active patterns BEFORE asking the LLM, so a pattern
that surfaces n times naturally bypasses planning altogether for the n+1th
similar request.

This module is dependency-light. We use the same embedding API the rest
of the system relies on.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CodingPattern, Episode
from app.llm.provider import LLMProvider
from app.memory.utils import cosine
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span

logger = get_logger("learning.pattern_evolver")


# ── tunables ──
MIN_CLUSTER_SIZE = 3
SIM_THRESHOLD = 0.82
LOOKBACK_DAYS = 60
TOPN_EPISODES = 200


@dataclass(slots=True)
class EvolutionStats:
    episodes_examined: int
    patterns_created: int
    patterns_updated: int
    duration_ms: int


class PatternEvolver:
    def __init__(self, *, llm: LLMProvider, tracer: Tracer) -> None:
        self._llm = llm
        self._tracer = tracer

    async def evolve(
        self,
        db: AsyncSession,
        *,
        user_id: UUID | None = None,
    ) -> EvolutionStats:
        async with trace_span(
            self._tracer,
            "learning.pattern_evolve",
            SpanKind.SYSTEM,
            payload={"user_id": str(user_id) if user_id else None},
        ) as span:
            started = time.perf_counter()
            episodes = await self._recent_successful(db, user_id=user_id)
            span["payload"]["episodes"] = len(episodes)

            created = 0
            updated = 0
            if len(episodes) < MIN_CLUSTER_SIZE:
                return EvolutionStats(
                    episodes_examined=len(episodes),
                    patterns_created=0,
                    patterns_updated=0,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

            # Group by intent first.
            groups: dict[str, list[Episode]] = defaultdict(list)
            for ep in episodes:
                groups[ep.intent or "code"].append(ep)

            for intent, eps in groups.items():
                clusters = await self._cluster(eps)
                for cluster in clusters:
                    if len(cluster) < MIN_CLUSTER_SIZE:
                        continue
                    pattern_data = _summarize_cluster(intent, cluster)
                    pattern_row, was_created = await self._upsert_pattern(
                        db,
                        user_id=user_id,
                        name=pattern_data["name"],
                        trigger=pattern_data["trigger"],
                        template=pattern_data["template"],
                        validation=pattern_data["validation"],
                        languages=pattern_data["languages"],
                        confidence=pattern_data["confidence"],
                        cluster_size=len(cluster),
                    )
                    created += 1 if was_created else 0
                    updated += 0 if was_created else 1

            stats = EvolutionStats(
                episodes_examined=len(episodes),
                patterns_created=created,
                patterns_updated=updated,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            span["payload"].update(
                {"patterns_created": created, "patterns_updated": updated}
            )
            return stats

    async def _recent_successful(
        self, db: AsyncSession, *, user_id: UUID | None
    ) -> list[Episode]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
        q = (
            select(Episode)
            .where(Episode.outcome == "success", Episode.created_at >= cutoff)
            .order_by(Episode.created_at.desc())
            .limit(TOPN_EPISODES)
        )
        if user_id is not None:
            q = q.where(Episode.user_id == user_id)
        rows = (await db.execute(q)).scalars().all()
        return list(rows)

    async def _cluster(self, episodes: Sequence[Episode]) -> list[list[Episode]]:
        """Single-pass agglomerative clustering on summary embeddings.

        Falls back gracefully if any embedding call fails.
        """
        if not episodes:
            return []
        texts = [(ep.summary or ep.title or ep.input)[:1500] for ep in episodes]
        try:
            emb = await self._llm.embed(texts)
            vectors = emb.vectors
        except Exception as exc:  # noqa: BLE001
            logger.warning("evolve_embed_failed", error=str(exc))
            return [[ep] for ep in episodes]

        clusters: list[list[Episode]] = []
        cluster_centroids: list[list[float]] = []
        for ep, v in zip(episodes, vectors, strict=False):
            placed = False
            for i, c in enumerate(cluster_centroids):
                if cosine(v, c) >= SIM_THRESHOLD:
                    clusters[i].append(ep)
                    # Update centroid (running mean).
                    n = len(clusters[i])
                    cluster_centroids[i] = [
                        (c[j] * (n - 1) + v[j]) / n for j in range(len(c))
                    ]
                    placed = True
                    break
            if not placed:
                clusters.append([ep])
                cluster_centroids.append(list(v))
        return clusters

    async def _upsert_pattern(
        self,
        db: AsyncSession,
        *,
        user_id: UUID | None,
        name: str,
        trigger: dict[str, Any],
        template: dict[str, Any],
        validation: dict[str, Any],
        languages: list[str],
        confidence: float,
        cluster_size: int,
    ) -> tuple[CodingPattern, bool]:
        # Match on user + name (a fingerprint of intent + keywords).
        q = select(CodingPattern).where(CodingPattern.name == name)
        if user_id is not None:
            q = q.where(CodingPattern.user_id == user_id)
        existing = (await db.execute(q)).scalar_one_or_none()
        if existing is not None:
            existing.use_count = max(existing.use_count, cluster_size)
            existing.success_count = max(existing.success_count, cluster_size)
            existing.confidence = max(existing.confidence, confidence)
            existing.template = template or existing.template
            existing.validation = validation or existing.validation
            existing.languages = sorted(set(existing.languages or []) | set(languages or []))
            existing.is_active = existing.confidence >= 0.6
            await db.flush()
            return existing, False

        row = CodingPattern(
            user_id=user_id,
            name=name,
            trigger=trigger,
            template=template,
            validation=validation,
            languages=languages,
            use_count=cluster_size,
            success_count=cluster_size,
            confidence=confidence,
            is_active=confidence >= 0.6,
        )
        db.add(row)
        await db.flush()
        return row, True


# ── helpers ──


_TOK = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def _summarize_cluster(intent: str, cluster: list[Episode]) -> dict[str, Any]:
    # Most-recent episode is canonical.
    latest = max(cluster, key=lambda e: e.created_at)
    keywords = _top_keywords([e.input + " " + (e.summary or "") for e in cluster], n=5)
    languages = _detect_languages(cluster)
    name = f"{intent}:{':'.join(keywords[:3]) or 'general'}"

    trigger = {"intent": intent, "keywords": keywords}
    template = {
        "summary": (latest.summary or latest.title)[:600],
        "example_input": latest.input[:400],
        "example_actions": (latest.actions or [])[:8],
    }
    validation = {
        "success_predicate": _extract_predicate(latest),
    }
    cohesion = min(0.95, 0.5 + 0.05 * len(cluster))  # bigger cluster → more confident
    return {
        "name": name,
        "trigger": trigger,
        "template": template,
        "validation": validation,
        "languages": languages,
        "confidence": cohesion,
    }


def _extract_predicate(ep: Episode) -> str:
    plan = ep.plan or {}
    subtasks = plan.get("subtasks") or []
    for s in subtasks:
        pred = s.get("success_predicate")
        if pred:
            return pred
    return "result matches user's success criteria"


def _top_keywords(texts: list[str], *, n: int = 5) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    stop = {
        "the", "and", "with", "from", "into", "that", "this", "have", "your", "you", "for",
        "are", "was", "were", "is", "to", "of", "a", "an", "in", "on", "be", "by", "it",
        "as", "or", "if", "but", "do", "done", "also", "then", "than", "i", "we", "they",
    }
    for t in texts:
        for tok in _TOK.findall(t.lower()):
            if len(tok) < 3 or tok in stop:
                continue
            counts[tok] += 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def _detect_languages(cluster: list[Episode]) -> list[str]:
    langs: set[str] = set()
    keys = ("python", "typescript", "javascript", "java", "go", "rust", "c++", "sql", "shell", "html", "css")
    for ep in cluster:
        text = (ep.input + " " + (ep.summary or "")).lower()
        for k in keys:
            if k in text:
                langs.add(k)
    return sorted(langs)
