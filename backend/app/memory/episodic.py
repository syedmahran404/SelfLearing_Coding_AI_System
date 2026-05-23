"""Episodic memory — task outcomes.

Each non-trivial task (chat turn that goes through the orchestrator) becomes
an Episode row. The summary is embedded so we can answer:

    "have I seen something like this before, and what worked?"

This is the input the Reflector + Planner consult to avoid re-discovering
known solutions and to recognize recurring failure modes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Episode
from app.db.qdrant import QdrantStore, VectorPoint
from app.llm.provider import LLMProvider
from app.observability import get_logger

logger = get_logger("memory.episodic")


@dataclass(slots=True)
class EpisodeHit:
    episode: Episode
    score: float


class EpisodicMemory:
    """Long-term task-outcome store.

    - rows live in `episodes` (Postgres)
    - summary embeddings live in the *memories* qdrant collection under
      `kind="episode"` so a single retrieval pipeline can fetch both
    """

    def __init__(self, *, qdrant: QdrantStore, llm: LLMProvider) -> None:
        self._qdrant = qdrant
        self._llm = llm
        self._collection = qdrant.memories

    # ── start / finish ──────────────────────────────────────────────────
    async def start(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        intent: str,
        title: str,
        input_text: str,
        project_id: UUID | None = None,
        session_id: UUID | None = None,
        trace_id: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> Episode:
        ep = Episode(
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
            trace_id=trace_id,
            title=title,
            intent=intent,
            input=input_text,
            plan=plan or {},
            outcome="pending",
        )
        session.add(ep)
        await session.flush()
        return ep

    async def finish(
        self,
        session: AsyncSession,
        *,
        episode_id: UUID,
        outcome: str,
        score: float,
        confidence: float,
        actions: list[dict[str, Any]] | None = None,
        summary: str | None = None,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> Episode:
        ep = (
            await session.execute(select(Episode).where(Episode.id == episode_id))
        ).scalar_one()
        ep.outcome = outcome
        ep.score = float(score)
        ep.confidence = float(confidence)
        if actions:
            ep.actions = actions
        if summary:
            ep.summary = summary
        ep.tokens_used = int(tokens_used)
        ep.cost_usd = float(cost_usd)
        ep.duration_ms = int(duration_ms)
        ep.finished_at = _utcnow()

        # Embed the summary so future similar-task lookups work.
        if summary:
            try:
                emb = await self._llm.embed([summary])
                if emb.vectors:
                    point_id = f"ep_{ep.id.hex}"[:32]
                    await self._qdrant.upsert(
                        self._collection,
                        [
                            VectorPoint(
                                id=point_id,
                                vector=emb.vectors[0],
                                payload={
                                    "user_id": str(ep.user_id),
                                    "project_id": str(ep.project_id) if ep.project_id else None,
                                    "kind": "episode",
                                    "intent": ep.intent,
                                    "outcome": ep.outcome,
                                    "score": ep.score,
                                    "episode_id": str(ep.id),
                                    "created_at": time.time(),
                                },
                            )
                        ],
                    )
                    ep.summary_vector_id = point_id
            except Exception as exc:  # noqa: BLE001
                logger.warning("episode_embed_failed", error=str(exc))

        await session.flush()
        return ep

    # ── retrieval ───────────────────────────────────────────────────────
    async def query_similar(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        text: str,
        top_k: int = 5,
        intent: str | None = None,
        project_id: UUID | None = None,
        only_outcomes: list[str] | None = None,
    ) -> list[EpisodeHit]:
        if not text:
            return []
        try:
            emb = await self._llm.embed([text])
            qvec = emb.vectors[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("episode_query_embed_failed", error=str(exc))
            return []

        must: dict[str, Any] = {"user_id": str(user_id), "kind": "episode"}
        if intent is not None:
            must["intent"] = intent
        if project_id is not None:
            must["project_id"] = str(project_id)
        if only_outcomes:
            must["outcome"] = list(only_outcomes)

        hits = await self._qdrant.search(self._collection, qvec, top_k=top_k * 2, filter_must=must)
        if not hits:
            return []

        ids = [h.payload.get("episode_id") for h in hits if h.payload.get("episode_id")]
        if not ids:
            return []
        rows = (
            (
                await session.execute(
                    select(Episode).where(Episode.id.in_([UUID(i) for i in ids]))
                )
            )
            .scalars()
            .all()
        )
        by_id = {str(r.id): r for r in rows}

        out: list[EpisodeHit] = []
        for h in hits:
            row = by_id.get(h.payload.get("episode_id", ""))
            if row is None:
                continue
            out.append(EpisodeHit(episode=row, score=float(h.score)))
        out.sort(key=lambda e: e.score, reverse=True)
        return out[:top_k]


def _utcnow():  # type: ignore[no-untyped-def]
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
