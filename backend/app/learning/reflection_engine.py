"""ReflectionEngine — turn Reflector outputs into durable Lessons.

Responsibilities:
- Persist a `Lesson` row when the Reflector emits a `lesson` string. Dedup
  by `(user_id, content_sha)`.
- Embed the lesson and upsert into Qdrant `memories` collection so it's
  retrievable in future ContextBuilder runs.
- Update lesson impact counters when a previously-applied lesson appears
  in the retrieved context of a successful run (handled via
  `feedback_lesson`).

This module deliberately keeps a narrow surface — the orchestrator hands
it a `Reflection` and a result; the engine doesn't drive flow control.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Lesson
from app.db.qdrant import VectorPoint
from app.llm.provider import LLMProvider
from app.memory.service import MemoryService
from app.memory.utils import content_sha, normalize_content
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span
from app.schemas.agent_io import Reflection

logger = get_logger("learning.reflection")


class ReflectionEngine:
    def __init__(
        self,
        *,
        memory: MemoryService,
        llm: LLMProvider,
        tracer: Tracer,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._tracer = tracer

    # ── Lessons ──────────────────────────────────────────────────────────
    async def persist_reflection(
        self,
        db: AsyncSession,
        *,
        reflection: Reflection,
        user_id: UUID,
        project_id: UUID | None = None,
        source_episode_id: UUID | None = None,
        applies_when: dict[str, Any] | None = None,
    ) -> Lesson | None:
        """Persist the lesson contained in `reflection`, if any.

        Returns the Lesson row (existing or new), or None if there's no lesson
        to persist.
        """
        if not reflection.lesson or not reflection.lesson.strip():
            return None

        async with trace_span(
            self._tracer,
            "learning.persist_lesson",
            SpanKind.MEMORY,
            payload={"len": len(reflection.lesson)},
        ) as span:
            normalized = normalize_content(reflection.lesson)
            sha = content_sha(normalized)
            existing = (
                await db.execute(
                    select(Lesson).where(Lesson.user_id == user_id, Lesson.content_sha == sha)
                )
            ).scalar_one_or_none()
            if existing is not None:
                # Bump impact: the same insight has surfaced again.
                existing.impact = min(1.0, existing.impact + 0.05)
                existing.confidence = max(existing.confidence, reflection.confidence)
                await db.flush()
                span["payload"]["status"] = "deduped"
                return existing

            tags = _derive_tags(reflection)
            vector_id: str | None = None
            try:
                emb = await self._llm.embed([normalized])
                if emb.vectors:
                    vector_id = sha[:32]
                    await self._memory._qdrant.upsert(  # type: ignore[attr-defined]
                        self._memory._qdrant.memories,  # type: ignore[attr-defined]
                        [
                            VectorPoint(
                                id=vector_id,
                                vector=emb.vectors[0],
                                payload={
                                    "user_id": str(user_id),
                                    "project_id": str(project_id) if project_id else None,
                                    "kind": "lesson",
                                    "tags": tags,
                                    "content_sha": sha,
                                    "source": "reflection",
                                    "created_at": time.time(),
                                },
                            )
                        ],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("lesson_embed_failed", error=str(exc))

            row = Lesson(
                user_id=user_id,
                project_id=project_id,
                rule=normalized,
                applies_when=applies_when or {},
                tags=tags,
                source_episode_id=source_episode_id,
                confidence=reflection.confidence,
                impact=0.1,
                vector_id=vector_id,
                content_sha=sha,
            )
            db.add(row)
            await db.flush()
            span["payload"]["status"] = "created"
            span["payload"]["id"] = str(row.id)
            return row

    # ── Lesson impact tracking ──────────────────────────────────────────
    async def feedback_lesson(
        self,
        db: AsyncSession,
        *,
        lesson_id: UUID,
        outcome: bool,
    ) -> None:
        """Record that a previously-retrieved lesson was followed by success
        (True) or failure (False)."""
        row = (
            await db.execute(select(Lesson).where(Lesson.id == lesson_id))
        ).scalar_one_or_none()
        if row is None:
            return
        if outcome:
            row.success_count += 1
            row.confidence = min(1.0, row.confidence + 0.02)
            row.impact = min(1.0, row.impact + 0.05)
        else:
            row.failure_count += 1
            row.confidence = max(0.0, row.confidence - 0.05)
            row.impact = max(0.0, row.impact - 0.02)
        await db.flush()


def _derive_tags(reflection: Reflection) -> list[str]:
    """Extract a few crude tags from the reflection content."""
    tags: list[str] = []
    text = (reflection.lesson or "") + " " + (reflection.root_cause or "")
    text_lc = text.lower()
    for kw in (
        ("python", "python"),
        ("typescript", "typescript"),
        ("javascript", "javascript"),
        ("test", "testing"),
        ("pytest", "pytest"),
        ("import", "imports"),
        ("type", "typing"),
        ("async", "async"),
        ("docker", "docker"),
        ("sql", "sql"),
        ("api", "api"),
    ):
        if kw[0] in text_lc and kw[1] not in tags:
            tags.append(kw[1])
    return tags[:6]
