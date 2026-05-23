"""Async SQLAlchemy session management.

Uses asyncpg under the hood. The engine is created once at startup
(via `init_engine`) and reused by all requests through a session factory.

We deliberately use `expire_on_commit=False` so that ORM objects returned
from a request handler can be safely serialized after the session closes.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings
from app.observability.logger import get_logger

logger = get_logger("db.session")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# Module-level holders, initialized via init_engine().
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    """Initialize the async engine and session factory.

    Idempotent: calling twice returns the existing engine.
    """
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    logger.info("db_engine_init", url=_redact(settings.database_url))

    engine_kwargs: dict[str, Any] = {
        "echo": settings.app_debug and settings.is_dev,
        "pool_pre_ping": True,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 1800,
    }

    _engine = create_async_engine(settings.database_url, **engine_kwargs)
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _engine


async def shutdown_engine() -> None:
    """Dispose the engine (called from FastAPI lifespan shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        logger.info("db_engine_shutdown")
        await _engine.dispose()
    _engine = None
    _session_factory = None


def _redact(url: str) -> str:
    """Hide credentials in logged URLs."""
    if "@" not in url:
        return url
    head, tail = url.split("@", 1)
    if "://" in head and ":" in head.split("://", 1)[1]:
        scheme, rest = head.split("://", 1)
        user = rest.split(":", 1)[0]
        return f"{scheme}://{user}:***@{tail}"
    return f"***@{tail}"


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an `AsyncSession`.

    Commits on clean exit, rolls back on exception, always closes.
    """
    if _session_factory is None:
        raise RuntimeError("DB engine not initialized; call init_engine() first")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def session_factory() -> async_sessionmaker[AsyncSession]:
    """Direct accessor used by background workers (outside HTTP request scope)."""
    if _session_factory is None:
        raise RuntimeError("DB engine not initialized; call init_engine() first")
    return _session_factory
