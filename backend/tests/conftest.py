"""Shared test fixtures.

Tests do not require Postgres, Redis, or Qdrant — we wire in the
deterministic `LocalProvider` and stub the storage layers in-memory.
This keeps `make test` runnable with no external services.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _env_overrides(tmp_path_factory):
    """Force LLM_PROVIDER=local + isolated sandbox root for the whole session."""
    sandbox = tmp_path_factory.mktemp("slcai_sandbox")
    overrides = {
        "LLM_PROVIDER": "local",
        "APP_ENV": "test",
        "APP_LOG_LEVEL": "WARNING",
        "DATABASE_URL": "postgresql+asyncpg://x:x@localhost:5432/x",  # never connected
        "REDIS_URL": "redis://localhost:6379/0",
        "QDRANT_URL": "http://localhost:6333",
        "SANDBOX_ROOT": str(sandbox),
        "OBSERVABILITY_TRACE_TO_DB": "false",
        "OBSERVABILITY_TRACE_TO_STDOUT": "false",
    }
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    # Reset cached settings so the new env takes effect.
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_settings.cache_clear()


@pytest.fixture
def settings():
    from app.config import get_settings

    return get_settings()


@pytest.fixture
def local_llm(settings):
    from app.llm.local_provider import LocalProvider

    return LocalProvider(settings)


@pytest.fixture
def tracer():
    from app.observability.tracing import Tracer

    return Tracer()


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    p = tmp_path / "workdir"
    p.mkdir(parents=True, exist_ok=True)
    return p
