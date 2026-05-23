"""Application configuration.

Single source of truth for all runtime settings. Loaded from environment
variables (or `.env` during local dev). Validation happens at import time so
mis-configuration fails loudly at boot rather than during the first request.

The settings object is intentionally a plain Pydantic model — no global
mutable state, no module-level side effects. Components that need it accept
it via dependency injection (`Depends(get_settings)` in FastAPI, or
constructor args elsewhere).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["openai", "anthropic", "local"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
NetworkPolicy = Literal["allow", "deny"]


class Settings(BaseSettings):
    """All runtime settings, validated at import time.

    Environment variables map 1:1 with field names (case-insensitive). Field
    aliases use the same UPPER_SNAKE_CASE names declared in `.env.example`.
    """

    # ── Application ──
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: LogLevel = Field(default="INFO", alias="APP_LOG_LEVEL")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_cors_origins: str = Field(
        default="http://localhost:5173",
        alias="APP_CORS_ORIGINS",
    )

    # ── LLM ──
    llm_provider: LLMProviderName = Field(default="local", alias="LLM_PROVIDER")
    llm_default_model: str = Field(default="gpt-4o-mini", alias="LLM_DEFAULT_MODEL")
    llm_planner_model: str = Field(default="gpt-4o-mini", alias="LLM_PLANNER_MODEL")
    llm_coder_model: str = Field(default="gpt-4o", alias="LLM_CODER_MODEL")
    llm_embedding_model: str = Field(
        default="text-embedding-3-small", alias="LLM_EMBEDDING_MODEL"
    )
    llm_max_tokens_per_task: int = Field(default=120_000, alias="LLM_MAX_TOKENS_PER_TASK")
    llm_max_tokens_per_session: int = Field(
        default=1_000_000, alias="LLM_MAX_TOKENS_PER_SESSION"
    )
    llm_request_timeout_s: int = Field(default=120, alias="LLM_REQUEST_TIMEOUT_S")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")

    # ── Postgres ──
    database_url: str = Field(
        default="postgresql+asyncpg://slcai:slcai_dev_password@localhost:5432/slcai",
        alias="DATABASE_URL",
    )

    # ── Redis ──
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_short_term_ttl_s: int = Field(default=3600, alias="REDIS_SHORT_TERM_TTL_S")
    redis_token_budget_ttl_s: int = Field(default=900, alias="REDIS_TOKEN_BUDGET_TTL_S")

    # ── Qdrant ──
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_memories: str = Field(
        default="slcai_memories", alias="QDRANT_COLLECTION_MEMORIES"
    )
    qdrant_collection_rag: str = Field(default="slcai_rag", alias="QDRANT_COLLECTION_RAG")
    qdrant_vector_size: int = Field(default=1536, alias="QDRANT_VECTOR_SIZE")

    # ── Sandbox / tools ──
    sandbox_root: Path = Field(default=Path("/tmp/slcai_sandbox"), alias="SANDBOX_ROOT")
    sandbox_default_timeout_s: int = Field(default=30, alias="SANDBOX_DEFAULT_TIMEOUT_S")
    sandbox_max_memory_mb: int = Field(default=512, alias="SANDBOX_MAX_MEMORY_MB")
    sandbox_max_cpu_s: int = Field(default=20, alias="SANDBOX_MAX_CPU_S")
    sandbox_network_default: NetworkPolicy = Field(
        default="deny", alias="SANDBOX_NETWORK_DEFAULT"
    )

    # ── Safety ──
    safety_max_recursion_depth: int = Field(default=4, alias="SAFETY_MAX_RECURSION_DEPTH")
    safety_min_confidence_to_execute: float = Field(
        default=0.55, alias="SAFETY_MIN_CONFIDENCE_TO_EXECUTE"
    )
    safety_hallucination_block: bool = Field(default=True, alias="SAFETY_HALLUCINATION_BLOCK")
    safety_dryrun_file_writes: bool = Field(default=False, alias="SAFETY_DRYRUN_FILE_WRITES")

    # ── Memory lifecycle ──
    memory_dedup_sim_threshold: float = Field(
        default=0.95, alias="MEMORY_DEDUP_SIM_THRESHOLD"
    )
    memory_decay_halflife_days: float = Field(
        default=30.0, alias="MEMORY_DECAY_HALFLIFE_DAYS"
    )
    memory_lifecycle_interval_s: int = Field(
        default=3600, alias="MEMORY_LIFECYCLE_INTERVAL_S"
    )

    # ── Observability ──
    observability_trace_to_db: bool = Field(default=True, alias="OBSERVABILITY_TRACE_TO_DB")
    observability_trace_to_stdout: bool = Field(
        default=True, alias="OBSERVABILITY_TRACE_TO_STDOUT"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Validators ──
    @field_validator("safety_min_confidence_to_execute")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("safety_min_confidence_to_execute must be in [0,1]")
        return v

    @field_validator("memory_dedup_sim_threshold")
    @classmethod
    def _sim_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("memory_dedup_sim_threshold must be in [0,1]")
        return v

    @field_validator("qdrant_vector_size")
    @classmethod
    def _vec_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("qdrant_vector_size must be positive")
        return v

    # ── Convenience ──
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.app_cors_origins.split(",") if o.strip()]

    @property
    def is_dev(self) -> bool:
        return self.app_env.lower() in {"development", "dev", "local"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor.

    Cached so import-side calls and FastAPI's `Depends` share the same
    instance. Tests can call `get_settings.cache_clear()` to force a reload.
    """
    return Settings()  # type: ignore[call-arg]
