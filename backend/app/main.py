"""FastAPI application entrypoint.

Composition root: every long-lived dependency (DB engine, Redis client,
Qdrant store, LLM provider, agent registry, tool registry, orchestrator) is
constructed exactly once in the lifespan handler and stashed on `app.state`.
Routes pick them up via `Depends(...)` accessors.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.db.qdrant import init_qdrant, shutdown_qdrant
from app.db.redis_client import init_redis, shutdown_redis
from app.db.session import init_engine, shutdown_engine
from app.observability import (
    Tracer,
    bind_trace,
    configure_logging,
    get_logger,
    new_trace_id,
    reset_trace,
)
from app.observability.tracing import TraceEvent

logger = get_logger("main")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: init resources on startup, dispose on shutdown."""
    settings: Settings = get_settings()
    configure_logging(settings.app_log_level, json_logs=not settings.is_dev)
    logger.info(
        "app_startup",
        version=__version__,
        env=settings.app_env,
        llm_provider=settings.llm_provider,
    )

    # Tracer (in-process pub/sub).
    tracer = Tracer()
    if settings.observability_trace_to_stdout:
        tracer.add_global_subscriber(_stdout_trace_subscriber)

    # Persistent resources.
    init_engine(settings)
    init_redis(settings)
    await init_qdrant(settings)

    # Metrics collector + DB trace persister.
    from app.db.session import session_factory
    from app.observability.metrics import MetricsCollector
    from app.observability.persistence import TracePersister

    metrics = MetricsCollector()
    tracer.add_global_subscriber(metrics)
    persister: TracePersister | None = None
    if settings.observability_trace_to_db:
        persister = TracePersister(session_factory=session_factory())
        persister.attach(tracer)

    # Stash on app.state for route access.
    app.state.settings = settings
    app.state.tracer = tracer
    app.state.metrics = metrics
    app.state.trace_persister = persister

    # Late imports avoid circular initialization between modules that depend
    # on settings/db being ready.
    from app.api.routes import register_routes
    from app.bootstrap import build_runtime

    runtime = await build_runtime(settings, tracer)
    app.state.runtime = runtime
    register_routes(app)

    logger.info("app_ready")
    try:
        yield
    finally:
        logger.info("app_shutdown")
        with contextlib.suppress(Exception):
            await runtime.shutdown()
        if persister is not None:
            with contextlib.suppress(Exception):
                await persister.shutdown()
        await shutdown_qdrant()
        await shutdown_redis()
        await shutdown_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Self-Learning Coding AI",
        version=__version__,
        description="Autonomous coding agent with multi-agent orchestration & layered memory.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Bind a trace_id for every HTTP request.

        Honor an inbound `x-trace-id` header (useful when the frontend wants
        to subscribe to events for a specific run before issuing the request).
        """
        inbound = request.headers.get("x-trace-id")
        trace_id = inbound if inbound and len(inbound) <= 64 else new_trace_id()
        token = bind_trace(trace_id)
        try:
            response = await call_next(request)
            response.headers["x-trace-id"] = trace_id
            return response
        finally:
            reset_trace(token)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": str(exc)},
        )

    return app


async def _stdout_trace_subscriber(event: TraceEvent) -> None:
    """Cheap stdout subscriber for local dev — pretty-prints span events."""
    logger.info(
        "trace",
        kind=event.kind.value,
        name=event.name,
        phase=event.phase.value,
        span=event.span_id[:8],
        parent=(event.parent_span_id or "")[:8] or None,
        duration_ms=event.duration_ms,
        tokens_in=event.tokens_in,
        tokens_out=event.tokens_out,
        error=event.error,
    )


app = create_app()
