"""Structured logging.

Uses `structlog` over the stdlib logger. Every log record carries the active
`trace_id` (when one is bound) so traces and logs cross-reference cleanly.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from app.config import LogLevel


def _trace_id_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach the active trace_id (if any) to every log event."""
    # Lazy import to avoid a cycle at module-load time.
    from app.observability.tracing import current_trace_id

    tid = current_trace_id()
    if tid is not None:
        event_dict.setdefault("trace_id", tid)
    return event_dict


def configure_logging(level: LogLevel = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog + stdlib logging.

    Call once at app startup. Idempotent.
    """
    # Configure stdlib logging — structlog wraps it.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level),
    )

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _trace_id_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Use a stable `name` (typically the module path) for every logger so log
    queries can filter by source.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
