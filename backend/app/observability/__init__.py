"""Observability primitives: structured logging, tracing, metrics.

This is the *backbone* of debugging an autonomous system. Every agent call,
tool run, retrieval, and reflection emits a structured event that can be
queried by `trace_id` to reconstruct an entire run.
"""

from app.observability.logger import configure_logging, get_logger
from app.observability.metrics import MetricsCollector, MetricsSnapshot
from app.observability.persistence import TracePersister
from app.observability.tracing import (
    TraceEvent,
    Tracer,
    bind_trace,
    current_trace_id,
    new_trace_id,
    reset_trace,
    trace_span,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "TraceEvent",
    "Tracer",
    "current_trace_id",
    "new_trace_id",
    "trace_span",
    "bind_trace",
    "reset_trace",
    "MetricsCollector",
    "MetricsSnapshot",
    "TracePersister",
]
