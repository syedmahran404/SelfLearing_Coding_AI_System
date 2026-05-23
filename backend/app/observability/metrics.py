"""In-memory metrics aggregator.

Subscribes to the `Tracer` and aggregates rolling counters by kind/name
(agent.<n>, tool.<n>, llm, memory.<op>, …). Exposed via `snapshot()` for
the metrics endpoint and the frontend dashboard.

Goals:
- Bounded: per-name top-N by count (LRU via OrderedDict).
- Cheap: every event is one dict-key lookup + a few additions.
- Resettable: tests can call `reset()` between scenarios.

Not a replacement for Prometheus — it's the *quick-look* dashboard that
ships with the app. For real production, point a Prometheus exporter at
the same Tracer.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.observability.tracing import SpanPhase, TraceEvent


@dataclass(slots=True)
class _Counters:
    starts: int = 0
    ends: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms_total: float = 0.0
    duration_ms_max: float = 0.0


@dataclass(slots=True)
class MetricsSnapshot:
    uptime_s: float
    by_kind: dict[str, dict[str, float]]
    by_name: dict[str, dict[str, float]]
    totals: dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Aggregator subscriber for the Tracer."""

    MAX_NAMES = 512  # LRU cap; events under unseen names beyond this are bucketed into "other"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_kind: dict[str, _Counters] = {}
        self._by_name: OrderedDict[str, _Counters] = OrderedDict()
        self._started_at = time.time()

    async def __call__(self, event: TraceEvent) -> None:
        with self._lock:
            kc = self._by_kind.setdefault(event.kind.value, _Counters())
            nc = self._touch_name(event.name)
            self._update(kc, event)
            self._update(nc, event)

    def _touch_name(self, name: str) -> _Counters:
        if name in self._by_name:
            self._by_name.move_to_end(name)
            return self._by_name[name]
        if len(self._by_name) >= self.MAX_NAMES:
            other = self._by_name.setdefault("__other__", _Counters())
            return other
        c = _Counters()
        self._by_name[name] = c
        return c

    @staticmethod
    def _update(c: _Counters, event: TraceEvent) -> None:
        if event.phase == SpanPhase.START:
            c.starts += 1
            return
        if event.phase == SpanPhase.ERROR:
            c.errors += 1
        if event.phase == SpanPhase.END:
            c.ends += 1
        if event.tokens_in is not None:
            c.tokens_in += int(event.tokens_in)
        if event.tokens_out is not None:
            c.tokens_out += int(event.tokens_out)
        if event.cost_usd is not None:
            c.cost_usd += float(event.cost_usd)
        if event.duration_ms is not None:
            c.duration_ms_total += float(event.duration_ms)
            c.duration_ms_max = max(c.duration_ms_max, float(event.duration_ms))

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            by_kind = {k: _to_dict(v) for k, v in self._by_kind.items()}
            by_name = {k: _to_dict(v) for k, v in self._by_name.items()}
            totals = {
                "tokens_in": sum(v.tokens_in for v in self._by_kind.values()),
                "tokens_out": sum(v.tokens_out for v in self._by_kind.values()),
                "cost_usd": round(sum(v.cost_usd for v in self._by_kind.values()), 6),
                "errors": sum(v.errors for v in self._by_kind.values()),
                "spans_ended": sum(v.ends for v in self._by_kind.values()),
            }
        return MetricsSnapshot(
            uptime_s=time.time() - self._started_at,
            by_kind=by_kind,
            by_name=by_name,
            totals=totals,
        )

    def reset(self) -> None:
        with self._lock:
            self._by_kind.clear()
            self._by_name.clear()
            self._started_at = time.time()


def _to_dict(c: _Counters) -> dict[str, float]:
    avg = c.duration_ms_total / c.ends if c.ends else 0.0
    return {
        "starts": c.starts,
        "ends": c.ends,
        "errors": c.errors,
        "tokens_in": c.tokens_in,
        "tokens_out": c.tokens_out,
        "cost_usd": round(c.cost_usd, 6),
        "duration_ms_total": round(c.duration_ms_total, 3),
        "duration_ms_max": round(c.duration_ms_max, 3),
        "duration_ms_avg": round(avg, 3),
    }
