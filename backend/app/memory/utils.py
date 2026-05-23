"""Memory-level helpers shared across short/long/episodic/lifecycle modules."""
from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import datetime, timezone

# ─── content normalization ─────────────────────────────────────────────────

_WS = re.compile(r"\s+")


def normalize_content(text: str) -> str:
    """Whitespace-collapse and trim. Used as input to `content_sha`."""
    return _WS.sub(" ", text or "").strip()


def content_sha(text: str) -> str:
    """Stable sha256 of the normalized text."""
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


# ─── decay model ───────────────────────────────────────────────────────────


def utility_score(
    *,
    base: float,
    last_accessed_at: datetime | float,
    access_count: int,
    success_count: int,
    failure_count: int,
    halflife_days: float,
    now: float | None = None,
) -> float:
    """Return a utility ∈ [0,1].

    Formula:
        utility = base * recency * frequency * success_ratio

    where:
        recency        = 0.5 ^ (age_days / halflife)
        frequency      = log2(1 + access_count) / log2(1 + access_count + 4)  -- soft saturation
        success_ratio  = (success + α) / (success + failure + 2α)             -- Laplace smoothing

    Bounded into [0, 1] for stable comparisons.
    """
    now_ts = now if now is not None else time.time()
    if isinstance(last_accessed_at, datetime):
        ts = last_accessed_at.timestamp() if last_accessed_at.tzinfo else (
            last_accessed_at.replace(tzinfo=timezone.utc).timestamp()
        )
    else:
        ts = float(last_accessed_at)

    age_days = max(0.0, (now_ts - ts) / 86400.0)
    halflife = max(0.5, halflife_days)
    recency = math.pow(0.5, age_days / halflife)

    ac = max(0, access_count)
    frequency = (math.log2(1 + ac)) / (math.log2(1 + ac + 4) + 1e-9) if ac > 0 else 0.25

    alpha = 1.0
    success_ratio = (success_count + alpha) / (success_count + failure_count + 2 * alpha)

    score = base * recency * frequency * success_ratio
    return max(0.0, min(1.0, score))


# ─── tiny vector math (avoid pulling numpy where unnecessary) ──────────────


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
