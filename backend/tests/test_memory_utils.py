"""Memory utility helpers — pure functions, no I/O."""
from __future__ import annotations

import time

from app.memory.utils import (
    content_sha,
    cosine,
    normalize_content,
    utility_score,
)


def test_normalize_content_collapses_whitespace():
    assert normalize_content("  hello\n  world  ") == "hello world"


def test_content_sha_is_deterministic_and_normalization_aware():
    a = content_sha("hello world")
    b = content_sha("hello   world\n")
    assert a == b


def test_cosine_basic():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert -1.001 < cosine([1, 0], [-1, 0]) < -0.999


def test_utility_score_in_range_and_decays_with_age():
    now = time.time()
    fresh = utility_score(
        base=0.7,
        last_accessed_at=now,
        access_count=5,
        success_count=4,
        failure_count=1,
        halflife_days=30,
        now=now,
    )
    old = utility_score(
        base=0.7,
        last_accessed_at=now - 86400 * 365,  # 1 year ago
        access_count=5,
        success_count=4,
        failure_count=1,
        halflife_days=30,
        now=now,
    )
    assert 0.0 <= fresh <= 1.0
    assert 0.0 <= old <= 1.0
    assert fresh > old


def test_utility_score_failure_reduces():
    now = time.time()
    good = utility_score(
        base=0.5, last_accessed_at=now, access_count=10,
        success_count=10, failure_count=0, halflife_days=30, now=now,
    )
    bad = utility_score(
        base=0.5, last_accessed_at=now, access_count=10,
        success_count=0, failure_count=10, halflife_days=30, now=now,
    )
    assert good > bad
