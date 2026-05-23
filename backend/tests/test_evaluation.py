"""Run-level evaluation metrics."""
from __future__ import annotations

from app.learning.evaluation import score_run


def test_all_pass_first_try_scores_high():
    m = score_run(
        statuses=[("passed", 1, 0.9), ("passed", 1, 0.8), ("passed", 1, 0.95)],
    )
    assert m.score >= 0.9
    assert m.passed == 3 and m.failed == 0
    assert 0.8 <= m.confidence <= 1.0


def test_repeated_attempts_penalize_score():
    m1 = score_run(statuses=[("passed", 1, 0.9)])
    m2 = score_run(statuses=[("passed", 4, 0.9)])
    assert m2.score < m1.score


def test_failures_drag_score_down():
    m = score_run(
        statuses=[("passed", 1, 0.8), ("failed", 2, 0.4), ("failed", 2, 0.4)],
    )
    assert m.score < 0.5
    assert m.failed == 2


def test_token_budget_abort_caps_score():
    m_ok = score_run(statuses=[("passed", 1, 0.9)])
    m_aborted = score_run(statuses=[("passed", 1, 0.9)], aborted_token_budget=True)
    assert m_aborted.score < m_ok.score
    assert m_aborted.aborted_token_budget is True
