"""Evaluation metrics — pure helpers used by the orchestrator and learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class EvaluationMetrics:
    passed: int
    failed: int
    skipped: int
    total_subtasks: int
    avg_attempts: float
    score: float            # weighted in [0,1]
    confidence: float       # mean confidence of subtask outputs
    aborted_token_budget: bool


def score_run(
    *,
    statuses: Iterable[tuple[str, int, float]],
    aborted_token_budget: bool = False,
) -> EvaluationMetrics:
    """Compute run-level metrics.

    `statuses` is an iterable of (status, attempts, confidence) per subtask.
    Status values: passed | failed | skipped | running.

    Score formula:
        score = passed_ratio - 0.25 * failed_ratio - 0.05 * (avg_attempts - 1)

    Bounded to [0, 1]. Penalizes long retry chains so a barely-passing run
    after 4 reflections doesn't look as good as one that worked first try.
    """
    passed = failed = skipped = 0
    total_attempts = 0
    confidences: list[float] = []
    n = 0
    for status, attempts, conf in statuses:
        n += 1
        if status == "passed":
            passed += 1
        elif status == "failed":
            failed += 1
        elif status == "skipped":
            skipped += 1
        total_attempts += max(1, int(attempts))
        confidences.append(float(conf))

    if n == 0:
        return EvaluationMetrics(
            passed=0,
            failed=0,
            skipped=0,
            total_subtasks=0,
            avg_attempts=0.0,
            score=0.0,
            confidence=0.0,
            aborted_token_budget=aborted_token_budget,
        )

    passed_ratio = passed / n
    failed_ratio = failed / n
    avg_attempts = total_attempts / n
    score = passed_ratio - 0.25 * failed_ratio - 0.05 * max(0.0, avg_attempts - 1.0)
    if aborted_token_budget:
        score *= 0.7
    score = max(0.0, min(1.0, score))
    confidence = sum(confidences) / n if confidences else 0.0
    return EvaluationMetrics(
        passed=passed,
        failed=failed,
        skipped=skipped,
        total_subtasks=n,
        avg_attempts=avg_attempts,
        score=score,
        confidence=confidence,
        aborted_token_budget=aborted_token_budget,
    )
