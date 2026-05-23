"""Confidence calibration.

We store `ConfidenceSample(predicted, actual)` rows whenever an evaluator
emits a confidence and we later observe the binary outcome (passed/failed).
This module fits a simple monotone calibration curve over those samples
and exposes a `calibrate(raw)` function used by the orchestrator before
confidence-gating side-effecting actions.

We use *isotonic-flavored bucketing*: predictions are sorted into B
quantile buckets, the per-bucket empirical pass-rate becomes the
calibrated value, and the curve is enforced to be monotone non-decreasing
via a left-to-right pass. This is dependency-free and stable on small N.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ConfidenceSample
from app.observability import get_logger

logger = get_logger("learning.confidence")


@dataclass(slots=True)
class CalibrationCurve:
    """Piecewise-linear curve from raw → calibrated confidence."""

    xs: list[float]   # ascending raw predictions (bucket midpoints)
    ys: list[float]   # monotone non-decreasing calibrated values
    n: int            # samples used to fit
    fit_at: datetime

    def apply(self, raw: float) -> float:
        if not self.xs:
            return raw
        x = max(0.0, min(1.0, float(raw)))
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]
        i = bisect.bisect_left(self.xs, x)
        x0, x1 = self.xs[i - 1], self.xs[i]
        y0, y1 = self.ys[i - 1], self.ys[i]
        if x1 == x0:
            return y0
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


class ConfidenceCalibrator:
    """Fits + applies a calibration curve.

    Callers:
    - `record(...)` after each subtask outcome.
    - `refit_from_db(session)` periodically (e.g. nightly worker).
    - `calibrate(raw)` synchronously on the hot path.

    Until the first fit, `calibrate` is the identity — calibration only
    helps once enough samples exist.
    """

    def __init__(self, *, lookback_days: int = 60, min_samples: int = 30, buckets: int = 10) -> None:
        self._lookback = lookback_days
        self._min_samples = max(8, int(min_samples))
        self._buckets = max(4, int(buckets))
        self._curve: CalibrationCurve | None = None

    @property
    def curve(self) -> CalibrationCurve | None:
        return self._curve

    async def record(
        self,
        session: AsyncSession,
        *,
        agent: str,
        intent: str,
        predicted: float,
        actual: float,
        user_id=None,
    ) -> None:
        """Insert a sample. `actual` is 1.0 for pass, 0.0 for fail (we accept
        partial credit in [0,1] for evaluator-scored runs)."""
        row = ConfidenceSample(
            user_id=user_id,
            agent=agent,
            intent=intent,
            predicted=max(0.0, min(1.0, float(predicted))),
            actual=max(0.0, min(1.0, float(actual))),
        )
        session.add(row)
        await session.flush()

    async def refit_from_db(self, session: AsyncSession) -> CalibrationCurve | None:
        """Re-fit the curve from recent samples in the DB."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._lookback)
        rows = (
            (
                await session.execute(
                    select(ConfidenceSample.predicted, ConfidenceSample.actual)
                    .where(ConfidenceSample.created_at >= cutoff)
                    .order_by(ConfidenceSample.created_at.asc())
                )
            )
            .all()
        )
        if len(rows) < self._min_samples:
            logger.info("calibration_skipped_min_samples", n=len(rows))
            return self._curve
        samples = [(float(r[0]), float(r[1])) for r in rows]
        self._curve = self.fit(samples, buckets=self._buckets)
        logger.info(
            "calibration_refit",
            n=self._curve.n,
            buckets=self._buckets,
            min=self._curve.ys[0] if self._curve.ys else None,
            max=self._curve.ys[-1] if self._curve.ys else None,
        )
        return self._curve

    @staticmethod
    def fit(samples: Sequence[tuple[float, float]], *, buckets: int = 10) -> CalibrationCurve:
        """Quantile-bucketed empirical curve, monotonized left-to-right."""
        if not samples:
            return CalibrationCurve(xs=[], ys=[], n=0, fit_at=datetime.now(timezone.utc))
        sorted_samples = sorted(samples, key=lambda s: s[0])
        n = len(sorted_samples)
        b = max(1, min(buckets, n))
        # Equal-frequency buckets.
        bucket_size = max(1, n // b)
        xs: list[float] = []
        ys: list[float] = []
        for k in range(b):
            lo = k * bucket_size
            hi = n if k == b - 1 else min(n, (k + 1) * bucket_size)
            sub = sorted_samples[lo:hi]
            if not sub:
                continue
            mid_x = sum(s[0] for s in sub) / len(sub)
            mean_y = sum(s[1] for s in sub) / len(sub)
            xs.append(mid_x)
            ys.append(mean_y)
        # Enforce monotonicity (left-to-right pool-adjacent-violators-lite).
        for i in range(1, len(ys)):
            if ys[i] < ys[i - 1]:
                # Pool with previous: average and propagate back.
                avg = (ys[i - 1] + ys[i]) / 2.0
                ys[i] = avg
                ys[i - 1] = avg
                # one back-pass to settle
                for j in range(i - 2, -1, -1):
                    if ys[j + 1] < ys[j]:
                        m = (ys[j] + ys[j + 1]) / 2.0
                        ys[j] = m
                        ys[j + 1] = m
                    else:
                        break
        return CalibrationCurve(xs=xs, ys=ys, n=n, fit_at=datetime.now(timezone.utc))

    def calibrate(self, raw: float) -> float:
        if self._curve is None:
            return max(0.0, min(1.0, float(raw)))
        return self._curve.apply(raw)
