"""Confidence calibration curve fitter."""
from __future__ import annotations

from app.learning.confidence import ConfidenceCalibrator


def test_calibrate_identity_when_no_curve():
    c = ConfidenceCalibrator()
    assert c.calibrate(0.7) == 0.7


def test_fit_produces_monotone_curve():
    # Construct samples where high predicted confidence really does pass more.
    samples = []
    for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Each predicted bucket has 10 outcomes whose mean ≈ x.
        for _ in range(10):
            samples.append((x, 1.0 if x > 0.5 else 0.0))
    curve = ConfidenceCalibrator.fit(samples, buckets=5)
    assert curve.n == len(samples)
    # Monotone non-decreasing.
    assert all(curve.ys[i] <= curve.ys[i + 1] for i in range(len(curve.ys) - 1))
    # Apply is bounded.
    for x in [0.0, 0.5, 1.0]:
        v = curve.apply(x)
        assert 0.0 <= v <= 1.0


def test_apply_interpolates_within_bucket_range():
    # Two-bucket curve: (0.2 -> 0.1), (0.8 -> 0.9).
    samples = [(0.2, 0.1)] * 10 + [(0.8, 0.9)] * 10
    curve = ConfidenceCalibrator.fit(samples, buckets=2)
    mid = curve.apply(0.5)
    assert curve.ys[0] <= mid <= curve.ys[-1]
