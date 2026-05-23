"""Self-improvement engine.

Three cooperating pieces:

- `ReflectionEngine`  — turns Reflector outputs into durable Lessons,
                        applies them to memory, records confidence samples.
- `PatternEvolver`    — periodically distills repeated successful episodes
                        into reusable CodingPatterns.
- `ConfidenceCalibrator` — fits a curve over (predicted, actual) samples;
                           callers pass raw confidence through it before
                           gating actions.

These are the *only* writers to lessons / patterns / confidence-calibration
tables. The MemoryAgent writes general facts; learning writes structured
rules and templates.
"""

from app.learning.confidence import ConfidenceCalibrator
from app.learning.evaluation import EvaluationMetrics, score_run
from app.learning.pattern_evolver import PatternEvolver
from app.learning.reflection_engine import ReflectionEngine

__all__ = [
    "ConfidenceCalibrator",
    "EvaluationMetrics",
    "score_run",
    "PatternEvolver",
    "ReflectionEngine",
]
