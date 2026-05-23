"""Safety layer.

Cross-cutting checks that wrap the rest of the system. Each module here
is small and pure; the orchestrator and route handlers compose them.

- `validators`        : structural validation of agent outputs before we
                        act on them.
- `guards`            : execution caps (depth, tokens, wallclock, circuit
                        breaker on repeated failure).
- `hallucination`     : extracts API references from generated code and
                        checks them against the RAG / project index.
- `rollback`          : staged-write area; promote on success, discard on
                        failure.
- `integrity`         : content-addressed hashes for memory writes; tamper
                        detection at read time.
"""
from app.safety.guards import CircuitBreaker, ExecutionGuard, GuardDecision, GuardError
from app.safety.hallucination import HallucinationCheck, HallucinationGuard
from app.safety.integrity import IntegrityError, content_fingerprint, verify_fingerprint
from app.safety.rollback import RollbackArea, StagedFile
from app.safety.validators import (
    ValidationError,
    validate_agent_output,
    validate_code_change,
)

__all__ = [
    "validate_agent_output",
    "validate_code_change",
    "ValidationError",
    "ExecutionGuard",
    "GuardDecision",
    "GuardError",
    "CircuitBreaker",
    "HallucinationGuard",
    "HallucinationCheck",
    "RollbackArea",
    "StagedFile",
    "content_fingerprint",
    "verify_fingerprint",
    "IntegrityError",
]
