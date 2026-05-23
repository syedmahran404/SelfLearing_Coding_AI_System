"""Safety layer — validators, guards, hallucination, integrity."""
from __future__ import annotations

import pytest

from app.safety.guards import CircuitBreaker, ExecutionGuard, GuardDecision, GuardError
from app.safety.hallucination import HallucinationGuard
from app.safety.integrity import (
    IntegrityError,
    content_fingerprint,
    verify_fingerprint,
)
from app.safety.validators import (
    ValidationError,
    code_health_score,
    validate_code_change,
    validate_memory_write,
    validate_path,
)
from app.schemas.agent_io import CodeChange


def test_validate_path_rejects_traversal_and_absolute():
    with pytest.raises(ValidationError):
        validate_path("../etc/passwd")
    with pytest.raises(ValidationError):
        validate_path("/abs")
    assert validate_path("a/b/c.py") == "a/b/c.py"


def test_validate_code_change_requires_content_for_create():
    c = CodeChange(path="a.py", operation="create")
    with pytest.raises(ValidationError):
        validate_code_change(c)
    c2 = CodeChange(path="a.py", operation="create", new_content="x = 1\n")
    assert validate_code_change(c2).path == "a.py"


def test_validate_memory_write_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        validate_memory_write("hello", "weird_kind")
    content, kind = validate_memory_write("  hello  ", "fact")
    assert content == "hello" and kind == "fact"


def test_execution_guard_aborts_on_token_overrun():
    g = ExecutionGuard(max_depth=4, max_tokens=100, max_wallclock_s=10)
    g.add_tokens(150)
    decision, _ = g.check()
    assert decision == GuardDecision.ABORT
    with pytest.raises(GuardError):
        g.assert_allow()


def test_execution_guard_backoff_on_consecutive_failures():
    g = ExecutionGuard(max_depth=4, max_tokens=10_000, max_wallclock_s=10)
    for _ in range(4):
        g.record_failure()
    decision, _ = g.check()
    assert decision == GuardDecision.BACKOFF


def test_circuit_breaker_trips_then_recovers():
    b = CircuitBreaker(threshold=3, cooldown_s=60)
    assert not b.is_open("agent", "intent")
    b.record_failure("agent", "intent")
    b.record_failure("agent", "intent")
    tripped = b.record_failure("agent", "intent")
    assert tripped is True
    assert b.is_open("agent", "intent")
    b.record_success("agent", "intent")
    assert not b.is_open("agent", "intent")


def test_integrity_fingerprint_round_trip():
    fp = content_fingerprint("hello   world\n")
    verify_fingerprint(content="hello world", expected_sha=fp.sha256)
    with pytest.raises(IntegrityError):
        verify_fingerprint(content="totally different", expected_sha=fp.sha256)


def test_code_health_score_bounded():
    s = code_health_score("a = 1\n")
    assert 0.0 <= s <= 1.0
    bad = "TODO\n" * 20 + ("x" * 300 + "\n") * 5
    assert code_health_score(bad) < 1.0


@pytest.mark.asyncio
async def test_hallucination_extracts_python_imports_and_calls():
    g = HallucinationGuard()
    text = "import widgetfoo\nwidgetfoo.do_thing()\nresult = numpy.array([1,2,3])\n"
    check = await g.check_code(text, language="python")
    # numpy is in the allowlist; widgetfoo is not.
    assert any(r.startswith("widgetfoo") for r in check.references)
    assert any("numpy" in v for v in check.verified)


def test_path_rejects_dotdot_segments():
    with pytest.raises(ValidationError):
        validate_path("a/../b")
