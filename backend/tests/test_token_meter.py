"""Token meter — estimation + cost tracking."""
from __future__ import annotations

from app.llm.token_meter import TokenMeter, estimate_messages_tokens, estimate_tokens
from app.llm.types import ChatMessage, Role


def test_estimate_tokens_nonzero():
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("") == 0


def test_estimate_messages_includes_overhead():
    msgs = [
        ChatMessage(role=Role.SYSTEM, content="system"),
        ChatMessage(role=Role.USER, content="hi"),
    ]
    n = estimate_messages_tokens(msgs)
    assert n >= estimate_tokens("system") + estimate_tokens("hi")


def test_meter_records_per_purpose():
    m = TokenMeter()
    m.record(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50, purpose="planner")
    m.record(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=80, purpose="coder")
    snap = m.snapshot()
    assert snap["totals"]["gpt-4o-mini"]["prompt"] == 300
    assert snap["totals"]["gpt-4o-mini"]["completion"] == 130
    assert "planner" in snap["cost_by_purpose"]
    assert "coder" in snap["cost_by_purpose"]
    assert snap["cost_usd"] >= 0


def test_meter_handles_unknown_model_gracefully():
    m = TokenMeter()
    cost = m.record(model="some-future-model", prompt_tokens=1000, completion_tokens=500)
    # Unknown model → zero pricing, zero cost.
    assert cost == 0.0
