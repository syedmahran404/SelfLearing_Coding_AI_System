"""Intent router — heuristic short-circuit when input is unambiguous."""
from __future__ import annotations

import pytest

from app.orchestration.router import IntentRouter
from app.schemas.agent_io import TaskIntent


@pytest.mark.asyncio
async def test_qa_for_questions(local_llm):
    r = IntentRouter(local_llm)
    intent, conf = await r.classify("What is the difference between let and const?")
    assert intent == TaskIntent.QA
    assert conf > 0


@pytest.mark.asyncio
async def test_code_for_imperative(local_llm):
    r = IntentRouter(local_llm)
    intent, _ = await r.classify("Write a Python function that reverses a string")
    assert intent == TaskIntent.CODE


@pytest.mark.asyncio
async def test_debug_for_stack_traces(local_llm):
    r = IntentRouter(local_llm)
    intent, _ = await r.classify("ImportError: cannot find module foo. Fix this.")
    assert intent == TaskIntent.DEBUG


@pytest.mark.asyncio
async def test_refactor_for_refactor_verbs(local_llm):
    r = IntentRouter(local_llm)
    intent, _ = await r.classify("Refactor this module to use pathlib")
    assert intent == TaskIntent.REFACTOR
