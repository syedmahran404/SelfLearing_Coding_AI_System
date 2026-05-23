"""LocalProvider — deterministic-stub contract."""
from __future__ import annotations

import pytest

from app.llm.types import ChatMessage, CompletionRequest, Role


@pytest.mark.asyncio
async def test_complete_returns_non_empty_content(local_llm):
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role=Role.USER, content="Hello")],
    )
    resp = await local_llm.complete(req)
    assert resp.content
    assert resp.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_stream_emits_chunks_and_final_usage(local_llm):
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role=Role.USER, content="Tell me about pytest")],
    )
    parts: list[str] = []
    final_usage = None
    async for chunk in local_llm.stream(req):
        if chunk.delta:
            parts.append(chunk.delta)
        if chunk.usage is not None:
            final_usage = chunk.usage
    assert "".join(parts)
    assert final_usage is not None
    assert final_usage.total_tokens > 0


@pytest.mark.asyncio
async def test_embed_vectors_are_stable_and_unit_norm(local_llm):
    a1 = await local_llm.embed(["hello world"])
    a2 = await local_llm.embed(["hello world"])
    b = await local_llm.embed(["different text"])
    assert a1.vectors[0] == a2.vectors[0]  # determinism
    assert a1.vectors[0] != b.vectors[0]
    norm_sq = sum(x * x for x in a1.vectors[0])
    assert 0.99 < norm_sq < 1.01  # L2-normalized
    assert len(a1.vectors[0]) == local_llm.settings.qdrant_vector_size


@pytest.mark.asyncio
async def test_response_schema_returns_valid_json(local_llm):
    schema = {
        "type": "object",
        "required": ["passed", "score"],
        "properties": {
            "passed": {"type": "boolean"},
            "score": {"type": "number"},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
    }
    req = CompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role=Role.USER, content="evaluate")],
        response_schema=schema,
    )
    resp = await local_llm.complete(req)
    import json

    parsed = json.loads(resp.content)
    assert "passed" in parsed
    assert isinstance(parsed["score"], (int, float))
