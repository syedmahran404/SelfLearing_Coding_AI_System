"""Deterministic local LLM provider.

Used for:
- Tests (no network, fast, reproducible)
- Local dev when no API key is configured
- CI / smoke runs

Behavior:
- `complete` / `stream` produce a *useful* output by inspecting the last
  user message and applying simple rules:
    * if the last message asks for JSON and provides a schema, return a
      valid JSON document matching the schema (best-effort)
    * if it looks like a code request, return a small, syntactically valid
      placeholder snippet
    * otherwise echo a short summary of the messages

- `embed` produces stable hash-derived vectors so memory tests can verify
  retrieval without depending on a real embedding model.

This is *not* a stand-in for real intelligence — it's a contract-compatible
stub that lets the rest of the system run end-to-end.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.llm.provider import LLMProvider
from app.llm.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingResponse,
    Role,
    Usage,
)


class LocalProvider(LLMProvider):
    name = "local"

    def __init__(self, settings: Settings, *, vector_size: int | None = None) -> None:
        super().__init__(settings)
        self._vector_size = vector_size or settings.qdrant_vector_size

    # ── completion ──
    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        text = self._generate(req)
        prompt_tokens = self.estimate(req.messages, req.model)
        completion_tokens = max(1, len(text) // 4)
        cost = self.meter.record(
            model=req.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            purpose=req.purpose,
        )
        return CompletionResponse(
            content=text,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
            ),
            model=req.model,
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[CompletionChunk]:  # type: ignore[override]
        text = self._generate(req)
        # Emit in small chunks to mimic streaming.
        chunk = 24
        for i in range(0, len(text), chunk):
            await asyncio.sleep(0)  # yield to loop
            yield CompletionChunk(delta=text[i : i + chunk])
        prompt_tokens = self.estimate(req.messages, req.model)
        completion_tokens = max(1, len(text) // 4)
        cost = self.meter.record(
            model=req.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            purpose=req.purpose,
        )
        yield CompletionChunk(
            delta="",
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
            ),
        )

    # ── embeddings ──
    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        m = model or self.settings.llm_embedding_model
        vectors = [self._hash_vector(t) for t in texts]
        total_tokens = sum(max(1, len(t) // 4) for t in texts)
        cost = self.meter.record(model=m, prompt_tokens=total_tokens, completion_tokens=0)
        return EmbeddingResponse(
            vectors=vectors,
            model=m,
            usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens, cost_usd=cost),
        )

    # ── generation core ──
    def _generate(self, req: CompletionRequest) -> str:
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == Role.USER), ""
        )
        last_user_lc = last_user.lower()

        if req.response_schema is not None:
            return json.dumps(_synthesize_from_schema(req.response_schema), separators=(",", ":"))

        if any(k in last_user_lc for k in ("write", "implement", "code", "function", "class")):
            return _placeholder_code(last_user)

        if "plan" in last_user_lc:
            return _placeholder_plan(last_user)

        return _placeholder_answer(last_user)

    # ── deterministic vectors ──
    def _hash_vector(self, text: str) -> list[float]:
        """Stable vector derived from a SHA-256 hash. Unit-norm."""
        size = self._vector_size
        out: list[float] = []
        seed = text.encode("utf-8") or b"\x00"
        h = hashlib.sha256(seed).digest()
        i = 0
        while len(out) < size:
            # Stretch by re-hashing.
            h = hashlib.sha256(h + i.to_bytes(2, "little")).digest()
            for j in range(0, len(h), 2):
                if len(out) >= size:
                    break
                v = int.from_bytes(h[j : j + 2], "little") / 65535.0
                out.append(v - 0.5)  # center around 0
            i += 1
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]


# ── schema-driven synthesis ──


def _synthesize_from_schema(schema: dict[str, Any]) -> Any:
    """Produce a value that satisfies a small subset of JSON-schema."""
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        out: dict[str, Any] = {}
        props = schema.get("properties", {})
        required = set(schema.get("required", list(props.keys())))
        for name, sub in props.items():
            if name in required:
                out[name] = _synthesize_from_schema(sub)
        return out
    if t == "array":
        item = schema.get("items", {"type": "string"})
        # Honor minItems if present, else 1 element.
        n = max(1, int(schema.get("minItems", 1)))
        return [_synthesize_from_schema(item) for _ in range(n)]
    if t == "string":
        return schema.get("default", "ok")
    if t == "integer":
        return int(schema.get("default", 0))
    if t == "number":
        return float(schema.get("default", 0.0))
    if t == "boolean":
        return bool(schema.get("default", False))
    return None


# ── canned outputs (kept short) ──


def _placeholder_code(prompt: str) -> str:
    # Try to detect language.
    if re.search(r"\b(python|django|flask|fastapi|pytest)\b", prompt, re.I):
        return (
            "```python\n"
            "def solve(x):\n"
            "    \"\"\"Auto-generated placeholder by LocalProvider.\"\"\"\n"
            "    return x\n"
            "```"
        )
    if re.search(r"\b(typescript|tsx|javascript|jsx|node|react)\b", prompt, re.I):
        return (
            "```ts\n"
            "export function solve<T>(x: T): T {\n"
            "  // Auto-generated placeholder by LocalProvider.\n"
            "  return x;\n"
            "}\n"
            "```"
        )
    return "```\n# placeholder\n```"


def _placeholder_plan(prompt: str) -> str:
    return (
        "1) Understand the request\n"
        "2) Identify required inputs\n"
        "3) Implement the smallest correct solution\n"
        "4) Verify with a test case\n"
        "5) Reflect and record the lesson"
    )


def _placeholder_answer(prompt: str) -> str:
    head = (prompt or "").strip().splitlines()[0:1]
    excerpt = head[0] if head else ""
    if len(excerpt) > 80:
        excerpt = excerpt[:80] + "…"
    return f"(local-provider) acknowledged: {excerpt}"
