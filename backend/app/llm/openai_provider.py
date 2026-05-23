"""OpenAI-compatible provider.

Speaks the OpenAI Chat Completions + Embeddings APIs over plain HTTP via
`httpx`. We avoid the official SDK to keep the dependency surface small
and to make swapping in OpenAI-compatible endpoints (Azure, vLLM, Together,
Groq, etc.) trivial via `OPENAI_BASE_URL`.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import Settings
from app.llm.provider import LLMProvider
from app.llm.retry import with_retry
from app.llm.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingResponse,
    Usage,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in .env or switch LLM_PROVIDER to 'local'."
            )
        self._base_url = (settings.openai_base_url or "https://api.openai.com").rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=settings.llm_request_timeout_s,
            headers=self._headers,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── chat completion ──
    @with_retry()
    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        body = self._build_body(req, stream=False)
        r = await self._client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        msg = choice["message"]
        usage_d = data.get("usage", {}) or {}
        prompt_tokens = int(usage_d.get("prompt_tokens", 0))
        completion_tokens = int(usage_d.get("completion_tokens", 0))
        cost = self.meter.record(
            model=req.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            purpose=req.purpose,
        )
        return CompletionResponse(
            content=msg.get("content") or "",
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=cost,
            ),
            model=req.model,
            raw=data,
        )

    @with_retry()
    async def stream(self, req: CompletionRequest) -> AsyncIterator[CompletionChunk]:  # type: ignore[override]
        body = self._build_body(req, stream=True)
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for raw in r.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                delta = (ch.get("delta") or {}).get("content") or ""
                finish = ch.get("finish_reason")
                yield CompletionChunk(delta=delta, finish_reason=finish)
        # OpenAI streaming doesn't return usage; meter records on the assumption
        # the caller used `stream_to_response` which relies on the heuristic.

    # ── embeddings ──
    @with_retry()
    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        m = model or self.settings.llm_embedding_model
        body = {"model": m, "input": texts}
        r = await self._client.post("/v1/embeddings", json=body)
        r.raise_for_status()
        data = r.json()
        vectors = [item["embedding"] for item in data["data"]]
        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        cost = self.meter.record(model=m, prompt_tokens=prompt_tokens, completion_tokens=0)
        return EmbeddingResponse(
            vectors=vectors,
            model=m,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
                cost_usd=cost,
            ),
        )

    # ── helpers ──
    def _build_body(self, req: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [self._msg_to_dict(m) for m in req.messages],
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": stream,
        }
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.stop:
            body["stop"] = req.stop
        if req.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured", "schema": req.response_schema},
            }
        return body

    @staticmethod
    def _msg_to_dict(m: Any) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        return d
