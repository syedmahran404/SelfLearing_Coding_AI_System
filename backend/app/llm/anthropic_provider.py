"""Anthropic (Claude) provider.

Implements the Anthropic Messages API (`/v1/messages`) over httpx. Maps
our `ChatMessage` shape into Anthropic's expected schema:
- system messages collapsed into a top-level `system` string
- the rest mapped to `messages: [{role, content}]`

Embeddings: Anthropic does not currently expose a public embedding API.
We fall back to the configured embedding model — if that's also Anthropic
we raise a clear error so callers know to use a separate provider for
embeddings.
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
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingResponse,
    Role,
    Usage,
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        self._base_url = (settings.anthropic_base_url or "https://api.anthropic.com").rstrip("/")
        self._headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
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
        r = await self._client.post("/v1/messages", json=body)
        r.raise_for_status()
        data = r.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage_d = data.get("usage", {}) or {}
        prompt_tokens = int(usage_d.get("input_tokens", 0))
        completion_tokens = int(usage_d.get("output_tokens", 0))
        cost = self.meter.record(
            model=req.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            purpose=req.purpose,
        )
        return CompletionResponse(
            content=text,
            finish_reason=data.get("stop_reason"),
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
        async with self._client.stream("POST", "/v1/messages", json=body) as r:
            r.raise_for_status()
            event: str | None = None
            async for raw in r.aiter_lines():
                if not raw:
                    event = None
                    continue
                if raw.startswith("event:"):
                    event = raw[6:].strip()
                    continue
                if not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event == "content_block_delta":
                    delta = obj.get("delta", {}).get("text", "")
                    if delta:
                        yield CompletionChunk(delta=delta)
                elif event == "message_delta":
                    stop = obj.get("delta", {}).get("stop_reason")
                    if stop:
                        yield CompletionChunk(delta="", finish_reason=stop)
                elif event == "message_stop":
                    break

    # ── embeddings ──
    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse:
        raise NotImplementedError(
            "Anthropic does not expose a public embeddings API. "
            "Use a dedicated embedding provider (e.g. OpenAI text-embedding-3-small) "
            "for vector operations."
        )

    # ── helpers ──
    def _build_body(self, req: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        sys_parts: list[str] = []
        msgs: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == Role.SYSTEM:
                sys_parts.append(m.content)
            else:
                role = "assistant" if m.role == Role.ASSISTANT else "user"
                msgs.append({"role": role, "content": m.content})

        body: dict[str, Any] = {
            "model": req.model,
            "messages": msgs,
            "max_tokens": req.max_tokens or 4096,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": stream,
        }
        if sys_parts:
            body["system"] = "\n\n".join(sys_parts)
        if req.stop:
            body["stop_sequences"] = req.stop
        return body

    @staticmethod
    def _msg(role: Role, text: str) -> ChatMessage:
        return ChatMessage(role=role, content=text)
