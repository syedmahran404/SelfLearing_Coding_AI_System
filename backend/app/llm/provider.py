"""LLMProvider abstract base class.

The contract every concrete provider implements. Three operations:
- `complete`        — non-streaming chat completion
- `stream`          — async-iterator of `CompletionChunk`s
- `embed`           — embeddings for one or more texts

Concrete providers also own a `TokenMeter` so cost/usage tracking is
centralized.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.config import Settings
from app.llm.token_meter import TokenMeter, estimate_messages_tokens
from app.llm.types import (
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingResponse,
)


class LLMProvider(ABC):
    """Provider-agnostic LLM interface."""

    name: str = "abstract"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.meter = TokenMeter()

    # ── Pre-flight estimate (no network) ──
    def estimate(self, messages: list[ChatMessage], model: str | None = None) -> int:
        return estimate_messages_tokens(messages, model or self.settings.llm_default_model)

    # ── Operations ──
    @abstractmethod
    async def complete(self, req: CompletionRequest) -> CompletionResponse: ...

    @abstractmethod
    def stream(self, req: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """Return an async iterator over completion chunks. Implementations are
        usually `async def` generators; declared as plain method so concrete
        classes can return `AsyncIterator[...]`."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str], model: str | None = None) -> EmbeddingResponse: ...

    # ── Convenience: aggregate streaming into a CompletionResponse ──
    async def stream_to_response(self, req: CompletionRequest) -> CompletionResponse:
        parts: list[str] = []
        finish: str | None = None
        usage = None
        async for chunk in self.stream(req):
            if chunk.delta:
                parts.append(chunk.delta)
            if chunk.finish_reason is not None:
                finish = chunk.finish_reason
            if chunk.usage is not None:
                usage = chunk.usage
        from app.llm.types import Usage

        if usage is None:
            text = "".join(parts)
            prompt = self.estimate(req.messages, req.model)
            completion = self.meter.cost_of(req.model, 0, 0)  # noqa: F841 — not used here
            usage = Usage(
                prompt_tokens=prompt,
                completion_tokens=max(1, len(text) // 4),
                total_tokens=prompt + max(1, len(text) // 4),
            )
        return CompletionResponse(
            content="".join(parts),
            finish_reason=finish,
            usage=usage,
            model=req.model,
        )

    # ── Lifecycle ──
    async def aclose(self) -> None:
        """Override if the provider holds an HTTP client / pool."""
        return None
