"""LLM provider abstraction.

Single typed interface (`LLMProvider`) implemented by:
- OpenAIProvider     — production
- AnthropicProvider  — production (alternate)
- LocalProvider      — deterministic stub for tests / offline dev

`build_llm_provider(settings)` selects the implementation from
`settings.llm_provider`. The rest of the app holds an `LLMProvider` and
never imports a concrete class — making provider swaps a config change.
"""
from __future__ import annotations

from app.config import Settings
from app.llm.provider import LLMProvider
from app.llm.token_meter import TokenMeter, estimate_tokens
from app.llm.types import (
    ChatMessage,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    EmbeddingResponse,
    Role,
    Usage,
)

__all__ = [
    "build_llm_provider",
    "LLMProvider",
    "TokenMeter",
    "estimate_tokens",
    "ChatMessage",
    "CompletionChunk",
    "CompletionRequest",
    "CompletionResponse",
    "EmbeddingResponse",
    "Role",
    "Usage",
]


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Factory: select and construct a provider from settings."""
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if provider == "local":
        from app.llm.local_provider import LocalProvider

        return LocalProvider(settings)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
