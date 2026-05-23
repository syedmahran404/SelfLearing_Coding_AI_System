"""Provider-agnostic data types for chat + embedding calls.

Pydantic models so requests can be validated/serialized at boundaries.
Kept deliberately small — the surface area we cross is `chat` + `embed`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: Role
    content: str
    name: str | None = None
    # Tool-call metadata (provider-dependent, opaque to most call sites).
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Tracked separately because cost is provider/model dependent.
    cost_usd: float = 0.0


class CompletionRequest(BaseModel):
    """Provider-agnostic chat-completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = None
    stop: list[str] | None = None
    # JSON-schema for structured outputs; providers map it appropriately.
    response_schema: dict[str, Any] | None = None
    # Generic key for tracing / billing scopes.
    purpose: str | None = None


class CompletionChunk(BaseModel):
    """One streaming chunk."""

    delta: str = ""
    finish_reason: str | None = None
    # Cumulative token estimate (filled in on the final chunk when known).
    usage: Usage | None = None


class CompletionResponse(BaseModel):
    """Aggregated (non-streaming) result."""

    content: str
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str
    raw: dict[str, Any] | None = None  # provider-specific extras for debugging


class EmbeddingResponse(BaseModel):
    """Result of an embed call."""

    vectors: list[list[float]]
    model: str
    usage: Usage = Field(default_factory=Usage)
