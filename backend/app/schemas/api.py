"""HTTP request/response schemas for the public API."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── chat ──


class ChatRequest(BaseModel):
    """POST /chat — one user turn."""

    message: str = Field(min_length=1)
    session_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    stream: bool = True


class ChatResponseChunk(BaseModel):
    """One streamed chunk over SSE."""

    type: str  # "delta" | "trace" | "done" | "error" | "memory" | "tool"
    data: Any = None


# ── projects ──


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    slug: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9-_]*$")
    description: str | None = None
    languages: list[str] = Field(default_factory=list)
    repo_root: str | None = None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    description: str | None = None
    languages: list[str] = Field(default_factory=list)
    repo_root: str | None = None
    created_at: datetime
    updated_at: datetime


# ── sessions ──


class SessionCreate(BaseModel):
    title: str | None = None
    project_id: uuid.UUID | None = None


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    project_id: uuid.UUID | None
    started_at: datetime
    last_active_at: datetime
    tokens_used: int


# ── memory ──


class MemoryQuery(BaseModel):
    query: str
    project_id: uuid.UUID | None = None
    kind: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class MemoryWrite(BaseModel):
    kind: str
    content: str
    tags: list[str] = Field(default_factory=list)
    project_id: uuid.UUID | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: str
    content: str
    summary: str | None = None
    tags: list[str]
    confidence: float
    utility: float
    access_count: int
    success_count: int
    failure_count: int
    created_at: datetime
    last_accessed_at: datetime
    score: float | None = None  # set by retriever when returned from search
