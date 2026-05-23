"""POST /chat — streamed orchestrator runs over Server-Sent Events.

Wire shape (each chunk is one SSE event):
    event: status | plan | subtask | tool | evaluation | reflection
         | answer | memory | done | error
    data:  JSON

The frontend `useStream` hook subscribes once and renders events as they
arrive. Trace events go to `/traces/{trace_id}/stream` (separate route).

We honor an inbound `x-trace-id` header so a client can pre-subscribe to
the trace channel before sending the request.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.deps import db_session, get_runtime, require_orchestrator
from app.bootstrap import Runtime
from app.observability import get_logger
from app.orchestration.orchestrator import RunContext
from app.schemas.api import ChatRequest

logger = get_logger("api.chat")

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    x_trace_id: str | None = Header(default=None, alias="x-trace-id"),
    db: AsyncSession = Depends(db_session),
    rt: Runtime = Depends(require_orchestrator),
) -> EventSourceResponse:
    """Run the orchestrator and stream events as SSE."""
    assert rt.orchestrator is not None
    trace_id = x_trace_id or uuid.uuid4().hex
    ctx = RunContext(
        request=body.message,
        db=db,
        user_id=body.user_id,
        project_id=body.project_id,
        session_id=body.session_id,
        trace_id=trace_id,
    )

    # Push the user turn into short-term memory for context-builder visibility.
    if rt.memory is not None and body.session_id is not None:
        try:
            await rt.memory.push_short_term(
                body.session_id, role="user", content=body.message
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("short_term_push_failed", error=str(exc))

    async def event_source():
        try:
            yield _sse("ready", {"trace_id": trace_id})
            assistant_buf: list[str] = []
            async for ev in rt.orchestrator.run_stream(ctx):
                if ev.type == "answer":
                    answer = (ev.data or {}).get("answer", "")
                    if isinstance(answer, str):
                        assistant_buf.append(answer)
                yield _sse(ev.type, ev.data)
            # Persist the assistant turn into short-term memory.
            if rt.memory is not None and body.session_id is not None and assistant_buf:
                try:
                    await rt.memory.push_short_term(
                        body.session_id,
                        role="assistant",
                        content="\n".join(assistant_buf),
                        agent="orchestrator",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("assistant_short_term_push_failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat_stream_unhandled", error=str(exc))
            yield _sse("error", {"message": str(exc)})

    return EventSourceResponse(event_source(), headers={"x-trace-id": trace_id})


def _sse(event: str, data: Any) -> dict[str, str]:
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False, default=str),
    }
