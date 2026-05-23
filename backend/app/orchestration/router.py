"""Intent router.

Cheap, deterministic-first classification:
1. Strong keyword heuristics (free, instant).
2. If ambiguous, ask the planner-tier LLM with a short schema-constrained
   prompt.

We bias toward QA when the request looks like a question (has ?, starts
with what/why/how/when), CODE when it asks for code-writing verbs,
DEBUG on stack-trace markers, RESEARCH on docs/api lookups.
"""
from __future__ import annotations

import re
from typing import Any

from app.llm.provider import LLMProvider
from app.llm.types import ChatMessage, CompletionRequest, Role
from app.observability import get_logger
from app.schemas.agent_io import TaskIntent

logger = get_logger("orchestration.router")

_QA = re.compile(
    r"^\s*(what|why|how|when|where|who|which|is|are|does|do|can|could|should|will|would)\b",
    re.I,
)
_CODE = re.compile(
    r"\b(write|implement|create|build|add|generate|make|refactor|rewrite|extract|extend|"
    r"convert|migrate|scaffold|set up|setup)\b",
    re.I,
)
_DEBUG = re.compile(
    r"\b(error|exception|traceback|stack trace|fail(s|ing|ed)?|broken|bug|crash|panic|"
    r"500|segfault)\b",
    re.I,
)
_RESEARCH = re.compile(
    r"\b(documentation|docs|api(?!\s*key)|library|framework|reference|spec|rfc)\b",
    re.I,
)
_EXPLAIN = re.compile(r"\b(explain|describe|walk me through|what does .+ do)\b", re.I)
_REFACTOR = re.compile(
    r"\b(refactor|clean up|reorganize|rename|deduplicate|simplify|optimize)\b", re.I
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["intent", "confidence"],
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in TaskIntent]},
        "confidence": {"type": "number"},
    },
}


class IntentRouter:
    def __init__(self, llm: LLMProvider, *, model: str | None = None) -> None:
        self._llm = llm
        self._model = model or llm.settings.llm_planner_model

    async def classify(self, request: str) -> tuple[TaskIntent, float]:
        """Return (intent, confidence). Heuristic first; LLM only on ambiguity."""
        intent, conf = self._heuristic(request)
        if conf >= 0.7:
            return intent, conf

        # LLM-backed classification — small and cheap.
        try:
            req = CompletionRequest(
                model=self._model,
                messages=[
                    ChatMessage(
                        role=Role.SYSTEM,
                        content=(
                            "Classify a coding-assistant request into one intent: "
                            "qa | code | debug | refactor | research | explain. "
                            "Reply ONLY with JSON {intent, confidence}."
                        ),
                    ),
                    ChatMessage(role=Role.USER, content=request[:1500]),
                ],
                temperature=0.0,
                max_tokens=120,
                response_schema=_SCHEMA,
                purpose="orchestration.router",
            )
            resp = await self._llm.complete(req)
            data = _safe_parse_intent(resp.content)
            if data:
                try:
                    return TaskIntent(data["intent"]), float(data.get("confidence", 0.6))
                except (ValueError, KeyError):
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("router_llm_failed", error=str(exc))

        return intent, conf  # heuristic fallback

    # ── heuristic ──
    def _heuristic(self, request: str) -> tuple[TaskIntent, float]:
        s = request.strip()
        if not s:
            return TaskIntent.QA, 0.5

        scores: dict[TaskIntent, float] = {i: 0.0 for i in TaskIntent}

        if "?" in s or _QA.search(s):
            scores[TaskIntent.QA] += 0.6
        if _CODE.search(s):
            scores[TaskIntent.CODE] += 0.7
        if _DEBUG.search(s):
            scores[TaskIntent.DEBUG] += 0.8
        if _REFACTOR.search(s):
            scores[TaskIntent.REFACTOR] += 0.7
        if _RESEARCH.search(s):
            scores[TaskIntent.RESEARCH] += 0.5
        if _EXPLAIN.search(s):
            scores[TaskIntent.EXPLAIN] += 0.6

        # Code patterns dominate over QA when both fire.
        if scores[TaskIntent.DEBUG] > 0:
            scores[TaskIntent.QA] *= 0.4
            scores[TaskIntent.CODE] *= 0.6

        intent = max(scores, key=lambda k: scores[k])
        conf = scores[intent]
        if conf == 0.0:
            return TaskIntent.QA, 0.4
        return intent, min(0.95, conf)


def _safe_parse_intent(text: str) -> dict | None:
    import json

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "intent" in data:
            return data
    except json.JSONDecodeError:
        pass
    # extract braces
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            data = json.loads(text[s : e + 1])
            if isinstance(data, dict) and "intent" in data:
                return data
        except json.JSONDecodeError:
            pass
    return None
