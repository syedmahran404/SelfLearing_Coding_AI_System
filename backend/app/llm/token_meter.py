"""Token counting + cost tracking.

Uses tiktoken when available; falls back to a fast heuristic so the path is
always cheap and deterministic in tests.

The TokenMeter is held on the LLMProvider; agents can ask
`provider.meter.estimate(messages)` before issuing a call to budget context.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from app.llm.types import ChatMessage


# Approximate per-1k-token prices in USD. These are deliberately conservative
# defaults; downstream cost reporting is for observability, not billing.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # model               : (input_per_1k, output_per_1k)
    "gpt-4o":              (0.0050, 0.0150),
    "gpt-4o-mini":         (0.00015, 0.0006),
    "gpt-4-turbo":         (0.01, 0.03),
    "claude-3-5-sonnet":   (0.003, 0.015),
    "claude-3-5-haiku":    (0.0008, 0.004),
    "claude-3-opus":       (0.015, 0.075),
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
}


@dataclass(slots=True)
class TokenCounter:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


@lru_cache(maxsize=8)
def _get_encoding(model: str):  # type: ignore[no-untyped-def]
    """Return a tiktoken encoding for the given model, or None on failure."""
    try:
        import tiktoken
    except Exception:  # noqa: BLE001
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            return None


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return a token count for `text`. Tiktoken when possible, else heuristic."""
    if not text:
        return 0
    enc = _get_encoding(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # noqa: BLE001
            pass
    # Heuristic: ~4 chars per token, with a minimum of 1.
    return max(1, len(text) // 4)


def estimate_messages_tokens(
    messages: Iterable[ChatMessage], model: str = "gpt-4o-mini"
) -> int:
    """Sum tokens across a chat-message list. Adds a small per-message overhead."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.content, model)
        total += 4  # role + delimiters overhead
        if m.name:
            total += estimate_tokens(m.name, model)
    return total


class TokenMeter:
    """Tracks cumulative tokens + cost across a process.

    Thread-safe. Per-purpose breakdown supports per-task budgeting and
    cost dashboards.
    """

    def __init__(self, pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self._pricing = pricing or dict(_DEFAULT_PRICING)
        self._lock = threading.Lock()
        self._totals: dict[str, TokenCounter] = {}
        self._cost_usd: float = 0.0
        self._cost_by_purpose: dict[str, float] = {}

    def price_for(self, model: str) -> tuple[float, float]:
        """Return (input, output) USD per 1k tokens. Falls back to zeros."""
        # Match by exact, then by prefix (e.g. "gpt-4o-2024-05-13" → "gpt-4o").
        if model in self._pricing:
            return self._pricing[model]
        for prefix, price in self._pricing.items():
            if model.startswith(prefix):
                return price
        return (0.0, 0.0)

    def cost_of(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        in_p, out_p = self.price_for(model)
        return (prompt_tokens / 1000.0) * in_p + (completion_tokens / 1000.0) * out_p

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str | None = None,
    ) -> float:
        cost = self.cost_of(model, prompt_tokens, completion_tokens)
        with self._lock:
            tc = self._totals.setdefault(model, TokenCounter())
            tc.prompt += prompt_tokens
            tc.completion += completion_tokens
            self._cost_usd += cost
            if purpose:
                self._cost_by_purpose[purpose] = self._cost_by_purpose.get(purpose, 0.0) + cost
        return cost

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "totals": {
                    m: {"prompt": c.prompt, "completion": c.completion, "total": c.total}
                    for m, c in self._totals.items()
                },
                "cost_usd": round(self._cost_usd, 6),
                "cost_by_purpose": {k: round(v, 6) for k, v in self._cost_by_purpose.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._totals.clear()
            self._cost_usd = 0.0
            self._cost_by_purpose.clear()
