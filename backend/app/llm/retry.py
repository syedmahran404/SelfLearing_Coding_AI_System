"""Retry helpers for provider calls.

Exposes a single decorator: `with_retry`. We don't retry on:
- AuthenticationError-like exceptions (4xx that the user must fix)
- ContextLengthExceeded (retrying won't help)
- Cancelled tasks (cooperative cancellation)

We do retry on:
- Network errors / timeouts
- 429 / 5xx responses

Backoff is exponential with jitter, capped to keep total wallclock bounded.
"""
from __future__ import annotations

import asyncio
import random
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from app.observability.logger import get_logger

logger = get_logger("llm.retry")

T = TypeVar("T")

# Strings that indicate the call is unretriable; matched case-insensitively.
_UNRETRIABLE_TOKENS = (
    "authentication",
    "invalid api key",
    "unauthorized",
    "permission",
    "context length",
    "too many tokens",
    "model not found",
    "billing",
)


class RetryableError(Exception):
    """Wraps a downstream error to request a retry."""


def _is_retriable(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return False
    msg = (str(exc) or repr(exc)).lower()
    if any(tok in msg for tok in _UNRETRIABLE_TOKENS):
        return False
    return True


def with_retry(
    *,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Async decorator: retries the wrapped coroutine with exponential backoff."""

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt == attempts or not _is_retriable(exc):
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay *= 0.5 + random.random()  # jitter ~[0.5x .. 1.5x]
                    logger.warning(
                        "llm_call_retry",
                        attempt=attempt,
                        delay_s=round(delay, 3),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
            # Unreachable: loop either returns or re-raises.
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
