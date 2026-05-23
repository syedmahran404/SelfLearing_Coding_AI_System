"""Memory integrity helpers.

Memory rows carry `content_sha` (sha256 of normalized content). Two helpers:

- `content_fingerprint(text)`     : produce the canonical hash + bytes-len
                                    + char-len envelope used at write time.
- `verify_fingerprint(row)`       : recompute and compare; raise on
                                    mismatch (tampering, encoding drift).

This is a lightweight defense against silent corruption: if a memory row
is mutated outside our code path, reads via `verify_fingerprint` will
detect and refuse to use it.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.memory.utils import content_sha as _hash_text
from app.memory.utils import normalize_content
from app.observability import get_logger

logger = get_logger("safety.integrity")


class IntegrityError(RuntimeError):
    """Raised when a stored content's hash no longer matches its bytes."""


@dataclass(slots=True)
class Fingerprint:
    sha256: str
    char_len: int
    byte_len: int


def content_fingerprint(text: str) -> Fingerprint:
    normalized = normalize_content(text or "")
    return Fingerprint(
        sha256=_hash_text(normalized),
        char_len=len(normalized),
        byte_len=len(normalized.encode("utf-8")),
    )


def verify_fingerprint(*, content: str, expected_sha: str) -> Fingerprint:
    """Recompute fingerprint and assert match. Returns the recomputed FP."""
    fp = content_fingerprint(content)
    if fp.sha256 != expected_sha:
        logger.warning("integrity_mismatch", expected=expected_sha[:16], actual=fp.sha256[:16])
        raise IntegrityError(
            f"content sha mismatch: expected {expected_sha[:16]}…, got {fp.sha256[:16]}…"
        )
    return fp
