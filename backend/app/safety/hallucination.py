"""Hallucination guard.

Extracts API references from generated code and verifies them against:
  1. The project index (`ProjectSymbol.qualified_name` / `name`).
  2. The RAG corpus (chunk payload `name`).
  3. A small allowlist of stdlib / well-known library names.

Anything still unmatched is flagged. By default, with
`SAFETY_HALLUCINATION_BLOCK=true` (env), the orchestrator can refuse to
execute tools or write code that contains *only* unverified APIs.

Goal: catch the obvious "the LLM invented a function" pattern; we do NOT
attempt to verify type signatures or argument arity here. That's the
Evaluator's job (it runs the code).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ProjectSymbol
from app.observability import get_logger

logger = get_logger("safety.hallucination")


# Identifier reference that follows a dot or `from X import Y` style.
_PY_IMPORT_FROM = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+([\w\s,]+)", re.MULTILINE)
_PY_IMPORT = re.compile(r"^\s*import\s+([\w\.]+)(?:\s+as\s+\w+)?", re.MULTILINE)
_PY_DOTTED_CALL = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\s*\(")
_TS_IMPORT = re.compile(r"^\s*import\s+(?:\{([^}]*)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)


# Names we never flag (stdlib + ubiquitous third-party).
_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    # Python stdlib (subset)
    "os", "sys", "re", "json", "math", "random", "datetime", "time", "pathlib",
    "typing", "collections", "itertools", "functools", "asyncio", "dataclasses",
    "enum", "io", "subprocess", "logging", "hashlib", "uuid", "tempfile", "shutil",
    "argparse", "contextlib", "abc", "string", "decimal", "fractions",
    # Common Python libs
    "numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow", "pytest", "fastapi",
    "pydantic", "sqlalchemy", "httpx", "requests", "redis", "qdrant_client",
    "structlog", "tiktoken", "rank_bm25", "matplotlib", "seaborn", "boto3",
    # Stdlib / common JS-TS
    "console", "Math", "JSON", "Promise", "Object", "Array", "Number", "String",
    "Map", "Set", "Date", "Error", "Symbol",
    "react", "next", "express", "lodash", "axios", "zod",
    # System / globals
    "self", "cls", "__main__", "True", "False", "None", "null", "undefined",
)


@dataclass(slots=True)
class HallucinationCheck:
    """Result of checking a code blob."""

    references: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    flagged: bool = False

    @property
    def confidence(self) -> float:
        """Fraction of references we could match."""
        if not self.references:
            return 1.0
        return len(self.verified) / len(self.references)


class HallucinationGuard:
    """Verify that referenced APIs exist somewhere we trust."""

    def __init__(
        self,
        *,
        rag_symbol_lookup: callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        # `rag_symbol_lookup(name) -> bool` — provided by callers that want
        # to consult the RAG corpus. Keeps this module dependency-light.
        self._rag = rag_symbol_lookup

    async def check_code(
        self,
        text: str,
        *,
        db: AsyncSession | None = None,
        project_id: UUID | None = None,
        language: str = "python",
        block_threshold: float = 0.6,
    ) -> HallucinationCheck:
        """Extract → verify → score."""
        refs = list(self._extract(text, language=language))
        check = HallucinationCheck(references=refs)
        if not refs:
            return check

        # Allowlist filter.
        unmatched = [r for r in refs if not _allowlisted(r)]
        check.verified = [r for r in refs if _allowlisted(r)]

        # Project index lookup, if available.
        if db is not None and project_id is not None and unmatched:
            verified = await self._verify_via_project(db, project_id, unmatched)
            check.verified.extend(verified)
            unmatched = [r for r in unmatched if r not in verified]

        # RAG corpus lookup if a callable is provided.
        if unmatched and self._rag is not None:
            verified_via_rag: list[str] = []
            for ref in unmatched:
                try:
                    if await self._rag(ref):
                        verified_via_rag.append(ref)
                except Exception:  # noqa: BLE001
                    continue
            check.verified.extend(verified_via_rag)
            unmatched = [r for r in unmatched if r not in verified_via_rag]

        check.unverified = unmatched
        check.flagged = check.confidence < block_threshold
        if check.flagged:
            logger.info(
                "hallucination_flagged",
                refs=len(refs),
                verified=len(check.verified),
                unverified=unmatched[:8],
                confidence=round(check.confidence, 3),
            )
        return check

    # ── extraction ──
    def _extract(self, text: str, *, language: str) -> Iterable[str]:
        if language == "python":
            yield from self._extract_python(text)
        elif language in {"typescript", "javascript"}:
            yield from self._extract_jsts(text)
        else:
            return

    def _extract_python(self, text: str) -> Iterable[str]:
        seen: set[str] = set()
        for m in _PY_IMPORT.finditer(text):
            mod = m.group(1).split(".")[0]
            if mod and mod not in seen:
                seen.add(mod)
                yield mod
        for m in _PY_IMPORT_FROM.finditer(text):
            base = m.group(1).split(".")[0]
            if base and base not in seen:
                seen.add(base)
                yield base
            for name in (n.strip() for n in m.group(2).split(",")):
                if name and name not in seen:
                    seen.add(name)
                    yield name
        for m in _PY_DOTTED_CALL.finditer(text):
            ref = m.group(1)
            if ref not in seen:
                seen.add(ref)
                yield ref

    def _extract_jsts(self, text: str) -> Iterable[str]:
        seen: set[str] = set()
        for m in _TS_IMPORT.finditer(text):
            named = m.group(1) or ""
            default = m.group(2) or ""
            mod = m.group(3)
            mod_root = mod.split("/")[0]
            for n in [default, *(s.strip() for s in named.split(","))]:
                n = n.strip()
                if n and n not in seen:
                    seen.add(n)
                    yield n
            if mod_root and mod_root not in seen:
                seen.add(mod_root)
                yield mod_root

    # ── DB lookup ──
    async def _verify_via_project(
        self,
        db: AsyncSession,
        project_id: UUID,
        candidates: list[str],
    ) -> list[str]:
        """Match each candidate against ProjectSymbol.name or qualified_name."""
        if not candidates:
            return []
        # Split into name root and tail (e.g. "foo.bar.baz" → "baz").
        roots = {c.split(".")[0] for c in candidates}
        tails = {c.rsplit(".", 1)[-1] for c in candidates}
        rows = (
            (
                await db.execute(
                    select(ProjectSymbol.name, ProjectSymbol.qualified_name).where(
                        ProjectSymbol.project_id == project_id,
                        (
                            ProjectSymbol.name.in_(roots | tails)
                            | ProjectSymbol.qualified_name.in_(candidates)
                        ),
                    )
                )
            ).all()
        )
        names = {row[0] for row in rows}
        qnames = {row[1] for row in rows}

        verified: list[str] = []
        for c in candidates:
            head = c.split(".")[0]
            tail = c.rsplit(".", 1)[-1]
            if c in qnames or head in names or tail in names:
                verified.append(c)
        return verified


def _allowlisted(ref: str) -> bool:
    head = ref.split(".")[0]
    if head in _ALLOWLIST_PREFIXES:
        return True
    return False
