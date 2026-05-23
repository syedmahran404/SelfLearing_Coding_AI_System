"""Language-aware chunkers.

Each chunker yields a stream of `Chunk` objects describing a region of
source content along with provenance (file path, line range, kind, name).

Strategy by source type:
- ``.py``     : `ast` walks; functions and classes become chunks.
- ``.md``     : split by heading hierarchy (H1/H2/H3).
- ``.ts/.tsx/.js/.jsx``: regex-based extraction of top-level
  function/class/export declarations. (Avoids a tree-sitter dep; good
  enough for retrieval grounding — false negatives just fall back to
  windowed chunking.)
- ``.json/.yaml/.toml``: top-level keys.
- everything else: windowed (sliding line-count) chunking.

A single function with `chunk(text, source_uri)` dispatches by extension.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ── data type ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Chunk:
    """One indexable region of content."""

    source_uri: str
    text: str
    language: str
    kind: str  # function | class | section | window | toplevel | doc
    name: str | None = None
    line_start: int = 0
    line_end: int = 0
    extra: dict = field(default_factory=dict)


# ── public entrypoint ─────────────────────────────────────────────────────


def chunk(text: str, source_uri: str, *, language: str | None = None) -> list[Chunk]:
    """Dispatch to the right chunker based on extension or explicit language."""
    if not text:
        return []

    lang = (language or _detect_language(source_uri)).lower()

    if lang == "python":
        out = list(_chunk_python(text, source_uri))
        return out or list(_chunk_window(text, source_uri, lang))
    if lang == "markdown":
        out = list(_chunk_markdown(text, source_uri))
        return out or list(_chunk_window(text, source_uri, lang))
    if lang in {"javascript", "typescript"}:
        out = list(_chunk_jsts(text, source_uri, lang))
        return out or list(_chunk_window(text, source_uri, lang))
    if lang == "json":
        out = list(_chunk_json(text, source_uri))
        return out or list(_chunk_window(text, source_uri, lang))
    return list(_chunk_window(text, source_uri, lang))


# ── language detection ────────────────────────────────────────────────────

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "markdown",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "shell",
}


def _detect_language(source_uri: str) -> str:
    ext = Path(source_uri).suffix.lower()
    return _EXT_LANG.get(ext, "text")


# ── windowed (fallback) ───────────────────────────────────────────────────


def _chunk_window(
    text: str, source_uri: str, language: str, *, lines_per_chunk: int = 60, overlap: int = 8
) -> Iterator[Chunk]:
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return
    step = max(1, lines_per_chunk - overlap)
    for start in range(0, n, step):
        end = min(n, start + lines_per_chunk)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        yield Chunk(
            source_uri=source_uri,
            text=body,
            language=language,
            kind="window",
            line_start=start + 1,
            line_end=end,
        )
        if end >= n:
            break


# ── Python (AST) ──────────────────────────────────────────────────────────


def _chunk_python(text: str, source_uri: str) -> Iterator[Chunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    lines = text.splitlines()

    # Module-level docstring as a "doc" chunk.
    doc = ast.get_docstring(tree)
    if doc:
        yield Chunk(
            source_uri=source_uri,
            text=doc,
            language="python",
            kind="doc",
            name="<module>",
            line_start=1,
            line_end=min(len(lines), 5),
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            line_start = node.lineno
            line_end = getattr(node, "end_lineno", line_start)
            body = "\n".join(lines[line_start - 1 : line_end])
            sig = _python_signature(node, lines)
            yield Chunk(
                source_uri=source_uri,
                text=body.strip(),
                language="python",
                kind=kind,
                name=node.name,
                line_start=line_start,
                line_end=line_end,
                extra={"signature": sig, "docstring": ast.get_docstring(node)},
            )
            # For classes, also emit each method as its own chunk.
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cls_line_start = child.lineno
                        cls_line_end = getattr(child, "end_lineno", cls_line_start)
                        m_body = "\n".join(lines[cls_line_start - 1 : cls_line_end])
                        yield Chunk(
                            source_uri=source_uri,
                            text=m_body.strip(),
                            language="python",
                            kind="function",
                            name=f"{node.name}.{child.name}",
                            line_start=cls_line_start,
                            line_end=cls_line_end,
                            extra={
                                "signature": _python_signature(child, lines),
                                "docstring": ast.get_docstring(child),
                                "parent": node.name,
                            },
                        )


def _python_signature(node: ast.AST, lines: list[str]) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""
    line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
    return line.strip()


# ── Markdown ──────────────────────────────────────────────────────────────


_MD_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def _chunk_markdown(text: str, source_uri: str) -> Iterator[Chunk]:
    lines = text.splitlines()
    sections: list[tuple[int, str, int]] = []  # (level, title, line_no)
    for i, line in enumerate(lines):
        m = _MD_HEADING.match(line)
        if m:
            sections.append((len(m.group(1)), m.group(2).strip(), i))

    if not sections:
        # Treat the whole doc as one section.
        body = text.strip()
        if body:
            yield Chunk(
                source_uri=source_uri,
                text=body,
                language="markdown",
                kind="section",
                name=Path(source_uri).stem,
                line_start=1,
                line_end=len(lines),
            )
        return

    sections.append((1, "", len(lines)))  # sentinel
    for k in range(len(sections) - 1):
        lvl, title, ln = sections[k]
        next_ln = sections[k + 1][2]
        body = "\n".join(lines[ln:next_ln]).strip()
        if not body:
            continue
        yield Chunk(
            source_uri=source_uri,
            text=body,
            language="markdown",
            kind="section",
            name=title,
            line_start=ln + 1,
            line_end=next_ln,
            extra={"level": lvl},
        )


# ── JavaScript / TypeScript (regex heuristic) ─────────────────────────────


_JS_DECL = re.compile(
    r"""
    ^[ \t]*
    (?:export\s+(?:default\s+)?)?
    (?:async\s+)?
    (?P<kind>function|class|const|let|var)\s+
    (?P<name>[A-Za-z_$][\w$]*)
    """,
    re.VERBOSE | re.MULTILINE,
)


def _chunk_jsts(text: str, source_uri: str, language: str) -> Iterator[Chunk]:
    lines = text.splitlines()
    matches = list(_JS_DECL.finditer(text))
    if not matches:
        return

    # Compute char-offset → line-no map.
    char_to_line: list[int] = []
    cum = 0
    for i, line in enumerate(lines, start=1):
        for _ in range(len(line) + 1):
            char_to_line.append(i)
        cum += len(line) + 1

    starts: list[int] = [m.start() for m in matches]
    starts.append(len(text))  # sentinel

    for k, m in enumerate(matches):
        s = starts[k]
        e = starts[k + 1]
        body = text[s:e].strip()
        if not body:
            continue
        line_start = char_to_line[s] if s < len(char_to_line) else 1
        line_end = char_to_line[e - 1] if e - 1 < len(char_to_line) else line_start
        kind_word = m.group("kind")
        kind = (
            "class"
            if kind_word == "class"
            else "function"
            if kind_word in {"function"}
            else "toplevel"
        )
        yield Chunk(
            source_uri=source_uri,
            text=body,
            language=language,
            kind=kind,
            name=m.group("name"),
            line_start=line_start,
            line_end=line_end,
        )


# ── JSON (top-level keys) ─────────────────────────────────────────────────


def _chunk_json(text: str, source_uri: str) -> Iterator[Chunk]:
    """Cheap heuristic: emit one chunk per top-level key. If parsing fails
    we fall back to windowed chunking via the dispatcher."""
    import json

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(obj, dict):
        yield Chunk(
            source_uri=source_uri,
            text=text.strip(),
            language="json",
            kind="toplevel",
            name=None,
        )
        return
    for k, v in obj.items():
        body = json.dumps({k: v}, indent=2, ensure_ascii=False, default=str)
        yield Chunk(
            source_uri=source_uri,
            text=body,
            language="json",
            kind="toplevel",
            name=str(k),
        )
