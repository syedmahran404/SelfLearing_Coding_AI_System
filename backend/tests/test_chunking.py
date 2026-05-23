"""RAG chunker — language-aware chunking guarantees."""
from __future__ import annotations

from app.rag.chunking import chunk


def test_python_extracts_functions_and_classes():
    src = (
        '"""module doc"""\n'
        "def foo(x):\n"
        '    """foo doc"""\n'
        "    return x\n"
        "\n"
        "class Bar:\n"
        '    """bar doc"""\n'
        "    def baz(self):\n"
        "        return 42\n"
    )
    chunks = chunk(src, "file://m.py")
    kinds = {c.kind for c in chunks}
    assert {"function", "class"}.issubset(kinds)
    names = {c.name for c in chunks if c.name}
    assert "foo" in names
    assert "Bar" in names
    # Methods are emitted under "Class.method" form.
    assert any(n == "Bar.baz" for n in names)


def test_python_invalid_falls_back_to_window():
    src = "def x(:\n    not python\n"
    chunks = chunk(src, "file://broken.py")
    # Either zero chunks or windowed fallback. Never a crash.
    assert all(c.kind in {"window", "doc", "function", "class"} for c in chunks)


def test_markdown_splits_by_heading():
    src = (
        "# Title\n\nintro\n\n"
        "## A\n\ntext a\n\n"
        "## B\n\ntext b\n"
    )
    chunks = chunk(src, "file://README.md")
    titles = [c.name for c in chunks if c.name]
    assert "Title" in titles
    assert "A" in titles
    assert "B" in titles


def test_jsts_extracts_top_level_decls():
    src = (
        "export function add(a, b) { return a + b; }\n"
        "export class M { run() {} }\n"
    )
    chunks = chunk(src, "file://m.ts")
    names = {c.name for c in chunks if c.name}
    assert {"add", "M"}.issubset(names)


def test_unknown_language_falls_back_to_window():
    src = "\n".join(f"line {i}" for i in range(120))
    chunks = chunk(src, "file://x.txt")
    assert all(c.kind == "window" for c in chunks)
    assert len(chunks) >= 1
