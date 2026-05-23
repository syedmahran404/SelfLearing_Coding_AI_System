"""Python AST extractor — symbols + import/inheritance/call edges."""
from __future__ import annotations

from app.project.ast_python import extract_python


def test_module_symbol_emitted():
    src = '"""mod doc"""\n'
    res = extract_python("pkg/m.py", src)
    mods = [s for s in res.symbols if s.kind == "module"]
    assert len(mods) == 1
    assert mods[0].qualified_name == "pkg.m"
    assert mods[0].docstring == "mod doc"


def test_function_class_method_emitted():
    src = (
        "def top():\n    return 1\n\n"
        "class C:\n    def m(self):\n        top()\n"
    )
    res = extract_python("pkg/m.py", src)
    qnames = {s.qualified_name for s in res.symbols}
    assert "pkg.m.top" in qnames
    assert "pkg.m.C" in qnames
    assert "pkg.m.C.m" in qnames

    # Call edge: pkg.m.C.m → top
    calls = [
        e for e in res.edges
        if e.relation == "calls" and e.src_qname == "pkg.m.C.m"
    ]
    assert any(e.dst_unresolved == "top" for e in calls)


def test_import_edges_recorded():
    src = (
        "import os\n"
        "from typing import List\n"
        "from .sibling import helper\n"
    )
    res = extract_python("pkg/sub/m.py", src)
    imports = [e for e in res.edges if e.relation == "imports"]
    targets = {e.dst_unresolved for e in imports}
    assert "os" in targets
    assert any("typing" in t for t in targets)
    # Relative resolution: from .sibling import helper inside pkg.sub.m → pkg.sub.sibling.helper
    assert any("pkg.sub.sibling.helper" == t for t in targets)


def test_inheritance_edge_recorded():
    src = "class A: pass\nclass B(A): pass\n"
    res = extract_python("pkg/m.py", src)
    inh = [e for e in res.edges if e.relation == "inherits"]
    assert any(e.src_qname == "pkg.m.B" and e.dst_unresolved == "A" for e in inh)


def test_syntax_error_returns_empty_symbols_with_error():
    res = extract_python("pkg/broken.py", "def x(:\n  pass\n")
    # Module symbol still emitted; nothing else.
    kinds = {s.kind for s in res.symbols}
    assert "module" in kinds
    assert "function" not in kinds
    assert res.parse_error is not None
