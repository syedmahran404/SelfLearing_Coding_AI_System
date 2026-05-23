"""Python-specific extractor: symbols + edges from `ast`.

Output is uniform with what other languages will eventually produce
(see `RawSymbol`, `RawEdge`). Where Python's resolution model is rich
enough we can resolve imports to project files; otherwise we record the
target as `dst_unresolved` (a fully-qualified or relative module path),
which the orchestrator can treat as "external".
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

# ── output shapes ────────────────────────────────────────────────────────


@dataclass(slots=True)
class RawSymbol:
    file_path: str
    language: str
    kind: str           # function | class | method | import | module
    name: str
    qualified_name: str
    line_start: int = 0
    line_end: int = 0
    signature: str | None = None
    docstring: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class RawEdge:
    src_qname: str
    relation: str        # imports | calls | inherits
    dst_qname: str | None = None       # filled if we can resolve in-project
    dst_unresolved: str | None = None  # always populated (string)
    weight: float = 1.0


@dataclass(slots=True)
class ExtractResult:
    symbols: list[RawSymbol]
    edges: list[RawEdge]
    parse_error: str | None = None


# ── extractor ─────────────────────────────────────────────────────────────


def extract_python(rel_path: str, source: str, *, project_root: str | None = None) -> ExtractResult:
    """Parse a Python file and return its symbols + edges.

    `rel_path` is the file's path relative to the project root; it
    determines the module's qualified name (e.g. `app/db/models.py` →
    `app.db.models`). On parse error we return zero symbols and the error
    text so the indexer can record it.
    """
    module_qname = _module_qname(rel_path)
    symbols: list[RawSymbol] = [
        RawSymbol(
            file_path=rel_path,
            language="python",
            kind="module",
            name=Path(rel_path).stem,
            qualified_name=module_qname,
            line_start=1,
            line_end=max(1, source.count("\n") + 1),
        )
    ]
    edges: list[RawEdge] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ExtractResult(symbols=symbols, edges=edges, parse_error=str(exc))

    module_doc = ast.get_docstring(tree)
    if module_doc:
        symbols[0].docstring = module_doc

    # Collect imports first (needed to resolve calls in some cases).
    import_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                local = alias.asname or alias.name.split(".", 1)[0]
                import_aliases[local] = target
                edges.append(RawEdge(src_qname=module_qname, relation="imports", dst_unresolved=target))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            if level:
                # Relative import — resolve to absolute against module_qname
                base_parts = module_qname.split(".")
                # Drop `level` parts from the *module* (not file) — Python relative
                # import semantics: `from . import X` inside `pkg.sub` → `pkg.X`.
                abs_parts = base_parts[: -level] if level <= len(base_parts) else []
                if module:
                    abs_parts.append(module)
                module = ".".join(abs_parts)
            for alias in node.names:
                local = alias.asname or alias.name
                full = f"{module}.{alias.name}" if module else alias.name
                import_aliases[local] = full
                edges.append(
                    RawEdge(src_qname=module_qname, relation="imports", dst_unresolved=full)
                )

    # Functions, classes, methods.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _emit_function(symbols, edges, node, source, module_qname, import_aliases, parent_qname=None)
        elif isinstance(node, ast.ClassDef):
            cls_qname = f"{module_qname}.{node.name}"
            symbols.append(
                RawSymbol(
                    file_path=rel_path,
                    language="python",
                    kind="class",
                    name=node.name,
                    qualified_name=cls_qname,
                    line_start=node.lineno,
                    line_end=getattr(node, "end_lineno", node.lineno),
                    signature=_signature_line(source, node.lineno),
                    docstring=ast.get_docstring(node),
                )
            )
            # Inheritance edges.
            for base in node.bases:
                base_name = _name_or_attr(base)
                if base_name:
                    edges.append(
                        RawEdge(
                            src_qname=cls_qname,
                            relation="inherits",
                            dst_unresolved=_resolve(base_name, import_aliases),
                        )
                    )
            # Methods.
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _emit_function(
                        symbols,
                        edges,
                        child,
                        source,
                        module_qname,
                        import_aliases,
                        parent_qname=cls_qname,
                    )

    return ExtractResult(symbols=symbols, edges=edges)


# ── helpers ──────────────────────────────────────────────────────────────


def _emit_function(
    symbols: list[RawSymbol],
    edges: list[RawEdge],
    node: ast.AST,
    source: str,
    module_qname: str,
    import_aliases: dict[str, str],
    *,
    parent_qname: str | None,
) -> None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    qname = f"{parent_qname}.{node.name}" if parent_qname else f"{module_qname}.{node.name}"
    kind = "method" if parent_qname else "function"
    symbols.append(
        RawSymbol(
            file_path="",  # filled by indexer (it knows rel_path)
            language="python",
            kind=kind,
            name=node.name,
            qualified_name=qname,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            signature=_signature_line(source, node.lineno),
            docstring=ast.get_docstring(node),
            extra={"is_async": isinstance(node, ast.AsyncFunctionDef), "parent": parent_qname},
        )
    )

    # Walk the body for `Call` edges. Best-effort name resolution only.
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = _call_target(child)
            if target:
                resolved = _resolve(target, import_aliases)
                edges.append(
                    RawEdge(
                        src_qname=qname,
                        relation="calls",
                        dst_unresolved=resolved,
                    )
                )


def _call_target(call: ast.Call) -> str | None:
    """Return a dotted name for the callee, when statically determinable."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _name_or_attr(func)
    return None


def _name_or_attr(node: ast.AST) -> str | None:
    """Render `Name` or `Attribute` chains as a dotted string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _name_or_attr(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return None


def _resolve(name: str, aliases: dict[str, str]) -> str:
    """If `name` starts with an imported alias, expand it. Else return as-is."""
    if not name:
        return name
    head, _, tail = name.partition(".")
    if head in aliases:
        target = aliases[head]
        return f"{target}.{tail}" if tail else target
    return name


def _signature_line(source: str, line_no: int) -> str:
    lines = source.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def _module_qname(rel_path: str) -> str:
    """`a/b/c.py` → `a.b.c`; strip trailing `.__init__`."""
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or p.stem
