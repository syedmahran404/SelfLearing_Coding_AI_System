"""ProjectIndexer — walk a repo, extract symbols + edges, persist + embed.

This is the heavy-but-cheap version: dependency-light, AST-driven for
Python, and gracefully skips files in other languages (their content is
still ingested by `RagService.ingest_directory` for vector recall — see
the indexer's docstring on the call site).

Two operations:
- `index(project, root)` : (re)index a project directory, full pass.
- `query(project, term)` : symbol/qualified-name + neighborhood search.

Idempotency
-----------
Re-running `index` deletes all existing rows for `project_id` first, then
re-inserts. Edges are resolved in-memory after symbols are inserted: a
final pass walks `_pending_edges` and links those whose `dst_unresolved`
matches an inserted symbol's `qualified_name`. Anything still unresolved
is recorded (likely external, e.g. `numpy.array`).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import Project, ProjectEdge, ProjectSymbol
from app.db.qdrant import QdrantStore, VectorPoint
from app.llm.provider import LLMProvider
from app.observability import Tracer, get_logger
from app.observability.tracing import SpanKind, trace_span
from app.project.ast_python import RawEdge, RawSymbol, extract_python

logger = get_logger("project.indexer")


_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}

_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", ".cache", ".tox", "site-packages",
}

MAX_FILE_BYTES = 200_000


@dataclass(slots=True)
class IndexStats:
    files_seen: int
    files_indexed: int
    symbols: int
    edges_total: int
    edges_resolved: int
    parse_errors: int
    duration_ms: int


class ProjectIndexer:
    """AST-driven project indexer."""

    def __init__(
        self,
        *,
        settings: Settings,
        qdrant: QdrantStore,
        llm: LLMProvider | None,
        tracer: Tracer,
    ) -> None:
        self._settings = settings
        self._qdrant = qdrant
        self._llm = llm
        self._tracer = tracer

    # ── index ──
    async def index(
        self,
        db: AsyncSession,
        *,
        project: Project,
        root: Path | str | None = None,
    ) -> IndexStats:
        root_path = Path(root or project.repo_root or ".").resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"project root not found: {root_path}")

        async with trace_span(
            self._tracer,
            "project.index",
            SpanKind.SYSTEM,
            payload={"project_id": str(project.id), "root": str(root_path)},
        ) as span:
            started = time.perf_counter()

            # Hard reset for the project to keep indexing idempotent.
            await db.execute(delete(ProjectEdge).where(ProjectEdge.project_id == project.id))
            await db.execute(delete(ProjectSymbol).where(ProjectSymbol.project_id == project.id))
            await db.flush()

            # Walk and extract.
            files = list(_walk(root_path))
            files_seen = len(files)
            extracted: list[tuple[str, list[RawSymbol], list[RawEdge], str | None]] = []
            for rel_path, abs_path in files:
                lang = _LANG_BY_EXT.get(abs_path.suffix.lower(), "")
                if lang != "python":
                    continue  # only Python full-AST today; others fall through to RAG
                try:
                    text = abs_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if len(text.encode("utf-8")) > MAX_FILE_BYTES:
                    continue
                result = extract_python(rel_path, text)
                # Stamp file_path on each symbol (extractor leaves it blank).
                for s in result.symbols:
                    s.file_path = rel_path
                extracted.append((rel_path, result.symbols, result.edges, result.parse_error))

            # Insert symbols, indexed by qualified_name → id for edge resolution.
            qname_to_id: dict[str, UUID] = {}
            symbol_count = 0
            parse_errors = 0
            symbol_rows_for_embedding: list[tuple[ProjectSymbol, str]] = []

            for rel_path, syms, _, parse_error in extracted:
                if parse_error:
                    parse_errors += 1
                for s in syms:
                    new_id = uuid.uuid4()
                    row = ProjectSymbol(
                        id=new_id,
                        project_id=project.id,
                        file_path=s.file_path,
                        language=s.language,
                        kind=s.kind,
                        name=s.name,
                        qualified_name=s.qualified_name,
                        line_start=s.line_start,
                        line_end=s.line_end,
                        signature=s.signature,
                        docstring=s.docstring,
                        extra=s.extra,
                    )
                    db.add(row)
                    qname_to_id[s.qualified_name] = new_id
                    symbol_count += 1
                    if s.kind in {"function", "method", "class"}:
                        embed_text = _symbol_text(s)
                        if embed_text:
                            symbol_rows_for_embedding.append((row, embed_text))

            await db.flush()

            # Edge resolution.
            edges_total = 0
            edges_resolved = 0
            for _, _, raws, _ in extracted:
                for e in raws:
                    edges_total += 1
                    src_id = qname_to_id.get(e.src_qname)
                    if src_id is None:
                        # Symbol may be at module-scope under a top-level import edge.
                        src_id = qname_to_id.get(e.src_qname.rsplit(".", 1)[0])
                    if src_id is None:
                        continue
                    dst_id = (
                        qname_to_id.get(e.dst_unresolved or "")
                        if e.dst_unresolved
                        else None
                    )
                    if dst_id:
                        edges_resolved += 1
                    db.add(
                        ProjectEdge(
                            project_id=project.id,
                            src_id=src_id,
                            dst_id=dst_id,
                            dst_unresolved=e.dst_unresolved if not dst_id else None,
                            relation=e.relation,
                            weight=e.weight,
                        )
                    )

            await db.flush()

            # Embeddings — best-effort. Individual failures don't fail the whole index.
            if self._llm is not None and symbol_rows_for_embedding:
                await self._embed_symbols(symbol_rows_for_embedding, project_id=project.id)
                await db.flush()

            stats = IndexStats(
                files_seen=files_seen,
                files_indexed=len(extracted),
                symbols=symbol_count,
                edges_total=edges_total,
                edges_resolved=edges_resolved,
                parse_errors=parse_errors,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            span["payload"].update(
                {
                    "symbols": stats.symbols,
                    "edges_total": stats.edges_total,
                    "edges_resolved": stats.edges_resolved,
                    "files_indexed": stats.files_indexed,
                    "parse_errors": stats.parse_errors,
                }
            )
            return stats

    # ── query ──
    async def query(
        self,
        db: AsyncSession,
        *,
        project_id: UUID,
        term: str,
        kind: str | None = None,
        top_k: int = 10,
    ) -> list[ProjectSymbol]:
        """Symbol/qualified-name search by name fragment or vector similarity.

        Hybrid: substring match on `name` (cheap) ∪ vector recall via the
        existing `rag` collection (when available). Combine + dedup by id.
        """
        seen: dict[UUID, ProjectSymbol] = {}

        # 1) name substring match (case-insensitive).
        q = (
            select(ProjectSymbol)
            .where(ProjectSymbol.project_id == project_id)
            .where(ProjectSymbol.name.ilike(f"%{term}%"))
        )
        if kind is not None:
            q = q.where(ProjectSymbol.kind == kind)
        rows = (await db.execute(q.limit(top_k))).scalars().all()
        for r in rows:
            seen[r.id] = r

        if len(seen) >= top_k:
            return list(seen.values())[:top_k]

        # 2) vector recall on rag collection — best-effort; skipped if no LLM.
        if self._llm is not None:
            try:
                emb = await self._llm.embed([term])
                if emb.vectors:
                    qvec = emb.vectors[0]
                    hits = await self._qdrant.search(
                        self._qdrant.rag,
                        qvec,
                        top_k=top_k * 2,
                        filter_must={"project_id": str(project_id)},
                    )
                    sym_ids = [
                        h.payload.get("symbol_id")
                        for h in hits
                        if h.payload.get("symbol_id")
                    ]
                    if sym_ids:
                        more = (
                            await db.execute(
                                select(ProjectSymbol).where(
                                    ProjectSymbol.id.in_(
                                        [UUID(s) for s in sym_ids if isinstance(s, str)]
                                    )
                                )
                            )
                        ).scalars().all()
                        for r in more:
                            seen[r.id] = r
            except Exception as exc:  # noqa: BLE001
                logger.warning("project_query_vector_failed", error=str(exc))

        out = list(seen.values())
        # Stable order: substring matches first, then by name length.
        out.sort(key=lambda r: (term.lower() not in r.name.lower(), len(r.name)))
        return out[:top_k]

    # ── neighbors ──
    async def neighbors(
        self,
        db: AsyncSession,
        *,
        symbol_id: UUID,
        depth: int = 1,
        max_nodes: int = 40,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return outbound + inbound neighborhoods up to `depth`."""
        out_ids: set[UUID] = set()
        in_ids: set[UUID] = set()
        frontier_out: set[UUID] = {symbol_id}
        frontier_in: set[UUID] = {symbol_id}
        for _ in range(max(0, depth)):
            next_out: set[UUID] = set()
            next_in: set[UUID] = set()
            if frontier_out:
                edges = (
                    await db.execute(
                        select(ProjectEdge.dst_id, ProjectEdge.relation).where(
                            ProjectEdge.src_id.in_(frontier_out),
                            ProjectEdge.dst_id.is_not(None),
                        )
                    )
                ).all()
                for dst_id, _ in edges:
                    if dst_id and dst_id not in out_ids:
                        out_ids.add(dst_id)
                        next_out.add(dst_id)
                        if len(out_ids) >= max_nodes:
                            break
            if frontier_in:
                edges = (
                    await db.execute(
                        select(ProjectEdge.src_id).where(ProjectEdge.dst_id.in_(frontier_in))
                    )
                ).all()
                for (src_id,) in edges:
                    if src_id and src_id not in in_ids:
                        in_ids.add(src_id)
                        next_in.add(src_id)
                        if len(in_ids) >= max_nodes:
                            break
            frontier_out = next_out
            frontier_in = next_in

        ids = (out_ids | in_ids) - {symbol_id}
        nodes = (
            (
                await db.execute(
                    select(ProjectSymbol).where(ProjectSymbol.id.in_(ids))
                )
            )
            .scalars()
            .all()
        )
        return {
            "outbound": [_node_dict(n) for n in nodes if n.id in out_ids],
            "inbound": [_node_dict(n) for n in nodes if n.id in in_ids],
        }

    # ── architecture map ──
    async def architecture_map(
        self,
        db: AsyncSession,
        *,
        project_id: UUID,
    ) -> list[dict[str, Any]]:
        """Connected-component grouping over the import-edge subgraph.

        Returns a list of clusters: [{cluster_id, files, size, centrality}].
        Cheap and stable; if you need finer modularity (Louvain etc.) add
        a real graph lib.
        """
        modules = (
            (
                await db.execute(
                    select(ProjectSymbol).where(
                        ProjectSymbol.project_id == project_id,
                        ProjectSymbol.kind == "module",
                    )
                )
            )
            .scalars()
            .all()
        )
        adj: dict[UUID, set[UUID]] = {m.id: set() for m in modules}

        edges = (
            (
                await db.execute(
                    select(ProjectEdge.src_id, ProjectEdge.dst_id).where(
                        ProjectEdge.project_id == project_id,
                        ProjectEdge.relation == "imports",
                        ProjectEdge.dst_id.is_not(None),
                    )
                )
            )
            .all()
        )
        for src_id, dst_id in edges:
            if src_id in adj and dst_id in adj:
                adj[src_id].add(dst_id)
                adj[dst_id].add(src_id)

        seen: set[UUID] = set()
        clusters: list[list[ProjectSymbol]] = []
        by_id = {m.id: m for m in modules}
        for m in modules:
            if m.id in seen:
                continue
            stack = [m.id]
            comp: list[ProjectSymbol] = []
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                comp.append(by_id[node])
                stack.extend(adj.get(node, ()))
            clusters.append(comp)

        clusters.sort(key=len, reverse=True)
        return [
            {
                "cluster_id": i,
                "size": len(c),
                "files": sorted(s.file_path for s in c),
                "central": _most_central(c, adj),
            }
            for i, c in enumerate(clusters)
        ]

    # ── helpers ──
    async def _embed_symbols(
        self,
        rows: list[tuple[ProjectSymbol, str]],
        *,
        project_id: UUID,
    ) -> None:
        # Batch embed.
        texts = [t for _, t in rows]
        try:
            emb = await self._llm.embed(texts)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("project_embed_failed", error=str(exc))
            return
        points: list[VectorPoint] = []
        ts = time.time()
        for (row, text), vec in zip(rows, emb.vectors, strict=False):
            if not vec:
                continue
            point_id = uuid.uuid5(uuid.NAMESPACE_URL, f"sym:{row.id.hex}").hex
            row.vector_id = point_id
            points.append(
                VectorPoint(
                    id=point_id,
                    vector=vec,
                    payload={
                        "project_id": str(project_id),
                        "kind": "project_symbol",
                        "chunk_kind": row.kind,
                        "language": row.language,
                        "source_uri": f"file://{row.file_path}",
                        "name": row.qualified_name,
                        "line_start": row.line_start,
                        "line_end": row.line_end,
                        "symbol_id": str(row.id),
                        "text": text,
                        "created_at": ts,
                    },
                )
            )
        if points:
            try:
                await self._qdrant.upsert(self._qdrant.rag, points)
            except Exception as exc:  # noqa: BLE001
                logger.warning("project_upsert_failed", error=str(exc))


def _walk(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        yield (p.relative_to(root).as_posix(), p)


def _symbol_text(s: RawSymbol) -> str:
    head = s.signature or s.qualified_name
    doc = s.docstring or ""
    return f"{head}\n{doc}".strip()


def _node_dict(n: ProjectSymbol) -> dict[str, Any]:
    return {
        "id": str(n.id),
        "name": n.name,
        "qualified_name": n.qualified_name,
        "kind": n.kind,
        "file_path": n.file_path,
        "lines": [n.line_start, n.line_end],
    }


def _most_central(comp: list[ProjectSymbol], adj: dict) -> str | None:
    """Highest-degree module qualified name in the component."""
    if not comp:
        return None
    best = max(comp, key=lambda m: len(adj.get(m.id, ())))
    return best.qualified_name


# Avoid unused imports in some paths.
_ = asyncio
