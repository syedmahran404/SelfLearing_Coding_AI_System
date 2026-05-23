"""Project understanding engine.

Builds a structured view of a repository:
- `indexer`           : walks the tree, runs language extractors, persists
                        ProjectSymbol + ProjectEdge rows, embeds symbol
                        chunks for semantic search.
- `ast_python`        : extracts functions/classes/imports/calls from .py
                        via the stdlib `ast` module.
- `dependency_graph`  : query helpers (callers, callees, neighbors,
                        impact analysis).
- `architecture_map`  : groups files by import-graph community detection
                        for "where to put this code" decisions.
"""

from app.project.indexer import IndexStats, ProjectIndexer

__all__ = ["ProjectIndexer", "IndexStats"]
