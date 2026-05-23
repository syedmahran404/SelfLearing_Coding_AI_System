"""Retrieval-augmented generation.

Components:
- `chunking`   : language-aware chunkers (Python AST, Markdown headings,
                 generic windowed). Each chunk knows its source.
- `ingestion`  : load → chunk → embed → upsert to Qdrant.
- `retriever`  : hybrid retrieval (vector + BM25) with a reranker.
- `service`    : RagService facade — the only thing agents import.

Why hybrid retrieval?
    Vectors catch concepts, BM25 catches exact symbol names. For coding
    workflows the second is decisive — a function name miss can derail a
    whole task. Re-ranking blends them with provenance-aware penalties
    (stale `source_version`, off-language chunks) into a single score.
"""

from app.rag.service import RagService

__all__ = ["RagService"]
