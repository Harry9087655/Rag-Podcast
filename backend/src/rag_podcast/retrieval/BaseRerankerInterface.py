"""Reranker interface — reserved, defined now, not called.

``Reranker`` is the second swappable pipeline axis (after ``Recaller``).  It is
defined in this task but never invoked: the ``search()`` body carries a
commented slot for it, and inserting the M4 cross-encoder is one line.  The
exact ``rerank_score`` convention (its scale vs ``score``, and any sign flip
under ``retrieval_score_mode == "distance"``) is an M4 concern — the field is
reserved, the semantics are not.

Design contract: .trellis/tasks/09-12-retrieval/design.md (D3)
"""

from __future__ import annotations

from typing import Protocol

from .BaseRecallerInterface import SearchResult


class Reranker(Protocol):
    """Abstract reranking interface: re-rank candidate docs by query relevance."""

    async def rerank(
        self,
        query: str,
        docs: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Re-rank ``docs`` for ``query`` and return the best ``top_k``."""
        ...
