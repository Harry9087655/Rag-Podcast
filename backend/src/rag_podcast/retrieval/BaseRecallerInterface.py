"""Retrieval interfaces: the result type and the recall contract.

Every stage of the search pipeline is a pure ``list[SearchResult] →
list[SearchResult]`` transform.  ``SearchResult`` is the uniform document the
pipeline carries end-to-end; ``Recaller`` is the swappable axis that produces
the candidate list.  Concrete recallers return ``SearchResult`` with a ``score``
field — the ``Recaller`` Protocol is the contract, not shared scoring code.

Design contract: .trellis/tasks/09-12-retrieval/design.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from sqlalchemy.ext.asyncio import AsyncSession


class RetrievalError(Exception):
    """Raised when retrieval itself fails (bad strategy, recall error, …)."""


@dataclass
class SearchResult:
    """One chunk match, self-contained for the ``qa``/``frontend`` consumer.

    Every field needed to render a source is filled by recall's single join
    (``podcast_name``/``episode_title`` come from the joined rows, not a
    downstream re-query).  ``score`` is the recall score per
    ``settings.retrieval_score_mode`` (cosine distance by default);
    ``rerank_score`` is reserved for the M4 reranker and takes over
    ``effective_score`` once set.
    """

    chunk_id: int
    text: str
    start: float
    end: float
    podcast_id: int
    podcast_name: str
    episode_id: int
    episode_title: str
    score: float
    rerank_score: float | None = None
    parent_id: int | None = None
    level: Literal["leaf", "parent"] = "leaf"
    merged_child_count: int = 0
    child_chunk_ids: list[int] = field(default_factory=list)

    @property
    def effective_score(self) -> float:
        """The score a consumer should use: rerank when present, else recall."""
        return self.rerank_score if self.rerank_score is not None else self.score


class Recaller(Protocol):
    """Abstract recall interface: turn an embedded query into ranked candidates.

    Async because recall is DB I/O (``asyncpg``).  The caller owns the
    ``session``; the vector is already embedded by ``search``.  A future
    ``HybridRecaller`` (dense + tsvector RRF) drops in behind this Protocol
    without reshaping the pipeline.
    """

    async def recall(
        self,
        session: AsyncSession,
        vector: list[float],
        *,
        top_k: int,
        podcast_id: int | None = None,
        episode_id: int | None = None,
    ) -> list[SearchResult]:
        """Return up to ``top_k`` chunks nearest ``vector``, best-first.

        When ``podcast_id`` is set, results are restricted to that podcast;
        when ``episode_id`` is set, to that single episode; ``None`` on both
        searches the whole library.
        """
        ...
