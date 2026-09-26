"""Concrete dense recall: pgvector cosine distance over ``chunk.embedding``.

One SQL statement — ``chunk ⋈ episode ⋈ podcast`` — returns fully-joined,
self-contained ``SearchResult`` rows, so no later pipeline stage touches the
DB.  The operator is pinned to ``<=>`` (cosine distance), which is what the
HNSW ``vector_cosine_ops`` index (``0005``) is built for; only the score
*value* exposed on ``SearchResult`` is transformed by ``_score_from_distance``.

Design contract: .trellis/tasks/09-12-retrieval/design.md (D5)
"""

from __future__ import annotations

import logging
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.chunk import Chunk
from ..models.episode import Episode
from ..models.podcast import Podcast
from .BaseRecallerInterface import SearchResult

logger = logging.getLogger(__name__)


def _score_from_distance(
    distance: float, mode: Literal["distance", "similarity"]
) -> float:
    """Map a cosine distance to the configured score presentation.

    ``<=>`` returns cosine *distance* (0 = identical, 2 = opposite);
    ``similarity`` flips it to ``1 - distance`` so higher-is-better consumers
    can use it unchanged.  Ordering is unaffected either way — ``<=> ASC`` is
    already best-first for both modes.
    """
    return distance if mode == "distance" else 1.0 - distance


class PgvectorRecaller:
    """Recall via ``chunk.embedding <=> vector`` ordered ascending."""

    async def recall(
        self,
        session: AsyncSession,
        vector: list[float],
        *,
        top_k: int,
        podcast_id: int | None = None,
        episode_id: int | None = None
    ) -> list[SearchResult]:
        distance = Chunk.embedding.cosine_distance(vector)
        stmt = (
            select(Chunk, distance.label("distance"), Episode.title, Podcast.name)
            .join(Episode, Episode.id == Chunk.episode_id)
            .join(Podcast, Podcast.id == Chunk.podcast_id)
            .order_by(distance)
            .limit(top_k)
        )
        if podcast_id is not None:
            stmt = stmt.where(Chunk.podcast_id == podcast_id)
        if episode_id is not None:
            stmt = stmt.where(Chunk.episode_id == episode_id)

        rows = (await session.execute(stmt)).all()

        score_mode = settings.retrieval_score_mode
        return [
            SearchResult(
                chunk_id=chunk.id,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                podcast_id=chunk.podcast_id,
                podcast_name=podcast_name,
                episode_id=chunk.episode_id,
                episode_title=episode_title,
                score=_score_from_distance(float(dist), score_mode),
                parent_id=chunk.parent_id,
            )
            for chunk, dist, episode_title, podcast_name in rows
        ]
