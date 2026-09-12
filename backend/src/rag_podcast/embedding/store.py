"""Chunk persistence: replace an episode's chunks atomically.

Library code — ``replace_chunks`` is the write half of the single-transaction
boundary (R2).  It deletes and inserts but **never commits**; the worker owns
the transaction and the one ``commit()`` that makes ``done`` a trustworthy
claim.

Design contract: .trellis/tasks/09-05-embedding-pipeline/design.md
"""

from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..indexing.BaseChunkerInterface import ChunkSpan
from ..models.chunk import Chunk
from ..models.episode import Episode


async def replace_chunks(
    session: AsyncSession,
    episode: Episode,
    spans: list[ChunkSpan],
    embeddings: list[list[float]],
) -> int:
    """Delete the episode's existing chunks and insert ``spans`` + ``embeddings``.

    ``spans`` and ``embeddings`` are zipped positionally, so the guard below is
    the safety rail against attaching the wrong vector to the wrong text.

    Does not commit — the caller owns the transaction boundary (R2).  Returns
    the number of rows written (``len(spans)``).
    """
    if len(spans) != len(embeddings):
        raise ValueError(
            f"span/embedding count mismatch: {len(spans)} spans vs "
            f"{len(embeddings)} embeddings"
        )

    await session.execute(
        delete(Chunk)
        .where(Chunk.episode_id == episode.id)
        .execution_options(synchronize_session=False)
    )

    chunks = [
        Chunk(
            episode_id=episode.id,
            podcast_id=episode.podcast_id,
            language=episode.language or "en",
            text=span.text,
            embedding=embeddings[i],
            start=span.start,
            end=span.end,
        )
        for i, span in enumerate(spans)
    ]
    session.add_all(chunks)

    return len(spans)
