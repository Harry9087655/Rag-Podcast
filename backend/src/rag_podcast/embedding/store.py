"""Chunk persistence: replace an episode's chunks and parent chunks atomically.

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
from ..indexing.parent_grouper import ParentSpan
from ..models.chunk import Chunk
from ..models.episode import Episode
from ..models.parent_chunk import ParentChunk


async def replace_chunks(
    session: AsyncSession,
    episode: Episode,
    leaves: list[ChunkSpan],
    embeddings: list[list[float]],
    parents: list[ParentSpan],
) -> int:
    """Delete the episode's chunks/parents and insert leaves + parents.

    ``leaves`` and ``embeddings`` are zipped positionally, so the guard below
    is the safety rail against attaching the wrong vector to the wrong text.
    ``parents`` carry ``leaf_indices`` (indices into ``leaves``), which is how
    each leaf gets its ``parent_id``.

    Does not commit — the caller owns the transaction boundary (R2).  Returns
    the number of leaf rows written (``len(leaves)``).
    """
    if len(leaves) != len(embeddings):
        raise ValueError(
            f"leaf/embedding count mismatch: {len(leaves)} leaves vs "
            f"{len(embeddings)} embeddings"
        )

    await session.execute(
        delete(Chunk)
        .where(Chunk.episode_id == episode.id)
        .execution_options(synchronize_session=False)
    )
    await session.execute(
        delete(ParentChunk)
        .where(ParentChunk.episode_id == episode.id)
        .execution_options(synchronize_session=False)
    )

    # Insert parents first and flush so their ids exist for the leaves below.
    parent_rows = [
        ParentChunk(
            episode_id=episode.id,
            podcast_id=episode.podcast_id,
            text=parent.text,
            start=parent.start,
            end=parent.end,
        )
        for parent in parents
    ]
    session.add_all(parent_rows)
    await session.flush()

    # Map each leaf index to its parent's new id (parents cover every leaf).
    leaf_parent_id: dict[int, int] = {}
    for parent_row, parent in zip(parent_rows, parents):
        for leaf_index in parent.leaf_indices:
            leaf_parent_id[leaf_index] = parent_row.id

    chunks = [
        Chunk(
            episode_id=episode.id,
            podcast_id=episode.podcast_id,
            language=episode.language or "en",
            text=leaf.text,
            embedding=embeddings[i],
            start=leaf.start,
            end=leaf.end,
            parent_id=leaf_parent_id.get(i),
        )
        for i, leaf in enumerate(leaves)
    ]
    session.add_all(chunks)

    return len(leaves)
