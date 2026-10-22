"""Auto-merge retrieved sibling leaves back into their parent chunk.

Retrieval happens at the leaf level (the only embedded/retrieved unit). When
≥ ``threshold`` leaves sharing one ``parent_id`` all land in the candidate
list, they are replaced by their time-contiguous parent block — one longer,
more listenable source instead of N fragmentary neighbours.

Design contract: .trellis/tasks/09-17-supermew-chunking-retrieval/design.md §5.3
"""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.parent_chunk import ParentChunk
from .BaseRecallerInterface import SearchResult

logger = logging.getLogger(__name__)


def _inherit_score(scores: list[float]) -> float:
    """The parent takes its best child's score.

    Under ``distance`` (smaller is better) that is ``min``; under
    ``similarity`` it is ``max``. Either way the parent lands exactly where its
    best child did, so the merged list stays ordered without a re-sort.
    """
    if settings.retrieval_score_mode == "similarity":
        return max(scores)
    return min(scores)


def _parent_result(parent: ParentChunk, members: list[SearchResult]) -> SearchResult:
    """Build the parent ``SearchResult`` that replaces ``members``.

    ``chunk_id`` is the parent block's own id (its identity downstream); text,
    start and end come from the parent row so the merged span is one
    contiguous block. podcast/episode titles are inherited from a member —
    every member shares the same episode, so ``members[0]`` is representative.
    ``parent_id`` stays ``None``: a parent has no parent of its own.
    """
    return SearchResult(
        chunk_id=parent.id,
        text=parent.text,
        start=parent.start,
        end=parent.end,
        podcast_id=parent.podcast_id,
        podcast_name=members[0].podcast_name,
        episode_id=parent.episode_id,
        episode_title=members[0].episode_title,
        score=_inherit_score([m.score for m in members]),
        level="parent",
        merged_child_count=len(members),
        child_chunk_ids=[m.chunk_id for m in members],
    )


async def auto_merge(
    docs: list[SearchResult],
    session: AsyncSession,
    threshold: int,
) -> list[SearchResult]:
    """Replace ≥``threshold`` same-parent sibling leaves with their parent block.

    ``docs`` arrive in recall order (best-first). Leaves without a
    ``parent_id`` (pre-``0007`` flat data) are passed through untouched, so a
    partially-migrated corpus still behaves like flat retrieval.

    The parent replaces the *first* leaf of its group (preserving that slot in
    the ordering); the group's remaining leaves are dropped as already covered.
    Parents are fetched in one batched query, never per-group.
    """
    if not docs or threshold < 2:
        return docs

    # Group leaves by parent, remembering each group's first index so the
    # parent can take that slot in the replayed list.
    groups: dict[int, list[SearchResult]] = defaultdict(list)
    first_index: dict[int, int] = {}
    for index, doc in enumerate(docs):
        if doc.parent_id is None:
            continue
        if doc.parent_id not in first_index:
            first_index[doc.parent_id] = index
        groups[doc.parent_id].append(doc)

    merge_ids = [pid for pid, members in groups.items() if len(members) >= threshold]
    if not merge_ids:
        return docs

    rows = (
        await session.execute(select(ParentChunk).where(ParentChunk.id.in_(merge_ids)))
    ).scalars().all()
    parent_by_id = {row.id: row for row in rows}

    merged: list[SearchResult] = []
    for index, doc in enumerate(docs):
        pid = doc.parent_id
        if pid in parent_by_id:
            if index == first_index[pid]:
                merged.append(_parent_result(parent_by_id[pid], groups[pid]))
            # Else: a later sibling of a group already emitted — covered by it.
            continue
        merged.append(doc)

    return merged
