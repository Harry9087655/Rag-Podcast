"""Group contiguous leaves into parent chunks.

Parents are time-contiguous merges of adjacent leaf chunks (the only embedded
and retrieved unit). The grouper runs the same greedy "nearest-desired + hard
max" gate as ``SizeBasedChunker``, but with no min floor — a parent has only a
desired token target and hard token/duration ceilings, so the gate is simply
"close when appending would breach a max or move further from the target".

Time continuity is guaranteed by construction: a parent is always a run of
*consecutive* leaves, so its ``start``/``end`` are the first leaf's start and
the last leaf's end, with no gaps to validate at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .BaseChunkerInterface import ChunkSpan


@dataclass
class ParentSpan:
    """One contiguous merge of leaf chunks, ready to store without embedding.

    ``text`` is the separator-joined concatenation of its leaves' text;
    ``start``/``end`` are the first leaf's start and last leaf's end;
    ``leaf_indices`` are the indices into the input ``leaves`` list that this
    parent covers (the store layer uses them to set each leaf's ``parent_id``).
    """

    text: str
    start: float
    end: float
    leaf_indices: list[int]


def group_into_parents(
    leaves: list[ChunkSpan],
    *,
    desired_tokens: int,
    max_tokens: int,
    max_duration: float,
    count: Callable[[str], int],
    language: str | None = "en",
) -> list[ParentSpan]:
    """Merge consecutive leaves into parent chunks.

    ``leaves`` must be in chronological order and cover the episode
    contiguously (the chunker's contract). The returned parents cover the same
    span, each a run of consecutive leaves.

    ``count`` is a token-counting callable (the chunker's ``count_tokens``).
    Counts run once per leaf and are summed in the accumulation loop — never by
    re-tokenizing the growing buffer, which would be O(n^2). ``language`` only
    selects the join separator (empty for CJK, space otherwise), matching
    ``SizeBasedChunker``.
    """
    if not leaves:
        return []

    separator = "" if (language or "en").lower() != "en" else " "

    parents: list[ParentSpan] = []
    buffer: list[int] = []  # indices into `leaves`
    buffer_tokens = 0

    for index, leaf in enumerate(leaves):
        if not buffer:
            buffer, buffer_tokens = [index], count(leaf.text)
            continue

        leaf_tokens = count(leaf.text)
        next_tokens = buffer_tokens + leaf_tokens

        # Hard-max OR gate across the two dimensions: either ceiling alone
        # forces a close. The token max is the one expected to fire in
        # practice; the duration max stays active as a low-token-density net.
        breaches_max = (
            leaf.end - leaves[buffer[0]].start > max_duration
            or next_tokens > max_tokens
        )

        # Nearest-target stopping: the cumulative token count only grows as
        # leaves are appended, so the error is V-shaped in position — the
        # first leaf where "one more makes it worse" is the local optimum.
        overshoots = abs(desired_tokens - buffer_tokens) < abs(
            desired_tokens - next_tokens
        )

        if breaches_max or overshoots:
            parents.append(_make_parent(leaves, buffer, separator))
            buffer, buffer_tokens = [index], leaf_tokens
        else:
            buffer.append(index)
            buffer_tokens = next_tokens

    if buffer:
        parents.append(_make_parent(leaves, buffer, separator))

    return parents


def _make_parent(
    leaves: list[ChunkSpan], indices: list[int], separator: str
) -> ParentSpan:
    # Leaf text is already stripped at chunk time (_make_span), so a plain
    # separator join reconstructs the parent text without extra whitespace.
    text = separator.join(leaves[i].text for i in indices)
    return ParentSpan(
        text=text,
        start=leaves[indices[0]].start,
        end=leaves[indices[-1]].end,
        leaf_indices=indices,
    )
