"""Tests for ``group_into_parents`` (indexing, parent chunk grouping).

Reuses the same ``FakeTokenizer`` as ``test_size_based_chunker`` so token
counts are exact and readable, and so "what counts as a token" has one
definition across the indexing tests. The grouper only reads
``encode(text, add_special_tokens=False).ids`` length, so the fake's offset
logic (needed by the chunker) is irrelevant here.
"""

from __future__ import annotations

from rag_podcast.indexing.BaseChunkerInterface import ChunkSpan
from rag_podcast.indexing.parent_grouper import ParentSpan, group_into_parents
from tests.test_size_based_chunker import FakeTokenizer

TOKENIZER = FakeTokenizer()

DEFAULTS = {"desired_tokens": 300, "max_tokens": 500, "max_duration": 360.0}


def _leaf(start: float, end: float, n_tokens: int, prefix: str = "w") -> ChunkSpan:
    """A leaf worth exactly *n_tokens* under ``FakeTokenizer``."""
    text = " ".join(f"{prefix}{i}" for i in range(n_tokens))
    return ChunkSpan(text=text, start=start, end=end, segments=[])


def _tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)


def _group(leaves: list[ChunkSpan], **overrides) -> list[ParentSpan]:
    bounds = {**DEFAULTS, **overrides}
    return group_into_parents(leaves, count=_tokens, language="en", **bounds)


def test_empty_leaves_returns_empty_list():
    assert group_into_parents([], count=_tokens, **DEFAULTS) == []


def test_single_leaf_becomes_single_parent():
    leaves = [_leaf(0.0, 60.0, 50)]

    parents = _group(leaves)

    assert len(parents) == 1
    assert parents[0].leaf_indices == [0]
    assert parents[0].start == 0.0
    assert parents[0].end == 60.0


def test_groups_leaves_to_nearest_desired():
    # Five 100-token leaves against desired=300: the error is V-shaped, so the
    # first leaf where one more moves away from 300 closes the parent.
    leaves = [_leaf(i * 60.0, (i + 1) * 60.0, 100) for i in range(5)]

    parents = _group(leaves, desired_tokens=300)

    assert [p.leaf_indices for p in parents] == [[0, 1, 2], [3, 4]]
    assert [_tokens(p.text) for p in parents] == [300, 200]


def test_closes_when_next_leaf_would_breach_max_tokens():
    # Two 300-token leaves, max 500: 600 would breach, so they stay apart.
    leaves = [
        _leaf(0.0, 60.0, 300, prefix="a"),
        _leaf(60.0, 120.0, 300, prefix="b"),
    ]

    parents = _group(leaves, desired_tokens=300, max_tokens=500)

    assert [p.leaf_indices for p in parents] == [[0], [1]]


def test_closes_when_next_leaf_would_breach_max_duration():
    # Low token density: the duration max is the only ceiling in play.
    leaves = [
        _leaf(0.0, 200.0, 50, prefix="a"),
        _leaf(200.0, 400.0, 50, prefix="b"),
    ]

    parents = _group(leaves, desired_tokens=300, max_duration=300.0)

    assert [p.leaf_indices for p in parents] == [[0], [1]]


def test_parent_text_is_leaf_text_concatenation():
    leaves = [
        _leaf(0.0, 10.0, 3, prefix="a"),
        _leaf(10.0, 20.0, 3, prefix="b"),
    ]

    parents = _group(leaves, desired_tokens=100)

    assert len(parents) == 1
    assert parents[0].text == "a0 a1 a2 b0 b1 b2"


def test_parent_start_end_cover_its_leaves():
    leaves = [
        _leaf(0.0, 60.0, 100, prefix="a"),
        _leaf(60.0, 120.0, 100, prefix="b"),
        _leaf(120.0, 180.0, 100, prefix="c"),
    ]

    parents = _group(leaves, desired_tokens=300, max_tokens=500)

    assert len(parents) == 1
    assert parents[0].start == leaves[0].start == 0.0
    assert parents[0].end == leaves[-1].end == 180.0


def test_parents_cover_all_leaves_contiguously():
    leaves = [_leaf(i * 10.0, (i + 1) * 10.0, 20) for i in range(6)]

    parents = _group(leaves, desired_tokens=50, max_tokens=100)

    # Flattened leaf_indices reconstruct the full, ordered index sequence.
    flat = [i for parent in parents for i in parent.leaf_indices]
    assert flat == list(range(6))
    # First parent starts where the first leaf starts; last parent ends where
    # the last leaf ends.
    assert parents[0].start == leaves[0].start == 0.0
    assert parents[-1].end == leaves[-1].end == 60.0
    # Each parent's start/end are exactly its first/last leaf's.
    for parent in parents:
        first, last = leaves[parent.leaf_indices[0]], leaves[parent.leaf_indices[-1]]
        assert parent.start == first.start
        assert parent.end == last.end


def test_parent_token_counts_stay_within_max():
    leaves = [_leaf(i * 60.0, (i + 1) * 60.0, 100) for i in range(6)]

    parents = _group(leaves, desired_tokens=300, max_tokens=500)

    for parent in parents:
        assert _tokens(parent.text) <= 500
