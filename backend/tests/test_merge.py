"""Tests for ``auto_merge`` (retrieval, leaf → parent merge).

The merge function's DB touch is a single batched ``SELECT parent_chunk WHERE
id IN (...)``; the stub session below returns pre-seeded parent rows without a
real database, so the ordering / grouping / dropping logic is exercised purely.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_podcast.retrieval.BaseRecallerInterface import SearchResult
from rag_podcast.retrieval.merge import _inherit_score, auto_merge


def _doc(chunk_id: int, *, parent_id: int | None, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        start=float(chunk_id),
        end=float(chunk_id) + 1.0,
        podcast_id=1,
        podcast_name="pod",
        episode_id=1,
        episode_title="ep",
        score=score,
        parent_id=parent_id,
    )


@dataclass
class _Parent:
    id: int
    text: str
    start: float
    end: float
    podcast_id: int
    episode_id: int


class _Result:
    def __init__(self, parents: list[_Parent]):
        self._parents = parents

    def scalars(self):
        return self

    def all(self):
        return self._parents


class _Session:
    """Stub AsyncSession: returns the seeded parents for the id IN (...) query."""

    def __init__(self, parents: list[_Parent]):
        self._parents = parents

    async def execute(self, stmt):  # noqa: ANN001 — stub ignores the statement
        return _Result(self._parents)


PARENT = _Parent(id=1000, text="PARENT-TEXT", start=0.0, end=60.0, podcast_id=1, episode_id=1)


async def test_no_docs_returns_empty():
    assert await auto_merge([], _Session([]), 2) == []


async def test_threshold_below_2_passes_through():
    docs = [_doc(1, parent_id=100, score=0.1), _doc(2, parent_id=100, score=0.2)]
    result = await auto_merge(docs, _Session([]), 1)
    assert result == docs


async def test_flat_data_without_parent_id_passes_through():
    docs = [_doc(1, parent_id=None, score=0.1), _doc(2, parent_id=None, score=0.2)]
    result = await auto_merge(docs, _Session([]), 2)
    assert result == docs


async def test_merge_replaces_group_and_drops_siblings():
    docs = [
        _doc(10, parent_id=PARENT.id, score=0.1),
        _doc(11, parent_id=PARENT.id, score=0.2),
        _doc(20, parent_id=200, score=0.3),  # alone → below threshold
    ]

    result = await auto_merge(docs, _Session([PARENT]), 2)

    assert len(result) == 2
    assert result[0].chunk_id == PARENT.id
    assert result[0].level == "parent"
    assert result[0].merged_child_count == 2
    assert result[0].child_chunk_ids == [10, 11]
    assert result[0].text == "PARENT-TEXT"
    assert result[0].start == 0.0
    assert result[0].end == 60.0
    assert result[0].parent_id is None
    assert result[1].chunk_id == 20  # untouched single leaf


async def test_parent_inherits_best_child_score_under_distance():
    docs = [
        _doc(10, parent_id=PARENT.id, score=0.1),
        _doc(11, parent_id=PARENT.id, score=0.3),
    ]
    result = await auto_merge(docs, _Session([PARENT]), 2)
    assert result[0].score == 0.1  # min under distance


def test_parent_inherits_best_child_score_under_similarity(monkeypatch):
    monkeypatch.setattr("rag_podcast.retrieval.merge.settings.retrieval_score_mode", "similarity")
    assert _inherit_score([0.1, 0.3]) == 0.3  # max under similarity


async def test_multiple_groups_merge_independently():
    p1 = _Parent(id=1000, text="P1", start=0.0, end=10.0, podcast_id=1, episode_id=1)
    p2 = _Parent(id=2000, text="P2", start=20.0, end=30.0, podcast_id=1, episode_id=1)
    docs = [
        _doc(10, parent_id=p1.id, score=0.1),
        _doc(11, parent_id=p1.id, score=0.2),
        _doc(20, parent_id=p2.id, score=0.15),
        _doc(21, parent_id=p2.id, score=0.25),
    ]

    result = await auto_merge(docs, _Session([p1, p2]), 2)

    assert [r.chunk_id for r in result] == [1000, 2000]
    assert [r.level for r in result] == ["parent", "parent"]
    assert [r.merged_child_count for r in result] == [2, 2]
    assert [r.child_chunk_ids for r in result] == [[10, 11], [20, 21]]
