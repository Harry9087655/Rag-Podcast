"""Exercise real QA routing with fake DB/model boundaries, without network I/O."""

import asyncio
from dataclasses import asdict
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from rag_podcast.qa import core


@pytest.fixture
def qa(monkeypatch):
    ctx = core.EpisodeContext(
        episode_id=1, episode_title="Episode", podcast_id=2,
        podcast_name="Podcast", language="en",
        full_transcript=[core.TranscriptSegment(text="Evidence", start=1, end=2)],
    )
    chunks = [
        core.SourceRef(
            text="Parent evidence", start=3, end=5, podcast_id=2,
            podcast_name="Podcast", episode_id=1, episode_title="Episode",
            chunk_id=20, level="parent", child_chunk_ids=[21, 22],
        ),
        core.SourceRef(
            text="Leaf evidence", start=6, end=7, podcast_id=2,
            podcast_name="Podcast", episode_id=1, episode_title="Episode",
            chunk_id=30,
        ),
    ]
    load = AsyncMock(return_value=ctx)
    search = AsyncMock(return_value=chunks)
    agentic = AsyncMock(return_value=("Full answer", [
        {"found": True, "text": "Evidence", "start": 1, "end": 2},
    ]))
    chat = AsyncMock(return_value="Search answer [2] [1] [2]")
    monkeypatch.setattr(core, "_load_episode_context", load)
    monkeypatch.setattr(core, "_get_context", search)
    monkeypatch.setattr(core, "_chat_agentic", agentic)
    monkeypatch.setattr(core, "_chat", chat)
    monkeypatch.setattr(core, "count_tokens", lambda messages: 10)
    monkeypatch.setattr(core, "_input_budget", lambda: 10)
    return load, search, agentic, chat, chunks


async def test_full_context_observed_before_model_and_legacy_result_unchanged(qa):
    load, search, agentic, chat, _ = qa
    arms = []

    def observe(arm):
        agentic.assert_not_awaited()
        arms.append(arm)

    session = object()
    observed = await core.answer("Question", session=session, episode_id=1, on_arm=observe)
    legacy = await core.answer("Question", session=session, episode_id=1)

    assert arms == ["full-context"]
    assert asdict(observed) == asdict(legacy) == {
        "answer": "Full answer",
        "sources": [{
            "text": "Evidence", "start": 1, "end": 2, "podcast_id": 2,
            "podcast_name": "Podcast", "episode_id": 1, "episode_title": "Episode",
            "chunk_id": None, "level": "leaf", "child_chunk_ids": [],
        }],
    }
    load.assert_awaited_with(session, 1)
    search.assert_not_awaited()
    chat.assert_not_awaited()


@pytest.mark.parametrize("route", ["over_budget", "no_transcript", "podcast", "library"])
async def test_search_observed_before_retrieval_and_preserves_citation_order(qa, monkeypatch, route):
    load, search, agentic, chat, chunks = qa
    scope = {"episode_id": 1}
    if route == "over_budget":
        monkeypatch.setattr(core, "count_tokens", lambda messages: 11)
    elif route == "no_transcript":
        load.return_value = None
    elif route == "podcast":
        scope = {"podcast_id": 2}
    else:
        scope = {}
    arms = []

    def observe(arm):
        search.assert_not_awaited()
        arms.append(arm)

    session = object()
    result = await core.answer("Question", session=session, top_k=7, on_arm=observe, **scope)
    legacy = await core.answer("Question", session=session, top_k=7, **scope)

    assert arms == ["search"]
    assert asdict(result) == asdict(legacy)
    assert result.answer == "Search answer [2] [1] [2]"
    assert result.sources == [chunks[1], chunks[0]]
    assert result.sources[1].child_chunk_ids == [21, 22]
    search.assert_awaited_with(
        session, episode_id=scope.get("episode_id"), podcast_id=scope.get("podcast_id"),
        top_k=7, question="Question",
    )
    agentic.assert_not_awaited()
    assert chat.await_count == 2
    if route in ("podcast", "library"):
        load.assert_not_awaited()


@pytest.mark.parametrize("arm", ["full-context", "search"])
async def test_empty_sources_still_report_actual_arm(qa, arm):
    _, search, agentic, chat, _ = qa
    search.return_value = []
    agentic.return_value = ("", [])
    arms = []
    scope = {"episode_id": 1} if arm == "full-context" else {}
    result = await core.answer("Question", session=object(), on_arm=arms.append, **scope)
    assert arms == [arm]
    assert result == core.AnswerResult("", [])
    chat.assert_not_awaited()


@pytest.mark.parametrize("stage", ["agentic", "retrieval", "chat"])
@pytest.mark.parametrize("error_type", [core.QAError, asyncio.CancelledError])
async def test_selected_arm_survives_work_failure_or_cancellation(qa, stage, error_type):
    _, search, agentic, chat, _ = qa
    target = {"agentic": agentic, "retrieval": search, "chat": chat}[stage]
    error = error_type("failed")
    target.side_effect = error
    scope = {"episode_id": 1} if stage == "agentic" else {}
    arms = []
    with pytest.raises(error_type) as caught:
        await core.answer("Question", session=object(), on_arm=arms.append, **scope)
    assert caught.value is error
    assert arms == ["full-context" if stage == "agentic" else "search"]


async def test_database_failure_before_routing_does_not_invent_arm(qa):
    load, search, agentic, chat, _ = qa
    error = SQLAlchemyError("database failed")
    load.side_effect = error
    observe = Mock()
    with pytest.raises(SQLAlchemyError) as caught:
        await core.answer("Question", session=object(), episode_id=1, on_arm=observe)
    assert caught.value is error
    observe.assert_not_called()
    search.assert_not_awaited()
    agentic.assert_not_awaited()
    chat.assert_not_awaited()


@pytest.mark.parametrize("arm", ["full-context", "search"])
async def test_callback_error_propagates_before_selected_work(qa, arm):
    _, search, agentic, chat, _ = qa
    error = RuntimeError("observer failed")
    observe = Mock(side_effect=error)
    scope = {"episode_id": 1} if arm == "full-context" else {}
    with pytest.raises(RuntimeError) as caught:
        await core.answer("Question", session=object(), on_arm=observe, **scope)
    assert caught.value is error
    observe.assert_called_once_with(arm)
    search.assert_not_awaited()
    agentic.assert_not_awaited()
    chat.assert_not_awaited()
