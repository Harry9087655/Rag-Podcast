"""Adapter projections and error isolation with fake answer/session boundaries."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag_podcast.evaluation.adapters import LiveAdapter, PredictionFileAdapter
from rag_podcast.evaluation.contracts import (
    Case,
    ContractError,
    Dataset,
    DatasetFingerprintMismatch,
    Observation,
    ObservationBundle,
    Scope,
    Source,
)
from rag_podcast.embedding.embedder import EmbeddingError
from rag_podcast.qa import core
from rag_podcast.retrieval.BaseRecallerInterface import RetrievalError

FP = "fingerprint-1"


def _source(**kw):
    base = dict(
        text="evidence", start=1.0, end=2.0, podcast_id=2,
        podcast_name="Podcast", episode_id=1, episode_title="Episode",
        chunk_id=10,
    )
    base.update(kw)
    return core.SourceRef(**base)


def _case(cid, **kw):
    base = dict(
        id=cid, tags=["single_fact"], critical=False, question=f"Q {cid}",
        expected_arm="search", scope=Scope(), gold_chunks=[1],
        gold_segments=None, reference_answer="A", required_claims=["F"],
    )
    base.update(kw)
    return Case(**base)


def _dataset(*cases):
    return Dataset(
        dataset_name="d", dataset_version="1", schema_version="dataset_v1",
        cases=list(cases),
    )


def _session_log():
    opened = []
    closed = []

    @asynccontextmanager
    async def factory():
        session = object()
        opened.append(session)
        try:
            yield session
        finally:
            closed.append(session)

    return factory, opened, closed


def _answer_fn(arm="search", results=None):
    """Fake answer: records scope/kwargs, emits ``arm``, then pops ``results``.

    Each result is either an ``AnswerResult`` (returned) or an ``Exception``
    (raised after the arm is emitted).
    """
    calls = []

    async def fn(question, *, session=None, episode_id=None, podcast_id=None,
                 history=None, top_k=None, on_arm=None):
        calls.append(dict(
            question=question, episode_id=episode_id, podcast_id=podcast_id,
            history=history, top_k=top_k,
        ))
        if on_arm is not None:
            on_arm(arm)
        if results is None:
            return core.AnswerResult("", [])
        item = results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return fn, calls


# ---- LiveAdapter: projection, arms, scope, k ----

async def test_live_observes_both_arms_and_scope_and_k():
    full = _case("c1", expected_arm="full-context", scope=Scope(episode_id=1))
    search = _case("c2", scope=Scope(podcast_id=2))
    answer_fn, calls = _answer_fn(arm="search", results=[
        core.AnswerResult("full answer", [_source(chunk_id=None)]),
        core.AnswerResult("search answer", [_source(chunk_id=10)]),
    ])
    factory, _, _ = _session_log()
    adapter = LiveAdapter(answer_fn, factory)

    bundle = await adapter.execute(_dataset(full, search), FP, k=5)

    assert bundle.provenance == "live"
    assert bundle.dataset_fingerprint == FP
    assert [o.arm for o in bundle.observations] == ["search", "search"]
    assert [o.answer_text for o in bundle.observations] == ["full answer", "search answer"]
    # Scope and k are forwarded verbatim; history is always None.
    assert calls[0]["episode_id"] == 1 and calls[0]["podcast_id"] is None
    assert calls[1]["episode_id"] is None and calls[1]["podcast_id"] == 2
    assert all(c["top_k"] == 5 and c["history"] is None for c in calls)


async def test_live_projects_source_whitelist_keeping_parent_identity():
    parent = _source(chunk_id=20, level="parent", child_chunk_ids=[21, 22])
    answer_fn, _ = _answer_fn(results=[core.AnswerResult("a", [parent])])
    factory, _, _ = _session_log()
    bundle = await LiveAdapter(answer_fn, factory).execute(
        _dataset(_case("c1")), FP, k=5,
    )

    projected = bundle.observations[0].sources[0]
    assert projected == Source(
        chunk_id=20, start=1.0, end=2.0, episode_id=1, text="evidence",
        level="parent", child_chunk_ids=[21, 22],
    )
    # Whitelist: no podcast_name / episode_title / podcast_id leak through.
    assert set(projected.model_dump()) == {
        "chunk_id", "start", "end", "episode_id", "text", "level", "child_chunk_ids",
    }


async def test_live_empty_answer_and_sources_are_success_observations():
    answer_fn, _ = _answer_fn(results=[core.AnswerResult("", [])])
    factory, _, _ = _session_log()
    bundle = await LiveAdapter(answer_fn, factory).execute(
        _dataset(_case("c1")), FP, k=5,
    )
    obs = bundle.observations[0]
    assert obs.answer_text == "" and obs.sources == [] and obs.error is None


async def test_duration_is_nonnegative_int_and_measured():
    async def slow(question, *, session=None, episode_id=None, podcast_id=None,
                   history=None, top_k=None, on_arm=None):
        if on_arm is not None:
            on_arm("search")
        await asyncio.sleep(0.02)
        return core.AnswerResult("a", [])

    factory, _, _ = _session_log()
    obs = (await LiveAdapter(slow, factory).execute(
        _dataset(_case("c1")), FP, k=5,
    )).observations[0]
    assert isinstance(obs.duration_ms, int) and obs.duration_ms >= 0
    assert obs.duration_ms > 0


# ---- LiveAdapter: error isolation ----

@pytest.mark.parametrize("error_type", [core.QAError, EmbeddingError, RetrievalError])
async def test_provider_failure_is_isolated_and_batch_continues(error_type):
    answer_fn, calls = _answer_fn(results=[
        error_type("provider blew up"),
        core.AnswerResult("ok", [_source(chunk_id=2)]),
    ])
    factory, _, _ = _session_log()
    bundle = await LiveAdapter(answer_fn, factory).execute(
        _dataset(_case("c1"), _case("c2")), FP, k=5,
    )

    failed, ok = bundle.observations
    assert failed.error == "provider_failed"
    assert failed.answer_text == "" and failed.sources == []
    assert "blew up" not in failed.model_dump_json()  # no raw exception leak
    assert ok.error is None and ok.answer_text == "ok"
    assert len(calls) == 2


@pytest.mark.parametrize("error_type", [core.QAError, EmbeddingError, RetrievalError])
async def test_provider_failure_before_arm_terminates(error_type):
    async def fn(question, *, session=None, episode_id=None, podcast_id=None,
                 history=None, top_k=None, on_arm=None):
        raise error_type("no route chosen")

    factory, _, _ = _session_log()
    with pytest.raises(error_type):
        await LiveAdapter(fn, factory).execute(_dataset(_case("c1")), FP, k=5)


async def test_case_timeout_after_arm_becomes_observation():
    async def fn(question, *, session=None, episode_id=None, podcast_id=None,
                 history=None, top_k=None, on_arm=None):
        if on_arm is not None:
            on_arm("search")
        await asyncio.sleep(10)
        return core.AnswerResult("never", [])

    factory, _, _ = _session_log()
    bundle = await LiveAdapter(fn, factory).execute(
        _dataset(_case("c1"), _case("c2")), FP, k=5, case_timeout=0.01,
    )
    assert [o.error for o in bundle.observations] == ["case_timeout", "case_timeout"]
    assert all(o.answer_text == "" and o.sources == [] for o in bundle.observations)


async def test_case_timeout_before_arm_terminates():
    async def fn(question, *, session=None, episode_id=None, podcast_id=None,
                 history=None, top_k=None, on_arm=None):
        await asyncio.sleep(10)
        return core.AnswerResult("never", [])

    factory, _, _ = _session_log()
    with pytest.raises(TimeoutError):
        await LiveAdapter(fn, factory).execute(
            _dataset(_case("c1")), FP, k=5, case_timeout=0.01,
        )


async def test_unexpected_and_session_errors_propagate():
    factory_ok, _, _ = _session_log()
    with pytest.raises(RuntimeError):
        await LiveAdapter(_answer_fn(results=[RuntimeError("boom")])[0], factory_ok).execute(
            _dataset(_case("c1")), FP, k=5,
        )

    @asynccontextmanager
    async def broken_factory():
        raise RuntimeError("cannot connect")
        yield  # pragma: no cover

    answer_fn, calls = _answer_fn()
    with pytest.raises(RuntimeError, match="cannot connect"):
        await LiveAdapter(answer_fn, broken_factory).execute(
            _dataset(_case("c1")), FP, k=5,
        )
    assert calls == []


async def test_cancellation_propagates_and_session_closes():
    async def fn(question, *, session=None, episode_id=None, podcast_id=None,
                 history=None, top_k=None, on_arm=None):
        if on_arm is not None:
            on_arm("search")
        raise asyncio.CancelledError

    factory, opened, closed = _session_log()
    with pytest.raises(asyncio.CancelledError):
        await LiveAdapter(fn, factory).execute(_dataset(_case("c1")), FP, k=5)
    assert opened and opened == closed  # session released before propagating


async def test_session_factory_per_case_and_closed_on_success_and_failure():
    answer_fn, _ = _answer_fn(results=[
        core.QAError("fail"),
        core.AnswerResult("ok", []),
    ])
    factory, opened, closed = _session_log()
    await LiveAdapter(answer_fn, factory).execute(
        _dataset(_case("c1"), _case("c2")), FP, k=5,
    )
    assert len(opened) == len(closed) == 2  # one fresh session per case, all closed


async def test_missing_or_repeated_arm_is_contract_failure():
    async def no_arm(question, *, session=None, episode_id=None, podcast_id=None,
                     history=None, top_k=None, on_arm=None):
        return core.AnswerResult("a", [])

    factory, _, _ = _session_log()
    with pytest.raises(ContractError, match="without emitting an arm"):
        await LiveAdapter(no_arm, factory).execute(_dataset(_case("c1")), FP, k=5)

    async def twice(question, *, session=None, episode_id=None, podcast_id=None,
                    history=None, top_k=None, on_arm=None):
        on_arm("full-context")
        on_arm("search")
        return core.AnswerResult("a", [])

    with pytest.raises(ContractError, match="more than once"):
        await LiveAdapter(twice, factory).execute(_dataset(_case("c1")), FP, k=5)


async def test_invalid_projection_data_is_not_swallowed():
    bad = SimpleNamespace(
        answer="a",
        sources=[SimpleNamespace(chunk_id=1, start=0.0, end=1.0, episode_id=1,
                                 text="t", level="bogus", child_chunk_ids=[])],
    )
    answer_fn, _ = _answer_fn(results=[bad])
    factory, _, _ = _session_log()
    with pytest.raises(ValidationError):
        await LiveAdapter(answer_fn, factory).execute(_dataset(_case("c1")), FP, k=5)


async def test_invalid_k_and_timeout_rejected_before_any_call():
    answer_fn, calls = _answer_fn()
    factory, _, _ = _session_log()
    adapter = LiveAdapter(answer_fn, factory)
    with pytest.raises(ContractError, match="positive integer"):
        await adapter.execute(_dataset(_case("c1")), FP, k=0)
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ContractError, match="case_timeout"):
            await adapter.execute(_dataset(_case("c1")), FP, k=5, case_timeout=bad)
    assert calls == []


# ---- PredictionFileAdapter: replay ----

def _bundle(*observations, fingerprint=FP, provenance="live"):
    return ObservationBundle(
        dataset_fingerprint=fingerprint, provenance=provenance,
        observations=list(observations),
    )


def _obs(cid, arm="search", error=None):
    return Observation(case_id=cid, arm=arm, answer_text="a", sources=[],
                       duration_ms=1, error=error)


def test_prediction_replays_bundle_preserving_provenance(tmp_path):
    path = tmp_path / "obs.json"
    bundle = _bundle(_obs("c1", "search"), _obs("c2", "full-context"),
                     provenance="contract_smoke")
    path.write_text(bundle.model_dump_json(), encoding="utf-8")

    replay = PredictionFileAdapter(path).execute(_dataset(_case("c1"), _case("c2")), FP)
    assert replay == bundle
    assert replay.provenance == "contract_smoke"


def test_prediction_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "obs.json"
    path.write_text(_bundle(_obs("c1")).model_dump_json(), encoding="utf-8")
    with pytest.raises(DatasetFingerprintMismatch):
        PredictionFileAdapter(path).execute(_dataset(_case("c1")), "other-fp")


def test_prediction_rejects_duplicate_and_unknown_ids(tmp_path):
    path = tmp_path / "obs.json"
    path.write_text(_bundle(_obs("c1"), _obs("c1")).model_dump_json(), encoding="utf-8")
    with pytest.raises(ContractError, match="Duplicate observation"):
        PredictionFileAdapter(path).execute(_dataset(_case("c1")), FP)

    path.write_text(_bundle(_obs("c1"), _obs("ghost")).model_dump_json(), encoding="utf-8")
    with pytest.raises(ContractError, match="Unknown observation"):
        PredictionFileAdapter(path).execute(_dataset(_case("c1")), FP)


def test_prediction_allows_missing_observations(tmp_path):
    path = tmp_path / "obs.json"
    path.write_text(_bundle(_obs("c1")).model_dump_json(), encoding="utf-8")
    replay = PredictionFileAdapter(path).execute(_dataset(_case("c1"), _case("c2")), FP)
    assert [o.case_id for o in replay.observations] == ["c1"]


def test_prediction_bad_file_is_contract_error(tmp_path):
    path = tmp_path / "obs.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ContractError):
        PredictionFileAdapter(path).execute(_dataset(_case("c1")), FP)
