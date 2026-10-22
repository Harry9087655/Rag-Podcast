"""End-to-end run/replay orchestration over fake answer/session boundaries."""

import json
from contextlib import asynccontextmanager

import pytest

from rag_podcast.evaluation import runner
from rag_podcast.evaluation.contracts import ContractError, Observation, ObservationBundle
from rag_podcast.evaluation.loaders import EVALS_DIR, dataset_fingerprint, load_validated_dataset
from rag_podcast.qa import core

GATES = EVALS_DIR / "gates_v1.json"


def _write_dataset(tmp_path):
    raw = {
        "dataset_name": "qa-eval-v1", "dataset_version": "1", "schema_version": "dataset_v1",
        "cases": [{
            "id": "sample", "tags": ["single_fact"], "critical": True,
            "question": "Question", "expected_arm": "search", "scope": {},
            "gold_chunks": [1], "gold_segments": None,
            "reference_answer": "Answer", "required_claims": ["Fact"],
        }],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, dataset_fingerprint(raw)


def _answer(arm="search", answer="answer", chunk_id=1):
    async def fn(question, *, session=None, episode_id=None, podcast_id=None,
                 history=None, top_k=None, on_arm=None):
        if on_arm is not None:
            on_arm(arm)
        return core.AnswerResult(answer, [core.SourceRef(
            text="evidence", start=0.0, end=1.0, podcast_id=2, podcast_name="P",
            episode_id=1, episode_title="E", chunk_id=chunk_id,
        )])
    return fn


@asynccontextmanager
async def _session():
    yield object()


async def test_run_then_score_replay_is_byte_identical(tmp_path):
    dataset, _ = _write_dataset(tmp_path)
    live = await runner.run(
        dataset, GATES, output_dir=tmp_path / "run1",
        answer_fn=_answer(), session_factory=_session,
    )
    assert live.exit_code == 0
    assert (tmp_path / "run1" / "report.json").exists()

    replay = await runner.run(
        dataset, GATES,
        observations_path=tmp_path / "run1" / "observations.json",
        output_dir=tmp_path / "run2",
    )
    assert (tmp_path / "run1" / "report.json").read_text(encoding="utf-8") == \
        (tmp_path / "run2" / "report.json").read_text(encoding="utf-8")
    assert (tmp_path / "run1" / "report.md").read_text(encoding="utf-8") == \
        (tmp_path / "run2" / "report.md").read_text(encoding="utf-8")


async def test_fail_on_regression_returns_1_and_still_writes_report(tmp_path):
    dataset, _ = _write_dataset(tmp_path)
    outcome = await runner.run(
        dataset, GATES, output_dir=tmp_path / "run",
        answer_fn=_answer(), session_factory=_session,
        fail_on_regression=True,
    )
    assert outcome.report.passed is False
    assert outcome.exit_code == 1
    assert (tmp_path / "run" / "report.json").exists()


async def test_run_provenance_live_and_score_preserves_smoke(tmp_path):
    dataset, fingerprint = _write_dataset(tmp_path)
    live = await runner.run(
        dataset, GATES, output_dir=tmp_path / "run",
        answer_fn=_answer(), session_factory=_session,
    )
    assert live.report.metadata["provenance"] == "live"

    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(ObservationBundle(
        dataset_fingerprint=fingerprint, provenance="contract_smoke",
        observations=[Observation(case_id="sample", arm="search", answer_text="",
                                  sources=[], duration_ms=1, error=None)],
    ).model_dump_json(), encoding="utf-8")

    replay = await runner.run(
        dataset, GATES, observations_path=smoke_path, output_dir=tmp_path / "run2",
    )
    assert replay.report.metadata["provenance"] == "contract_smoke"


async def test_save_baseline_rejection_keeps_report(tmp_path):
    dataset, _ = _write_dataset(tmp_path)

    async def failing(question, *, session=None, episode_id=None, podcast_id=None,
                      history=None, top_k=None, on_arm=None):
        if on_arm is not None:
            on_arm("search")
        raise core.QAError("provider blew up")

    outcome = await runner.run(
        dataset, GATES, output_dir=tmp_path / "run",
        answer_fn=failing, session_factory=_session, save_baseline=True,
    )
    assert outcome.exit_code == 2
    assert outcome.baseline_saved is False
    assert (tmp_path / "run" / "report.json").exists()
    assert not (tmp_path / "run" / "baseline.json").exists()


async def test_save_baseline_on_score_is_contract_error(tmp_path):
    dataset, _ = _write_dataset(tmp_path)
    with pytest.raises(ContractError, match="live run"):
        await runner.run(
            dataset, GATES, observations_path=tmp_path / "obs.json",
            output_dir=tmp_path / "run", save_baseline=True,
        )


async def test_input_inside_output_dir_is_rejected(tmp_path):
    dataset, _ = _write_dataset(tmp_path)  # dataset lives under tmp_path
    with pytest.raises(ContractError, match="inside the output directory"):
        await runner.run(
            dataset, GATES, output_dir=tmp_path,
            answer_fn=_answer(), session_factory=_session,
        )


async def test_live_run_requires_answer_and_session(tmp_path):
    dataset, _ = _write_dataset(tmp_path)
    with pytest.raises(ContractError, match="answer_fn"):
        await runner.run(dataset, GATES, output_dir=tmp_path / "run")
