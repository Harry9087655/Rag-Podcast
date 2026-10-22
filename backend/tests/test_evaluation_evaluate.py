"""Offline tests through the actual evaluation orchestrator."""

import asyncio
import importlib

import pytest

from rag_podcast.evaluation.contracts import (
    Case, CasePassPolicy, ContractError, Dataset,
    JudgeScores, Observation, ObservationBundle,
    Source,
)
from rag_podcast.evaluation.evaluate import evaluate
from rag_podcast.evaluation.loaders import EVALS_DIR, load_baseline, load_gate_policy, save_baseline
from rag_podcast.evaluation.render import render_json, render_markdown


@pytest.fixture
def inputs():
    dataset = Dataset(
        dataset_name="synthetic", dataset_version="1", schema_version="1",
        cases=[
            Case(id="search", tags=["z", "shared"], critical=True, question="Find?",
                 expected_arm="search", scope={}, gold_chunks=[1, 2], gold_segments=None,
                 reference_answer="Found.", required_claims=[]),
            Case(id="context", tags=["shared"], critical=False, question="When?",
                 expected_arm="full-context", scope={"episode_id": 1}, gold_chunks=None,
                 gold_segments=[{"start": 2, "end": 3}],
                 reference_answer="Then.", required_claims=[]),
            Case(id="unknown", tags=["no_knowledge"], critical=False, question="Unknown?",
                 expected_arm="search", scope={}, gold_chunks=None, gold_segments=None,
                 reference_answer="Unknown.", required_claims=[]),
        ],
    )
    bundle = ObservationBundle(
        dataset_fingerprint="fingerprint", provenance="live", observations=[
            Observation(case_id="search", arm="search", answer_text="Found.", sources=[
                Source(chunk_id=100, start=0, end=4, episode_id=1, text="Found.",
                       level="parent", child_chunk_ids=[1, 2]),
            ]),
            Observation(case_id="context", arm="full-context", answer_text="Then.", sources=[
                Source(chunk_id=None, start=1, end=4, episode_id=1, text="Then."),
            ]),
            Observation(case_id="unknown", arm="search", answer_text="Unknown.", sources=[]),
        ],
    )
    return dataset, bundle, load_gate_policy(EVALS_DIR / "gates_v1.json")


async def run(inputs, **kwargs):
    dataset, bundle, policy = inputs
    return await evaluate(dataset, "fingerprint", bundle, policy, **kwargs)


async def test_complete_report_scores_metadata_and_rendering(inputs):
    report = await run(inputs)
    assert report.passed
    assert [case.case_id for case in report.cases] == ["search", "context", "unknown"]
    assert all(case.passed for case in report.cases)
    assert report.metrics["recall@5"].value == 1
    assert report.metrics["recall@5"].eligible_cases == 1
    assert report.metrics["citation_hit"].value == 1
    assert report.metrics["provider_failure_rate"].eligible_cases == 3
    assert report.slices["shared"]["case_pass_rate"].eligible_cases == 2
    assert report.metadata == {
        "provenance": "live", "k": 5, "observed_cases": 3,
        "missing_observation_cases": 0, "provider_failed_cases": 0,
        "arm_search_cases": 2, "arm_full_context_cases": 1,
        "judge_requested": False, "judge_attempted_cases": 0,
        "judge_succeeded_cases": 0, "judge_failed_cases": 0,
        "judge_skipped_cases": 3, "judge_available": False,
    }
    assert render_json(report) == render_json(await run(inputs))
    assert "required_provenance" in render_markdown(report)


async def test_order_and_inputs_are_preserved(inputs):
    dataset, bundle, policy = inputs
    before = [item.model_dump() for item in inputs]
    report = await run(inputs)
    assert [item.model_dump() for item in inputs] == before
    bundle.observations.reverse()
    policy.metric_gates.reverse()
    assert render_json(report) == render_json(await run(inputs))
    assert [case.case_id for case in report.cases] == [case.id for case in dataset.cases]


@pytest.mark.parametrize("error", ["provider_failed", ""])
async def test_missing_and_provider_failures_have_distinct_denominators(inputs, error):
    dataset, bundle, policy = inputs
    bundle.observations[0].error = error
    bundle.observations.pop(1)
    report = await run(inputs)
    assert report.cases[0].metrics["gold_chunk_coverage"] == 1
    assert report.cases[0].checks == {"provider_failed": True}
    assert report.cases[1].checks == {"missing_observation": True}
    assert report.cases[1].metrics == {}
    assert report.metrics["case_pass_rate"].value == pytest.approx(1 / 3)
    assert report.metrics["provider_failure_rate"].value == 0.5
    assert report.metrics["provider_failure_rate"].eligible_cases == 2
    assert report.metadata["observed_cases"] == 2
    assert report.metadata["missing_observation_cases"] == 1
    assert report.metadata["provider_failed_cases"] == 1
    assert not report.passed


@pytest.mark.parametrize("empty_dataset", [False, True])
async def test_no_observations_fail_without_fabricated_scores(inputs, empty_dataset):
    dataset, bundle, _ = inputs
    bundle.observations.clear()
    if empty_dataset:
        dataset.cases.clear()
    report = await run(inputs)
    assert not report.passed
    assert report.metrics["provider_failure_rate"].eligible_cases == 0
    assert report.metadata["missing_observation_cases"] == len(dataset.cases)
    assert all(case.checks == {"missing_observation": True} for case in report.cases)


async def test_dynamic_rank_and_explicit_case_policy(inputs):
    _, bundle, policy = inputs
    policy.k = 10
    for gate in policy.metric_gates:
        gate.metric = gate.metric.replace("@5", "@10")
    bundle.observations[0].arm = "full-context"  # Actual arm is informational.
    report = await run(inputs)
    assert report.passed
    assert "precision@10" in report.cases[0].checks
    assert report.metadata["k"] == 10
    explicit = CasePassPolicy(search={"precision@5": 0.6})
    report = await run(inputs, policy=explicit)
    assert not report.cases[0].passed
    assert report.cases[0].checks == {"missing_metric:precision@5": True}


@pytest.mark.parametrize("kind", ["dataset", "observation", "unknown", "policy"])
async def test_preflight_errors_happen_before_scoring_or_judge(inputs, monkeypatch, kind):
    dataset, bundle, policy = inputs
    module = importlib.import_module("rag_podcast.evaluation.evaluate")

    def forbidden(*args, **kwargs):
        pytest.fail("scoring/Judge must not start")

    monkeypatch.setattr(module, "score_case", forbidden)
    monkeypatch.setattr(module, "judge_report", forbidden)
    if kind == "dataset":
        dataset.cases.extend([dataset.cases[2], dataset.cases[0]])
        expected = "Duplicate dataset case_ids: ['search', 'unknown']"
    elif kind == "observation":
        bundle.observations[0].case_id = "alien"
        bundle.observations.append(bundle.observations[1])
        expected = "Duplicate observation case_ids: ['context']"
    elif kind == "unknown":
        bundle.observations[0].case_id = "z"
        bundle.observations[1].case_id = "a"
        expected = "Unknown observation case_ids: ['a', 'z']"
    else:
        policy.k = -1
        expected = "k must be a positive integer"
    with pytest.raises(ContractError) as error:
        await run(inputs, run_judge=True)
    assert expected in str(error.value)


@pytest.mark.parametrize("enabled", [False, True])
async def test_judge_enrichment_is_serial_and_deterministically_isolated(inputs, enabled):
    original = await run(inputs)
    calls = []

    async def judge(case, obs):
        calls.append(case.id)
        await asyncio.sleep(0)
        if case.id == "context":
            return None
        return JudgeScores(
            correctness=1, groundedness=1, relevance=1, completeness=1,
            context_relevance=1, unsupported_claim_rate=0,
            conflict_disclosure_rate=0, reason="Supported.",
        )

    report = await run(inputs, run_judge=enabled, judge_fn=judge)
    assert calls == (["search", "context", "unknown"] if enabled else [])
    for field in ("metrics", "slices", "gates", "passed"):
        assert getattr(report, field) == getattr(original, field)
    assert [case.passed for case in report.cases] == [case.passed for case in original.cases]
    assert report.metadata["judge_attempted_cases"] == (3 if enabled else 0)
    assert report.metadata["judge_failed_cases"] == (1 if enabled else 0)


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_judge_unexpected_errors_propagate_and_stop(inputs, error_type):
    calls = []

    async def judge(case, obs):
        calls.append(case.id)
        raise error_type()

    with pytest.raises(error_type):
        await run(inputs, run_judge=True, judge_fn=judge)
    assert calls == ["search"]


async def test_baseline_roundtrip_and_regressions(inputs, tmp_path):
    report = await run(inputs)
    path = tmp_path / "baseline.json"
    save_baseline(report, path)
    baseline = load_baseline(path)
    assert baseline.k == 5
    assert baseline.eligible_cases["recall@5"] == 1
    assert (await run(inputs, baseline=baseline)).passed
    inputs[1].observations.pop(0)
    regressed = await run(inputs, baseline=baseline)
    assert not regressed.passed
    statuses = {gate.name: gate.status for gate in regressed.gates}
    assert statuses["critical:search:current"] == "FAIL"
    assert statuses["critical:search:regression"] == "FAIL"
    assert statuses["metric:recall@5:eligible"] == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("k", 10), ("provenance", "contract_smoke"), ("case_passed", {}),
    ("metrics", {}),
])
async def test_incompatible_baseline_never_calls_judge(inputs, tmp_path, field, value):
    path = tmp_path / "baseline.json"
    save_baseline(await run(inputs), path)
    baseline = load_baseline(path)
    setattr(baseline, field, value)

    async def forbidden(case, obs):
        pytest.fail("Judge must not start")

    with pytest.raises(ContractError):
        await run(inputs, baseline=baseline, run_judge=True, judge_fn=forbidden)


async def test_smoke_report_preserves_fail_and_no_baseline(inputs):
    inputs[1].provenance = "contract_smoke"
    report = await run(inputs)
    assert not report.passed
    assert report.gates[0].name == "required_provenance"
    assert report.gates[0].status == "FAIL"
    assert any(gate.status == "NO_BASELINE" for gate in report.gates)


async def test_judge_skips_missing_and_provider_failed_cases(inputs):
    inputs[1].observations[0].error = "provider_failed"
    inputs[1].observations.pop(1)
    calls = []

    async def judge(case, obs):
        calls.append(case.id)
        return None

    report = await run(inputs, run_judge=True, judge_fn=judge)
    assert calls == ["unknown"]
    assert report.metadata["judge_attempted_cases"] == 1
    assert report.metadata["judge_succeeded_cases"] == 0
    assert report.metadata["judge_failed_cases"] == 1
    assert report.metadata["judge_skipped_cases"] == 2
    assert report.metadata["judge_available"] is False


async def test_judge_does_not_change_saved_baseline(inputs, tmp_path):
    async def judge(case, obs):
        return JudgeScores(
            correctness=0, groundedness=0, relevance=0, completeness=0,
            context_relevance=0, unsupported_claim_rate=1,
            conflict_disclosure_rate=1, reason="Low quality.",
        )

    plain = await run(inputs)
    judged = await run(inputs, run_judge=True, judge_fn=judge)
    before = [item.model_dump() for item in inputs]
    path = tmp_path / "baseline.json"
    save_baseline(plain, path)
    original = load_baseline(path).model_dump(exclude={"created_at"})
    save_baseline(judged, path)
    assert load_baseline(path).model_dump(exclude={"created_at"}) == original
    assert [item.model_dump() for item in inputs] == before
