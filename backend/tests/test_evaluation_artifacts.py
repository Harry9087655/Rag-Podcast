"""Run-artifact persistence: directories, atomic writes, completion marker, baseline candidates."""

import pytest

from rag_podcast.evaluation import artifacts
from rag_podcast.evaluation.contracts import (
    CaseResult, ContractError, JudgeScores, MetricValue, Observation,
    ObservationBundle, Report,
)
from rag_podcast.evaluation.loaders import EVALS_DIR, load_gate_policy, load_observations


@pytest.fixture(scope="module")
def policy():
    return load_gate_policy(EVALS_DIR / "gates_v1.json")


def _report(**metadata_overrides):
    metrics = {
        "case_pass_rate": MetricValue(value=1.0, eligible_cases=2),
        "citation_hit": MetricValue(value=1.0, eligible_cases=1),
        "citation_precision": MetricValue(value=1.0, eligible_cases=1),
        "recall@5": MetricValue(value=1.0, eligible_cases=1),
        "precision@5": MetricValue(value=1.0, eligible_cases=1),
        "gold_chunk_coverage": MetricValue(value=1.0, eligible_cases=1),
        "provider_failure_rate": MetricValue(value=0.0, eligible_cases=2),
        "mrr@5": MetricValue(value=1.0, eligible_cases=1),
        "ndcg@5": MetricValue(value=1.0, eligible_cases=1),
    }
    metadata = {
        "provenance": "live", "k": 5,
        "provider_failed_cases": 0, "missing_observation_cases": 0,
    }
    metadata.update(metadata_overrides)
    return Report(
        dataset_name="d", dataset_fingerprint="fp", case_count=2,
        metrics=metrics, slices={},
        cases=[
            CaseResult(case_id="c1", passed=True, metrics={}, checks={}),
            CaseResult(case_id="c2", passed=True, metrics={}, checks={}),
        ],
        gates=[], passed=True, metadata=metadata,
    )


def _bundle():
    return ObservationBundle(
        dataset_fingerprint="fp", provenance="live",
        observations=[Observation(case_id="c1", arm="search", answer_text="a",
                                   sources=[], duration_ms=1, error=None)],
    )


# ---- directory resolution ----

def test_create_run_dir_is_unique_and_rejects_existing(tmp_path):
    first = artifacts.create_run_dir(tmp_path)
    second = artifacts.create_run_dir(tmp_path)
    assert first != second and first.exists() and second.exists()

    with pytest.raises(ContractError, match="already exists"):
        artifacts.create_run_dir(tmp_path, explicit=first)

    manual = artifacts.create_run_dir(tmp_path, explicit=tmp_path / "manual")
    assert manual == tmp_path / "manual" and manual.exists()


# ---- bundle and report writing ----

def test_write_bundle_roundtrips(tmp_path):
    bundle = _bundle()
    artifacts.write_bundle(tmp_path, bundle)
    assert load_observations(tmp_path / "observations.json") == bundle


def test_write_report_writes_markdown_then_json_completion_marker(tmp_path, monkeypatch):
    report = _report()
    written = []
    real_write = artifacts._atomic_write

    def spy(path, text):
        written.append(path.name)
        return real_write(path, text)

    monkeypatch.setattr(artifacts, "_atomic_write", spy)
    artifacts.write_report(tmp_path, report)
    assert written == ["report.md", "report.json"]
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()


def test_write_report_renders_both_before_writing(tmp_path, monkeypatch):
    def boom(report):
        raise RuntimeError("render failed")

    monkeypatch.setattr(artifacts, "render_json", boom)
    with pytest.raises(RuntimeError):
        artifacts.write_report(tmp_path, _report())
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "report.md").exists()


def test_write_report_content_contains_no_run_dir_path(tmp_path):
    artifacts.write_report(tmp_path, _report())
    content = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in content


def test_write_failure_propagates_when_directory_missing(tmp_path):
    with pytest.raises(OSError):
        artifacts.write_bundle(tmp_path / "missing", _bundle())


# ---- baseline candidates ----

def test_baseline_candidate_rejects_provider_failed_and_missing(tmp_path, policy):
    with pytest.raises(ContractError, match="provider_failed"):
        artifacts.write_baseline_candidate(tmp_path, _report(provider_failed_cases=1), policy)
    with pytest.raises(ContractError, match="missing observations"):
        artifacts.write_baseline_candidate(tmp_path, _report(missing_observation_cases=1), policy)


def test_baseline_candidate_rejects_policy_metric_without_eligible_cases(tmp_path, policy):
    report = _report()
    report.metrics["gold_chunk_coverage"] = MetricValue(value=1.0, eligible_cases=0)
    with pytest.raises(ContractError, match="gold_chunk_coverage"):
        artifacts.write_baseline_candidate(tmp_path, report, policy)


def test_baseline_candidate_success_excludes_judge_and_is_read_only(tmp_path, policy):
    report = _report()
    report.cases[0].judge = JudgeScores(
        correctness=1.0, groundedness=1.0, relevance=1.0, completeness=1.0,
        context_relevance=1.0, unsupported_claim_rate=0.0,
        conflict_disclosure_rate=0.0, reason="ok",
    )
    before = report.model_dump()

    artifacts.write_baseline_candidate(tmp_path, report, policy)

    baseline_text = (tmp_path / "baseline.json").read_text(encoding="utf-8")
    assert "judge" not in baseline_text
    assert '"case_pass_rate"' in baseline_text
    assert report.model_dump() == before
