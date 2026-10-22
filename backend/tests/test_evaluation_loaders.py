"""Strict policy parsing and baseline persistence; no real evaluation assets."""

import json
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from rag_podcast.evaluation.contracts import (
    Baseline, CaseResult, ContractError, GatePolicy, MetricGate, MetricValue, Report,
)
from rag_podcast.evaluation.loaders import load_baseline, load_gate_policy, save_baseline


def baseline_data():
    return {
        "dataset_fingerprint": "original-fingerprint", "provenance": "live",
        "k": 10,
        "metrics": {"recall@10": 0.8}, "eligible_cases": {"recall@10": 1},
        "case_passed": {"a": True, "b": False},
        "created_at": "2026-10-17T00:00:00+00:00",
    }


def report_fixture():
    return Report(
        dataset_name="synthetic", dataset_fingerprint="original-fingerprint",
        case_count=2,
        metrics={
            "recall@10": MetricValue(value=0.8, eligible_cases=1),
            "case_pass_rate": MetricValue(value=0.5, eligible_cases=2),
            "correctness": MetricValue(value=1, eligible_cases=2),
            "unknown": MetricValue(value=1, eligible_cases=2),
            "recall@0": MetricValue(value=1, eligible_cases=2),
        },
        slices={}, gates=[], passed=False, metadata={"provenance": "live", "k": 10},
        cases=[CaseResult(case_id=name, passed=passed, metrics={}, checks={})
               for name, passed in [("a", True), ("b", False)]],
    )


@pytest.mark.parametrize("field", ["minimum", "maximum", "max_regression"])
@pytest.mark.parametrize("value", [True, False, "0.5", float("nan"), float("inf"),
                                  -float("inf"), -0.01, 1.01])
def test_metric_gate_rejects_invalid_numbers(field, value):
    with pytest.raises(ValidationError):
        MetricGate(metric="case_pass_rate", **{field: value})


@pytest.mark.parametrize("value", [0, 1, 0.15, None])
def test_metric_gate_accepts_numeric_bounds_and_optional_null(value):
    gate = MetricGate(metric="case_pass_rate", minimum=value, max_regression=value)
    assert gate.minimum == value


@pytest.mark.parametrize("field,value", [
    ("k", True), ("k", "5"), ("k", 5.0),
    ("critical_no_regression", 1), ("critical_no_regression", "true"),
])
def test_policy_loader_rejects_coercion_with_path(tmp_path, field, value):
    path = tmp_path / "policy.json"
    data = {"metric_gates": [], "k": 5, "critical_no_regression": True,
            "required_provenance": ["live"], field: value}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        load_gate_policy(path)
    assert str(path) in str(exc.value)


def test_policy_semantics_stay_out_of_contract_and_defaults_are_independent():
    # Positive k and mandatory true are semantic checks performed by gate.py.
    first = GatePolicy(metric_gates=[], k=0, critical_no_regression=False)
    second = GatePolicy(metric_gates=[])
    first.required_provenance.append("live")
    assert second.required_provenance == []


@pytest.mark.parametrize("field,value", [
    ("provenance", "unknown"),
    *[("metrics", {"recall@10": v})
      for v in [True, "0.8", float("nan"), float("inf"), -0.1, 1.1]],
    *[("eligible_cases", {"recall@10": v}) for v in [True, "1", 1.0, -1]],
    *[("case_passed", {"a": v}) for v in [1, "true"]],
])
def test_baseline_loader_rejects_invalid_values_with_path(tmp_path, field, value):
    path = tmp_path / "baseline.json"
    data = baseline_data()
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        load_baseline(path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("change", ["old_format", "missing_key", "extra_key", "extra_field"])
def test_baseline_loader_rejects_incomplete_contract(tmp_path, change):
    data = baseline_data()
    if change == "old_format":
        del data["eligible_cases"]
    elif change == "missing_key":
        data["eligible_cases"] = {}
    elif change == "extra_key":
        data["eligible_cases"]["precision@10"] = 1
    else:
        data["unexpected"] = True
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        load_baseline(path)
    assert str(path) in str(exc.value)


def test_missing_baseline_and_corrupt_json_are_distinct(tmp_path):
    path = tmp_path / "baseline.json"
    assert load_baseline(path) is None
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        load_baseline(path)
    assert str(path) in str(exc.value)


def test_baseline_round_trip_preserves_denominators_and_filters_metrics(tmp_path):
    report = report_fixture()
    before = report.model_dump()
    path = tmp_path / "baseline.json"
    save_baseline(report, path)
    baseline = load_baseline(path)
    assert baseline.metrics == {"recall@10": 0.8, "case_pass_rate": 0.5}
    assert baseline.eligible_cases == {"recall@10": 1, "case_pass_rate": 2}
    assert baseline.case_passed == {"a": True, "b": False}
    assert baseline.dataset_fingerprint == report.dataset_fingerprint
    assert baseline.provenance == "live"
    assert baseline.k == 10
    assert baseline.k == 10
    assert datetime.fromisoformat(baseline.created_at).utcoffset() == timedelta(0)
    assert report.model_dump() == before
    assert path.read_text(encoding="utf-8") == (
        json.dumps(baseline.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize("name", [
    "case_pass_rate", "citation_hit", "citation_precision", "gold_chunk_coverage",
    "provider_failure_rate", "recall@1", "precision@10", "mrr@10", "ndcg@10",
])
def test_save_retains_registered_metric_families(tmp_path, name):
    report = report_fixture()
    report.metrics = {name: MetricValue(value=0, eligible_cases=1)}
    path = tmp_path / "baseline.json"
    save_baseline(report, path)
    assert load_baseline(path).metrics == {name: 0}


@pytest.mark.parametrize("name", [
    "correctness", "groundedness", "relevance", "completeness", "context_relevance",
    "unsupported_claim_rate", "conflict_disclosure_rate", "recall@01", "mrr@0",
])
def test_save_filters_judge_and_unregistered_metrics(tmp_path, name):
    report = report_fixture()
    report.metrics[name] = MetricValue(value=1, eligible_cases=2)
    path = tmp_path / "baseline.json"
    save_baseline(report, path)
    baseline = load_baseline(path)
    assert name not in baseline.metrics
    assert name not in baseline.eligible_cases


@pytest.mark.parametrize("field,value", [
    ("value", True), ("value", "0.8"), ("value", float("nan")),
    ("value", float("inf")), ("value", -0.1), ("value", 1.1),
    ("eligible_cases", True), ("eligible_cases", "1"),
    ("eligible_cases", -1), ("eligible_cases", 3),
])
def test_save_rejects_invalid_metric_before_overwriting(tmp_path, field, value):
    report = report_fixture()
    # Bypass parsing: the save boundary must also defend against mutated objects.
    setattr(report.metrics["recall@10"], field, value)
    path = tmp_path / "baseline.json"
    path.write_text("keep existing baseline", encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        save_baseline(report, path)
    assert str(path) in str(exc.value)
    assert path.read_text(encoding="utf-8") == "keep existing baseline"


@pytest.mark.parametrize("change", ["missing_provenance", "bad_provenance", "bad_passed",
                                   "duplicate_case", "case_count"])
def test_save_rejects_invalid_report_without_creating_file(tmp_path, change):
    report = report_fixture()
    if change == "missing_provenance":
        report.metadata.clear()
    elif change == "bad_provenance":
        report.metadata["provenance"] = "unknown"
    elif change == "bad_passed":
        report.cases[0].passed = 1
    elif change == "duplicate_case":
        report.cases[1].case_id = "a"
    else:
        report.case_count = 3
    path = tmp_path / "baseline.json"
    with pytest.raises(ContractError):
        save_baseline(report, path)
    assert not path.exists()


def test_zero_eligible_is_preserved_without_fabricating_denominator(tmp_path):
    report = report_fixture()
    report.metrics["recall@10"].eligible_cases = 0
    path = tmp_path / "baseline.json"
    save_baseline(report, path)
    assert load_baseline(path).eligible_cases["recall@10"] == 0


def test_baseline_requires_eligible_cases_even_for_direct_construction():
    data = baseline_data()
    del data["eligible_cases"]
    with pytest.raises(ValidationError):
        Baseline(**data)


@pytest.mark.parametrize("value", [None, True, "5", 5.0, 0, -1])
def test_baseline_requires_strict_positive_k(tmp_path, value):
    data = baseline_data()
    if value is None:
        del data["k"]
    else:
        data["k"] = value
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractError, match="k"):
        load_baseline(path)


def test_save_requires_explicit_k_before_overwriting(tmp_path):
    report = report_fixture()
    del report.metadata["k"]
    path = tmp_path / "baseline.json"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(ContractError, match="k"):
        save_baseline(report, path)
    assert path.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("value", [True, "10", 10.0, 0, -1])
def test_save_rejects_invalid_k_before_overwriting(tmp_path, value):
    report = report_fixture()
    report.metadata["k"] = value
    path = tmp_path / "baseline.json"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(ContractError, match="k"):
        save_baseline(report, path)
    assert path.read_text(encoding="utf-8") == "preserve"
