"""Tests for policy completeness, dynamic k, and immutable guardrails."""

import pytest

from rag_podcast.evaluation.contracts import (
    Baseline, CaseResult, ContractError, GatePolicy, MetricGate, MetricValue,
)
from rag_podcast.evaluation.gate import evaluate_gates, validate_gate_policy


def _policy(k=5):
    floors = {
        "case_pass_rate": 0.5, "citation_hit": 0.5,
        "citation_precision": 0.5, f"recall@{k}": 0.4,
        f"precision@{k}": 0.4, "gold_chunk_coverage": 0.6,
    }
    return GatePolicy(
        k=k, required_provenance=["live"],
        metric_gates=[
            MetricGate(metric=n, minimum=v, max_regression=0.15)
            for n, v in floors.items()
        ] + [MetricGate(metric="provider_failure_rate", maximum=0.3)],
    )


@pytest.mark.parametrize("k", [1, 5, 10])
def test_valid_dynamic_policy_does_not_mutate_input(k):
    policy = _policy(k)
    before = policy.model_dump()
    assert validate_gate_policy(policy) == []
    assert policy.model_dump() == before


def test_missing_errors_are_sorted():
    policy = _policy()
    policy.metric_gates = []
    assert validate_gate_policy(policy) == [f"Missing required metric: {n}" for n in sorted([
        "case_pass_rate", "citation_hit", "citation_precision", "recall@5",
        "precision@5", "gold_chunk_coverage", "provider_failure_rate",
    ])]


def test_duplicate_unknown_and_wrong_k():
    policy = _policy(10)
    policy.metric_gates.extend([
        MetricGate(metric="recall@10", minimum=0.4, max_regression=0.15),
        MetricGate(metric="correctness", minimum=0.5),
        MetricGate(metric="recall@5", minimum=0.4),
    ])
    assert validate_gate_policy(policy) == [
        "Duplicate metric: recall@10",
        "Unknown or non-deterministic metric: correctness",
        "Unknown or non-deterministic metric: recall@5",
    ]


@pytest.mark.parametrize("field,value,expected", [
    ("minimum", None, "minimum is required"),
    ("minimum", 0.49, "minimum must be at least 0.5"),
    ("maximum", 0.9, "maximum is not allowed"),
    ("max_regression", None, "max_regression is required"),
    ("max_regression", 0.16, "max_regression must not exceed 0.15"),
    ("minimum", float("nan"), "finite number"),
    ("minimum", float("inf"), "finite number"),
    ("minimum", -0.1, "finite number"),
    ("minimum", 1.1, "finite number"),
    ("minimum", True, "finite number"),
    ("minimum", "0.5", "finite number"),
])
def test_invalid_thresholds(field, value, expected):
    policy = _policy()
    # Assignment bypasses parsing to test the function's own checks.
    setattr(policy.metric_gates[0], field, value)
    assert any(expected in e for e in validate_gate_policy(policy))


def test_lower_is_better_and_optional_metrics():
    policy = _policy(10)
    provider = policy.metric_gates[-1]
    provider.max_regression = 0.15
    policy.metric_gates.extend([
        MetricGate(metric="mrr@10", minimum=0, max_regression=0),
        MetricGate(metric="ndcg@10", minimum=1),
    ])
    assert validate_gate_policy(policy) == []
    provider.maximum = 0.31
    provider.minimum = 0
    errors = validate_gate_policy(policy)
    assert any("maximum must not exceed 0.3" in e for e in errors)
    assert any("minimum is not allowed" in e for e in errors)


@pytest.mark.parametrize("k", [0, -1, True, 2.5, "5"])
def test_invalid_k(k):
    policy = _policy()
    policy.k = k
    assert validate_gate_policy(policy)[0] == "k must be a positive integer"


def test_global_errors_precede_missing_metrics():
    policy = _policy()
    policy.critical_no_regression = False
    policy.required_provenance = ["contract_smoke", "live"]
    policy.metric_gates = []
    errors = validate_gate_policy(policy)
    assert errors[:2] == [
        "critical_no_regression must be true",
        "required_provenance must be exactly ['live']",
    ]
    assert len(errors) == 9


def _run_inputs(k=5):
    policy = _policy(k)
    metrics = {
        g.metric: MetricValue(value=0.1 if g.metric == "provider_failure_rate" else 0.8,
                              eligible_cases=2)
        for g in policy.metric_gates
    }
    cases = [CaseResult(case_id=i, passed=True, metrics={}, checks={}) for i in ("b", "a")]
    baseline = Baseline(
        dataset_fingerprint="synthetic", k=k, provenance="live",
        metrics={n: v.value for n, v in metrics.items()},
        eligible_cases={n: v.eligible_cases for n, v in metrics.items()},
        case_passed={c.case_id: c.passed for c in cases}, created_at="2026-10-17T00:00:00Z",
    )
    return policy, metrics, cases, baseline


def _gates(policy, metrics, cases, baseline=None, critical_ids=None, provenance="live"):
    return {g.name: g for g in evaluate_gates(
        metrics, cases, {"a"} if critical_ids is None else critical_ids,
        policy, baseline, provenance=provenance,
    )}


@pytest.mark.parametrize("k", [1, 5, 10])
def test_complete_gate_order_fields_and_input_immutability(k):
    policy, metrics, cases, baseline = _run_inputs(k)
    before = (policy.model_dump(), {n: v.model_dump() for n, v in metrics.items()},
              [c.model_dump() for c in cases], baseline.model_dump())
    gates = _gates(policy, metrics, cases, baseline, {"b", "a"})
    expected = ["required_provenance"]
    for rule in sorted(policy.metric_gates, key=lambda g: g.metric):
        suffixes = ["coverage", "maximum" if rule.maximum is not None else "minimum"]
        if rule.max_regression is not None:
            suffixes.append("regression")
        suffixes.append("eligible")
        expected.extend(f"metric:{rule.metric}:{suffix}" for suffix in suffixes)
    expected.extend(f"critical:{i}:{s}" for i in ("a", "b") for s in ("current", "regression"))
    assert list(gates) == expected
    assert all(g.status == "PASS" for g in gates.values())
    absolute = gates["metric:citation_hit:minimum"]
    assert (absolute.actual, absolute.baseline, absolute.threshold) == (0.8, None, 0.5)
    eligible = gates["metric:citation_hit:eligible"]
    assert (eligible.actual, eligible.baseline, eligible.threshold) == (2, 2, 2)
    reordered = policy.model_copy(deep=True)
    reordered.metric_gates.reverse()
    assert _gates(reordered, dict(reversed(list(metrics.items()))), list(reversed(cases)),
                  baseline, {"a", "b"}) == gates
    assert before == (policy.model_dump(), {n: v.model_dump() for n, v in metrics.items()},
                      [c.model_dump() for c in cases], baseline.model_dump())


def test_no_baseline_keeps_absolute_failure_and_smoke_failure():
    policy, metrics, cases, _ = _run_inputs()
    metrics["citation_hit"].value = 0.4
    gates = _gates(policy, metrics, cases, provenance="contract_smoke")
    assert gates["required_provenance"].status == "FAIL"
    assert gates["metric:citation_hit:minimum"].status == "FAIL"
    regression = gates["metric:citation_hit:regression"]
    assert (regression.status, regression.actual, regression.baseline, regression.threshold) == (
        "NO_BASELINE", 0.4, None, 0.15)
    eligible = gates["metric:citation_hit:eligible"]
    assert (eligible.status, eligible.actual, eligible.threshold) == ("NO_BASELINE", 2, None)


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("with_baseline", [True, False])
def test_uncovered_scores_never_use_placeholder_values(missing, with_baseline):
    policy, metrics, cases, baseline = _run_inputs()
    if missing:
        del metrics["provider_failure_rate"]
    else:
        metrics["provider_failure_rate"] = MetricValue(value=0, eligible_cases=0)
    policy.metric_gates[-1].max_regression = 0.15
    gates = _gates(policy, metrics, cases, baseline if with_baseline else None)
    prefix = "metric:provider_failure_rate:"
    assert gates[prefix + "coverage"].status == "FAIL"
    absolute = gates[prefix + "maximum"]
    assert absolute.status == "FAIL" and absolute.actual is None
    assert absolute.detail == ("missing_metric" if missing else "no_eligible_cases")
    regression = gates[prefix + "regression"]
    assert regression.status == ("FAIL" if with_baseline else "NO_BASELINE")
    assert regression.actual is None
    assert gates[prefix + "eligible"].status == ("FAIL" if with_baseline else "NO_BASELINE")


@pytest.mark.parametrize("metric,previous,current,budget,status", [
    ("citation_hit", 0.8, 0.65, 0.15, "PASS"),
    ("citation_hit", 0.8, 0.649999, 0.15, "FAIL"),
    ("citation_hit", 0.8, 0.9, 0, "PASS"),
    ("citation_hit", 0.8, 0.8, 0, "PASS"),
    ("provider_failure_rate", 0.1, 0.25, 0.15, "PASS"),
    ("provider_failure_rate", 0.1, 0.250001, 0.15, "FAIL"),
    ("provider_failure_rate", 0.1, 0, 0, "PASS"),
])
def test_regression_directions_and_float_boundaries(metric, previous, current, budget, status):
    policy, metrics, cases, baseline = _run_inputs()
    next(g for g in policy.metric_gates if g.metric == metric).max_regression = budget
    baseline.metrics[metric] = previous
    metrics[metric].value = current
    gate = _gates(policy, metrics, cases, baseline)[f"metric:{metric}:regression"]
    assert (gate.status, gate.actual, gate.baseline, gate.threshold) == (status, current, previous, budget)
    assert "direction=" in gate.detail and "delta=" in gate.detail


@pytest.mark.parametrize("metric,value,status", [
    ("citation_hit", 0.5, "PASS"), ("citation_hit", 0.499999, "FAIL"),
    ("provider_failure_rate", 0.3, "PASS"), ("provider_failure_rate", 0.300001, "FAIL"),
])
def test_absolute_boundaries(metric, value, status):
    policy, metrics, cases, _ = _run_inputs()
    metrics[metric].value = value
    suffix = "maximum" if metric == "provider_failure_rate" else "minimum"
    assert _gates(policy, metrics, cases)[f"metric:{metric}:{suffix}"].status == status


def test_improving_mean_does_not_hide_eligible_shrinkage():
    policy, metrics, cases, baseline = _run_inputs()
    metrics["citation_hit"] = MetricValue(value=0.9, eligible_cases=1)
    gates = _gates(policy, metrics, cases, baseline)
    assert gates["metric:citation_hit:regression"].status == "PASS"
    assert gates["metric:citation_hit:eligible"].status == "FAIL"


@pytest.mark.parametrize("previous,current,regression", [
    (True, False, "FAIL"), (False, False, "PASS"),
    (False, True, "PASS"), (True, True, "PASS"), (None, False, "NO_BASELINE"),
])
def test_critical_current_and_regression_are_independent(previous, current, regression):
    policy, metrics, cases, baseline = _run_inputs()
    next(c for c in cases if c.case_id == "a").passed = current
    baseline.case_passed["a"] = previous if previous is not None else True
    gates = _gates(policy, metrics, cases, baseline if previous is not None else None)
    assert gates["critical:a:current"].status == ("PASS" if current else "FAIL")
    assert gates["critical:a:regression"].status == regression


def test_no_critical_and_optional_metrics():
    policy, metrics, cases, _ = _run_inputs()
    policy.metric_gates.append(MetricGate(metric="mrr@5", minimum=0.2))
    metrics["mrr@5"] = MetricValue(value=0.5, eligible_cases=1)
    gates = _gates(policy, metrics, cases, critical_ids=set())
    assert "metric:mrr@5:regression" not in gates
    assert gates["metric:mrr@5:minimum"].status == "PASS"
    assert gates["critical_no_regression"].model_dump() == dict(
        name="critical_no_regression", status="SKIP", actual=None, baseline=None,
        threshold=None, detail="no_critical_cases")


@pytest.mark.parametrize("field,value", [
    ("value", True), ("value", "0.5"), ("value", float("nan")),
    ("value", float("inf")), ("value", -0.1), ("value", 1.1),
    ("eligible_cases", True), ("eligible_cases", 1.0),
    ("eligible_cases", -1), ("eligible_cases", 3),
])
def test_invalid_current_metric_is_contract_error(field, value):
    policy, metrics, cases, _ = _run_inputs()
    setattr(metrics["citation_hit"], field, value)
    with pytest.raises(ContractError, match="citation_hit"):
        _gates(policy, metrics, cases)


@pytest.mark.parametrize("change", [
    "provenance", "k", "old_k", "missing_case", "extra_case", "passed", "metric",
    "denominator", "zero", "too_many", "bool_count", "nan", "bool_score",
])
def test_damaged_baseline_is_not_no_baseline(change):
    policy, metrics, cases, baseline = _run_inputs()
    if change == "provenance": baseline.provenance = "contract_smoke"
    elif change == "k": baseline.k = 10
    elif change == "old_k": del baseline.k
    elif change == "missing_case": del baseline.case_passed["a"]
    elif change == "extra_case": baseline.case_passed["extra"] = True
    elif change == "passed": baseline.case_passed["a"] = 1
    elif change == "metric":
        del baseline.metrics["citation_hit"]
        del baseline.eligible_cases["citation_hit"]
    elif change == "denominator": del baseline.eligible_cases["citation_hit"]
    elif change == "zero": baseline.eligible_cases["citation_hit"] = 0
    elif change == "too_many": baseline.eligible_cases["citation_hit"] = 3
    elif change == "bool_count": baseline.eligible_cases["citation_hit"] = True
    elif change == "nan": baseline.metrics["citation_hit"] = float("nan")
    else: baseline.metrics["citation_hit"] = True
    with pytest.raises(ContractError):
        _gates(policy, metrics, cases, baseline)


def test_duplicate_results_missing_critical_and_invalid_policy():
    policy, metrics, cases, _ = _run_inputs()
    with pytest.raises(ContractError, match="Duplicate"):
        _gates(policy, metrics, cases + [cases[0]])
    with pytest.raises(ContractError, match="Missing critical"):
        _gates(policy, metrics, cases, critical_ids={"missing"})
    policy.metric_gates = []
    with pytest.raises(ContractError, match="Missing required"):
        _gates(policy, metrics, cases)


def test_judge_and_case_metric_changes_do_not_affect_gates():
    policy, metrics, cases, baseline = _run_inputs()
    expected = _gates(policy, metrics, cases, baseline)
    for case in cases:
        case.metrics = {"correctness": 0}
        case.checks = {"diagnostic": "changed"}
        case.judge = object()  # A sentinel proves gate never reads Judge fields.
    assert _gates(policy, metrics, cases, baseline) == expected
