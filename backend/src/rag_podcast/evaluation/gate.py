"""Validate deterministic evaluation gate policies without I/O."""

import math
import re
from typing import Literal

from .contracts import (
    Baseline,
    CaseResult,
    ContractError,
    GatePolicy,
    GateResult,
    MetricValue,
)

EPS = 1e-12


def is_deterministic_metric(metric: str) -> bool:
    """Recognize registered metric families, including positive dynamic ranks."""
    return metric in {
        "case_pass_rate", "citation_hit", "citation_precision",
        "gold_chunk_coverage", "provider_failure_rate",
    } or re.fullmatch(r"(?:recall|precision|mrr|ndcg)@[1-9][0-9]*", metric) is not None


def validate_gate_policy(gate_policy: GatePolicy) -> list[str]:
    """Return all policy violations, or an empty list for a valid policy.

    This checks rules, not evaluation scores. Callers decide whether to
    raise ContractError. Ranked metric names follow policy.k.
    """
    errors: list[str] = []

    # Check global rules before checking individual metric gates.
    if type(gate_policy.k) is not int or gate_policy.k <= 0:
        errors.append("k must be a positive integer")
    if gate_policy.critical_no_regression is not True:
        errors.append("critical_no_regression must be true")
    if gate_policy.required_provenance != ["live"]:
        errors.append("required_provenance must be exactly ['live']")

    minimum_floors = {
        "case_pass_rate": 0.5,
        "citation_hit": 0.5,
        "citation_precision": 0.5,
        f"recall@{gate_policy.k}": 0.4,
        f"precision@{gate_policy.k}": 0.4,
        "gold_chunk_coverage": 0.6,
    }
    required_metrics = set(minimum_floors) | {"provider_failure_rate"}
    optional_metrics = {f"mrr@{gate_policy.k}", f"ndcg@{gate_policy.k}"}
    allowed_metrics = required_metrics | optional_metrics

    configured_metrics = {gate.metric for gate in gate_policy.metric_gates}
    for metric in sorted(required_metrics - configured_metrics):
        errors.append(f"Missing required metric: {metric}")

    seen: set[str] = set()
    for gate in gate_policy.metric_gates:
        metric = gate.metric
        if metric in seen:
            errors.append(f"Duplicate metric: {metric}")
        seen.add(metric)

        if not is_deterministic_metric(metric) or metric not in allowed_metrics:
            errors.append(f"Unknown or non-deterministic metric: {metric}")
            continue

        # Validate numbers before comparing bounds. None is checked below.
        invalid_fields: set[str] = set()
        for field in ("minimum", "maximum", "max_regression"):
            value = getattr(gate, field)
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                errors.append(f"{metric}: {field} must be a finite number in [0, 1]")
                invalid_fields.add(field)

        if metric == "provider_failure_rate":
            if gate.minimum is not None:
                errors.append(f"{metric}: minimum is not allowed for a lower-is-better metric")
            if gate.maximum is None:
                errors.append(f"{metric}: maximum is required")
            elif "maximum" not in invalid_fields and gate.maximum > 0.3:
                errors.append(f"{metric}: maximum must not exceed 0.3")
        else:
            if gate.maximum is not None:
                errors.append(f"{metric}: maximum is not allowed for a higher-is-better metric")
            if gate.minimum is None:
                errors.append(f"{metric}: minimum is required")
            elif "minimum" not in invalid_fields and metric in minimum_floors:
                floor = minimum_floors[metric]
                if gate.minimum < floor:
                    errors.append(f"{metric}: minimum must be at least {floor}")

        if gate.max_regression is None:
            if metric in minimum_floors:
                errors.append(f"{metric}: max_regression is required")
        elif "max_regression" not in invalid_fields and gate.max_regression > 0.15:
            errors.append(f"{metric}: max_regression must not exceed 0.15")

    return errors


def validate_baseline_compatibility(
    baseline: Baseline,
    policy: GatePolicy,
    case_ids: set[str],
    *,
    provenance: Literal["contract_smoke", "live"],
) -> None:
    """Check a supplied baseline without current scores or external work.

    The caller validates policy and fingerprint first. Gate and runner share
    these checks; missing baseline handling remains with the caller.
    """
    case_count = len(case_ids)
    if type(getattr(baseline, "k", None)) is not int or baseline.k != policy.k:
        raise ContractError("baseline k must match policy.k")
    if baseline.provenance != provenance:
        raise ContractError("baseline provenance must match current provenance")

    baseline_ids = set(baseline.case_passed)
    current_ids = case_ids
    if baseline_ids != current_ids:
        missing_ids = sorted(current_ids - baseline_ids)
        extra_ids = sorted(baseline_ids - current_ids)
        raise ContractError(
            f"baseline case IDs must match current results: "
            f"missing={missing_ids}, extra={extra_ids}"
        )

    for case_id in sorted(baseline.case_passed):
        if type(baseline.case_passed[case_id]) is not bool:
            raise ContractError(f"baseline {case_id}: case_passed must be a bool")

    if baseline.metrics.keys() != baseline.eligible_cases.keys():
        raise ContractError("baseline metrics and eligible_cases must have identical keys")

    policy_metrics = {gate.metric for gate in policy.metric_gates}
    missing_metrics = sorted(policy_metrics - baseline.metrics.keys())
    if missing_metrics:
        raise ContractError(f"baseline missing policy metrics: {missing_metrics}")

    for metric_name in sorted(baseline.metrics):
        value = baseline.metrics[metric_name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ContractError(
                f"baseline {metric_name}: value must be a finite number in [0, 1]"
            )

        eligible_cases = baseline.eligible_cases[metric_name]
        # 政策使用的指标必须有有效分母；其他指标可以保留零分母。
        minimum_count = 1 if metric_name in policy_metrics else 0
        if (
            type(eligible_cases) is not int
            or not minimum_count <= eligible_cases <= case_count
        ):
            raise ContractError(
                f"baseline {metric_name}: eligible_cases must be an integer "
                f"in [{minimum_count}, {case_count}]"
            )


def evaluate_gates(
    metrics: dict[str, MetricValue],
    case_results: list[CaseResult],
    critical_ids: set[str],
    policy: GatePolicy,
    baseline: Baseline | None = None,
    *,
    provenance: Literal["contract_smoke", "live"],
) -> list[GateResult]:
    """Validate comparison inputs, then return every gate in stable order."""
    # 1. 先校验 policy 本身是否合法
    errors = validate_gate_policy(policy)
    if errors:
        raise ContractError("; ".join(errors))

    # 2. 校验 case_results 里 case_id 不重复以及critical_id缺失

    results_by_id: dict[str, CaseResult] = {}
    for result in case_results:
        if result.case_id in results_by_id:
            raise ContractError(f"Duplicate case_id: {result.case_id}")
        if type(result.passed) is not bool:
            raise ContractError(f"{result.case_id}: passed must be a bool")
        results_by_id[result.case_id] = result

    for critical_id in sorted(critical_ids):
        if critical_id not in results_by_id:
            raise ContractError(f"Missing critical case_id in case_results: {critical_id}")

    # 3. 校验当前指标输入；零分母和缺指标留到后续 coverage gate 判 FAIL。
    case_count = len(results_by_id)
    for metric_name in sorted(metrics):
        metric_value = metrics[metric_name]
        value = metric_value.value
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ContractError(f"{metric_name}: value must be a finite number in [0, 1]")

        eligible_cases = metric_value.eligible_cases
        if type(eligible_cases) is not int or not 0 <= eligible_cases <= case_count:
            raise ContractError(
                f"{metric_name}: eligible_cases must be an integer in [0, {case_count}]"
            )

    # 4. Shared with live preflight; preserve validation order before gates.
    if baseline is not None:
        validate_baseline_compatibility(
            baseline, policy, set(results_by_id), provenance=provenance,
        )

    gates = [GateResult(
        name="required_provenance",
        status="PASS" if provenance in policy.required_provenance else "FAIL",
        actual=None, baseline=None, threshold=None,
        detail=f"actual={provenance}; allowed={sorted(policy.required_provenance)}",
    )]

    for rule in sorted(policy.metric_gates, key=lambda gate: gate.metric):
        name = rule.metric
        current = metrics.get(name)
        count = current.eligible_cases if current is not None else None
        covered = count is not None and count > 0
        actual = current.value if current is not None and covered else None
        reason = "missing_metric" if current is None else "no_eligible_cases"
        gates.append(GateResult(
            name=f"metric:{name}:coverage", status="PASS" if covered else "FAIL",
            actual=count, baseline=None, threshold=1,
            detail="eligible_cases > 0" if covered else reason,
        ))

        higher_is_better = name != "provider_failure_rate"
        bound_name = "minimum" if higher_is_better else "maximum"
        limit = rule.minimum if higher_is_better else rule.maximum
        assert limit is not None  # Guaranteed by policy validation.
        absolute_passed = actual is not None and (
            actual + EPS >= limit if higher_is_better else actual <= limit + EPS
        )
        gates.append(GateResult(
            name=f"metric:{name}:{bound_name}", status="PASS" if absolute_passed else "FAIL",
            actual=actual, baseline=None, threshold=limit,
            detail=f"{bound_name}={limit}" if covered else reason,
        ))

        if rule.max_regression is not None:
            previous = baseline.metrics[name] if baseline is not None else None
            if baseline is None:
                status = "NO_BASELINE"
                detail = "no_baseline"
            elif actual is None:
                status = "FAIL"
                detail = reason
            else:
                assert previous is not None
                delta = previous - actual if higher_is_better else actual - previous
                status = "PASS" if delta <= rule.max_regression + EPS else "FAIL"
                direction = "higher_is_better" if higher_is_better else "lower_is_better"
                detail = f"direction={direction}; delta={delta:.12g}"
            gates.append(GateResult(
                name=f"metric:{name}:regression", status=status,
                actual=actual, baseline=previous, threshold=rule.max_regression, detail=detail,
            ))

        previous_count = baseline.eligible_cases[name] if baseline is not None else None
        if baseline is None:
            status = "NO_BASELINE"
            detail = "no_baseline"
        else:
            assert previous_count is not None
            status = "PASS" if count is not None and count >= previous_count else "FAIL"
            detail = "missing_metric" if count is None else "current_eligible >= baseline_eligible"
        gates.append(GateResult(
            name=f"metric:{name}:eligible", status=status,
            actual=count, baseline=previous_count, threshold=previous_count, detail=detail,
        ))

    if not critical_ids:
        gates.append(GateResult(
            name="critical_no_regression", status="SKIP",
            actual=None, baseline=None, threshold=None, detail="no_critical_cases",
        ))
    for case_id in sorted(critical_ids):
        passed = results_by_id[case_id].passed
        gates.append(GateResult(
            name=f"critical:{case_id}:current", status="PASS" if passed else "FAIL",
            actual=int(passed), baseline=None, threshold=1, detail="current_passed_required",
        ))
        previous_passed = baseline.case_passed[case_id] if baseline is not None else None
        if baseline is None:
            status = "NO_BASELINE"
        else:
            status = "FAIL" if previous_passed and not passed else "PASS"
        gates.append(GateResult(
            name=f"critical:{case_id}:regression", status=status, actual=int(passed),
            baseline=int(previous_passed) if previous_passed is not None else None,
            threshold=None,
            detail="no_baseline" if baseline is None else "no_true_to_false_regression",
        ))

    return gates

