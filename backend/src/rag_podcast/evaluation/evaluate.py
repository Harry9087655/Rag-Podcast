"""Evaluate in-memory observations before optional Judge enrichment."""

from collections import Counter
from collections.abc import Awaitable, Callable

from .contracts import (
    Baseline, Case, CasePassPolicy, CaseResult, ContractError, Dataset,
    DatasetFingerprintMismatch, GatePolicy, JudgeScores, Observation,
    ObservationBundle, Report,
)
from .gate import evaluate_gates, validate_gate_policy
from .judge import judge_report
from .metrics import aggregate, case_passed, score_case


async def evaluate(
    dataset: Dataset,
    dataset_fingerprint: str,
    bundle: ObservationBundle,
    gate_policy: GatePolicy,
    baseline: Baseline | None = None,
    run_judge: bool = False,
    policy: CasePassPolicy | None = None,
    *,
    judge_fn: Callable[[Case, Observation], Awaitable[JudgeScores | None]] | None = None,
) -> Report:
    """Score, aggregate, apply gates, then enrich a completed report.

    The caller supplies the fingerprint returned with the loaded dataset.
    Missing observations become failed case results; invalid identities,
    policies and baselines raise before any Judge call. Inputs are read-only.

    A supplied per-case policy is used as-is. The default preserves the
    configured precision minimums while following the gate policy's rank k.
    """
    if bundle.dataset_fingerprint != dataset_fingerprint:
        raise DatasetFingerprintMismatch("observation and dataset fingerprints differ")
    if baseline is not None and baseline.dataset_fingerprint != dataset_fingerprint:
        raise DatasetFingerprintMismatch("baseline and dataset fingerprints differ")

    case_counts = Counter(case.id for case in dataset.cases)
    duplicate_ids = sorted(case_id for case_id, count in case_counts.items() if count > 1)
    if duplicate_ids:
        raise ContractError(f"Duplicate dataset case_ids: {duplicate_ids}")

    observation_counts = Counter(obs.case_id for obs in bundle.observations)
    duplicate_ids = sorted(
        case_id for case_id, count in observation_counts.items() if count > 1
    )
    if duplicate_ids:
        raise ContractError(f"Duplicate observation case_ids: {duplicate_ids}")

    unknown_ids = sorted(observation_counts.keys() - case_counts.keys())
    if unknown_ids:
        raise ContractError(f"Unknown observation case_ids: {unknown_ids}")

    errors = validate_gate_policy(gate_policy)
    if errors:
        raise ContractError("; ".join(errors))

    observation_case_id = {obs.case_id: obs for obs in bundle.observations}
    case_policy = policy if policy is not None else CasePassPolicy(
        search={"gold_chunk_coverage": 1.0, f"precision@{gate_policy.k}": 0.6},
        full_context={"citation_hit": 1.0, "citation_precision": 0.8},
    )

    case_results: list[CaseResult] = []
    for c in dataset.cases:
        obs = observation_case_id.get(c.id, None)
        metrics = (
            score_case(c, obs, k=gate_policy.k)
            if obs is not None
            else {}
        )
        passed, checks = case_passed(c, obs, metrics, policy=case_policy)
        case_results.append(CaseResult(
            case_id=c.id,
            passed=passed,
            metrics=dict(sorted(metrics.items())),
            checks=dict(sorted(checks.items())),
            judge=None,
        ))
    metric_values, metric_values_by_tags = aggregate(dataset.cases, case_results)

    critical_ids = {c.id for c in dataset.cases if c.critical}
    gates = evaluate_gates(
        metrics=metric_values,
        case_results=case_results,
        critical_ids=critical_ids,
        policy=gate_policy,
        baseline=baseline,
        provenance=bundle.provenance,
    )
    report_passed = not any(gate.status == "FAIL" for gate in gates)

    report = Report(
        dataset_name=dataset.dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        case_count=len(dataset.cases),
        metrics=dict(sorted(metric_values.items())),
        slices={
            tag: dict(sorted(values.items()))
            for tag, values in sorted(metric_values_by_tags.items())
        },
        cases=case_results,
        gates=gates,
        passed=report_passed,
        metadata={
            "provenance": bundle.provenance,
            "k": gate_policy.k,
            "observed_cases": len(observation_case_id),
            "missing_observation_cases": len(dataset.cases) - len(observation_case_id),
            "provider_failed_cases": sum(
                obs.error is not None for obs in observation_case_id.values()
            ),
            "arm_search_cases": sum(
                obs.arm == "search" for obs in observation_case_id.values()
            ),
            "arm_full_context_cases": sum(
                obs.arm == "full-context" for obs in observation_case_id.values()
            ),
        },
    )

    return await judge_report(
        dataset=dataset,
        bundle=bundle,
        report=report,
        run_judge=run_judge,
        judge_fn=judge_fn,
    )
