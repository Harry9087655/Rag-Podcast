"""Per-case pass decisions, separate from scoring and aggregate gates."""

import math

from .contracts import Case, CasePassPolicy, ContractError, Observation


def evaluate_case_policy(
    case: Case,
    obs: Observation | None,
    metrics: dict[str, float],
    policy: CasePassPolicy,
) -> tuple[bool, dict[str, float | bool | str]]:
    """Apply minimums inclusively; missing required metrics fail closed.

    Ranked metric names must match score_case's k (e.g. precision@10).
    Missing observations and provider errors always fail, regardless of policy.
    """
    if obs is None:
        return False, {"missing_observation": True}
    if obs.error is not None:
        return False, {"provider_failed": True}

    if case.gold_chunks is not None:
        minimums = policy.search
    elif case.gold_segments is not None:
        minimums = policy.full_context
    else:
        abstained = len(obs.sources) == 0
        return abstained or not policy.require_abstention, {"abstained": abstained}

    passed = True
    checks: dict[str, float | bool | str] = {}
    for metric, minimum in minimums.items():
        if metric not in metrics:
            checks[f"missing_metric:{metric}"] = True
            passed = False
            continue
        value = metrics[metric]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ContractError(f"{metric}: value must be a finite number in [0, 1]")
        checks[metric] = value
        if value < minimum:
            passed = False
    return passed, checks
