"""Validate live inputs before opening a DB session or invoking providers."""

from collections import Counter
from typing import Literal

from .contracts import (
    Baseline, ContractError, Dataset, DatasetFingerprintMismatch, GatePolicy,
)
from .gate import validate_baseline_compatibility, validate_gate_policy


def validate_run_inputs(
    dataset: Dataset,
    dataset_fingerprint: str,
    policy: GatePolicy,
    baseline: Baseline | None = None,
    *,
    provenance: Literal["contract_smoke", "live"] = "live",
) -> None:
    """Validate fingerprint, identities, policy and baseline, in that order.

    File callers first use load_validated_dataset for schema/consistency. This
    memory-only guard also protects callers constructing a Dataset directly.
    Provenance identifies observation origin, including during later replay.
    """
    if baseline is not None and baseline.dataset_fingerprint != dataset_fingerprint:
        raise DatasetFingerprintMismatch("baseline and dataset fingerprints differ")
    counts = Counter(case.id for case in dataset.cases)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise ContractError(f"Duplicate dataset case_ids: {duplicates}")
    if not counts:
        raise ContractError("dataset must contain at least one case")
    errors = validate_gate_policy(policy)
    if errors:
        raise ContractError("; ".join(errors))
    if baseline is not None:
        validate_baseline_compatibility(baseline, policy, set(counts), provenance=provenance)
