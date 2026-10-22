"""Fingerprint validation precedes identity checks and all scoring."""

import importlib

import pytest

from rag_podcast.evaluation.contracts import (
    Baseline, Dataset, DatasetFingerprintMismatch, Observation, ObservationBundle,
)
from rag_podcast.evaluation.evaluate import evaluate
from rag_podcast.evaluation.loaders import EVALS_DIR, load_gate_policy


@pytest.mark.parametrize("bundle_bad,baseline_bad", [(True, False), (False, True), (True, True)])
async def test_fingerprint_priority_before_unknown_ids_and_scoring(
    monkeypatch, bundle_bad, baseline_bad,
):
    dataset = Dataset(dataset_name="empty", dataset_version="1", schema_version="1", cases=[])
    bundle = ObservationBundle(
        dataset_fingerprint="wrong" if bundle_bad else "original", provenance="live",
        observations=[Observation(case_id="unknown", arm="search", answer_text="", sources=[])],
    )
    baseline = Baseline(
        dataset_fingerprint="wrong" if baseline_bad else "original", k=5,
        provenance="live", metrics={}, eligible_cases={}, case_passed={}, created_at="fixed",
    )
    module = importlib.import_module("rag_podcast.evaluation.evaluate")

    def forbidden(*args, **kwargs):
        pytest.fail("must not score or invoke Judge")

    monkeypatch.setattr(module, "score_case", forbidden)
    monkeypatch.setattr(module, "judge_report", forbidden)
    with pytest.raises(DatasetFingerprintMismatch, match="observation" if bundle_bad else "baseline"):
        await evaluate(dataset, "original", bundle, load_gate_policy(EVALS_DIR / "gates_v1.json"),
                       baseline, run_judge=True)
