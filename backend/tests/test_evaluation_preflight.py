"""Shared runner validation rejects invalid inputs without DB/model dependencies."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag_podcast.evaluation import check_dataset, gate
from rag_podcast.evaluation.contracts import (
    Baseline, CaseResult, ContractError, Dataset, DatasetFingerprintMismatch, MetricValue,
)
from rag_podcast.evaluation.loaders import (
    EVALS_DIR, dataset_fingerprint, load_dataset, load_gate_policy, load_validated_dataset,
)
from rag_podcast.evaluation.preflight import validate_run_inputs


@pytest.fixture
def raw():
    return {
        "dataset_name": "qa-eval-v1", "dataset_version": "1", "schema_version": "dataset_v1",
        "cases": [{
            "id": "sample", "tags": ["single_fact"], "critical": True,
            "question": "Question", "expected_arm": "search", "scope": {},
            "gold_chunks": [1], "gold_segments": None,
            "reference_answer": "Answer", "required_claims": ["Fact"],
        }],
    }


@pytest.fixture
def schema():
    return json.loads(check_dataset.SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def inputs(raw):
    dataset = Dataset.model_validate(raw)
    policy = load_gate_policy(EVALS_DIR / "gates_v1.json")
    names = [rule.metric for rule in policy.metric_gates]
    fingerprint = dataset_fingerprint(raw)
    baseline = Baseline(
        dataset_fingerprint=fingerprint, k=policy.k, provenance="live",
        metrics={name: 0.5 for name in names}, eligible_cases={name: 1 for name in names},
        case_passed={"sample": True}, created_at="fixed",
    )
    return dataset, fingerprint, policy, baseline


def test_validated_load_reads_raw_once_and_keeps_raw_fingerprint(raw, tmp_path, monkeypatch):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    original = Path.read_text
    reads = []

    def read_once(self, *args, **kwargs):
        reads.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_once)
    dataset, fingerprint = load_validated_dataset(path)
    assert reads.count(path) == 1
    assert dataset.cases[0].conflicts == []
    assert fingerprint == dataset_fingerprint(raw)
    assert fingerprint != dataset_fingerprint(dataset.model_dump())
    assert (dataset, fingerprint) == load_dataset(path)


@pytest.mark.parametrize("kind,match", [
    ("empty", "SCHEMA FAIL"), ("shape", "SCHEMA FAIL"),
    ("unknown_tag", "SCHEMA FAIL"), ("scope", "SCHEMA FAIL"),
    ("extra", "SCHEMA FAIL"), ("duplicate", "duplicate id"),
    ("search_gold", "search needs"), ("context_gold", "full-context needs"),
    ("time", "start > end"), ("cross_podcast", "empty scope"),
    ("no_knowledge_gold", "both golds null"),
    ("no_knowledge_claims", "empty required_claims"),
])
def test_schema_and_consistency_errors_are_shared(raw, schema, tmp_path, kind, match):
    case = raw["cases"][0]
    if kind == "empty":
        raw["cases"] = []
    elif kind == "shape":
        raw["cases"] = "secret-payload"
    elif kind == "unknown_tag":
        case["tags"] = ["secret-payload"]
    elif kind == "scope":
        case["expected_arm"] = "full-context"
    elif kind == "extra":
        case["extra"] = "secret-payload"
    elif kind == "duplicate":
        raw["cases"].append(copy.deepcopy(case))
    elif kind == "search_gold":
        case["gold_chunks"] = []
    elif kind == "context_gold":
        case.update(expected_arm="full-context", scope={"episode_id": 1})
    elif kind == "time":
        case.update(expected_arm="full-context", scope={"episode_id": 1},
                    gold_chunks=None, gold_segments=[{"start": 2, "end": 1}])
    elif kind == "cross_podcast":
        case.update(tags=["cross_podcast"], scope={"podcast_id": 1})
    else:
        case["tags"] = ["no_knowledge"]
        if kind == "no_knowledge_claims":
            case["gold_chunks"] = None
    before = copy.deepcopy(raw)
    errors = check_dataset.validate_dataset(raw, schema)
    assert any(match in error for error in errors)
    assert "secret-payload" not in str(errors)
    assert raw == before
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractError, match=match):
        load_validated_dataset(path)


def test_schema_failure_never_enters_consistency(schema, monkeypatch):
    def forbidden(raw):
        pytest.fail("invalid shape must not enter consistency")
    monkeypatch.setattr(check_dataset, "consistency_errors", forbidden)
    assert check_dataset.validate_dataset(None, schema)[0].startswith("SCHEMA FAIL")


@pytest.mark.parametrize("contents", [b'{', b'\xff'])
def test_checker_bad_file_is_controlled_failure(tmp_path, monkeypatch, capsys, contents):
    path = tmp_path / "bad.json"
    path.write_bytes(contents)
    monkeypatch.setattr(sys, "argv", ["check_dataset", str(path)])
    assert check_dataset.main() == 1
    output = capsys.readouterr().out
    assert "VALIDATION FAIL" in output
    assert "Traceback" not in output


def test_checker_valid_and_empty_dataset(raw, tmp_path, monkeypatch, capsys):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_dataset", str(path)])
    assert check_dataset.main() == 0
    assert dataset_fingerprint(raw) in capsys.readouterr().out
    raw["cases"] = []
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert check_dataset.main() == 1
    assert "SCHEMA FAIL" in capsys.readouterr().out


def test_preflight_valid_or_missing_baseline_is_read_only(inputs):
    dataset, fingerprint, policy, baseline = inputs
    before = [item.model_dump() for item in (dataset, policy, baseline)]
    validate_run_inputs(dataset, fingerprint, policy, baseline)
    validate_run_inputs(dataset, fingerprint, policy)
    assert [item.model_dump() for item in (dataset, policy, baseline)] == before


@pytest.mark.parametrize("kind,match", [
    ("k", "k must match"), ("provenance", "provenance must match"),
    ("identity", "case IDs must match"), ("passed", "passed must be a bool"),
    ("keys", "identical keys"), ("missing_metric", "missing policy metrics"),
    ("nan", "finite number"), ("zero", "eligible_cases"),
    ("over", "eligible_cases"), ("bool", "eligible_cases"),
])
def test_baseline_checks_shared_by_gate_and_preflight(inputs, kind, match):
    dataset, fingerprint, policy, baseline = inputs
    metric = policy.metric_gates[0].metric
    if kind == "k":
        baseline.k = 10
    elif kind == "provenance":
        baseline.provenance = "contract_smoke"
    elif kind == "identity":
        baseline.case_passed = {"other": True}
    elif kind == "passed":
        baseline.case_passed["sample"] = 1
    elif kind == "keys":
        baseline.eligible_cases.pop(metric)
    elif kind == "missing_metric":
        baseline.metrics.pop(metric)
        baseline.eligible_cases.pop(metric)
    elif kind == "nan":
        baseline.metrics[metric] = float("nan")
    else:
        baseline.eligible_cases[metric] = {"zero": 0, "over": 2, "bool": True}[kind]
    metrics = {rule.metric: MetricValue(value=0.5, eligible_cases=1) for rule in policy.metric_gates}
    results = [CaseResult(case_id="sample", passed=True, metrics={}, checks={})]
    with pytest.raises(ContractError, match=match) as preflight_error:
        validate_run_inputs(dataset, fingerprint, policy, baseline)
    with pytest.raises(ContractError, match=match) as gate_error:
        gate.evaluate_gates(metrics, results, {"sample"}, policy, baseline, provenance="live")
    assert str(preflight_error.value) == str(gate_error.value)


def test_preflight_error_priority_and_empty_or_duplicate_dataset(inputs):
    dataset, fingerprint, policy, baseline = inputs
    baseline.dataset_fingerprint = "wrong"
    baseline.k = 10
    dataset.cases.append(dataset.cases[0].model_copy(deep=True))
    with pytest.raises(DatasetFingerprintMismatch):
        validate_run_inputs(dataset, fingerprint, policy, baseline)
    baseline.dataset_fingerprint = fingerprint
    with pytest.raises(ContractError, match="Duplicate dataset"):
        validate_run_inputs(dataset, fingerprint, policy, baseline)
    dataset.cases = []
    with pytest.raises(ContractError, match="at least one"):
        validate_run_inputs(dataset, fingerprint, policy)


def test_gate_preserves_current_metric_error_priority(inputs):
    _, _, policy, baseline = inputs
    baseline.k = 10
    bad = MetricValue(value=float("nan"), eligible_cases=1)
    with pytest.raises(ContractError, match="finite number"):
        gate.evaluate_gates(
            {"case_pass_rate": bad},
            [CaseResult(case_id="sample", passed=True, metrics={}, checks={})],
            {"sample"}, policy, baseline, provenance="live",
        )


def test_invalid_policy_precedes_baseline_compatibility(inputs):
    dataset, fingerprint, policy, baseline = inputs
    policy.critical_no_regression = False
    baseline.k = 10
    with pytest.raises(ContractError, match="critical_no_regression must be true"):
        validate_run_inputs(dataset, fingerprint, policy, baseline)


def test_baseline_error_order_and_replay_provenance(inputs):
    dataset, fingerprint, policy, baseline = inputs
    baseline.k = 10
    baseline.provenance = "contract_smoke"
    baseline.case_passed = {"unknown": True}
    with pytest.raises(ContractError, match="k must match"):
        validate_run_inputs(dataset, fingerprint, policy, baseline)
    baseline.k = policy.k
    with pytest.raises(ContractError, match="provenance must match"):
        validate_run_inputs(dataset, fingerprint, policy, baseline)
    with pytest.raises(ContractError, match="case IDs must match"):
        validate_run_inputs(dataset, fingerprint, policy, baseline, provenance="contract_smoke")
    baseline.case_passed = {"sample": True}
    validate_run_inputs(dataset, fingerprint, policy, baseline, provenance="contract_smoke")


def test_optional_baseline_metrics_can_keep_zero_eligible(inputs):
    dataset, fingerprint, policy, baseline = inputs
    baseline.metrics["mrr@5"] = 0.0
    baseline.eligible_cases["mrr@5"] = 0
    validate_run_inputs(dataset, fingerprint, policy, baseline)


def test_validation_imports_need_no_jsonschema_or_production_dependencies():
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"jsonschema", "openai", "sqlalchemy"} or fullname in {
            "rag_podcast.config", "rag_podcast.db", "rag_podcast.qa", "rag_podcast.retrieval",
        }:
            raise AssertionError("unexpected import: " + fullname)
sys.meta_path.insert(0, Block())
from rag_podcast.evaluation.check_dataset import validate_dataset
from rag_podcast.evaluation.loaders import EVALS_DIR, load_dataset, load_gate_policy
from rag_podcast.evaluation.preflight import validate_run_inputs
dataset, fingerprint = load_dataset(EVALS_DIR / "dataset_v1.json")
validate_run_inputs(dataset, fingerprint, load_gate_policy(EVALS_DIR / "gates_v1.json"))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
