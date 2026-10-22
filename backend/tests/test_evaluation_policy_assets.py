"""Gate asset contracts and loader/semantic validation boundaries."""

import copy
import json
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from rag_podcast.evaluation.contracts import ContractError, GatePolicy
from rag_podcast.evaluation.gate import validate_gate_policy
from rag_podcast.evaluation.loaders import EVALS_DIR, load_gate_policy


@pytest.fixture
def policy_data():
    return json.loads((EVALS_DIR / "gates_v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def schema_validator():
    schema = json.loads((EVALS_DIR / "schema/gates_v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def write_policy(tmp_path, data):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_initial_policy_matches_design(policy_data, schema_validator):
    floors = {
        "case_pass_rate": 0.5, "citation_hit": 0.5, "citation_precision": 0.5,
        "recall@5": 0.4, "precision@5": 0.4, "gold_chunk_coverage": 0.6,
    }
    assert policy_data == {
        "critical_no_regression": True, "required_provenance": ["live"], "k": 5,
        "metric_gates": [
            {"metric": name, "minimum": floor, "max_regression": 0.15}
            for name, floor in floors.items()
        ] + [{"metric": "provider_failure_rate", "maximum": 0.3}],
    }
    schema_validator.validate(policy_data)
    assert validate_gate_policy(load_gate_policy(EVALS_DIR / "gates_v1.json")) == []


@pytest.mark.parametrize("k", [1, 5, 10])
def test_dynamic_k_and_optional_metrics(tmp_path, policy_data, schema_validator, k):
    policy_data["k"] = k
    for gate in policy_data["metric_gates"]:
        gate["metric"] = gate["metric"].replace("@5", f"@{k}")
    policy_data["metric_gates"] += [
        {"metric": f"mrr@{k}", "minimum": 0, "max_regression": None},
        {"metric": f"ndcg@{k}", "minimum": 1, "max_regression": 0},
    ]
    schema_validator.validate(policy_data)
    assert load_gate_policy(write_policy(tmp_path, policy_data)).k == k


@pytest.mark.parametrize("field", ["k", "critical_no_regression", "required_provenance", "metric_gates"])
def test_required_root_fields(tmp_path, policy_data, schema_validator, field):
    del policy_data[field]
    assert not schema_validator.is_valid(policy_data)
    path = write_policy(tmp_path, policy_data)
    with pytest.raises(ContractError) as exc:
        load_gate_policy(path)
    assert str(path) in str(exc.value)
    assert field in str(exc.value)


@pytest.mark.parametrize("change", [
    "root_extra", "nested_extra", "unknown", "judge", "missing_minimum", "missing_maximum",
    "missing_budget", "null_budget", "reverse", "null_bound", "string", "bool",
    "negative", "above_one", "bad_k", "critical", "empty_provenance", "duplicate_provenance",
    "smoke_provenance", "rank_zero", "rank_leading_zero", "rank_newline",
])
def test_schema_and_loader_reject_structural_errors(tmp_path, policy_data, schema_validator, change):
    first = policy_data["metric_gates"][0]
    provider = policy_data["metric_gates"][-1]
    if change == "root_extra":
        policy_data["extra"] = 1
    elif change == "nested_extra":
        first["extra"] = 1
    elif change in {"unknown", "judge", "rank_zero", "rank_leading_zero", "rank_newline"}:
        first["metric"] = {"unknown": "typo", "judge": "correctness",
                           "rank_zero": "recall@0", "rank_leading_zero": "recall@01",
                           "rank_newline": "recall@5\n"}[change]
    elif change == "missing_minimum":
        del first["minimum"]
    elif change == "missing_maximum":
        del provider["maximum"]
    elif change == "missing_budget":
        del first["max_regression"]
    elif change == "null_budget":
        first["max_regression"] = None
    elif change == "reverse":
        first["maximum"] = 0.5
    elif change in {"null_bound", "string", "bool", "negative", "above_one"}:
        first["minimum"] = {"null_bound": None, "string": "0.5", "bool": True,
                            "negative": -0.1, "above_one": 1.1}[change]
    elif change == "bad_k":
        policy_data["k"] = True
    elif change == "critical":
        policy_data["critical_no_regression"] = False
    else:
        policy_data["required_provenance"] = {
            "empty_provenance": [], "duplicate_provenance": ["live", "live"],
            "smoke_provenance": ["live", "contract_smoke"],
        }[change]
    assert not schema_validator.is_valid(policy_data)
    path = write_policy(tmp_path, policy_data)
    with pytest.raises(ContractError) as exc:
        load_gate_policy(path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("index", range(7))
def test_removing_each_required_metric_is_semantic_error(tmp_path, policy_data, schema_validator, index):
    removed = policy_data["metric_gates"].pop(index)["metric"]
    schema_validator.validate(policy_data)
    with pytest.raises(ContractError, match=f"Missing required metric: {removed}"):
        load_gate_policy(write_policy(tmp_path, policy_data))


@pytest.mark.parametrize("index", range(7))
def test_each_absolute_guardrail_cannot_be_relaxed(tmp_path, policy_data, schema_validator, index):
    rule = policy_data["metric_gates"][index]
    if "minimum" in rule:
        rule["minimum"] -= 0.01
    else:
        rule["maximum"] += 0.01
    schema_validator.validate(policy_data)
    with pytest.raises(ContractError, match=rule["metric"]):
        load_gate_policy(write_policy(tmp_path, policy_data))


@pytest.mark.parametrize("index", range(7))
def test_each_regression_guardrail_cannot_be_relaxed(tmp_path, policy_data, schema_validator, index):
    policy_data["metric_gates"][index]["max_regression"] = 0.16
    schema_validator.validate(policy_data)
    with pytest.raises(ContractError, match="max_regression must not exceed 0.15"):
        load_gate_policy(write_policy(tmp_path, policy_data))


@pytest.mark.parametrize("change", ["duplicate", "wrong_k"])
def test_schema_defers_cross_entry_rules_to_loader(tmp_path, policy_data, schema_validator, change):
    if change == "duplicate":
        policy_data["metric_gates"].append(copy.deepcopy(policy_data["metric_gates"][0]))
    else:
        policy_data["k"] = 10
    schema_validator.validate(policy_data)
    with pytest.raises(ContractError):
        load_gate_policy(write_policy(tmp_path, policy_data))


def test_all_semantic_errors_keep_stable_order_and_path(tmp_path, policy_data):
    policy_data["critical_no_regression"] = False
    removed = policy_data["metric_gates"].pop(1)["metric"]
    policy_data["metric_gates"][0]["minimum"] = 0.49
    policy_data["metric_gates"].append(copy.deepcopy(policy_data["metric_gates"][-1]))
    errors = validate_gate_policy(GatePolicy.model_validate(policy_data))
    assert errors == [
        "critical_no_regression must be true", f"Missing required metric: {removed}",
        "case_pass_rate: minimum must be at least 0.5", "Duplicate metric: provider_failure_rate",
    ]
    path = write_policy(tmp_path, policy_data)
    with pytest.raises(ContractError) as exc:
        load_gate_policy(path)
    assert str(exc.value) == f"{path}: invalid gate policy: {'; '.join(errors)}"


@pytest.mark.parametrize("contents", ["{broken", "[]", "null", '"policy"'])
def test_bad_json_and_nonobjects_have_path_errors(tmp_path, contents):
    path = tmp_path / "policy.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ContractError) as exc:
        load_gate_policy(path)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonstandard_nonfinite_json_numbers_are_rejected(tmp_path, policy_data, value):
    policy_data["metric_gates"][0]["minimum"] = value
    with pytest.raises(ContractError):
        load_gate_policy(write_policy(tmp_path, policy_data))


def test_json_schema_integer_semantics_do_not_weaken_strict_loader(tmp_path, policy_data, schema_validator):
    # JSON Schema integers include 5.0; the Python contract deliberately requires int.
    policy_data["k"] = 5.0
    schema_validator.validate(policy_data)
    with pytest.raises(ContractError):
        load_gate_policy(write_policy(tmp_path, policy_data))


def test_runtime_policy_loading_does_not_import_jsonschema():
    script = '''
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "jsonschema" or name.startswith("jsonschema."):
        raise AssertionError("jsonschema must remain a dev-only dependency")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from rag_podcast.evaluation.loaders import EVALS_DIR, load_gate_policy
assert load_gate_policy(EVALS_DIR / "gates_v1.json").k == 5
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
