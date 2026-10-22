"""File loading and baseline persistence for the evaluation library.

Scoring works on in-memory contract objects. File / JSON / validation failures
surface here as a ``ContractError`` naming the file; the dataset checker uses
the stricter authoring loader without changing the scoring loader's behavior.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from rag_podcast.evaluation.contracts import (
    Baseline,
    ContractError,
    Dataset,
    GatePolicy,
    ObservationBundle,
    Report,
)
from rag_podcast.evaluation.gate import is_deterministic_metric, validate_gate_policy

# evaluation → rag_podcast → src → backend → <repo>；依赖 uv 的 editable 安装
# 仅供 CLI / check_dataset 做默认参数，下列函数一律收显式 path
EVALS_DIR = Path(__file__).resolve().parents[4] / "evals" / "qa"

M = TypeVar("M", bound=BaseModel)


def dataset_fingerprint(dataset: dict) -> str:
    # 唯一真源；必须对原始 dict 算（模型会补默认值，哈希会变）
    canonical = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"{path}: cannot read file ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: invalid JSON ({exc})") from exc
    except UnicodeError as exc:
        raise ContractError(f"{path}: invalid UTF-8") from exc


def _parse(model: type[M], data: dict, path: Path) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ContractError(f"{path}: does not match {model.__name__}\n{exc}") from exc


def load_dataset(path: Path) -> tuple[Dataset, str]:
    raw = _read_json(path)
    return _parse(Dataset, raw, path), dataset_fingerprint(raw)


def load_validated_dataset(
    path: Path, *, schema_path: Path | None = None,
) -> tuple[Dataset, str]:
    """Load once with schema/consistency checks for checker and future CLI.

    Unlike load_dataset, this authoring/runner boundary requires jsonschema.
    Keep the regular scoring loader independent of that dev dependency.
    """
    from .check_dataset import SCHEMA_PATH, validate_dataset

    raw = _read_json(path)
    schema = _read_json(schema_path if schema_path is not None else SCHEMA_PATH)
    errors = validate_dataset(raw, schema)
    if errors:
        raise ContractError(f"{path}: " + "; ".join(errors))
    try:
        dataset = Dataset.model_validate(raw)
    except ValidationError as exc:
        # Do not include Pydantic's raw input values in CLI diagnostics.
        raise ContractError(f"{path}: does not match Dataset") from exc
    return dataset, dataset_fingerprint(raw)


def load_observations(path: Path) -> ObservationBundle:
    return _parse(ObservationBundle, _read_json(path), path)


def load_gate_policy(path: Path) -> GatePolicy:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ContractError(f"{path}: gate policy must be an object")
    required = {"metric_gates", "critical_no_regression", "required_provenance", "k"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ContractError(f"{path}: missing required gate policy fields: {missing}")
    policy = _parse(GatePolicy, raw, path)
    errors = validate_gate_policy(policy)
    if errors:
        raise ContractError(f"{path}: invalid gate policy: {'; '.join(errors)}")
    return policy


def load_baseline(path: Path) -> Baseline | None:
    # 缺文件 → None（回归 gate 记 NO_BASELINE）；fingerprint 比对在 evaluate()
    if not path.exists():
        return None
    baseline = _parse(Baseline, _read_json(path), path)
    if baseline.metrics.keys() != baseline.eligible_cases.keys():
        raise ContractError(f"{path}: baseline metrics and eligible_cases must have identical keys")
    return baseline


def save_baseline(report: Report, path: Path) -> None:
    if "provenance" not in report.metadata:
        raise ContractError(f"{path}: report.metadata has no 'provenance'; cannot build baseline")
    if "k" not in report.metadata:
        raise ContractError(f"{path}: report.metadata has no 'k'; cannot build baseline")
    case_passed = {c.case_id: c.passed for c in report.cases}
    if len(case_passed) != len(report.cases):
        raise ContractError(f"{path}: duplicate case_id in report")
    if type(report.case_count) is not int or report.case_count != len(report.cases):
        raise ContractError(f"{path}: report.case_count must equal the number of cases")
    metrics = {name: mv for name, mv in report.metrics.items() if is_deterministic_metric(name)}
    baseline = _parse(Baseline, {
        "dataset_fingerprint": report.dataset_fingerprint,
        "k": report.metadata["k"],
        "provenance": report.metadata["provenance"],
        "metrics": {name: mv.value for name, mv in metrics.items()},
        "eligible_cases": {name: mv.eligible_cases for name, mv in metrics.items()},
        "case_passed": case_passed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, path)
    if any(count > report.case_count for count in baseline.eligible_cases.values()):
        raise ContractError(f"{path}: eligible_cases must not exceed report.case_count")
    text = json.dumps(baseline.model_dump(), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
