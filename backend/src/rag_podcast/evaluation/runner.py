"""Orchestrate one evaluation run end to end, owning no single step.

The runner sequences input validation, the chosen adapter, scoring, rendering
and artifact persistence. It imports no production code; the CLI injects the
live answer function and session factory for ``run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import artifacts
from .adapters import AnswerFn, LiveAdapter, PredictionFileAdapter, SessionFactory
from .contracts import ContractError, Report
from .evaluate import evaluate
from .loaders import EVALS_DIR, load_baseline, load_gate_policy, load_validated_dataset
from .preflight import validate_run_inputs


@dataclass
class RunOutcome:
    exit_code: int
    run_dir: Path
    dataset_fingerprint: str
    report: Report
    baseline_saved: bool = False


async def run(
    dataset_path: Path,
    gates_path: Path,
    *,
    observations_path: Path | None = None,
    baseline_path: Path | None = None,
    output_dir: Path | None = None,
    run_judge: bool = False,
    fail_on_regression: bool = False,
    case_timeout: float | None = None,
    save_baseline: bool = False,
    answer_fn: AnswerFn | None = None,
    session_factory: SessionFactory | None = None,
) -> RunOutcome:
    """Run a live evaluation or replay saved observations into a report.

    ``observations_path=None`` selects a live run (requires ``answer_fn`` and
    ``session_factory``); otherwise the bundle is replayed and its provenance
    preserved. Returns the CLI exit code plus the artifacts for a summary.
    """
    if save_baseline and observations_path is not None:
        raise ContractError("--save-baseline is only supported for a live run")
    if observations_path is None and (answer_fn is None or session_factory is None):
        raise ContractError("live run requires answer_fn and session_factory")

    dataset, fingerprint = load_validated_dataset(dataset_path)
    policy = load_gate_policy(gates_path)
    baseline = load_baseline(baseline_path) if baseline_path is not None else None

    if output_dir is not None:
        _ensure_inputs_outside(
            output_dir, dataset_path, gates_path, baseline_path, observations_path
        )
    run_dir = artifacts.create_run_dir(EVALS_DIR / "runs", explicit=output_dir)

    if observations_path is None:
        validate_run_inputs(dataset, fingerprint, policy, baseline, provenance="live")
        bundle = await LiveAdapter(answer_fn, session_factory).execute(
            dataset, fingerprint, k=policy.k, case_timeout=case_timeout,
        )
    else:
        bundle = PredictionFileAdapter(observations_path).execute(dataset, fingerprint)
        validate_run_inputs(
            dataset, fingerprint, policy, baseline, provenance=bundle.provenance
        )

    artifacts.write_bundle(run_dir, bundle)
    report = await evaluate(
        dataset, fingerprint, bundle, policy, baseline, run_judge=run_judge,
    )
    artifacts.write_report(run_dir, report)

    exit_code = 0
    if fail_on_regression and not report.passed:
        exit_code = 1

    baseline_saved = False
    if save_baseline:
        try:
            artifacts.write_baseline_candidate(run_dir, report, policy)
            baseline_saved = True
        except ContractError:
            # The report is already saved; a rejected candidate is a command
            # failure, not a hard crash.
            exit_code = 2

    return RunOutcome(
        exit_code=exit_code,
        run_dir=run_dir,
        dataset_fingerprint=fingerprint,
        report=report,
        baseline_saved=baseline_saved,
    )


def _ensure_inputs_outside(run_dir: Path, *paths: Path | None) -> None:
    resolved_run = run_dir.resolve()
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved == resolved_run or resolved_run in resolved.parents:
            raise ContractError(
                f"input file must not be inside the output directory: {path}"
            )
