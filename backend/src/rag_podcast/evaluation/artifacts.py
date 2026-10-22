"""Run-artifact persistence: the only evaluation module that writes files.

``create_run_dir`` resolves an exclusive per-run output directory;
``write_bundle`` / ``write_report`` / ``write_baseline_candidate`` publish the
run artifacts with atomic replacement and a ``report.json`` completion marker.
Scoring modules (evaluate/metrics/gate/judge/render) stay in-memory.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .contracts import ContractError, GatePolicy, ObservationBundle, Report
from .loaders import save_baseline
from .render import render_json, render_markdown


def create_run_dir(base: Path, *, explicit: Path | None = None) -> Path:
    """Resolve an exclusive output directory.

    An explicit path must not already exist (even when empty); the auto path
    gets a UTC-timestamp plus random suffix so two runs never collide and the
    command never depends on the host timezone.
    """
    if explicit is not None:
        if explicit.exists():
            raise ContractError(f"output directory already exists: {explicit}")
        explicit.mkdir(parents=True)
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"{stamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _atomic_write(path: Path, text: str) -> None:
    """Write UTF-8 to a sibling temp file, then atomically replace the target."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_bundle(run_dir: Path, bundle: ObservationBundle) -> None:
    _atomic_write(run_dir / "observations.json", bundle.model_dump_json(indent=2) + "\n")


def write_report(run_dir: Path, report: Report) -> None:
    """Render both report forms first, then publish; report.json lands last.

    ``report.json`` is the completion marker: its presence means the whole run
    finished. If rendering fails, nothing is written.
    """
    json_text = render_json(report)
    markdown_text = render_markdown(report)
    _atomic_write(run_dir / "report.md", markdown_text)
    _atomic_write(run_dir / "report.json", json_text + "\n")


def write_baseline_candidate(run_dir: Path, report: Report, policy: GatePolicy) -> None:
    """Save a deterministic baseline candidate for a complete, clean live run.

    Rejects candidates that cannot serve as a comparison baseline (provider
    failures, missing observations, or a policy metric with no eligible cases);
    ``save_baseline`` then extracts only deterministic metrics, never Judge data.
    """
    if report.metadata.get("provider_failed_cases", 0) > 0:
        raise ContractError("baseline candidate rejected: provider_failed cases present")
    if report.metadata.get("missing_observation_cases", 0) > 0:
        raise ContractError("baseline candidate rejected: missing observations present")
    policy_metrics = {gate.metric for gate in policy.metric_gates}
    for metric in sorted(policy_metrics):
        value = report.metrics.get(metric)
        if value is None or value.eligible_cases <= 0:
            raise ContractError(f"baseline candidate rejected: {metric} has no eligible cases")
    save_baseline(report, run_dir / "baseline.json")
