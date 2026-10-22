"""Render in-memory reports without file I/O or model dependencies."""

from .contracts import Report


def render_json(report: Report) -> str:
    return report.model_dump_json(indent=2, exclude_none=True)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def render_markdown(report: Report) -> str:
    lines = [
        f"# QA evaluation: {_cell(report.dataset_name)}", "",
        f"Fingerprint: `{report.dataset_fingerprint}`", "",
        f"Provenance: {_cell(report.metadata.get('provenance', 'unknown'))}", "",
        f"Result: {'PASS' if report.passed else 'FAIL'}", "",
        "## Metrics", "", "| Metric | Value | Eligible cases |",
        "|---|---:|---:|",
    ]
    for name, metric in sorted(report.metrics.items()):
        lines.append(f"| {_cell(name)} | {metric.value:.4f} | {metric.eligible_cases} |")
    for tag, metrics in sorted(report.slices.items()):
        lines.extend([
            "", f"## Slice: {_cell(tag)}", "", "| Metric | Value | Eligible cases |",
            "|---|---:|---:|",
        ])
        for name, metric in sorted(metrics.items()):
            lines.append(f"| {_cell(name)} | {metric.value:.4f} | {metric.eligible_cases} |")
    lines.extend([
        "", "## Gates", "", "| Gate | Status | Actual | Baseline | Threshold | Detail |",
        "|---|---|---|---|---|---|",
    ])
    for gate in report.gates:
        cells = [gate.name, gate.status, gate.actual, gate.baseline, gate.threshold, gate.detail]
        lines.append("| " + " | ".join(_cell(v) if v is not None else "—" for v in cells) + " |")
    lines.extend(["", "## Judge execution", ""])
    for name in (
        "judge_requested", "judge_attempted_cases", "judge_succeeded_cases",
        "judge_failed_cases", "judge_skipped_cases", "judge_available",
    ):
        lines.append(f"- {name}: {_cell(report.metadata.get(name, 'unknown'))}")
    for case in report.cases:
        lines.extend(["", f"## Case: {_cell(case.case_id)}", "",
                      f"Result: {'PASS' if case.passed else 'FAIL'}", ""])
        for name, value in sorted(case.metrics.items()):
            lines.append(f"- {_cell(name)}: {value:.4f}")
        for name, value in sorted(case.checks.items()):
            lines.append(f"- {_cell(name)}: {_cell(value)}")
        if case.judge is None:
            lines.extend(["", "Judge: no score (skipped or failed)."])
            continue
        lines.extend(["", "| Judge metric | Value | Direction |", "|---|---:|---|"])
        for name, value in case.judge.model_dump(exclude={"reason"}).items():
            direction = "Higher is better"
            label = name
            if name == "unsupported_claim_rate":
                direction = "Lower is better"
            elif name == "conflict_disclosure_rate":
                label += " (conflict omission rate / 冲突未披露率)"
                direction = "Lower is better"
            lines.append(f"| {label} | {value:.4f} | {direction} |")
        lines.extend(["", f"Reason: {_cell(case.judge.reason)}"])
    return "\n".join(lines) + "\n"
