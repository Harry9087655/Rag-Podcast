"""Validate ``dataset_v1.json`` and print its fingerprint.

Checks, in order:
1. JSON Schema (``schema/dataset_v1.schema.json``).
2. Consistency rules the schema cannot express:
   - case ids are unique;
   - full-context  ⇒ gold_chunks is null, gold_segments is non-empty;
   - search        ⇒ gold_chunks is non-empty, gold_segments is null;
   - no_knowledge  ⇒ both golds null, required_claims empty.
   - cross_podcast ⇒ scope is empty (whole library).
3. Prints case count, arm / tag / critical breakdown, and ``dataset_fingerprint``.

``dataset_fingerprint`` is imported from ``loaders`` (its single source of truth).
Data lives in ``evals/qa/`` at the repo root, not in this package.

Usage (from ``backend/``; ``jsonschema`` is a dev dependency):
    uv run python -m rag_podcast.evaluation.check_dataset [path/to/dataset.json]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from rag_podcast.evaluation.contracts import ContractError
from rag_podcast.evaluation.loaders import EVALS_DIR, load_validated_dataset

SCHEMA_PATH = EVALS_DIR / "schema" / "dataset_v1.schema.json"
DEFAULT_DATASET = EVALS_DIR / "dataset_v1.json"


def consistency_errors(dataset: dict) -> list[str]:
    errors: list[str] = []
    if not dataset["cases"]:
        errors.append("dataset must contain at least one case")
    seen: set[str] = set()
    for case in dataset["cases"]:
        cid = case["id"]
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)

        chunks, segments = case["gold_chunks"], case["gold_segments"]
        if "no_knowledge" in case["tags"]:
            if chunks is not None or segments is not None:
                errors.append(f"{cid}: no_knowledge case must have both golds null")
            if case["required_claims"]:
                errors.append(f"{cid}: no_knowledge case must have empty required_claims")
        elif case["expected_arm"] == "full-context":
            if chunks is not None or not segments:
                errors.append(f"{cid}: full-context needs gold_chunks=null and non-empty gold_segments")
        elif not chunks or segments is not None:
            errors.append(f"{cid}: search needs non-empty gold_chunks and gold_segments=null")

        if "cross_podcast" in case["tags"] and case.get("scope") != {}:
            errors.append(f"{cid}: cross_podcast case must use empty scope (whole library)")

        for seg in segments or []:
            if seg["start"] > seg["end"]:
                errors.append(f"{cid}: gold_segment start > end")
    return errors


def validate_dataset(raw: dict, schema: dict) -> list[str]:
    """Validate structure before consistency; diagnostics omit input values.

    jsonschema is a dev dependency, loaded only when validation is requested.
    Schema errors are trusted-structure paths and keywords, not raw payloads.
    """
    import jsonschema

    try:
        jsonschema.validate(raw, schema)
    except jsonschema.ValidationError as exc:
        return [f"SCHEMA FAIL: {exc.validator} at {list(exc.absolute_path)}"]
    return consistency_errors(raw)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    try:
        dataset, fingerprint = load_validated_dataset(path, schema_path=SCHEMA_PATH)
    except ContractError as exc:
        print(f"VALIDATION FAIL: {exc}")
        return 1

    cases = dataset.cases
    arms = Counter(c.expected_arm for c in cases)
    tags = Counter(t for c in cases for t in c.tags)
    critical = sum(c.critical for c in cases)

    print(f"OK: {path.name}")
    print(f"cases:       {len(cases)}")
    print(f"arms:        {dict(arms)}")
    print(f"critical:    {critical} ({critical / len(cases):.0%})")
    print(f"tags:        {dict(sorted(tags.items()))}")
    print(f"fingerprint: {fingerprint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
