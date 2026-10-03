"""Validate ``dataset_v1.json`` and print its fingerprint.

Checks, in order:
1. JSON Schema (``schema/dataset_v1.schema.json``).
2. Consistency rules the schema cannot express:
   - case ids are unique;
   - full-context  ⇒ gold_chunks is null, gold_segments is non-empty;
   - search        ⇒ gold_chunks is non-empty, gold_segments is null;
   - no_knowledge  ⇒ both golds null, required_claims empty.
3. Prints case count, arm / tag / critical breakdown, and ``dataset_fingerprint``.

The fingerprint algorithm is the single source of truth shared with the
``scoring`` module — do not change it without changing both.

Usage (from ``backend/`` so the venv has ``jsonschema``):
    uv run python ../evals/qa/check_dataset.py [path/to/dataset.json]
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import jsonschema

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema" / "dataset_v1.schema.json"
DEFAULT_DATASET = HERE / "dataset_v1.json"


def dataset_fingerprint(dataset: dict) -> str:
    canonical = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def consistency_errors(dataset: dict) -> list[str]:
    errors: list[str] = []
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

        for seg in segments or []:
            if seg["start"] > seg["end"]:
                errors.append(f"{cid}: gold_segment start > end ({seg})")
    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    dataset = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    try:
        jsonschema.validate(dataset, schema)
    except jsonschema.ValidationError as exc:
        print(f"SCHEMA FAIL: {exc.message} at {list(exc.absolute_path)}")
        return 1

    errors = consistency_errors(dataset)
    if errors:
        print("CONSISTENCY FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1

    cases = dataset["cases"]
    arms = Counter(c["expected_arm"] for c in cases)
    tags = Counter(t for c in cases for t in c["tags"])
    critical = sum(c["critical"] for c in cases)

    print(f"OK: {path.name}")
    print(f"cases:       {len(cases)}")
    print(f"arms:        {dict(arms)}")
    print(f"critical:    {critical} ({critical / len(cases):.0%})")
    print(f"tags:        {dict(sorted(tags.items()))}")
    print(f"fingerprint: {dataset_fingerprint(dataset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
