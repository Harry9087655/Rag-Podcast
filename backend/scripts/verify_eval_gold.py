"""Verify every gold in ``evals/qa/dataset_v1.json`` against the live DB (D3).

- search arm:       every ``gold_chunks`` id exists in ``chunk``; warns if the
                    chunk falls outside the case ``scope`` (episode / podcast).
- full-context arm: every ``gold_segments`` ``[start, end]`` equals a real
                    segment of ``scope.episode_id`` (cleaned with
                    ``clean_words``, same as ``qa/core.py::_display_segments``).

Prints PASS/FAIL per case; exits 0 only when all cases pass.

Usage:
    python scripts/verify_eval_gold.py [path/to/dataset.json]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from rag_podcast.cleaning.filler_words import clean_words
from rag_podcast.db import async_session
from rag_podcast.models.chunk import Chunk
from rag_podcast.models.episode import Episode

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evals" / "qa" / "dataset_v1.json"


async def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]

    chunk_ids = {i for c in cases for i in (c["gold_chunks"] or [])}
    episode_ids = {
        c["scope"]["episode_id"] for c in cases if c["gold_segments"]
    }

    async with async_session() as session:
        chunk_rows = (
            await session.execute(
                select(Chunk.id, Chunk.episode_id, Chunk.podcast_id).where(
                    Chunk.id.in_(chunk_ids)
                )
            )
        ).all()
        episode_rows = (
            await session.execute(
                select(Episode.id, Episode.language, Episode.transcript_data).where(
                    Episode.id.in_(episode_ids)
                )
            )
        ).all()

    chunks = {r.id: r for r in chunk_rows}
    spans = {
        r.id: {
            (s["start"], s["end"])
            for s in clean_words((r.transcript_data or {}).get("segments", []), r.language)
        }
        for r in episode_rows
    }

    failed = 0
    for case in cases:
        problems: list[str] = []
        warnings: list[str] = []
        scope = case["scope"]

        for cid in case["gold_chunks"] or []:
            row = chunks.get(cid)
            if row is None:
                problems.append(f"chunk {cid} not found")
                continue
            if "episode_id" in scope and row.episode_id != scope["episode_id"]:
                warnings.append(f"chunk {cid} is in episode {row.episode_id}, outside scope")
            if "podcast_id" in scope and row.podcast_id != scope["podcast_id"]:
                warnings.append(f"chunk {cid} is in podcast {row.podcast_id}, outside scope")

        for seg in case["gold_segments"] or []:
            if (seg["start"], seg["end"]) not in spans.get(scope["episode_id"], set()):
                problems.append(
                    f"segment [{seg['start']}, {seg['end']}] not in episode {scope['episode_id']}"
                )

        status = "FAIL" if problems else "PASS"
        failed += bool(problems)
        print(f"{status}  {case['id']}")
        for p in problems:
            print(f"      ✗ {p}")
        for w in warnings:
            print(f"      ! {w}")

    print(f"\n{len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
