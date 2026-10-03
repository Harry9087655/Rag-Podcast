"""Dump the QA-eval annotation inventory to ``evals/qa/_inventory/``.

Three files, all read from the live DB:

- ``chunks.json``    — every chunk (id / episode_id / podcast_id / episode_title
  / podcast_name / start / end / text) for search-arm ``gold_chunks`` annotation.
- ``episodes.json``  — every episode + its podcast (id / language / title / name
  / segment count) for ``scope`` selection.
- ``segments.json``  — every episode's cleaned sentence-level segments
  (episode_id / start / end / text) for full-context-arm ``gold_segments``.

The segment text is the *cleaned* display text (``cleaning.filler_words.clean_words``),
identical to what ``qa/core.py::_display_segments`` feeds the LLM, so annotations
match the text the model actually sees.

Usage:
    python scripts/dump_eval_inventory.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from rag_podcast.cleaning.filler_words import clean_words
from rag_podcast.db import async_session
from rag_podcast.models.chunk import Chunk
from rag_podcast.models.episode import Episode
from rag_podcast.models.podcast import Podcast

OUT_DIR = Path(__file__).resolve().parents[2] / "evals" / "qa" / "_inventory"


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_session() as session:
        chunk_rows = (
            await session.execute(
                select(
                    Chunk.id,
                    Chunk.episode_id,
                    Chunk.podcast_id,
                    Chunk.start,
                    Chunk.end,
                    Chunk.text,
                    Episode.title.label("episode_title"),
                    Podcast.name.label("podcast_name"),
                )
                .join(Episode, Episode.id == Chunk.episode_id)
                .join(Podcast, Podcast.id == Chunk.podcast_id)
                .order_by(Chunk.id)
            )
        ).all()

        episode_rows = (
            await session.execute(
                select(
                    Episode.id,
                    Episode.podcast_id,
                    Episode.language,
                    Episode.title,
                    Episode.transcript_data,
                    Podcast.name.label("podcast_name"),
                )
                .join(Podcast, Podcast.id == Episode.podcast_id)
                .order_by(Episode.id)
            )
        ).all()

    chunks = [
        {
            "id": r.id,
            "episode_id": r.episode_id,
            "podcast_id": r.podcast_id,
            "episode_title": r.episode_title,
            "podcast_name": r.podcast_name,
            "start": r.start,
            "end": r.end,
            "text": r.text,
        }
        for r in chunk_rows
    ]

    episodes = []
    segments = []
    for r in episode_rows:
        raw = (r.transcript_data or {}).get("segments", [])
        cleaned = clean_words(raw, r.language)
        episodes.append(
            {
                "id": r.id,
                "podcast_id": r.podcast_id,
                "podcast_name": r.podcast_name,
                "language": r.language,
                "title": r.title,
                "segment_count": len(cleaned),
            }
        )
        for s in cleaned:
            segments.append(
                {
                    "episode_id": r.id,
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                }
            )

    segments.sort(key=lambda s: (s["episode_id"], s["start"]))

    (OUT_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "episodes.json").write_text(
        json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"chunks:     {len(chunks)}")
    print(f"episodes:   {len(episodes)}")
    print(f"segments:   {len(segments)}")
    print(f"wrote to:   {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
