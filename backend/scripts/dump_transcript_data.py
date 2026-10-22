"""Dump every non-null ``Episode.transcript_data`` to a JSON file under ``backend/``.

Reads the ``transcript_data`` JSONB column from the ``episode`` table and writes
it to ``backend/transcript_data.json``, keyed by episode id, with lightweight
metadata alongside each transcript for identification.

Usage:
    python scripts/dump_transcript_data.py            # all episodes
    python scripts/dump_transcript_data.py 12         # a single episode
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from rag_podcast.db import async_session
from rag_podcast.models.episode import Episode

OUT_PATH = Path(__file__).resolve().parents[1] / "transcript_data.json"


async def main() -> None:
    episode_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

    query = select(Episode).where(Episode.transcript_data.is_not(None))
    if episode_id is not None:
        query = query.where(Episode.id == episode_id)

    async with async_session() as session:
        result = await session.execute(query)
        episodes = result.scalars().all()

    payload = {
        str(ep.id): {
            "episode_id": ep.id,
            "guid": ep.guid,
            "title": ep.title,
            "podcast_id": ep.podcast_id,
            "language": ep.language,
            "transcript_status": ep.transcript_status.value,
            "transcript_data": ep.transcript_data,
        }
        for ep in episodes
    }

    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(payload)} episode(s) to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
