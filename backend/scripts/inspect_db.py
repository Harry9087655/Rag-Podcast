import os
from pathlib import Path
import whisperx
from rag_podcast.config import settings
import asyncio
from sqlalchemy import select
from rag_podcast.db import async_session, get_session
from rag_podcast.models.episode import Episode
from rag_podcast.config import settings
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_db_id(episode_id: int) -> str:
    async with async_session() as session:
        episode = await session.get(Episode, episode_id)
        if episode is None:
            raise ValueError(f"Episode with id {episode_id} not found")
        return episode 

async def main() -> None:
    episode = await get_db_id(1)

    logger.info(f"Episode {episode.id}: {episode.title}")

    transcript = episode.transcript_data
    logger.info(
        f"Transcript length: {len(transcript['segments']) if transcript and 'segments' in transcript else 0}"
    )

    episode_2 = await get_db_id(2)

    logger.info(f"Episode 2: {episode_2.title}")

    transcript = episode_2.transcript_data
    logger.info(
        f"Transcript length: {len(transcript['segments']) if transcript and 'segments' in transcript else 0}"
    )
    print(transcript['segments'][0])

    episode_3 = await get_db_id(3)

    logger.info(f"Episode 3: {episode_3.title}")

    transcript = episode_3.transcript_data
    logger.info(
        f"Transcript length: {len(transcript['segments']) if transcript and 'segments' in transcript else 0}"
    )
    print(transcript['segments'][0])


if __name__ == "__main__":
    asyncio.run(main())

