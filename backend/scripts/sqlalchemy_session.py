import asyncio
import os
from sqlalchemy import select
from rag_podcast.db import async_session, get_session
from rag_podcast.models.episode import Episode
from rag_podcast.config import settings

async def get_audio_path(episode_id: int) -> str:
    async with async_session() as session:
        episode = await session.get(Episode, episode_id)
        if episode is None:
            raise ValueError(f"Episode with id {episode_id} not found")
        return os.path.join(settings.data_dir, episode.audio_local_path)

if __name__ == "__main__":
    episode_id = 1  # Replace with the actual episode ID you want to query
    print(f"{settings.data_dir}")
    audio_path = asyncio.run(get_audio_path(episode_id))
    print(f"Audio path for episode {episode_id}: {audio_path}")
    with open(audio_path, "rb") as f:
        data = f.read()
    print(f"Read {len(data)} bytes from {audio_path}")