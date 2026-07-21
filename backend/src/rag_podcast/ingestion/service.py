from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..models.episode import Episode, TranscriptStatus
from ..models.podcast import Podcast
from .downloader import DownloadError, download_audio
from .parser import ParsedEpisode, parse_feed

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    podcast: Podcast
    episodes: list[Episode]
    new_episodes: int
    skipped: int
    already_imported: bool


async def ingest_podcast(
    session: AsyncSession, rss_url: str, max_episodes: int = 1
) -> IngestResult:
    """Parse a feed, store podcast/episode rows, download audio.

    Mirrors PLAN.md §5.1's 7-step flow. Feed fetch/parse and the podcast
    insert are allowed to raise straight out of this function — the caller
    aborts the whole ingestion on those. Individual episode inserts and
    downloads are isolated so one bad episode doesn't sink the rest
    (PLAN.md §5.4).
    """
    rss_url = rss_url.strip()
    parsed = parse_feed(rss_url, max_episodes=max_episodes)
    podcast = await session.scalar(select(Podcast).where(Podcast.rss_url == rss_url))
    already_imported = podcast is not None
    if podcast is None:
        podcast = Podcast(
            rss_url=rss_url,
            name=parsed.name,
            author=parsed.author,
            cover_url=parsed.cover_url,
        )
        session.add(podcast)
        await session.flush()  # assigns podcast.id without committing yet

    existing_guids = set(
        (
            await session.scalars(
                select(Episode.guid).where(Episode.podcast_id == podcast.id)
            )
        ).all()
    )

    new_episodes: list[Episode] = []
    skipped = 0
    for parsed_episode in parsed.episodes:
        if parsed_episode.guid in existing_guids:
            skipped += 1
            continue

        episode = await _insert_episode(session, podcast.id, parsed_episode)
        if episode is None:
            skipped += 1
            continue
        new_episodes.append(episode)

    await session.commit()

    data_dir = Path(settings.data_dir)
    for episode in new_episodes:
        await _download_episode(session, episode, podcast.id, podcast.name, data_dir)

    return IngestResult(
        podcast=podcast,
        episodes=new_episodes,
        new_episodes=len(new_episodes),
        skipped=skipped,
        already_imported=already_imported,
    )


async def _insert_episode(
    session: AsyncSession, podcast_id: int, parsed_episode: ParsedEpisode
) -> Episode | None:
    """Insert one episode in a SAVEPOINT so a failure here only rolls back
    this row, not the podcast insert or sibling episodes already added."""
    try:
        async with session.begin_nested():
            episode = Episode(
                podcast_id=podcast_id,
                guid=parsed_episode.guid,
                title=parsed_episode.title,
                published_date=parsed_episode.published_date,
                enclosure_url=parsed_episode.enclosure_url,
                duration_seconds=parsed_episode.duration_seconds,
                transcript_status=TranscriptStatus.PENDING,
            )
            session.add(episode)
            await session.flush()
        return episode
    except Exception:
        logger.exception("Failed to insert episode guid=%s", parsed_episode.guid)
        return None


async def _download_episode(
    session: AsyncSession, episode: Episode, podcast_id: int, podcast_title: str, data_dir: Path
) -> None:
    try:
        path = download_audio(
            episode.enclosure_url,
            podcast_id=podcast_id,
            episode_id=episode.id,
            podcast_title=podcast_title,
            data_dir=data_dir,
        )
        episode.audio_local_path = str(path)
        episode.transcript_status = TranscriptStatus.DOWNLOADED
    except DownloadError:
        logger.exception("Failed to download audio for episode id=%s", episode.id)
        episode.transcript_status = TranscriptStatus.FAILED
    await session.commit()
