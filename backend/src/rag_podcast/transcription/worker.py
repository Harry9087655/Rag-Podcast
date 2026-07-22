"""Transcription worker: poll DB, transcribe one episode at a time.

Library code — the CLI entry point in ``scripts/run_transcription_worker.py``
wires everything together.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..models.episode import Episode, TranscriptStatus
from ..models.podcast import Podcast
from .transcriber import TranscribeError, Transcriber

logger = logging.getLogger(__name__)


async def reset_stale_processing(session: AsyncSession) -> int:
    """Reset any PROCESSING episodes back to DOWNLOADED.

    Handles unclean worker shutdown: if the worker crashes mid-transcription,
    those episodes would be stuck in PROCESSING forever.  Running this on
    startup makes the worker crash-safe.
    """
    result = await session.execute(
        update(Episode)
        .where(Episode.transcript_status == TranscriptStatus.PROCESSING)
        .values(transcript_status=TranscriptStatus.DOWNLOADED)
    )
    await session.commit()
    count = result.rowcount
    if count:
        logger.warning(
            "Reset %d episode(s) from PROCESSING back to DOWNLOADED "
            "(unclean shutdown recovery).",
            count,
        )
    return count


async def poll_downloaded(session: AsyncSession) -> Episode | None:
    """Return the next DOWNLOADED episode, or None if none are ready."""
    result = await session.execute(
        select(Episode)
        .where(Episode.transcript_status == TranscriptStatus.DOWNLOADED)
        .order_by(Episode.id)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def claim_episode(session: AsyncSession, episode_id: int) -> Episode | None:
    """Atomically claim an episode for transcription.

    Uses a status guard (WHERE status = 'downloaded') so even if two workers
    were running, only one would claim a given episode.
    """
    result = await session.execute(
        update(Episode)
        .where(
            Episode.id == episode_id,
            Episode.transcript_status == TranscriptStatus.DOWNLOADED,
        )
        .values(transcript_status=TranscriptStatus.PROCESSING)
        .returning(Episode)
    )
    await session.commit()
    return result.scalar_one_or_none()


async def transcribe_episode(
    session: AsyncSession,
    episode: Episode,
    transcriber: Transcriber,
    data_dir: Path,
) -> None:
    """Transcribe one episode: PROCESSING → transcribe → DONE or FAILED.

    The episode object should already have status PROCESSING (set by
    ``claim_episode``).  All changes are committed before this function
    returns, so the DB always reflects the latest state even if the
    worker crashes on a subsequent episode.
    """
    if not episode.audio_local_path:
        logger.error(
            "Episode id=%s has no audio_local_path — cannot transcribe.",
            episode.id,
        )
        episode.transcript_status = TranscriptStatus.FAILED
        await session.commit()
        return

    audio_path = data_dir / episode.audio_local_path
    logger.info(
        "Transcribing episode id=%s guid=%s path=%s",
        episode.id,
        episode.guid,
        audio_path,
    )

    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        episode.transcript_status = TranscriptStatus.FAILED
        await session.commit()
        return

    try:
        result = await transcriber.transcribe(audio_path)
    except TranscribeError as exc:
        logger.exception(
            "Transcription failed for episode id=%s: %s", episode.id, exc
        )
        episode.transcript_status = TranscriptStatus.FAILED
        await session.commit()
        return
    except Exception:
        logger.exception(
            "Unexpected error transcribing episode id=%s", episode.id
        )
        episode.transcript_status = TranscriptStatus.FAILED
        await session.commit()
        return

    episode.transcript_data = result
    episode.transcript_status = TranscriptStatus.DONE
    await session.commit()
    logger.info(
        "Episode id=%s transcribed successfully — %d segments.",
        episode.id,
        len(result.get("segments", ())),
    )


async def run_worker(
    transcriber: Transcriber,
    session_factory: async_sessionmaker[AsyncSession],
    data_dir: Path,
    poll_interval: int = 10,
) -> None:
    """Main worker loop.

    1. Reset any PROCESSING episodes left over from a crash.
    2. Poll for DOWNLOADED episodes; transcribe them one at a time.
    3. When the queue is empty, sleep *poll_interval* seconds.
    """
    logger.info(
        "Transcription worker starting — poll_interval=%ds data_dir=%s",
        poll_interval,
        data_dir,
    )

    async with session_factory() as session:
        await reset_stale_processing(session)

    try:
        while True:
            async with session_factory() as session:
                episode = await poll_downloaded(session)

                if episode is None:
                    logger.debug("No DOWNLOADED episodes — sleeping %ds.", poll_interval)
                    await asyncio.sleep(poll_interval)
                    continue

                # Claim before transcribing (no sleep between polls when
                # episodes are ready).
                claimed = await claim_episode(session, episode.id)
                if claimed is None:
                    logger.debug(
                        "Episode id=%s already claimed by another worker — skipping.",
                        episode.id,
                    )
                    continue

                await transcribe_episode(session, claimed, transcriber, data_dir)

    except asyncio.CancelledError:
        logger.info("Worker cancelled — shutting down.")
        raise
    except KeyboardInterrupt:
        logger.info("Worker interrupted — shutting down.")
    except Exception:
        logger.exception("Fatal worker error — shutting down.")
        raise
