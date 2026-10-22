"""Embedding worker: poll DB, index one episode at a time.

Library code — the CLI entry point (``embedding/cli.py``) wires the chunker and
settings together.  Mirrors ``transcription/worker.py``'s poll → claim → index
→ loop skeleton, but with the single-transaction boundary R2 requires: a
successful index commits DELETE + INSERT + ``index_status = done`` exactly once,
and a failure rolls back before marking ``failed`` in a separate commit.

Design contract: .trellis/tasks/09-05-embedding-pipeline/design.md
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..cleaning import clean_words
from ..config import settings
from ..indexing.BaseChunkerInterface import Chunker
from ..indexing.parent_grouper import group_into_parents
from ..models.episode import Episode, IndexStatus, TranscriptStatus
from .embedder import embed_batch
from .store import replace_chunks

logger = logging.getLogger(__name__)


async def reset_stale_processing(session: AsyncSession) -> int:
    """Reset any PROCESSING episodes back to PENDING.

    Handles unclean worker shutdown: if the worker crashes mid-index, those
    episodes would be stuck in PROCESSING forever.  Running this on startup
    makes the worker crash-safe.
    """
    result = await session.execute(
        update(Episode)
        .where(Episode.index_status == IndexStatus.PROCESSING)
        .values(index_status=IndexStatus.PENDING)
    )
    await session.commit()
    count = result.rowcount
    if count:
        logger.warning(
            "Reset %d episode(s) from PROCESSING back to PENDING "
            "(unclean shutdown recovery).",
            count,
        )
    return count


async def reset_failed(session: AsyncSession) -> int:
    """Reset FAILED episodes back to PENDING.

    Only called with ``--retry-failed`` — a failed episode is terminal
    otherwise, since the poll query looks for PENDING (R4).
    """
    result = await session.execute(
        update(Episode)
        .where(Episode.index_status == IndexStatus.FAILED)
        .values(index_status=IndexStatus.PENDING)
    )
    await session.commit()
    count = result.rowcount
    if count:
        logger.warning(
            "Reset %d episode(s) from FAILED back to PENDING (--retry-failed).",
            count,
        )
    return count


async def reset_done(session: AsyncSession) -> int:
    """Reset DONE episodes back to PENDING for a full reindex.

    The migration path for a re-chunk/re-embed (09-17 leaf/parent rework): every
    already-indexed episode must be rebuilt so its chunks — and now parents —
    reflect the new scheme. Runs once before the worker loop; the worker then
    re-indexes each as if new.
    """
    result = await session.execute(
        update(Episode)
        .where(Episode.index_status == IndexStatus.DONE)
        .values(index_status=IndexStatus.PENDING)
    )
    await session.commit()
    count = result.rowcount
    if count:
        logger.warning(
            "Reset %d episode(s) from DONE back to PENDING (full reindex).",
            count,
        )
    return count


async def poll_ready(
    session: AsyncSession, episode_id: int | None = None
) -> Episode | None:
    """Return the next episode ready to index, or None.

    Ready = ``transcript_status`` done AND ``index_status`` pending.
    ``episode_id`` narrows to one episode for debugging.
    """
    stmt = select(Episode).where(
        Episode.transcript_status == TranscriptStatus.DONE,
        Episode.index_status == IndexStatus.PENDING,
    )
    if episode_id is not None:
        stmt = stmt.where(Episode.id == episode_id)
    result = await session.execute(stmt.order_by(Episode.id).limit(1))
    return result.scalar_one_or_none()


async def claim_episode(session: AsyncSession, episode_id: int) -> Episode | None:
    """Atomically claim an episode for indexing (PENDING → PROCESSING).

    Status-guarded, so even two concurrent workers could never both claim the
    same episode.
    """
    result = await session.execute(
        update(Episode)
        .where(
            Episode.id == episode_id,
            Episode.index_status == IndexStatus.PENDING,
        )
        .values(index_status=IndexStatus.PROCESSING)
        .returning(Episode)
    )
    await session.commit()
    return result.scalar_one_or_none()


def resolve_token_tier(
    duration_seconds: float,
) -> tuple[dict[str, int], dict[str, int]]:
    """Pick ``(leaf, parent)`` token bounds for an episode's duration.

    First-match-wins in ``settings.audio_tier`` definition order (Short →
    Medium → Long); ``Long``'s ``max_episode_duration_seconds`` is ``inf``, so
    it is the catch-all and must stay last. Each half is the leaf's
    ``{min, desired, max}`` followed by the parent's ``{desired, max}`` (no
    parent min — a parent has no floor).
    """
    for cfg in settings.audio_tier.values():
        if duration_seconds <= cfg["max_episode_duration_seconds"]:
            leaf = {k: int(v) for k, v in cfg["leaf"].items()}
            parent = {k: int(v) for k, v in cfg["parent"].items()}
            return leaf, parent
    raise RuntimeError("no audio tier matched")  # unreachable: Long is inf


async def index_episode(
    session: AsyncSession,
    episode: Episode,
    chunker: Chunker,
) -> None:
    """Index one episode: clean → chunk → embed → replace → DONE or FAILED.

    The episode should already have status PROCESSING (set by
    ``claim_episode``).  A successful index holds its whole write set — the
    DELETE of old rows, the INSERT of new rows, and ``index_status = done`` —
    in ONE transaction and commits once (R2).  Any failure rolls that
    transaction back and marks the episode failed in a separate commit.

    Missing/empty transcript segments, an empty cleaned result, and an empty
    chunk list are all treated as ``failed``, never as a vacuous ``done``.
    """
    segments = (episode.transcript_data or {}).get("segments", []) or []
    if not segments:
        logger.error(
            "Episode id=%s has no transcript segments — cannot index.",
            episode.id,
        )
        episode.index_status = IndexStatus.FAILED
        await session.commit()
        return
    duration = episode.duration_seconds
    cleaned = clean_words(segments, episode.language)
    if not cleaned:
        logger.error(
            "Episode id=%s has no segments left after cleaning — cannot index.",
            episode.id,
        )
        episode.index_status = IndexStatus.FAILED
        await session.commit()
        return

    # Duration from the cleaned segments, not episode.duration_seconds (which
    # can be None).
    
    leaf_bounds, parent_bounds = resolve_token_tier(duration)
    leaves = chunker.build_chunks(
        cleaned,
        min_duration=settings.chunk_min_duration_seconds,
        max_duration=settings.chunk_max_duration_seconds,
        language=episode.language,
        **leaf_bounds,
    )
    if not leaves:
        logger.error("Episode id=%s produced no chunks — cannot index.", episode.id)
        episode.index_status = IndexStatus.FAILED
        await session.commit()
        return

    parents = group_into_parents(
        leaves,
        count=chunker.count_tokens,
        max_duration=settings.chunk_max_duration_seconds,
        language=episode.language,
        **parent_bounds,
    )

    logger.info(
        "Indexing episode id=%s — %d chunks, %d parents.",
        episode.id,
        len(leaves),
        len(parents),
    )

    try:
        vectors = await embed_batch(
            [leaf.text for leaf in leaves],
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout_seconds,
        )
        await replace_chunks(session, episode, leaves, vectors, parents)  # no commit
        episode.index_status = IndexStatus.DONE
        await session.commit()  # single commit: DELETE + INSERT + done (R2)
        logger.info(
            "Episode id=%s indexed — %d chunks stored.", episode.id, len(leaves)
        )
    except Exception:
        logger.exception("Indexing failed for episode id=%s", episode.id)
        await session.rollback()
        await session.execute(
            update(Episode)
            .where(Episode.id == episode.id)
            .values(index_status=IndexStatus.FAILED)
        )
        await session.commit()


async def run_worker(
    chunker: Chunker,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    poll_interval: int = 10,
    once: bool = False,
    episode_id: int | None = None,
    retry_failed: bool = False,
    reindex: bool = False,
) -> None:
    """Main worker loop.

    1. Reset PROCESSING (crash recovery); with ``retry_failed``, FAILED →
       PENDING; with ``reindex``, DONE → PENDING (full rebuild).
    2. Poll for ready episodes and index them one at a time.
    3. When the queue is empty: return if ``once``, else sleep
       ``poll_interval`` seconds.
    """
    logger.info(
        "Embedding worker starting — poll_interval=%ds once=%s episode_id=%s "
        "retry_failed=%s reindex=%s",
        poll_interval,
        once,
        episode_id,
        retry_failed,
        reindex,
    )

    async with session_factory() as session:
        await reset_stale_processing(session)
        if retry_failed:
            await reset_failed(session)
        if reindex:
            await reset_done(session)

    try:
        while True:
            async with session_factory() as session:
                episode = await poll_ready(session, episode_id)

                if episode is None:
                    if once:
                        logger.info("Ready queue drained — exiting (--once).")
                        return
                    logger.debug("No ready episodes — sleeping %ds.", poll_interval)
                    await asyncio.sleep(poll_interval)
                    continue

                claimed = await claim_episode(session, episode.id)
                if claimed is None:
                    logger.debug(
                        "Episode id=%s already claimed by another worker — skipping.",
                        episode.id,
                    )
                    continue

                await index_episode(session, claimed, chunker)

    except asyncio.CancelledError:
        logger.info("Worker cancelled — shutting down.")
        raise
    except KeyboardInterrupt:
        logger.info("Worker interrupted — shutting down.")
    except Exception:
        logger.exception("Fatal worker error — shutting down.")
        raise
