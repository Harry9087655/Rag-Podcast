"""Transcription worker CLI.

Runs as the ``worker`` service in ``docker-compose.yml`` (see
``backend/Dockerfile.worker``), which has GPU access via
``deploy.resources.reservations.devices``. Can also be run directly on the
host for local debugging::

    uv run --extra transcription python -m rag_podcast.transcription.cli

Reads database URL and WhisperX config from ``.env`` / ``config.Settings``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rag_podcast.config import settings
from rag_podcast.db import async_session
try:
    from rag_podcast.transcription.Transcriber import create_transcriber
    from rag_podcast.transcription.worker import run_worker
except ImportError as exc:
    raise ImportError(
        "WhisperX is not installed. The transcription CLI requires the [transcription] extra.\n"
        "Run: uv run --extra transcription python -m rag_podcast.transcription.cli"
    ) from exc


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    for noisy in ("sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll DB for DOWNLOADED episodes and transcribe via WhisperX.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Seconds between polls when no episodes are ready (default: from config).",
    )
    return parser.parse_args(argv)


async def _main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    poll_interval = (
        args.poll_interval
        if args.poll_interval is not None
        else settings.worker_poll_interval
    )

    transcriber = create_transcriber(
        model=settings.whisperx_model,
        device=settings.whisperx_device,
        compute_type=settings.whisperx_compute_type,
        funasr_model=settings.funasr_model,
        funasr_vad_model=settings.funasr_vad_model,
        funasr_punc_model=settings.funasr_punc_model,
        funasr_device=settings.funasr_device,
        lang_detect_window_seconds=settings.lang_detect_window_seconds,
    )

    data_dir = Path(settings.data_dir)

    await run_worker(
        transcriber=transcriber,
        session_factory=async_session,
        data_dir=data_dir,
        poll_interval=poll_interval,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
