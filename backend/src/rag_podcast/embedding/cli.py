"""Embedding worker CLI.

Runs as the ``indexer`` service in ``docker-compose.yml`` (reusing the api
image — no GPU, no optional extras).  Can also be run directly on the host for
local debugging::

    uv run python -m rag_podcast.embedding.cli --debug --once --episode-id <N>

Reads database URL and embedding config from ``.env`` / ``config.Settings``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rag_podcast.config import settings
from rag_podcast.db import async_session
from rag_podcast.embedding.worker import run_worker
from rag_podcast.indexing.Chunker import create_chunker


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
        description="Poll DB for ready episodes and embed + persist their chunks.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Seconds between polls when no episodes are ready (default: from config).",
    )
    parser.add_argument(
        "--strategy",
        default="size_based",
        help="Chunking strategy (default: size_based).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Drain the ready queue and exit instead of looping.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Index only this episode (for debugging).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset FAILED episodes to PENDING before starting.",
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

    chunker = create_chunker(args.strategy)

    await run_worker(
        chunker=chunker,
        session_factory=async_session,
        poll_interval=poll_interval,
        once=args.once,
        episode_id=args.episode_id,
        retry_failed=args.retry_failed,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
