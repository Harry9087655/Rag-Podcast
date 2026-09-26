"""Retrieval CLI — verify semantic search against the indexed corpus.

Runs a single query end-to-end (embed → recall → print) for spot-checking
acceptance criteria against already-indexed episodes::

    uv run python -m rag_podcast.retrieval.cli "what is ..." [--podcast-id N] [--top-k K]

Reads database + embedding config from ``.env`` / ``config.Settings``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rag_podcast.config import settings
from rag_podcast.db import async_session
from rag_podcast.retrieval.search import search


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
        description="Semantic search over the indexed chunk vectors.",
    )
    parser.add_argument("question", help="The query to search for.")
    parser.add_argument(
        "--podcast-id",
        type=int,
        default=None,
        help="Restrict results to this podcast (default: whole library).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results (default: settings.retrieval_top_k).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


async def _main() -> None:
    args = _parse_args()
    _configure_logging(args.debug)

    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k

    async with async_session() as session:
        results = await search(
            args.question,
            session=session,
            top_k=top_k,
            podcast_id=args.podcast_id,
        )

    if not results:
        print("No results.")
        return

    for i, result in enumerate(results, 1):
        print(
            f"{i}. [{result.score:.4f}] {result.podcast_name} / "
            f"{result.episode_title} ({result.start:.1f}s-{result.end:.1f}s)"
        )
        print(f"   {result.text}")
        print()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
