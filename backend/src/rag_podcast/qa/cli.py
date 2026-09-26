"""QA CLI — verify question answering over the indexed corpus.

Runs a single question end-to-end (route → LLM → cite) for spot-checking
acceptance criteria against already-indexed episodes::

    uv run python -m rag_podcast.qa.cli "问题" [--episode-id N] [--podcast-id N] [--top-k K]

Reads database + LLM config from ``.env`` / ``config.Settings``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rag_podcast.config import settings
from rag_podcast.db import async_session
from rag_podcast.qa.core import answer


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
        description="Q&A over the podcast corpus using retrieved chunks.",
    )

    parser.add_argument(
        "question",
        type=str,
        help="The user query.",
    )

    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Answer within this single episode (full-context when it fits).",
    )

    parser.add_argument(
        "--podcast-id",
        type=int,
        default=None,
        help="Answer across this whole podcast (cross-episode search).",
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
        ans = await answer(
            question=args.question,
            session=session,
            episode_id=args.episode_id,
            podcast_id=args.podcast_id,
            top_k=top_k,
        )

    print(f"Answer:\n{ans.answer}")
    print("-------------------------------------------------------")
    if not ans.sources:
        print("No sources.")
        return

    for i, s in enumerate(ans.sources, 1):
        print(
            f"{i}. {s.podcast_name} / {s.episode_title} "
            f"({s.start:.1f}s-{s.end:.1f}s)"
        )
        print(f"   {s.text}")
        print()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
