"""Semantic search: the single entry point over chunk vectors.

``search()`` is the seam ``qa``/``api`` call.  Behind it is a fixed pipeline
skeleton whose stages are pure ``list[SearchResult] → list[SearchResult]``
transforms, so future stages (rerank, threshold, hybrid recall) insert as one
line without reshaping the output type.

Design contract: .trellis/tasks/09-12-retrieval/design.md
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..embedding.embedder import embed_batch
from .BaseRecallerInterface import Recaller, SearchResult
from .Recaller import create_recaller
from .merge import auto_merge

# Prepended to the *query* only (never to passages — the corpus was embedded
# plain, which is correct for passages).  A single clearly-marked constant so
# M4 eval can flip it on/off trivially (BGE-M3 model card, PRD Q#5).
QUERY_INSTRUCTION_PREFIX = "Represent this sentence for searching relevant passages: "


async def _embed_query(query: str) -> list[float]:
    """Embed ``query`` through the corpus's exact model/endpoint config (R2).

    No separate query-embedding client or model knob — a second config path
    invites a silent query/corpus vector-space mismatch.
    """
    vectors = await embed_batch(
        [QUERY_INSTRUCTION_PREFIX + query],
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        batch_size=1,
        timeout=settings.embedding_timeout_seconds,
    )
    return vectors[0]


def resolve_candidate_k(top_k: int) -> int:
    """Over-fetch multiplier for future rerank; a no-op at the default of 1."""
    return top_k * settings.retrieval_candidate_multiplier


async def search(
    query: str,
    *,
    session: AsyncSession,
    top_k: int | None = None,
    podcast_id: int | None = None,
    episode_id: int | None = None,
    recaller: Recaller | None = None,
) -> list[SearchResult]:
    """Return up to ``top_k`` chunks nearest ``query``, best-first.

    Embeds the query, recalls candidates, and trims to ``top_k``.  With the
    default ``retrieval_candidate_multiplier = 1`` the recall returns exactly
    ``top_k`` and the trim is a no-op — the candidate slot exists for the M4
    reranker.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k
    if recaller is None:
        recaller = create_recaller()

    vector = await _embed_query(query)
    candidate_k = resolve_candidate_k(top_k)
    docs = await recaller.recall(
        session, vector, top_k=candidate_k, podcast_id=podcast_id, episode_id=episode_id
    )

    # Auto-merge sibling leaves into their parent block before trimming, so a
    # parent that absorbed several children still fits inside top_k.
    docs = await auto_merge(docs, session, settings.auto_merge_threshold)

    # M4: re-rank the over-fetched candidates before trimming.
    # docs = await reranker.rerank(query, docs, top_k)
    return docs[:top_k]
