"""Embedding client: turn cleaned text spans into BGE-M3 vectors.

Library code — ``embed_batch`` is the single seam between the chunker's
``ChunkSpan`` list and the persistence layer.  It talks to DeepInfra's
OpenAI-compatible ``/embeddings`` endpoint through the ``openai`` SDK.

Design contract: .trellis/tasks/09-05-embedding-pipeline/design.md
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024


class EmbeddingError(Exception):
    """Raised when an embedding request fails or returns invalid data."""


async def embed_batch(
    texts: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int = 32,
    timeout: float = 60.0,
) -> list[list[float]]:
    """Embed ``texts`` into ``EMBEDDING_DIM``-dim vectors in input order.

    Sends requests **sequentially** in batches of ``batch_size``.  Any
    failure — an API error, a timeout/connection drop, a response whose item
    count differs from the batch's, or a vector of the wrong dimension —
    raises :class:`EmbeddingError`.  There is no retry (PRD R3).

    Config is pre-resolved by the caller (``base_url`` / ``api_key`` /
    ``model``) rather than read from ``settings`` inside, so the function can
    also be driven from a throwaway probe script.
    """
    if not texts:
        return []

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                resp = await client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float",
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"embedding request failed for batch of {len(batch)} texts: {exc}"
                ) from exc

            # The OpenAI-compatible contract returns ``index`` precisely because
            # array order is not promised — sort before appending so the result
            # is in request order regardless.
            data = sorted(resp.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise EmbeddingError(
                    f"embedding response has {len(data)} items, expected {len(batch)}"
                )
            for item in data:
                vector = item.embedding
                if len(vector) != EMBEDDING_DIM:
                    raise EmbeddingError(
                        f"embedding vector has {len(vector)} dims, "
                        f"expected {EMBEDDING_DIM}"
                    )
                vectors.append(list(vector))

        return vectors
    finally:
        await client.close()
