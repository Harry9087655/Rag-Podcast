from __future__ import annotations

import logging

from .BaseChunkerInterface import ChunkSpan

logger = logging.getLogger(__name__)


class SimilarityBiasedChunker:
    """Phase 2 slot — reserved, not implemented yet.

    Planned behaviour: within the same min/max token and duration bounds
    ``SizeBasedChunker`` enforces, bias *which* candidate boundary a chunk
    closes at toward the point of highest embedding distance between
    adjacent units (punctuation-delimited sentences, not raw segments),
    instead of Phase 1's nearest-token-target criterion. Boundary vectors
    are ephemeral — computed via the existing hosted embedding API and
    never persisted — so a cheaper model than the storage one is an option.

    Not gated on the retrieval eval harness existing first; it is meant to
    drop in as a ``Chunker`` once there is a way to measure whether it
    actually beats Phase 1.
    """

    def build_chunks(
        self,
        segments: list[dict],
        *,
        min_tokens: int,
        max_tokens: int,
        desired_tokens: int,
        min_duration: float,
        max_duration: float,
        language: str | None = None,
    ) -> list[ChunkSpan]:
        raise NotImplementedError(
            "SimilarityBiasedChunker is a Phase 2 placeholder; "
            "use strategy='size_based' for now."
        )
