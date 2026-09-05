from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ChunkingError(Exception):
    """Raised when chunking fails (malformed segments, unsplittable span, …)."""


@dataclass
class ChunkSpan:
    """One contiguous slice of an episode, ready to embed and store.

    ``text`` is already cleaned: the caller runs ``clean_words`` over the
    whole transcript *before* chunking, so a chunker only ever concatenates
    the segments' own ``text`` fields — it never rebuilds text from a word
    list (that path is where the CJK "H e l l o" spacing bug lives, and it
    is handled once, upstream, in ``cleaning/``).

    ``segments`` keeps segment shape rather than a flat word list because
    word-level timestamps are what ``start``/``end`` are derived from, and
    the oversized-segment fallback needs per-word timing to pick a split
    point. Where a flat word list is actually wanted (contiguity checks),
    derive it on demand::

        [w for seg in span.segments for w in seg["words"]]
    """

    text: str
    start: float
    end: float
    segments: list[dict]


class Chunker(Protocol):
    """Abstract chunking interface.

    ``SizeBasedChunker`` closes chunks on token/duration bounds alone.
    A future ``SimilarityBiasedChunker`` will bias the boundary toward the
    point of highest embedding distance within those same bounds. Both
    return the same ``list[ChunkSpan]``, so downstream code (``embedder``,
    ``store``, ``worker``) imports only this module's types and never
    branches on which strategy ran.

    Synchronous by design: chunking is pure CPU work over an in-memory
    transcript, with no I/O to await.
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
        language: str | None = 'en',
    ) -> list[ChunkSpan]:
        """Chunk an episode's **already-cleaned** segments into spans.

        ``segments`` are WhisperX-shaped (see ``BaseTranscriberInterface``)
        and must arrive in chronological order. The returned spans cover the
        whole episode contiguously: flattening every span's ``segments`` back
        into words reconstructs the original word sequence with nothing
        skipped and nothing duplicated.

        The token bounds arrive pre-resolved — the caller picks a tier from
        the episode's runtime (``settings.audio_tier``), so implementations
        hold no tiering logic of their own. ``language`` is only a hint for
        joining words back into text when a span has to be synthesised
        mid-segment; ``None`` means "space-join", the safe default for
        space-delimited scripts.

        Raises:
            ChunkingError: if the segments cannot be chunked as given.
        """
        ...
