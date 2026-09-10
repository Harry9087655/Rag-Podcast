"""Chunker factory: build the configured chunking strategy once.

The single module that imports both concrete chunker strategies.  ``cli.py``
calls :func:`create_chunker` once at startup and passes the result into
``run_worker``, so the tokenizer is loaded exactly once per process rather
than once per episode — mirroring ``Transcriber.py::create_transcriber``'s
eager-load-once precedent.

Design contract: .trellis/tasks/09-05-embedding-pipeline/design.md
"""

from __future__ import annotations

from .BaseChunkerInterface import Chunker
from .SimilarityBiasedChunker import SimilarityBiasedChunker
from .SizeBasedChunker import SizeBasedChunker
from .tokenizer import get_tokenizer


def create_chunker(strategy: str = "size_based") -> Chunker:
    if strategy == "size_based":
        return SizeBasedChunker(get_tokenizer())
    if strategy == "similarity_biased":
        return SimilarityBiasedChunker()
    raise ValueError(f"unknown chunking strategy: {strategy!r}")
