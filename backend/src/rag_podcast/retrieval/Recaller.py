"""Recaller factory: build the configured recall strategy once.

The single module that imports the concrete recaller — mirroring
``Chunker.py::create_chunker`` / ``Transcriber.py::create_transcriber``.
``search()`` calls :func:`create_recaller` once and passes the result through
the pipeline, so the concrete class is never referenced elsewhere.

Design contract: .trellis/tasks/09-12-retrieval/design.md
"""

from __future__ import annotations

from .BaseRecallerInterface import Recaller
from .PgvectorRecaller import PgvectorRecaller


def create_recaller(strategy: str = "pgvector") -> Recaller:
    if strategy == "pgvector":
        return PgvectorRecaller()
    raise ValueError(f"unknown recall strategy: {strategy!r}")
