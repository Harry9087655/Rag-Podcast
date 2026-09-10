from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol
from ..config import settings
import numpy as np

logger = logging.getLogger(__name__)


class TranscribeError(Exception):
    """Raised when transcription fails (model crash, OOM, corrupt audio, …)."""


class Transcriber(Protocol):
    """Abstract transcription interface.

    ``LocalWhisperX`` wraps the whisperx library (sync, GPU-bound).
    A future ``WhisperXAPI`` would make HTTP calls (async, I/O-bound).
    The protocol is async so the worker doesn't need to change when we swap.
    """

    async def transcribe(self, audio_path: Path) -> dict:
        """Transcribe audio and return a WhisperX-shaped result dict.

        Matches ``whisperx.align()``'s actual return shape (there is no
        ``"model"`` key; ``"language"`` is merged back in separately since
        alignment drops it)::

            {
                "language": "en",
                "segments": [
                    {
                        "text": "…",
                        "start": 0.0,
                        "end": 3.2,
                        "words": [
                            {"word": "…", "start": 0.0, "end": 0.5, "score": 0.9},
                            …
                        ],
                        "avg_logprob": -0.16,  # omitted when unavailable
                    },
                    …
                ],
                "word_segments": [...],  # flattened words across all segments
            }

        ``avg_logprob`` (per segment) and ``start``/``end``/``score`` (per
        word) are only present when the underlying model actually produced
        them — e.g. a word whose characters couldn't be aligned may appear
        as bare ``{"word": "…"}``. Callers must not assume these keys exist.
        """
        ...