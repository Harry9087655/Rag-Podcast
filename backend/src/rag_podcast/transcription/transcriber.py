"""Transcriber abstraction: Protocol + LocalWhisperX implementation.

The Transcriber Protocol allows swapping implementations without changing
the worker loop.  LocalWhisperX wraps the whisperx library (sync, GPU-bound)
and runs transcription in ``asyncio.to_thread`` so it never blocks the
event loop permanently.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol

import whisperx

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
        """Transcribe audio and return a WhisperX result dict.

        Expected return shape::

            {
                "model": "small",
                "language": "en",
                "segments": [
                    {
                        "text": "…",
                        "start": 0.0,
                        "end": 3.2,
                        "words": [
                            {"word": "…", "start": 0.0, "end": 0.5},
                            …
                        ],
                    },
                    …
                ],
            }
        """
        ...


class LocalWhisperX:
    """Wraps the local whisperx library.

    Model is loaded eagerly at init (first transcription is fast).
    Transcription runs in ``asyncio.to_thread`` to keep the event loop free.
    """

    def __init__(
        self,
        model: str,
        device: str,
        compute_type: str,
        batch_size: int = 8,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._batch_size = batch_size

        logger.info(
            "Loading WhisperX model=%s device=%s compute_type=%s …",
            model,
            device,
            compute_type,
        )
        self._model = whisperx.load_model(model, device, compute_type=compute_type)
        logger.info("WhisperX model loaded.")

    async def transcribe(self, audio_path: Path) -> dict:
        """Transcribe *audio_path* and return aligned segments (word-level)."""
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    # ── sync helpers (run inside asyncio.to_thread) ────────────────────────

    def _transcribe_sync(self, audio_path: Path) -> dict:
        try:
            audio = whisperx.load_audio(str(audio_path))
        except Exception as exc:
            raise TranscribeError(
                f"Failed to load audio from {audio_path}: {exc}"
            ) from exc

        try:
            result = self._model.transcribe(audio, batch_size=self._batch_size)
        except Exception as exc:
            raise TranscribeError(
                f"WhisperX transcription failed for {audio_path}: {exc}"
            ) from exc

        logger.info(
            "Transcription complete — language=%s segments=%d",
            result.get("language", "?"),
            len(result.get("segments", ())),
        )

        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self._device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )
        except Exception as exc:
            raise TranscribeError(
                f"Alignment failed for {audio_path}: {exc}"
            ) from exc

        logger.info(
            "Alignment complete — %d aligned segments.", len(result.get("segments", ()))
        )
        return result
