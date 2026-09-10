
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Protocol
from ..config import settings
import numpy as np
import whisperx
from .BaseTranscriberInterface import TranscribeError

logger = logging.getLogger(__name__)




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
            transcribe_result = self._model.transcribe(
                audio, batch_size=self._batch_size
            )
        except Exception as exc:
            raise TranscribeError(
                f"WhisperX transcription failed for {audio_path}: {exc}"
            ) from exc

        language = transcribe_result.get("language")
        logger.info(
            "Transcription complete — language=%s segments=%d",
            language or "?",
            len(transcribe_result.get("segments", ())),
        )

        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=self._device
            )
            aligned = whisperx.align(
                transcribe_result["segments"],
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

        # whisperx.align() only returns {"segments": ..., "word_segments": ...}
        # — it drops "language", so merge it back in from the pre-align result.
        aligned["language"] = language

        logger.info(
            "Alignment complete — %d aligned segments.",
            len(aligned.get("segments", ())),
        )
        return aligned