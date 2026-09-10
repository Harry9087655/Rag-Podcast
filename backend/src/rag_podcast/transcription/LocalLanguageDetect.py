from __future__ import annotations
import random
import asyncio
import logging
from pathlib import Path
from typing import Protocol
from ..config import settings
import numpy as np
import whisperx
from .BaseTranscriberInterface import TranscribeError
from .LocalFunASR import LocalFunASR
from .LocalWhisperX import LocalWhisperX

logger = logging.getLogger(__name__)



ZH_LANGUAGE_CODES = frozenset({"zh"})


class LanguageRoutingTranscriber:
    """Routes each episode to WhisperX or FunASR based on detected language.

    Loads the episode's audio once, detects language via a small number of
    random windows run through the already-loaded WhisperX model (no extra
    model load), then dispatches the whole episode to whichever backend
    matches. Fully encapsulated behind the ``Transcriber`` Protocol so
    ``worker.py`` needs no changes.
    """

    def __init__(
        self,
        whisperx: LocalWhisperX,
        funasr: LocalFunASR,
        window_seconds: float = 30.0,
        num_detect_windows: int = 2,
    ) -> None:
        self._whisperx = whisperx
        self._funasr = funasr
        self._window_seconds = window_seconds
        self._num_detect_windows = num_detect_windows
        self._rng = random.Random()
        self.audio: np.ndarray | None = None  # set in transcribe() for get_audio_length_seconds()

    async def transcribe(self, audio_path: Path) -> dict:
        try:
            audio = await asyncio.to_thread(whisperx.load_audio, str(audio_path))
            self.audio = audio
            lang = await asyncio.to_thread(
                self.detect_language,
                self._whisperx._model,
                audio,
                self._window_seconds,
                self._rng,
                self._num_detect_windows,
            )
        except Exception as exc:
            raise TranscribeError(
                f"Language detection failed for {audio_path}: {exc}"
            ) from exc

        logger.info("Detected language=%s for %s", lang, audio_path)

        if lang in ZH_LANGUAGE_CODES:
            return await self._funasr.transcribe(audio_path)
        return await self._whisperx.transcribe(audio_path)



    def random_window(self,
        audio: np.ndarray,
        sample_rate: int,
        window_seconds: float,
        rng: random.Random,
    ) -> np.ndarray:
        """Return a random ``window_seconds``-length slice of *audio*.

        If *audio* is shorter than the requested window, the whole array is
        returned unchanged.
        """
        window_len = int(window_seconds * sample_rate)
        n = len(audio)
        if n <= window_len:
            return audio
        start = rng.randrange(0, n - window_len)
        return audio[start : start + window_len]

    def detect_language(self,
        whisperx_model,
        audio: np.ndarray,
        window_seconds: float,
        rng: random.Random,
        num_windows: int = 2,
        batch_size: int = 8,
    ) -> str:
        """Detect the episode's language via up to *num_windows* random samples.

        Draws independent random windows (rather than e.g. always the first
        N seconds) to avoid an unreliable guess from intro music/jingles.
        Returns ``"zh"`` as soon as ANY window detects Chinese (OR logic,
        short-circuits) — a single unlucky window landing on non-speech audio
        should not misroute a real Chinese episode back to WhisperX. If no
        window detects ``"zh"``, returns the last window's detected language
        (defaulting to ``"en"`` if missing).
        """
        last_lang = "en"
        for _ in range(num_windows):
            clip = self.random_window(audio, 16000, window_seconds, rng)
            result = whisperx_model.transcribe(clip, batch_size=batch_size)
            last_lang = result.get("language") or "en"
            if last_lang == "zh":
                return "zh"
        return last_lang

    def get_audio_length_seconds(self, sample_rate: int = 16000) -> float:
        """Return the length of *audio* in seconds."""
        if self.audio is None:
            raise ValueError("Audio not loaded; call transcribe() first.")
        return int(len(self.audio) / sample_rate)
