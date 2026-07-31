"""Transcriber abstraction: Protocol + LocalWhisperX implementation.

The Transcriber Protocol allows swapping implementations without changing
the worker loop.  LocalWhisperX wraps the whisperx library (sync, GPU-bound)
and runs transcription in ``asyncio.to_thread`` so it never blocks the
event loop permanently.
"""

from __future__ import annotations

import asyncio
import logging
import random
import string
from pathlib import Path
from typing import Protocol
from ..config import settings
import funasr
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import numpy as np
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


def create_transcriber(
    model: str,
    device: str,
    compute_type: str,
    *,
    batch_size: int = 8,
    funasr_model: str = settings.funasr_model,
    funasr_vad_model: str = settings.funasr_vad_model,
    funasr_punc_model: str = settings.funasr_punc_model,
    funasr_device: str = settings.funasr_device,
    lang_detect_window_seconds: float = settings.lang_detect_window_seconds,
) -> Transcriber:
    """Factory: build both underlying models and the language-routing wrapper.

    ``LocalWhisperX`` handles everything that isn't detected as Chinese;
    ``LocalFunASR`` handles Chinese. ``LanguageRoutingTranscriber`` performs
    the per-episode detection and dispatches to one or the other, fully
    behind the ``Transcriber`` Protocol so callers (the worker loop) never
    need to change.
    """
    whisperx_transcriber = LocalWhisperX(
        model=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
    )
    funasr_transcriber = LocalFunASR(
        model=funasr_model,
        vad_model=funasr_vad_model,
        punc_model=funasr_punc_model,
        device=funasr_device,
    )
    return LanguageRoutingTranscriber(
        whisperx_transcriber,
        funasr_transcriber,
        window_seconds=lang_detect_window_seconds,
    )


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


def random_window(
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


def detect_language(
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
        clip = random_window(audio, 16000, window_seconds, rng)
        result = whisperx_model.transcribe(clip, batch_size=batch_size)
        last_lang = result.get("language") or "en"
        if last_lang == "zh":
            return "zh"
    return last_lang


_FUNASR_PUNCTUATION = frozenset(
    string.punctuation + "，。！？；：“”‘’（）《》【】、…—～·"
)


def _funasr_to_segments(result: dict) -> dict:
    """Adapt FunASR's ``sentence_info``-shaped output to the WhisperX-shaped
    ``segments``/``words`` contract this codebase's downstream code expects
    (see ``Transcriber.transcribe``'s docstring for that contract).

    FunASR has no equivalent of WhisperX's per-segment ``avg_logprob`` or
    per-word ``score`` confidence values, so those keys are omitted rather
    than faked — consistent with the real contract, where both are only
    ever present when the underlying model actually produced them. The
    returned dict also has no top-level ``word_segments`` (the flattened
    word list); nothing in this codebase currently reads it.

    FunASR reports ``start``/``end``/``timestamp`` in milliseconds; the
    contract used elsewhere in this codebase is seconds. Each character in
    a sentence's ``text`` is paired with its own timestamp pair to build a
    per-character ``words`` list — this matches the char-level granularity
    ``cleaning/text_join.py`` and ``cleaning/filler_words.py`` already
    special-case for ``"zh"``.

    ``ct-punc`` (the punctuation-restoration model) inserts punctuation
    characters into ``text`` that have no corresponding entry in
    ``timestamp`` (which only covers recognized speech characters), so
    ``len(text) != len(timestamp)`` is an expected, not exceptional, case.
    Naively zipping the two would silently misalign every character after
    the first inserted punctuation mark. Punctuation characters are
    stripped out before zipping so the remaining (non-punctuation)
    characters line up positionally with their timestamps; ``segments[
    ].text`` still keeps the original punctuation-inclusive text verbatim,
    only the per-character ``words`` list omits punctuation. This mirrors
    the already-accepted lossiness of reconstructing text from ``words``
    elsewhere in this codebase (see ``cleaning/filler_words.py``'s
    docstring note that a fresh word-join doesn't preserve the original's
    exact punctuation/whitespace layout).
    """
    segments = []
    for sentence in result.get("sentence_info", ()):
        text = sentence.get("text", "")
        timestamps = sentence.get("timestamp", ())
        chars = [char for char in text if char not in _FUNASR_PUNCTUATION]
        if len(chars) != len(timestamps):
            logger.warning(
                "FunASR sentence_info text/timestamp length mismatch "
                "(text=%d chars, %d non-punctuation, %d timestamps) — "
                "word-level timestamps for this sentence may be misaligned: %r",
                len(text),
                len(chars),
                len(timestamps),
                text,
            )
        words = [
            {"word": char, "start": start_ms / 1000, "end": end_ms / 1000}
            for char, (start_ms, end_ms) in zip(chars, timestamps)
        ]
        segments.append(
            {
                "text": text,
                "start": sentence["start"] / 1000,
                "end": sentence["end"] / 1000,
                "words": words,
            }
        )
    return {"language": "zh", "segments": segments}


class LocalFunASR:
    """Wraps the local funasr library (Paraformer + VAD + punctuation).

    Model is loaded eagerly at init, mirroring ``LocalWhisperX``.
    Transcription runs in ``asyncio.to_thread`` to keep the event loop free.
    """

    def __init__(
        self,
        model: str,
        vad_model: str,
        punc_model: str,
        device: str,
    ) -> None:
        self._model_name = model
        self._device = device

        logger.info(
            "Loading FunASR model=%s vad_model=%s punc_model=%s device=%s …",
            model,
            vad_model,
            punc_model,
            device,
        )
        self._model = funasr.AutoModel(
            model=model,
            vad_model=vad_model,
            punc_model=punc_model,
            device=device,
            disable_update=True,
        )
        logger.info("FunASR model loaded.")

    async def transcribe(self, audio_path: Path) -> dict:
        """Transcribe *audio_path* and return WhisperX-shaped segments."""
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    # ── sync helpers (run inside asyncio.to_thread) ────────────────────────

    def _transcribe_sync(self, audio_path: Path) -> dict:
        try:
            result = self._model.generate(input=str(audio_path),
                                          batch_size_s=300,
                                          sentence_timestamp=True,
                                          pred_timestamp=True
                                          )
            logger.info("Type of FunASR result: %s", type(result))
            logger.info("Length of FunASR result: %d", len(result))
            for item in result:
                item["text"] = rich_transcription_postprocess(item["text"])
        except Exception as exc:
            raise TranscribeError(
                f"FunASR transcription failed for {audio_path}: {exc}"
            ) from exc

        adapted = _funasr_to_segments(result[0])
        logger.info(
            "FunASR transcription complete — %d segments.",
            len(adapted.get("segments", ())),
        )
        return adapted


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

    async def transcribe(self, audio_path: Path) -> dict:
        try:
            audio = await asyncio.to_thread(whisperx.load_audio, str(audio_path))
            lang = await asyncio.to_thread(
                detect_language,
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
