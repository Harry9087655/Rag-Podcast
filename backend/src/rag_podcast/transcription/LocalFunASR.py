from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Protocol
from ..config import settings
import funasr
from .BaseTranscriberInterface import TranscribeError

logger = logging.getLogger(__name__)

# Punctuation ``ct-punc`` can emit. Its own ``punc_list`` is just ``_，。？、``,
# but funasr's CT-Transformer renderer rewrites a mark to its half-width form
# whenever the token before it is ASCII ("CP.", "AI,"), so both widths have to
# be recognised. The remaining marks are defensive — they cost nothing and keep
# the tokenizer honest if a future punctuation model widens its inventory.
_FULL_WIDTH_PUNCTUATION = frozenset("，。？！、；：…")
_HALF_WIDTH_PUNCTUATION = frozenset(",.?!;:")

# Marks that close a segment. ``，`` and ``、`` deliberately stay *inside* a
# segment: chunks are assembled by merging whole segments, so splitting on every
# mark lets a chunk boundary land mid-sentence, while splitting only on
# sentence-final marks makes every chunk end on a complete sentence. Measured
# over a 108-minute episode this is 1008 segments (median 5.5s, max 54.6s)
# rather than 3238 (median 1.6s) — both comfortably inside
# ``chunk_max_duration_seconds``.
_SEGMENT_FINAL_PUNCTUATION = frozenset("。？！.?!")


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
            # ``rich_transcription_postprocess`` is deliberately NOT applied
            # here. It is a SenseVoice helper: on any ``<|...|>`` tag it injects
            # emoji into the text (see funasr's postprocess_utils), and those
            # characters carry no timestamp, which would break the token /
            # timestamp alignment ``_funasr_to_segments`` relies on.
            result = self._model.generate(input=str(audio_path),
                                          batch_size_s=300,
                                          sentence_timestamp=True,
                                          pred_timestamp=True
                                          )
        except Exception as exc:
            raise TranscribeError(
                f"FunASR transcription failed for {audio_path}: {exc}"
            ) from exc

        adapted = self._funasr_to_segments(result[0])
        logger.info(
            "FunASR transcription complete — %d segments.",
            len(adapted.get("segments", ())),
        )
        return adapted

    @staticmethod
    def _is_punctuation(char: str, next_char: str) -> bool:
        """Whether *char* is a mark ``ct-punc`` inserted, given the char after it.

        Full-width marks are unambiguous. The half-width forms are not: ``.``,
        ``,`` and ``?`` are ASCII, so the ASCII-run rule in ``_tokenize`` would
        otherwise swallow them into the preceding token ("CP." as one token).

        The lookahead resolves that, and is sound because funasr's renderer
        always puts a space between two adjacent ASCII tokens:

        * ``3. 5`` — an inserted period, next char is a space → punctuation.
        * ``3.5`` — a decimal the ASR emitted as one token, next char is an
          ASCII digit → part of the token.

        Note the test is *ASCII* alphanumeric: ``str.isalnum()`` alone is true
        for CJK characters too, which would misread the ``.`` in "CP.哎".
        """
        if char in _FULL_WIDTH_PUNCTUATION:
            return True
        if char not in _HALF_WIDTH_PUNCTUATION:
            return False
        return not (next_char.isascii() and next_char.isalnum())

    def _tokenize(self, text: str) -> list[tuple[str, str]]:
        """Split *text* into ``(token, trailing_punctuation)`` pairs.

        Mirrors funasr's ``split_words`` so that one token consumes exactly one
        ``timestamp`` entry — that is the whole point of this function, and the
        invariant ``_funasr_to_segments`` checks. The two rules are:

        * a non-ASCII (CJK) character is a token on its own;
        * a run of ASCII characters ("CP", "driver", "AI") is one token,
          terminated by whitespace or by a CJK character.

        Punctuation is not a token — it is attached to the token it follows, so
        the word list rejoins losslessly into the original text.
        """
        tokens: list[list[str]] = []
        ascii_run = ""

        def flush() -> None:
            nonlocal ascii_run
            if ascii_run:
                tokens.append([ascii_run, ""])
                ascii_run = ""

        for index, char in enumerate(text):
            if self._is_punctuation(char, text[index + 1 : index + 2]):
                flush()
                if tokens:
                    tokens[-1][1] += char
                # A mark with no token before it has nothing to attach to and
                # no timestamp of its own; dropping it keeps the count honest.
                continue
            if char.isspace():
                flush()
                continue
            if char.isascii():
                ascii_run += char
                continue
            flush()
            tokens.append([char, ""])
        flush()

        return [(token, punctuation) for token, punctuation in tokens]

    def _funasr_to_segments(self, result: dict) -> dict:
        """Adapt FunASR's native output to the WhisperX-shaped
        ``segments``/``words`` contract this codebase's downstream code expects
        (see ``Transcriber.transcribe``'s docstring for that contract).

        Built from ``result["text"]`` and ``result["timestamp"]`` only.
        ``result["sentence_info"]`` is deliberately ignored: funasr 1.3.30
        assembles it with a tokenizer that disagrees with the one its own
        punctuation array is indexed by, so its text drifts a token further out
        of place at every VAD-chunk boundary and its tail loses text entirely.
        The two fields used here share one tokenization (funasr's
        ``split_words``), which ``_tokenize`` reproduces. See
        ``funasr_segments_redesign.md`` for the full analysis.

        FunASR reports timestamps in milliseconds; the contract used elsewhere
        in this codebase is seconds.

        FunASR has no equivalent of WhisperX's per-segment ``avg_logprob`` or
        per-word ``score`` confidence values, so those keys are omitted rather
        than faked — consistent with the real contract, where both are only
        ever present when the underlying model actually produced them. The
        returned dict also has no top-level ``word_segments`` (the flattened
        word list); nothing in this codebase currently reads it.

        Each word keeps the punctuation that follows it ("对，"), the way
        WhisperX does for English, so ``join_words(words, "zh")`` reproduces the
        segment text exactly. ``cleaning.clean_words`` rebuilds a segment's
        ``text`` from its surviving words, and ``SizeBasedChunker`` looks for a
        sentence-final mark on a word to snap a cut onto — both are silently
        wrong if punctuation lives only in ``text``.
        """
        text = result.get("text", "")
        timestamps = result.get("timestamp") or []

        if not text.strip():
            if timestamps:
                logger.warning(
                    "FunASR returned %d timestamps but no text — emitting no "
                    "segments.",
                    len(timestamps),
                )
            return {"language": "zh", "segments": []}

        tokens = self._tokenize(text)
        if len(tokens) != len(timestamps):
            # Deterministic: retrying transcription cannot fix it. Failing the
            # episode is cheaper than storing a transcript whose timestamps are
            # plausible but wrong, which is exactly how the sentence_info bug
            # this replaces went unnoticed.
            raise TranscribeError(
                f"FunASR token/timestamp mismatch: {len(tokens)} tokens from "
                f"text, {len(timestamps)} timestamps. Text starts: {text[:120]!r}"
            )

        segments: list[dict] = []
        words: list[dict] = []
        for (token, punctuation), (start_ms, end_ms) in zip(tokens, timestamps):
            words.append(
                {
                    "word": token + punctuation,
                    "start": start_ms / 1000,
                    "end": end_ms / 1000,
                }
            )
            if punctuation and punctuation[-1] in _SEGMENT_FINAL_PUNCTUATION:
                segments.append(
                    {
                        "text": "".join(word["word"] for word in words),
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "words": words,
                    }
                )
                words = []

        if words:
            # Trailing run with no closing mark — funasr forces a sentence-final
            # mark onto the last token, so this is a safety net rather than the
            # common case.
            segments.append(
                {
                    "text": "".join(word["word"] for word in words),
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "words": words,
                }
            )

        return {"language": "zh", "segments": segments}