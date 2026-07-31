from __future__ import annotations

import random

import numpy as np
import pytest

from rag_podcast.transcription.transcriber import (
    LanguageRoutingTranscriber,
    _funasr_to_segments,
    detect_language,
    random_window,
)


# ── _funasr_to_segments ──────────────────────────────────────────────────


def test_funasr_to_segments_converts_ms_to_seconds_and_splits_per_character():
    result = {
        "text": "欢迎大家",
        "sentence_info": [
            {
                "text": "欢迎",
                "start": 880,
                "end": 1360,
                "timestamp": [[880, 1120], [1120, 1360]],
            },
            {
                "text": "大家",
                "start": 1360,
                "end": 1800,
                "timestamp": [[1360, 1580], [1580, 1800]],
            },
        ],
    }

    adapted = _funasr_to_segments(result)

    assert adapted == {
        "language": "zh",
        "segments": [
            {
                "text": "欢迎",
                "start": 0.88,
                "end": 1.36,
                "words": [
                    {"word": "欢", "start": 0.88, "end": 1.12},
                    {"word": "迎", "start": 1.12, "end": 1.36},
                ],
            },
            {
                "text": "大家",
                "start": 1.36,
                "end": 1.8,
                "words": [
                    {"word": "大", "start": 1.36, "end": 1.58},
                    {"word": "家", "start": 1.58, "end": 1.8},
                ],
            },
        ],
    }


def test_funasr_to_segments_empty_sentence_info_returns_empty_segments():
    assert _funasr_to_segments({"text": "", "sentence_info": []}) == {
        "language": "zh",
        "segments": [],
    }


def test_funasr_to_segments_missing_sentence_info_key_returns_empty_segments():
    assert _funasr_to_segments({"text": ""}) == {"language": "zh", "segments": []}


def test_funasr_to_segments_strips_punctuation_before_pairing_with_timestamps():
    # ct-punc inserts a comma into "text" with no corresponding timestamp
    # entry ("timestamp" only covers the two recognized speech characters).
    # A naive zip() would pair "迎" with the timestamp meant for "，" and
    # then run one character out of alignment for the rest of the sentence.
    result = {
        "text": "欢迎，大家",
        "sentence_info": [
            {
                "text": "欢迎，大家",
                "start": 880,
                "end": 1800,
                "timestamp": [[880, 1120], [1120, 1360], [1360, 1580], [1580, 1800]],
            },
        ],
    }

    adapted = _funasr_to_segments(result)

    assert adapted["segments"][0]["text"] == "欢迎，大家"
    assert [w["word"] for w in adapted["segments"][0]["words"]] == [
        "欢",
        "迎",
        "大",
        "家",
    ]
    assert adapted["segments"][0]["words"][2] == {
        "word": "大",
        "start": 1.36,
        "end": 1.58,
    }


def test_funasr_to_segments_logs_warning_on_residual_length_mismatch(caplog):
    # Even after stripping punctuation, a length mismatch can still occur
    # (e.g. a non-punctuation char FunASR didn't timestamp) — this should
    # be surfaced via a warning log rather than silently misaligning.
    result = {
        "text": "欢迎大家",
        "sentence_info": [
            {
                "text": "欢迎大家",
                "start": 880,
                "end": 1360,
                "timestamp": [[880, 1120], [1120, 1360]],
            },
        ],
    }

    with caplog.at_level("WARNING", logger="rag_podcast.transcription.transcriber"):
        adapted = _funasr_to_segments(result)

    assert len(adapted["segments"][0]["words"]) == 2
    assert any("mismatch" in message for message in caplog.messages)


# ── random_window ─────────────────────────────────────────────────────────


def test_random_window_shorter_than_window_returns_whole_array_unchanged():
    audio = np.arange(100, dtype=np.float32)
    rng = random.Random(0)

    result = random_window(audio, sample_rate=16000, window_seconds=30.0, rng=rng)

    assert result is audio


def test_random_window_equal_length_returns_whole_array_unchanged():
    sample_rate = 16000
    window_seconds = 1.0
    audio = np.arange(sample_rate, dtype=np.float32)
    rng = random.Random(0)

    result = random_window(audio, sample_rate=sample_rate, window_seconds=window_seconds, rng=rng)

    assert result is audio


def test_random_window_longer_than_window_returns_correct_length_slice_within_bounds():
    sample_rate = 16000
    window_seconds = 1.0
    window_len = int(window_seconds * sample_rate)
    audio = np.arange(sample_rate * 5, dtype=np.float32)
    rng = random.Random(42)

    result = random_window(audio, sample_rate=sample_rate, window_seconds=window_seconds, rng=rng)

    assert len(result) == window_len
    # The slice must be a contiguous run of the original array's values.
    start = int(result[0])
    assert np.array_equal(result, audio[start : start + window_len])
    assert 0 <= start <= len(audio) - window_len


def test_random_window_is_deterministic_given_seeded_rng():
    sample_rate = 16000
    audio = np.arange(sample_rate * 5, dtype=np.float32)

    result_a = random_window(audio, sample_rate, 1.0, random.Random(7))
    result_b = random_window(audio, sample_rate, 1.0, random.Random(7))

    assert np.array_equal(result_a, result_b)


# ── detect_language ────────────────────────────────────────────────────────


class _FakeWhisperXModel:
    """Fake whisperx model whose .transcribe() returns scripted results."""

    def __init__(self, languages: list[str]):
        self._languages = list(languages)
        self.calls = 0

    def transcribe(self, clip, batch_size=8):
        lang = self._languages[self.calls]
        self.calls += 1
        return {"language": lang}


def test_detect_language_short_circuits_when_first_window_is_zh():
    model = _FakeWhisperXModel(["zh", "en"])
    rng = random.Random(0)
    audio = np.arange(16000 * 60, dtype=np.float32)

    lang = detect_language(model, audio, window_seconds=30.0, rng=rng, num_windows=2)

    assert lang == "zh"
    assert model.calls == 1


def test_detect_language_short_circuits_when_only_second_window_is_zh():
    model = _FakeWhisperXModel(["en", "zh"])
    rng = random.Random(0)
    audio = np.arange(16000 * 60, dtype=np.float32)

    lang = detect_language(model, audio, window_seconds=30.0, rng=rng, num_windows=2)

    assert lang == "zh"
    assert model.calls == 2


def test_detect_language_returns_last_window_language_when_none_are_zh():
    model = _FakeWhisperXModel(["en", "fr"])
    rng = random.Random(0)
    audio = np.arange(16000 * 60, dtype=np.float32)

    lang = detect_language(model, audio, window_seconds=30.0, rng=rng, num_windows=2)

    assert lang == "fr"
    assert model.calls == 2


def test_detect_language_never_calls_more_than_num_windows_times():
    model = _FakeWhisperXModel(["en", "en", "en"])
    rng = random.Random(0)
    audio = np.arange(16000 * 60, dtype=np.float32)

    detect_language(model, audio, window_seconds=30.0, rng=rng, num_windows=2)

    assert model.calls == 2


def test_detect_language_defaults_to_en_when_language_key_missing():
    class _NoLangModel:
        def transcribe(self, clip, batch_size=8):
            return {}

    audio = np.arange(16000 * 60, dtype=np.float32)
    lang = detect_language(_NoLangModel(), audio, 30.0, random.Random(0), num_windows=1)

    assert lang == "en"


# ── LanguageRoutingTranscriber ──────────────────────────────────────────────


class _FakeDelegate:
    """Fake Transcriber-shaped stand-in recording whether it was called."""

    def __init__(self, sentinel: dict):
        self.sentinel = sentinel
        self.called_with = None

    async def transcribe(self, audio_path):
        self.called_with = audio_path
        return self.sentinel


@pytest.mark.asyncio
async def test_language_routing_transcriber_dispatches_to_funasr_when_zh(monkeypatch, tmp_path):
    whisperx_stub = _FakeDelegate({"language": "en", "segments": []})
    whisperx_stub._model = object()
    funasr_stub = _FakeDelegate({"language": "zh", "segments": []})

    router = LanguageRoutingTranscriber(whisperx_stub, funasr_stub)

    import rag_podcast.transcription.transcriber as transcriber_module

    monkeypatch.setattr(transcriber_module.whisperx, "load_audio", lambda path: np.zeros(10))
    monkeypatch.setattr(transcriber_module, "detect_language", lambda *a, **kw: "zh")

    audio_path = tmp_path / "episode.mp3"
    result = await router.transcribe(audio_path)

    assert result is funasr_stub.sentinel
    assert funasr_stub.called_with == audio_path
    assert whisperx_stub.called_with is None


@pytest.mark.asyncio
async def test_language_routing_transcriber_dispatches_to_whisperx_when_not_zh(monkeypatch, tmp_path):
    whisperx_stub = _FakeDelegate({"language": "en", "segments": []})
    whisperx_stub._model = object()
    funasr_stub = _FakeDelegate({"language": "zh", "segments": []})

    router = LanguageRoutingTranscriber(whisperx_stub, funasr_stub)

    import rag_podcast.transcription.transcriber as transcriber_module

    monkeypatch.setattr(transcriber_module.whisperx, "load_audio", lambda path: np.zeros(10))
    monkeypatch.setattr(transcriber_module, "detect_language", lambda *a, **kw: "en")

    audio_path = tmp_path / "episode.mp3"
    result = await router.transcribe(audio_path)

    assert result is whisperx_stub.sentinel
    assert whisperx_stub.called_with == audio_path
    assert funasr_stub.called_with is None


@pytest.mark.asyncio
async def test_language_routing_transcriber_wraps_detection_failure_as_transcribe_error(
    monkeypatch, tmp_path
):
    from rag_podcast.transcription.transcriber import TranscribeError

    whisperx_stub = _FakeDelegate({"language": "en", "segments": []})
    whisperx_stub._model = object()
    funasr_stub = _FakeDelegate({"language": "zh", "segments": []})

    router = LanguageRoutingTranscriber(whisperx_stub, funasr_stub)

    import rag_podcast.transcription.transcriber as transcriber_module

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(transcriber_module.whisperx, "load_audio", lambda path: np.zeros(10))
    monkeypatch.setattr(transcriber_module, "detect_language", _raise)

    with pytest.raises(TranscribeError):
        await router.transcribe(tmp_path / "episode.mp3")
