from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_podcast.cleaning.text_join import join_words
from rag_podcast.transcription.BaseTranscriberInterface import TranscribeError
from rag_podcast.transcription.LocalFunASR import LocalFunASR

FIXTURE = Path(__file__).parent / "fixtures" / "funasr_raw_excerpt.json"


@pytest.fixture
def adapter():
    """A LocalFunASR whose __init__ never ran.

    `_tokenize` / `_funasr_to_segments` are pure functions of their arguments —
    they touch no instance state — but they live on the class, and running the
    real __init__ would download and load three models.
    """
    return object.__new__(LocalFunASR)


def _stamps(count: int, step: int = 100) -> list[list[int]]:
    """`count` back-to-back [start_ms, end_ms] pairs."""
    return [[i * step, (i + 1) * step] for i in range(count)]


# ── _tokenize ────────────────────────────────────────────────────────────


def test_tokenize_gives_one_token_per_cjk_character(adapter):
    assert adapter._tokenize("你好吗") == [("你", ""), ("好", ""), ("吗", "")]


def test_tokenize_keeps_an_ascii_run_as_a_single_token(adapter):
    # "CP" is one timestamped token, not two — this is what makes the token
    # count line up with FunASR's per-token timestamp list.
    assert adapter._tokenize("看CP哎") == [("看", ""), ("CP", ""), ("哎", "")]


def test_tokenize_splits_adjacent_ascii_tokens_on_whitespace(adapter):
    assert adapter._tokenize("open AI") == [("open", ""), ("AI", "")]


def test_tokenize_attaches_punctuation_to_the_preceding_token(adapter):
    assert adapter._tokenize("对，好。") == [("对", "，"), ("好", "。")]


def test_tokenize_treats_half_width_mark_after_ascii_token_as_punctuation(adapter):
    # FunASR's renderer emits "." rather than "。" when the token before it is
    # ASCII. The next character is CJK, so it is a mark, not part of "CP".
    assert adapter._tokenize("的CP.哎") == [("的", ""), ("CP", "."), ("哎", "")]


def test_tokenize_keeps_a_decimal_point_inside_its_token(adapter):
    # Next character is an ASCII digit, so this is one ASR token, not "3" + ".".
    assert adapter._tokenize("3.5米") == [("3.5", ""), ("米", "")]


def test_tokenize_drops_a_leading_mark_that_has_no_token_to_attach_to(adapter):
    # It would otherwise need a timestamp of its own and break the count.
    assert adapter._tokenize("，你好") == [("你", ""), ("好", "")]


def test_tokenize_empty_text_gives_no_tokens(adapter):
    assert adapter._tokenize("") == []


# ── _funasr_to_segments ──────────────────────────────────────────────────


def test_converts_milliseconds_to_seconds(adapter):
    result = {"text": "你好。", "timestamp": [[0, 250], [250, 1500]]}

    adapted = adapter._funasr_to_segments(result)

    assert adapted["language"] == "zh"
    assert adapted["segments"] == [
        {
            "text": "你好。",
            "start": 0.0,
            "end": 1.5,
            "words": [
                {"word": "你", "start": 0.0, "end": 0.25},
                {"word": "好。", "start": 0.25, "end": 1.5},
            ],
        }
    ]


def test_splits_on_sentence_final_marks_only(adapter):
    # The comma stays inside its segment; the period closes one.
    result = {"text": "你好，早。晚。", "timestamp": _stamps(4)}

    texts = [s["text"] for s in adapter._funasr_to_segments(result)["segments"]]

    assert texts == ["你好，早。", "晚。"]


def test_half_width_period_also_closes_a_segment(adapter):
    result = {"text": "的CP.哎。", "timestamp": _stamps(3)}

    texts = [s["text"] for s in adapter._funasr_to_segments(result)["segments"]]

    assert texts == ["的CP.", "哎。"]


def test_trailing_run_without_a_closing_mark_is_still_emitted(adapter):
    result = {"text": "你好。晚", "timestamp": _stamps(3)}

    segments = adapter._funasr_to_segments(result)["segments"]

    assert [s["text"] for s in segments] == ["你好。", "晚"]
    assert segments[-1]["end"] == 0.3


def test_segment_span_comes_from_its_first_and_last_word(adapter):
    result = {"text": "你好。", "timestamp": [[500, 700], [900, 2000]]}

    segment = adapter._funasr_to_segments(result)["segments"][0]

    assert (segment["start"], segment["end"]) == (0.5, 2.0)


def test_token_timestamp_mismatch_raises(adapter):
    result = {"text": "你好吗。", "timestamp": _stamps(2)}

    with pytest.raises(TranscribeError, match="token/timestamp mismatch"):
        adapter._funasr_to_segments(result)


def test_missing_timestamp_key_with_text_raises(adapter):
    with pytest.raises(TranscribeError, match="token/timestamp mismatch"):
        adapter._funasr_to_segments({"text": "你好。"})


def test_empty_text_returns_no_segments(adapter):
    assert adapter._funasr_to_segments({"text": "", "timestamp": []}) == {
        "language": "zh",
        "segments": [],
    }


def test_whitespace_only_text_returns_no_segments_without_raising(adapter):
    assert adapter._funasr_to_segments({"text": "   "}) == {
        "language": "zh",
        "segments": [],
    }


def test_sentence_info_is_ignored_even_when_present(adapter):
    # funasr 1.3.30 fills sentence_info with drifted text; it must not be read.
    result = {
        "text": "你好。",
        "timestamp": [[0, 100], [100, 200]],
        "sentence_info": [{"text": "WRONG。", "start": 0, "end": 9999, "timestamp": []}],
    }

    segments = adapter._funasr_to_segments(result)["segments"]

    assert [s["text"] for s in segments] == ["你好。"]
    assert segments[0]["end"] == 0.2


# ── golden fixture: a real FunASR excerpt ────────────────────────────────


@pytest.fixture
def raw_excerpt():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_every_token_gets_exactly_one_timestamp(adapter, raw_excerpt):
    assert len(adapter._tokenize(raw_excerpt["text"])) == len(raw_excerpt["timestamp"])


def test_fixture_words_rejoin_losslessly_into_the_original_text(adapter, raw_excerpt):
    words = [
        w for s in adapter._funasr_to_segments(raw_excerpt)["segments"] for w in s["words"]
    ]

    assert join_words(words, "zh") == "".join(raw_excerpt["text"].split())


def test_fixture_reproduces_every_timestamp_in_order(adapter, raw_excerpt):
    # The bug this replaces dropped its last 1118 timestamp slots on a full
    # episode; nothing may be lost, reordered or invented.
    words = [
        w for s in adapter._funasr_to_segments(raw_excerpt)["segments"] for w in s["words"]
    ]
    rebuilt = [[round(w["start"] * 1000), round(w["end"] * 1000)] for w in words]

    assert rebuilt == raw_excerpt["timestamp"]


def test_fixture_segments_are_non_empty_and_ordered(adapter, raw_excerpt):
    segments = adapter._funasr_to_segments(raw_excerpt)["segments"]

    assert segments
    assert all(s["words"] and s["text"] for s in segments)
    assert all(a["end"] <= b["start"] for a, b in zip(segments, segments[1:]))
