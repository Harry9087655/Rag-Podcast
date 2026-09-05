"""Tests for ``SizeBasedChunker`` (indexing, Phase 1).

The chunker takes a real ``tokenizers.Tokenizer`` in production, but nothing
in ``build_chunks`` depends on BGE-M3's actual vocab — only on the length of
``encode(text, add_special_tokens=False).ids`` and on that encoding's
``offsets``. These tests inject a fake tokenizer so token counts are exact and
readable in the assertions, and so the suite needs neither a 17MB HF download
nor network access. One opt-in test at the bottom exercises the real
tokenizer.
"""

from __future__ import annotations

import copy
import logging
import os
import re

import pytest

from rag_podcast.indexing.BaseChunkerInterface import ChunkingError, ChunkSpan
from rag_podcast.indexing.SizeBasedChunker import SizeBasedChunker

# --------------------------------------------------------------------------
# Fake tokenizer
# --------------------------------------------------------------------------


class _Encoding:
    def __init__(self, ids: list[str], offsets: list[tuple[int, int]]) -> None:
        self.ids = ids
        # (start, end) character range of each token in the encoded text —
        # what `_token_prefix` walks to map tokens back onto words.
        self.offsets = offsets


class FakeTokenizer:
    """One token per Latin word, one per CJK ideograph.

    Mirrors the only two properties of BGE-M3 the chunker actually relies on —
    a script-neutral count that grows with content, and per-token character
    offsets into the encoded text — while staying trivially predictable:
    ``"a b c"`` is 3 tokens, ``"你好"`` is 2.

    ``add_special_tokens`` is accepted and ignored: the fake never wrapped its
    output in ``<s>``/``</s>`` in the first place, so both settings give the
    same count here. The real tokenizer does add them, which is why the
    chunker always passes ``False``.
    """

    _PATTERN = re.compile(r"[㐀-鿿]|[^\s㐀-鿿]+")

    def encode(self, text: str, add_special_tokens: bool = True) -> _Encoding:
        ids, offsets = [], []
        for match in self._PATTERN.finditer(text):
            start, end = match.span()
            # SentencePiece folds the preceding space into the word-initial
            # token ("▁world" in "Hello world" reports (5, 12), not (6, 12)),
            # and `_token_prefix` has to compensate for that when mapping
            # tokens back onto words. Reproduced here so a fixture without
            # network access still exercises the compensation.
            if start and text[start - 1].isspace():
                start -= 1
            ids.append(match.group())
            offsets.append((start, end))
        return _Encoding(ids, offsets)


@pytest.fixture
def chunker() -> SizeBasedChunker:
    return SizeBasedChunker(FakeTokenizer())


# --------------------------------------------------------------------------
# Segment builders
# --------------------------------------------------------------------------

# Wide bounds by default; each test narrows only the bound it is exercising,
# so a failure points at one gate rather than at "some threshold".
BOUNDS = {
    "min_tokens": 10,
    "max_tokens": 1000,
    "desired_tokens": 500,
    "min_duration": 20.0,
    "max_duration": 90.0,
}


def bounds(**overrides) -> dict:
    return {**BOUNDS, **overrides}


def _words(tokens: list[str], start: float, end: float) -> list[dict]:
    """Spread *tokens* evenly across [start, end] as WhisperX word dicts."""
    if not tokens:
        return []
    step = (end - start) / len(tokens)
    return [
        {"word": token, "start": start + i * step, "end": start + (i + 1) * step}
        for i, token in enumerate(tokens)
    ]


def _segment(
    start: float,
    end: float,
    tokens: list[str],
    *,
    text: str | None = None,
    separator: str = " ",
) -> dict:
    """A WhisperX-shaped segment whose text and words agree with each other."""
    return {
        "text": separator.join(tokens) if text is None else text,
        "start": start,
        "end": end,
        "words": _words(tokens, start, end),
    }


def _sized(start: float, end: float, n_tokens: int, prefix: str = "w") -> dict:
    """A segment worth exactly *n_tokens* under ``FakeTokenizer``."""
    return _segment(start, end, [f"{prefix}{i}" for i in range(n_tokens)])


def _tokens_of(span: ChunkSpan) -> int:
    return len(FakeTokenizer().encode(span.text).ids)


def _all_words(segments: list[dict]) -> list[dict]:
    return [word for segment in segments for word in segment.get("words", [])]


def _chunk_words(chunks: list[ChunkSpan]) -> list[dict]:
    return [word for chunk in chunks for word in _all_words(chunk.segments)]


# --------------------------------------------------------------------------
# Basic accumulation
# --------------------------------------------------------------------------


def test_empty_segments_returns_empty_list(chunker):
    assert chunker.build_chunks([], **BOUNDS) == []


def test_single_segment_becomes_one_chunk_even_below_both_mins(chunker):
    # Nothing is left to merge a trailing short buffer with, so the last
    # chunk is allowed to fall short of min_tokens and min_duration.
    segments = [_sized(0.0, 5.0, 3)]

    chunks = chunker.build_chunks(segments, **BOUNDS)

    assert len(chunks) == 1
    assert chunks[0].segments == segments


def test_merges_segments_until_both_mins_clear(chunker):
    # Three 8s / 6-token segments: neither min is met until all three are in.
    segments = [
        _segment(0.0, 8.0, ["Hello", "there", "and", "welcome", "to", "one."]),
        _segment(8.0, 16.0, ["This", "is", "the", "second", "segment", "two."]),
        _segment(16.0, 24.0, ["And", "here", "is", "segment", "number", "three."]),
    ]

    chunks = chunker.build_chunks(segments, **bounds(min_tokens=15, min_duration=20.0))

    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 24.0
    assert _tokens_of(chunks[0]) == 18


def test_closes_when_next_segment_would_breach_max_tokens(chunker):
    segments = [_sized(0.0, 30.0, 60), _sized(30.0, 60.0, 60, prefix="x")]

    chunks = chunker.build_chunks(segments, **bounds(max_tokens=100))

    assert len(chunks) == 2
    assert [_tokens_of(chunk) for chunk in chunks] == [60, 60]


def test_closes_when_next_segment_would_breach_max_duration(chunker):
    # Low token density: the token max never fires, so the duration max is
    # the only thing keeping these two 50s segments apart.
    segments = [_sized(0.0, 50.0, 20), _sized(50.0, 100.0, 20, prefix="x")]

    chunks = chunker.build_chunks(segments, **bounds(max_duration=90.0))

    assert len(chunks) == 2
    assert (chunks[0].start, chunks[0].end) == (0.0, 50.0)
    assert (chunks[1].start, chunks[1].end) == (50.0, 100.0)


def test_max_tokens_is_inclusive_not_exclusive(chunker):
    # Landing exactly on max_tokens is legal — the gate is `> max_tokens`.
    segments = [
        _sized(0.0, 25.0, 50, prefix="a"),
        _sized(25.0, 50.0, 50, prefix="b"),
        _sized(50.0, 75.0, 50, prefix="c"),
        _sized(75.0, 100.0, 50, prefix="d"),
    ]

    chunks = chunker.build_chunks(
        segments, **bounds(max_tokens=100, desired_tokens=100)
    )

    assert len(chunks) == 2
    assert [_tokens_of(chunk) for chunk in chunks] == [100, 100]


# --------------------------------------------------------------------------
# The AND-on-min veto: a max may be breached while a min is still open
# --------------------------------------------------------------------------


def test_min_duration_veto_forces_append_past_max_tokens(chunker):
    # Token-dense but only 5s in: min_duration has not cleared, so the
    # buffer must keep growing even though max_tokens is breached.
    segments = [_sized(0.0, 5.0, 30), _sized(5.0, 10.0, 30, prefix="x")]

    chunks = chunker.build_chunks(
        segments, **bounds(min_tokens=10, min_duration=20.0, max_tokens=40)
    )

    assert len(chunks) == 1
    assert _tokens_of(chunks[0]) == 60  # deliberately over max_tokens


def test_min_tokens_veto_forces_append_past_max_duration(chunker):
    # Long but nearly empty: min_tokens has not cleared, so the duration max
    # loses to the token floor.
    segments = [_sized(0.0, 30.0, 3), _sized(30.0, 60.0, 3, prefix="x")]

    chunks = chunker.build_chunks(
        segments, **bounds(min_tokens=10, min_duration=20.0, max_duration=45.0)
    )

    assert len(chunks) == 1
    assert chunks[0].end - chunks[0].start == 60.0  # deliberately over max_duration


# --------------------------------------------------------------------------
# Nearest-desired-target stopping inside the legal zone
# --------------------------------------------------------------------------


def test_stops_at_nearest_desired_token_target_not_at_max(chunker):
    # Six 30-token / 15s segments with desired_tokens=100 and both maxes far
    # away. 90 is closer to 100 than 120 is, so each chunk must close at
    # three segments rather than running on toward max_tokens.
    segments = [
        _sized(i * 15.0, (i + 1) * 15.0, 30, prefix=f"s{i}_") for i in range(6)
    ]

    chunks = chunker.build_chunks(
        segments,
        **bounds(desired_tokens=100, max_tokens=1000, max_duration=1000.0),
    )

    assert len(chunks) == 2
    assert [len(chunk.segments) for chunk in chunks] == [3, 3]
    assert [_tokens_of(chunk) for chunk in chunks] == [90, 90]


def test_extends_past_desired_target_while_it_still_improves(chunker):
    # Same shape, desired_tokens=150: 120 (four segments) is nearer than 90,
    # so the one-step lookahead must keep going rather than close early.
    segments = [
        _sized(i * 15.0, (i + 1) * 15.0, 30, prefix=f"s{i}_") for i in range(4)
    ]

    chunks = chunker.build_chunks(
        segments,
        **bounds(desired_tokens=150, max_tokens=1000, max_duration=1000.0),
    )

    assert len(chunks) == 1
    assert _tokens_of(chunks[0]) == 120


# --------------------------------------------------------------------------
# ChunkSpan shape: start / end / text / segments
# --------------------------------------------------------------------------


def test_span_bounds_come_from_word_timestamps_not_segment_boundaries(chunker):
    # WhisperX pads segment boundaries into surrounding silence; the span must
    # report where speech actually starts and stops (prd.md's ±2s audio-jump
    # bar is measured against that).
    segment = _segment(0.0, 30.0, ["one", "two", "three"])
    segment["words"] = _words(["one", "two", "three"], 1.5, 28.0)

    chunks = chunker.build_chunks([segment], **BOUNDS)

    assert chunks[0].start == 1.5
    assert chunks[0].end == 28.0


def test_span_bounds_fall_back_to_segment_boundaries_without_words(chunker):
    segments = [{"text": "No word alignment here.", "start": 4.0, "end": 26.0, "words": []}]

    chunks = chunker.build_chunks(segments, **BOUNDS)

    assert (chunks[0].start, chunks[0].end) == (4.0, 26.0)


def test_span_bounds_span_a_leading_unaligned_segment(chunker):
    # First segment carries no words, second does: start must come from the
    # first *word* available anywhere in the chunk, end from the last.
    segments = [
        {"text": "Unaligned intro.", "start": 0.0, "end": 10.0, "words": []},
        _segment(10.0, 30.0, ["aligned", "content", "here"]),
    ]

    chunks = chunker.build_chunks(segments, **bounds(min_tokens=1))

    assert len(chunks) == 1
    assert chunks[0].start == 10.0
    assert chunks[0].end == 30.0


def test_text_is_the_segments_own_text_space_joined_for_english(chunker):
    # Regression guard: .text is concatenated from the segments' (already
    # cleaned) text fields, never rebuilt from the word list.
    segments = [
        _segment(0.0, 12.0, ["Hello", "world."], text="  Hello world.  "),
        _segment(12.0, 24.0, ["Second", "one."], text="Second one."),
    ]

    chunks = chunker.build_chunks(segments, **bounds(min_tokens=1))

    assert chunks[0].text == "Hello world. Second one."


def test_text_is_joined_without_spaces_for_chinese(chunker):
    # The CJK spacing bug: a uniform space-join turns "你好" into "你 好".
    segments = [
        _segment(0.0, 12.0, list("你好世界。"), text="你好世界。", separator=""),
        _segment(12.0, 24.0, list("再见。"), text="再见。", separator=""),
    ]

    chunks = chunker.build_chunks(segments, **bounds(min_tokens=1, language="zh"))

    assert chunks[0].text == "你好世界。再见。"


def test_language_hint_is_case_insensitive(chunker):
    segments = [
        _segment(0.0, 12.0, list("你好"), text="你好", separator=""),
        _segment(12.0, 24.0, list("世界"), text="世界", separator=""),
    ]

    assert chunker.build_chunks(segments, **bounds(min_tokens=1, language="ZH"))[
        0
    ].text == "你好世界"
    assert chunker.build_chunks(segments, **bounds(min_tokens=1, language="EN"))[
        0
    ].text == "你好 世界"


def test_language_none_space_joins_like_english(chunker):
    # BaseChunkerInterface: "``None`` means space-join, the safe default for
    # space-delimited scripts."
    segments = [
        _segment(0.0, 12.0, ["Hello", "world."]),
        _segment(12.0, 24.0, ["Second", "one."]),
    ]

    chunks = chunker.build_chunks(segments, **bounds(min_tokens=1, language=None))

    assert chunks[0].text == "Hello world. Second one."


def test_span_segments_are_the_caller_s_own_segment_objects(chunker):
    # Whole-segment chunks carry the input dicts through by reference — no
    # copying, no synthetic rewrapping, so downstream code can still reach
    # avg_logprob and any other field the transcriber attached.
    segments = [_sized(0.0, 12.0, 6), _sized(12.0, 24.0, 6, prefix="x")]

    chunks = chunker.build_chunks(segments, **bounds(min_tokens=1))

    assert chunks[0].segments[0] is segments[0]
    assert chunks[0].segments[1] is segments[1]


def test_input_segments_are_never_mutated(chunker):
    # clean_words' convention: the worker still holds the original
    # transcript_data after chunking.
    segments = [
        _sized(0.0, 40.0, 20),
        _segment(40.0, 240.0, [f"w{i}" for i in range(200)]),  # forces the pre-pass
        _sized(240.0, 280.0, 20, prefix="z"),
    ]
    before = copy.deepcopy(segments)

    chunker.build_chunks(segments, **BOUNDS)

    assert segments == before


def test_make_span_rejects_zero_segments(chunker):
    with pytest.raises(ChunkingError):
        chunker._make_span([])


# --------------------------------------------------------------------------
# Per-word token prefix — the array cut placement is measured against
# --------------------------------------------------------------------------


def _prefix(chunker: SizeBasedChunker, words: list[dict], separator: str = " "):
    """Call ``_token_prefix`` with an explicit separator.

    ``build_chunks`` normally stores it on the instance once the episode's
    language is known; these tests exercise the helper on its own, so they set
    it directly rather than going through a whole chunking run.
    """
    chunker._separator = separator
    return chunker._token_prefix(words)


def _per_word(prefix: list[int]) -> list[int]:
    """Undo the prefix sum, so attribution is readable in an assertion."""
    return [later - earlier for earlier, later in zip(prefix, prefix[1:])]


def test_token_prefix_totals_match_a_direct_count(chunker):
    # The invariant the whole split rests on: the cut sweep and the
    # accumulation loop must be counting in the same units.
    words = [{"word": word} for word in ["one", "two", "three", "four"]]

    prefix = _prefix(chunker, words)

    assert len(prefix) == len(words) + 1
    assert prefix[0] == 0
    assert prefix[-1] == chunker._count(" ".join(w["word"] for w in words))


def test_token_prefix_is_non_decreasing(chunker):
    # The one-step-lookahead argument in the cut sweep needs monotonicity and
    # nothing else.
    words = [{"word": word} for word in ["a", "bb", "ccc", "dddd", "e"]]

    prefix = _prefix(chunker, words)

    assert all(earlier <= later for earlier, later in zip(prefix, prefix[1:]))


def test_token_prefix_attributes_multi_token_words_across_a_space(chunker):
    # Guards the `+ len(separator)` correction. SentencePiece folds the
    # preceding space into a word-initial token, so without that term every
    # boundary is credited one token early and this comes out as [2, 2, 1].
    #
    # The middle word is deliberately mixed-script: FakeTokenizer gives one
    # token per Latin word, so a Latin-only fixture is 1-token-per-word and
    # would pass whether or not the correction is present.
    words = [{"word": "hello"}, {"word": "你好a"}, {"word": "world"}]

    prefix = _prefix(chunker, words)

    assert _per_word(prefix) == [1, 3, 1]


def test_token_prefix_attributes_tokens_without_a_separator(chunker):
    # CJK path: no separator, so the correction term is 0 and each ideograph
    # is its own token.
    words = [{"word": character} for character in "你好世界"]

    prefix = _prefix(chunker, words, separator="")

    assert _per_word(prefix) == [1, 1, 1, 1]
    assert prefix[-1] == chunker._count("你好世界")


def test_token_prefix_counts_unaligned_words_as_zero(chunker):
    # WhisperX emits bare {"word": ...}-less entries for characters it could
    # not align. They must not shift the total or the attribution.
    words = [{"word": "one"}, {"start": 1.0, "end": 2.0}, {"word": "two"}]

    prefix = _prefix(chunker, words)

    assert _per_word(prefix) == [1, 0, 1]
    assert prefix[-1] == 2


def test_token_prefix_of_no_words_is_a_single_zero(chunker):
    assert _prefix(chunker, []) == [0]


# --------------------------------------------------------------------------
# Oversized-segment pre-pass
# --------------------------------------------------------------------------


def test_oversized_segment_is_split_into_equal_shares(chunker):
    # 200s against a 90s max is ceil(200/90) = 3 pieces of ~66.7s, not a
    # greedy 90 + 90 + 20.
    segment = _segment(0.0, 200.0, [f"w{i}" for i in range(200)])

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert len(chunks) == 3
    durations = [chunk.end - chunk.start for chunk in chunks]
    assert all(60.0 <= duration <= 70.0 for duration in durations), durations
    assert sum(durations) == pytest.approx(200.0)


def test_oversized_split_snaps_a_cut_onto_a_sentence_end(chunker):
    # Equal-share boundary lands at 66.7s; a sentence ends at 60.0s, inside
    # the 0.15 * 66.7 = 10s snap window, so the cut moves there.
    tokens = [f"w{i}" for i in range(200)]
    tokens[59] = "ends."
    segment = _segment(0.0, 200.0, tokens)

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert chunks[0].end == pytest.approx(60.0)
    assert chunks[0].text.endswith("ends.")


def test_oversized_split_ignores_punctuation_outside_the_snap_window(chunker):
    # Same setup, but the sentence ends at 40.0s — 26.7s from the target,
    # well beyond the window — so the arithmetic boundary wins.
    tokens = [f"w{i}" for i in range(200)]
    tokens[39] = "ends."
    segment = _segment(0.0, 200.0, tokens)

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert chunks[0].end == pytest.approx(67.0)


def test_oversized_split_snaps_on_standalone_cjk_punctuation(chunker):
    # WhisperX aligns CJK per character, so sentence-final marks arrive as
    # their own word entry rather than fused onto the previous word.
    tokens = [f"w{i}" for i in range(200)]
    tokens[59] = "。"
    segment = _segment(0.0, 200.0, tokens)

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert chunks[0].end == pytest.approx(60.0)


def test_oversized_segment_pieces_rejoin_the_normal_gate_and_merge(chunker):
    # The point of making this a pre-pass rather than an in-loop fallback:
    # pieces are ordinary segments, so pieces that fall under the mins merge
    # instead of standing as slivers. 200s with only 6 words gives three
    # 2-token pieces, none of which clears min_tokens.
    segment = _segment(0.0, 200.0, ["one", "two", "three", "four", "five", "six"])

    chunks = chunker.build_chunks([segment], **bounds(min_tokens=10))

    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == 200.0


def test_oversized_split_piece_text_is_joined_language_aware(chunker):
    # This is the one place text is rebuilt from words; a naive " ".join
    # reintroduces the CJK spacing bug.
    tokens = list("大家好欢迎收听我的播客节目今天我们要聊的话题") * 10
    segment = _segment(0.0, 200.0, tokens, text="".join(tokens), separator="")

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0, language="zh"))

    assert len(chunks) > 1
    assert " " not in chunks[0].text
    assert chunks[0].text.startswith("大家好欢迎收听")


def test_oversized_segment_without_words_is_emitted_whole(chunker, caplog):
    # Dropping audio content is worse than an over-long chunk.
    segment = {"text": "A long unaligned stretch.", "start": 0.0, "end": 200.0, "words": []}

    with caplog.at_level(logging.WARNING):
        chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0.0, 200.0)
    assert "emitting unsplit" in caplog.text


def test_oversized_segment_with_one_word_is_emitted_whole(chunker, caplog):
    segment = _segment(0.0, 200.0, ["mmmmm"])

    with caplog.at_level(logging.WARNING):
        chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert len(chunks) == 1
    assert "emitting unsplit" in caplog.text


def test_split_keeps_words_that_whisperx_could_not_align(chunker):
    # Unaligned characters arrive as bare {"word": ...} with no timestamps.
    # They must survive the split, not be dropped, and must not drag a
    # piece's bounds back to 0.0.
    segment = _segment(0.0, 200.0, [f"w{i}" for i in range(200)])
    for index in (5, 6, 120):
        segment["words"][index] = {"word": segment["words"][index]["word"]}

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    assert len(_chunk_words(chunks)) == 200
    assert all(chunk.start >= 0.0 and chunk.end > chunk.start for chunk in chunks)
    assert all(chunk.end - chunk.start < 200.0 for chunk in chunks)


def test_every_piece_of_a_split_carries_word_level_data(chunker):
    segment = _segment(0.0, 200.0, [f"w{i}" for i in range(200)])

    chunks = chunker.build_chunks([segment], **bounds(max_duration=90.0))

    for chunk in chunks:
        for piece in chunk.segments:
            assert piece["words"]
            assert piece["text"]


# --------------------------------------------------------------------------
# Token-driven split: trigger, piece count, and cut placement
#
# Everything above this point uses max_tokens=1000, so the token bound never
# fires and the split is always duration-triggered. On those fixtures every
# word is worth one token and time is spread evenly, which makes equal-time
# and equal-token cuts land in the same place — they would pass whether or
# not the split measures in tokens. These cases are the ones that can tell
# the difference.
# --------------------------------------------------------------------------


def test_token_only_trigger_splits_a_segment_that_is_short_but_dense(chunker):
    # 200 tokens in 30s against a 90s duration max: the old duration-only
    # trigger left this whole, and the accumulation loop could not fix it —
    # a segment is atomic to it. Now it splits into ceil(200 / 50) = 4.
    segment = _segment(0.0, 30.0, [f"w{i}" for i in range(200)])

    chunks = chunker.build_chunks(
        [segment],
        **bounds(max_tokens=50, max_duration=90.0, min_duration=1.0),
    )

    assert len(chunks) == 4
    assert [_tokens_of(chunk) for chunk in chunks] == [50, 50, 50, 50]
    # Duration was never the trigger: every piece sits far inside the bound.
    assert all(chunk.end - chunk.start < 90.0 for chunk in chunks)


def test_duration_trigger_still_fires_with_max_tokens_wide(chunker):
    # The low-density safety net. 60 tokens over 200s: the token term demands
    # a single piece, so ceil(200 / 90) = 3 is entirely the duration term's
    # doing. Guards against the token bound quietly replacing duration rather
    # than joining it.
    segment = _segment(0.0, 200.0, [f"w{i}" for i in range(60)])

    chunks = chunker.build_chunks([segment], **bounds(max_tokens=1000, max_duration=90.0))

    assert len(chunks) == 3
    assert [_tokens_of(chunk) for chunk in chunks] == [20, 20, 20]
    durations = [chunk.end - chunk.start for chunk in chunks]
    assert all(60.0 <= duration <= 70.0 for duration in durations), durations


def test_both_bounds_breached_takes_the_larger_piece_count(chunker):
    # 300 tokens over 180s: tokens demand ceil(300 / 60) = 5 pieces, duration
    # demands ceil(180 / 50) = 4. Taking the max is what puts every piece
    # under both ceilings — min(), or either term alone, gives 4 pieces of 75
    # tokens, which breaches max_tokens.
    segment = _segment(0.0, 180.0, [f"w{i}" for i in range(300)])

    chunks = chunker.build_chunks([segment], **bounds(max_tokens=60, max_duration=50.0))

    assert len(chunks) == 5
    assert all(_tokens_of(chunk) <= 60 for chunk in chunks)
    assert all(chunk.end - chunk.start <= 50.0 for chunk in chunks)


def _dense_then_sparse_segment(replacements: dict[int, str] | None = None) -> dict:
    """Forty 3-token words crammed into 0-10s, then thirty 1-token words to 60s.

    150 tokens either way, but 80% of them live in the first sixth of the
    segment. Equal-time and equal-token cuts land nowhere near each other
    here, which is the whole point of the fixture.

    *replacements* swaps a word by index, for the snap tests. Every
    substitution keeps that word's token weight, so the cut targets below do
    not move: ``"词密。"`` is three tokens like ``"词密集"`` (``。`` is outside
    ``FakeTokenizer``'s ideograph range, so it forms its own token), and
    ``"l5."`` is one token like ``"l5"``.
    """
    labels = ["词密集"] * 40 + [f"l{i}" for i in range(30)]
    for index, replacement in (replacements or {}).items():
        labels[index] = replacement

    words = _words(labels[:40], 0.0, 10.0) + _words(labels[40:], 10.0, 60.0)
    return {
        "text": " ".join(word["word"] for word in words),
        "start": 0.0,
        "end": 60.0,
        "words": words,
    }


def test_cuts_are_equal_in_tokens_not_in_seconds(chunker):
    # The step-5 guard. Three pieces of ~50 tokens each; if the sweep still
    # ran on gap_time the cuts would land near 20s and 40s, giving roughly
    # [120, 15, 15] tokens — the first piece alone over twice max_tokens.
    chunks = chunker.build_chunks(
        [_dense_then_sparse_segment()],
        **bounds(max_tokens=50, max_duration=1000.0, min_duration=1.0),
    )

    assert len(chunks) == 3
    # Near-equal in tokens: the residual spread is word granularity (each
    # heavy word is an indivisible 3 tokens), not a placement error.
    assert [_tokens_of(chunk) for chunk in chunks] == [51, 48, 51]
    # Deliberately unequal in seconds — a 13x spread between first and last.
    durations = [round(chunk.end - chunk.start, 2) for chunk in chunks]
    assert durations == [4.25, 4.0, 51.75]


# The two snap cases run on the uneven-density fixture on purpose. The
# existing snap tests above use a uniform segment, where a window of 0.15
# shares is the same set of words whether it is measured in tokens or in
# seconds — they pin where the window sits, but not what it is measured in.
# These two put the two units in direct conflict.


def test_snap_fires_on_a_sentence_end_inside_the_token_window(chunker):
    # 3 pieces of 50 tokens, so the window is 0.15 * 50 = 7.5 tokens around
    # the first target at 50. The sentence ends at cumulative token 45, five
    # inside the window, so the cut moves back onto it — even though at 3.75s
    # it is nowhere near the 20s equal-time boundary.
    segment = _dense_then_sparse_segment({14: "词密。"})

    chunks = chunker.build_chunks(
        [segment], **bounds(max_tokens=50, max_duration=1000.0, min_duration=1.0)
    )

    assert _tokens_of(chunks[0]) == 45
    assert chunks[0].text.endswith("词密。")


def test_snap_ignores_a_sentence_end_outside_the_token_window(chunker):
    # The mirror image: this sentence ends at exactly 20.0s, dead on the
    # equal-*time* boundary, but at cumulative token 126 — 76 tokens from the
    # 50-token target, ten times the window. It must be ignored, leaving the
    # arithmetic boundary at 51 tokens. A window measured in seconds snaps
    # here and yields a 126-token first piece.
    segment = _dense_then_sparse_segment({45: "l5."})

    chunks = chunker.build_chunks(
        [segment], **bounds(max_tokens=50, max_duration=1000.0, min_duration=1.0)
    )

    assert _tokens_of(chunks[0]) == 51
    assert not chunks[0].text.endswith("l5.")


# --------------------------------------------------------------------------
# Contiguity — the iron law
# --------------------------------------------------------------------------


def test_chunks_reconstruct_the_full_word_sequence(chunker):
    # Short segments that must merge, a normal run, and an oversized segment
    # that must split — flattening every chunk back to words must reproduce
    # the input exactly, nothing skipped, nothing duplicated.
    segments = [
        _sized(0.0, 5.0, 4, prefix="intro_"),
        _sized(5.0, 12.0, 6, prefix="intro2_"),
        _sized(12.0, 60.0, 90, prefix="main_"),
        _segment(60.0, 320.0, [f"long{i}" for i in range(260)]),
        _sized(320.0, 360.0, 40, prefix="outro_"),
    ]

    chunks = chunker.build_chunks(segments, **bounds(max_tokens=120, max_duration=90.0))

    assert [word["word"] for word in _chunk_words(chunks)] == [
        word["word"] for word in _all_words(segments)
    ]


def test_chunks_are_ordered_and_non_overlapping(chunker):
    segments = [
        _sized(i * 25.0, (i + 1) * 25.0, 40, prefix=f"s{i}_") for i in range(10)
    ]

    chunks = chunker.build_chunks(segments, **bounds(max_tokens=100, desired_tokens=80))

    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.end <= later.start
        assert earlier.start < earlier.end


# --------------------------------------------------------------------------
# Opt-in: the real BGE-M3 tokenizer (needs network on first run)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("RAG_PODCAST_REAL_TOKENIZER"),
    reason="set RAG_PODCAST_REAL_TOKENIZER=1 to download/load the BGE-M3 tokenizer",
)
def test_build_chunks_runs_against_the_real_bge_m3_tokenizer():
    from rag_podcast.indexing.tokenizer import get_tokenizer

    real = SizeBasedChunker(get_tokenizer())
    segments = [
        _segment(
            i * 20.0,
            (i + 1) * 20.0,
            ["Hello", "大家好,", "welcome", "to", "the", "show", "今天我们聊聊", "chunking."],
        )
        for i in range(20)
    ]

    chunks = real.build_chunks(
        segments,
        min_tokens=150,
        max_tokens=400,
        desired_tokens=275,
        min_duration=20.0,
        max_duration=360.0,
    )

    assert chunks
    assert [word["word"] for word in _chunk_words(chunks)] == [
        word["word"] for word in _all_words(segments)
    ]


@pytest.mark.skipif(
    not os.environ.get("RAG_PODCAST_REAL_TOKENIZER"),
    reason="set RAG_PODCAST_REAL_TOKENIZER=1 to download/load the BGE-M3 tokenizer",
)
def test_count_excludes_special_tokens_on_the_real_tokenizer():
    # BGE-M3's post-processor wraps every encode in <s>/</s>. _count must not
    # carry them, or every bound in settings.audio_tier is measured against
    # counts inflated by 2 per segment.
    from rag_podcast.indexing.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    real = SizeBasedChunker(tokenizer)

    assert real._count("hello world") == len(tokenizer.encode("hello world").ids) - 2


@pytest.mark.skipif(
    not os.environ.get("RAG_PODCAST_REAL_TOKENIZER"),
    reason="set RAG_PODCAST_REAL_TOKENIZER=1 to download/load the BGE-M3 tokenizer",
)
def test_token_prefix_matches_the_real_tokenizer_on_both_scripts():
    # FakeTokenizer cannot reproduce BGE-M3's actual subword behaviour — a
    # contraction splitting three ways ("we're" -> ▁we / ' / re), merges that
    # span two CJK "words" ("成" + "了" -> 成了). This pins the invariant
    # against the real thing on both separator paths.
    from rag_podcast.indexing.tokenizer import get_tokenizer

    real = SizeBasedChunker(get_tokenizer())

    english = [{"word": word} for word in ["between", "exactly", "we're", "not"]]
    prefix = _prefix(real, english)
    assert _per_word(prefix) == [1, 1, 3, 1]
    assert prefix[-1] == real._count(" ".join(w["word"] for w in english))

    chinese = [{"word": character} for character in "成了特牌金牌"]
    prefix = _prefix(real, chinese, separator="")
    assert prefix[0] == 0
    assert all(earlier <= later for earlier, later in zip(prefix, prefix[1:]))
    assert prefix[-1] == real._count("成了特牌金牌")
