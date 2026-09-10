"""Phase 1 chunker: closes chunks on token/duration bounds alone.

See design.md's "`SizeBasedChunker.py` (Phase 1)" section for the full
rationale. The short version:

- Greedy single pass over segments in order, accumulating into a buffer.
- Two-part closing gate — hard bounds first (OR on max, AND on min), then
  a nearest-token-target lookahead inside whatever zone that leaves legal.
- Token counts are computed once per segment and summed, never by
  re-tokenizing the growing buffer (that would be O(n^2)).
"""

from __future__ import annotations

import logging
import math
from rag_podcast.cleaning.text_join import join_words
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)

# Sentence-final punctuation, both shapes WhisperX produces: CJK marks tend
# to arrive as their own word token, Latin ones fused onto the preceding
# word ("welcome."). Used when splitting an over-long segment, to prefer a
# natural break over an exactly-even one.
SENTENCE_FINAL_PUNCTUATION: frozenset[str] = frozenset("。！？；.!?;")

# How far either side of an equal-share boundary to look for sentence-final
# punctuation. Wide enough to catch a nearby sentence end, narrow enough
# that pieces stay close to equal when sentences are long.
PUNCTUATION_SNAP_WINDOW = 5


def _split_segment(
    segment: dict,
    *,
    max_duration: float,
    language: str = 'en',
) -> list[dict]:
    """Split one over-long segment into roughly equal word-level pieces.

    The trigger is duration-only by design (BGE-M3's 8192-token ceiling
    leaves enormous headroom over budgets in the hundreds, so a
    token-dense-but-short segment is not a real overflow risk).

    Pieces are ``ceil(duration / max_duration)`` equal shares rather
    than greedy max-width windows. Greedy windowing leaves a remainder:
    a 361s segment against a 360s bound yields 360s + 1s, and because a
    piece is a segment the min bounds still apply to, that sliver is a
    visible artefact. Equal shares give 2 x 180.5s instead.

    Each piece is ONE synthetic segment dict wrapping its word
    sub-range, so the contiguity invariant still holds when callers
    flatten it back into words. ``avg_logprob`` is deliberately dropped:
    it described the original whole segment and no longer applies.
    """
    words = segment.get("words") or []
    duration = segment["end"] - segment["start"]
    n_pieces = min(math.ceil(duration / max_duration), len(words))

    if n_pieces < 2:
        # Nothing to cut at — a segment with no (or one) word-level
        # entry cannot be split. Emitting it whole keeps the transcript
        # complete; the alternative is dropping audio content.
        logger.warning(
            "segment at %.1fs runs %.1fs (over max_duration %.1fs) but has "
            "%d word(s) — emitting unsplit",
            segment["start"],
            duration,
            max_duration,
            len(words),
        )
        return [segment]

    # Each equal-share boundary independently picks the word nearest to
    # it. Two adjacent targets can land on the same word when the words
    # are sparse relative to the segment's length (long silence, music,
    # a stretch WhisperX barely aligned), so dedupe rather than forcing
    # distinct cuts: a collision means there is no honest boundary to
    # separate those shares, and emitting fewer, better-placed pieces
    # beats cutting at an arbitrary word far from the target. Dropping
    # 0 and len(words) rules out an empty leading/trailing piece.
    piece_duration = duration / n_pieces
    targets = [segment["start"] + i * piece_duration for i in range(1, n_pieces)] + [words[-1]['end']]
    targets_index = 0
    print(targets)
    left = 0
    res = []


    for right in range(len(words)-1):
        curr_diff = abs(words[right].get('end') - targets[targets_index])
        next_diff = abs(words[right+1].get('end') - targets[targets_index])
        print(targets_index)
        print(f"当前word：{words[right]}，diff：{curr_diff}")
        print(f"当前word：{words[right+1]}，diff：{next_diff}")
        if (next_diff > curr_diff) and targets_index < n_pieces:
            res.append(_synthetic_segment(words[left:right+1], language=language))
            targets_index += 1
            left = right + 1
    res.append(_synthetic_segment(words[left:], language=language))
    
    return res
             
        
        


    # cuts = sorted({_cut_index(words, t) for t in targets} - {0, len(words)})

    # bounds = [0, *cuts, len(words)]
    # return [
    #     _synthetic_segment(words[a:b], language)
    #     for a, b in zip(bounds, bounds[1:])
    # ]

def _cut_index(words: list[dict], target_time: float) -> int:
    """Index of the word boundary nearest *target_time*.

    The result is then snapped to sentence-final punctuation within
    ``PUNCTUATION_SNAP_WINDOW`` words, if any candidate qualifies —
    a slightly uneven split at a sentence end reads far better than an
    exactly even one mid-clause.

    Words missing ``start``/``end`` are not a special case: WhisperX
    emits bare ``{"word": ...}`` entries for characters it could not
    align, so boundary time falls back to the last known timestamp,
    which keeps those words in the transcript rather than dropping them.
    """

    def boundary_time(index: int) -> float:
        for word in reversed(words[:index]):
            if "end" in word:
                return word["end"]
        return float("-inf")

    candidates = range(1, len(words))
    nearest = min(candidates, key=lambda i: abs(boundary_time(i) - target_time))

    snapped = [
        i
        for i in candidates
        if abs(i - nearest) <= PUNCTUATION_SNAP_WINDOW and _ends_sentence(words[i - 1])
    ]
    return min(snapped, key=lambda i: abs(i - nearest)) if snapped else nearest


def _ends_sentence(word: dict) -> bool:
    """Whether *word* closes a sentence, in either shape WhisperX emits.

    CJK marks tend to arrive as their own token; Latin ones come fused
    onto the preceding word ("welcome."). Checking the last character
    covers both.
    """
    text = word.get("word", "").strip()
    return bool(text) and text[-1] in SENTENCE_FINAL_PUNCTUATION


def _synthetic_segment(words: list[dict], language: str | None) -> dict:
    """Wrap a word sub-range in a segment dict.

    This is the one place text IS rebuilt from words, so it must go
    through ``join_words(words, language)`` — a naive ``" ".join(...)``
    reintroduces the CJK spacing bug on Chinese episodes.
    """
    starts = [w["start"] for w in words if "start" in w]
    ends = [w["end"] for w in words if "end" in w]
    return {
        "text": join_words(words, language),
        "start": starts[0] if starts else 0.0,
        "end": ends[-1] if ends else 0.0,
        "words": words,
    }


if __name__ == "__main__":
    test = {"text": "您的半拿铁周刊，", "start": 0.33, "end": 9.41, 
            "words": [{"word": "您", "start": 0.33, "end": 2.11}, 
                      {"word": "的", "start": 8.31, "end": 8.43}, 
                      {"word": "半", "start": 8.43, "end": 8.59}, 
                      {"word": "拿", "start": 8.59, "end": 8.77}, 
                      {"word": "铁", "start": 8.77, "end": 9.01}, 
                      {"word": "周", "start": 9.01, "end": 9.17}, 
                      {"word": "刊", "start": 9.17, "end": 9.41}]}
    print(_split_segment(test, max_duration=4, language='zh'))


