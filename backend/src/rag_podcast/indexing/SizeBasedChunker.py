"""Phase 1 chunker: closes chunks on token/duration bounds alone.

See design.md's "`SizeBasedChunker.py` (Phase 1)" section for the full
rationale. The short version:

- Over-long segments are split up front, so the accumulation loop below
  never needs a special case for them.
- Greedy single pass over segments in order, accumulating into a buffer.
- Two-part closing gate — hard bounds first (OR on max, AND on min), then
  a nearest-token-target lookahead inside whatever zone that leaves legal.
- Token counts are computed once per segment and summed, never by
  re-tokenizing the growing buffer (that would be O(n^2)).
"""

from __future__ import annotations

import logging
import math

from tokenizers import Tokenizer

from .BaseChunkerInterface import ChunkingError, ChunkSpan

logger = logging.getLogger(__name__)

# Sentence-final punctuation, both shapes WhisperX produces: CJK marks tend
# to arrive as their own word token, Latin ones fused onto the preceding
# word ("welcome."). Used when splitting an over-long segment, to prefer a
# natural break over an exactly-even one.
SENTENCE_FINAL_PUNCTUATION: frozenset[str] = frozenset("。！？；.!?;")

# How far from an equal-share boundary to look for sentence-final
# punctuation, as a fraction of one piece's token count. Expressed in
# tokens because that is the unit the bound being satisfied is expressed in
# — a window in seconds would admit wildly different amounts of text
# depending on speaking pace, and the piece it produced could still breach
# max_tokens. Keeping this below 0.5 makes adjacent targets' snap windows
# provably disjoint: they sit one full share apart, so two targets can never
# be pulled onto the same punctuation mark.
PUNCTUATION_SNAP_RATIO = 0.15


class SizeBasedChunker:
    """Chunks an episode by token count and wall-clock duration.

    Input segments must already be cleaned (``cleaning.clean_words`` runs
    once on the whole transcript, upstream in the worker) and in
    chronological order.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        # Overwritten per call by build_chunks, once the episode's language is
        # known. Seeded here so the helpers that read it are safe to call on a
        # freshly constructed instance rather than raising AttributeError.
        self._separator = " "

    def _count(self, text: str) -> int:
        """Token count of *text*, excluding the tokenizer's special tokens.

        Every count in this file goes through here, so the accumulation loop
        and the oversized-segment split can never drift into different units.

        ``add_special_tokens=False`` is the whole point: BGE-M3's post-processor
        wraps each call in ``<s>``/``</s>``, so the default inflates every
        count by exactly 2. Summed over a chunk that is per-segment, not
        per-chunk — a 20-segment chunk overshoots a 275-token target by ~15%.
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def _token_prefix(self, words: list[dict]) -> list[int]:
        """Cumulative token counts over *words*: entry ``i`` covers ``words[:i]``.

        Returns ``len(words) + 1`` entries, so both ends are indexable without
        a bounds check — the same shape (and for the same reason) as the
        ``gap_time`` array in ``_split_oversized_segment``.

        Costs **one** encode call, not one per word. The words are joined once, 
        encoded once, and each token is attributed back to a word through the 
        encoding's character offsets.

        Two properties the caller depends on:

        - ``prefix[-1] == self._count(joined_text)`` — every token lands on
          exactly one word, so cut placement and buffer accumulation measure in
          identical units.
        - ``prefix`` is non-decreasing, which is all the cut sweep needs to
          keep its one-step-lookahead argument valid.

        Words carrying no ``"word"`` key (WhisperX emits those for characters it
        could not align) count as empty strings: zero tokens, no shift.
        """
        pieces = [word.get("word", "") for word in words]

        # Character index at which each word begins inside the joined text,
        # recorded in the same pass that would build that text.
        word_start: list[int] = []
        position = 0
        for index, piece in enumerate(pieces):
            if index:
                position += len(self._separator)
            word_start.append(position)
            position += len(piece)

        encoding = self._tokenizer.encode(
            self._separator.join(pieces), add_special_tokens=False
        )

        # One forward sweep. Both sequences run left to right, so a pointer
        # that only advances is enough to attribute every token.
        #
        # The `+ len(self._separator)` is not slack, it is a correction:
        # SentencePiece folds the preceding space into a word-initial token, so
        # "▁exactly" in "between exactly" reports offset (7, 15) while the word
        # itself starts at 8. Without the term every word boundary is credited
        # one token early — a constant skew down the whole array. It is 0 for
        # CJK, where the separator is empty and the problem cannot arise.
        #
        # On CJK the loop instead runs several times on a single token, because
        # one token can cover several single-character words ("成" + "了" merge
        # into "成了"). Such a token is credited entirely to the first word it
        # covers, leaving the rest at zero — a flat stretch in the prefix. That
        # is the right shape for cut placement rather than a rounding loss: a
        # cut inside a merged token is not representable, and on a tie the
        # sweep's strict `>` keeps advancing, so the cut lands after the whole
        # token. The total is unaffected either way.
        per_word = [0] * len(words)
        at = 0
        for token_start, _token_end in encoding.offsets:
            while (
                at + 1 < len(words)
                and word_start[at + 1] <= token_start + len(self._separator)
            ):
                at += 1
            per_word[at] += 1

        prefix = [0]
        for count in per_word:
            prefix.append(prefix[-1] + count)
        return prefix

    def build_chunks(
        self,
        segments: list[dict],
        #*,
        min_tokens: int,
        max_tokens: int,
        desired_tokens: int,
        min_duration: float,
        max_duration: float,
        language: str | None = 'en',
    ) -> list[ChunkSpan]:
        """See ``BaseChunkerInterface.Chunker.build_chunks``."""
        if not segments:
            return []

        # Normalise over-long segments up front. Everything below this then
        # treats every segment the same way, and the split pieces stay
        # eligible to merge with their neighbours — an end piece landing a
        # second or two long is absorbed by the next chunk instead of being
        # emitted as a standalone sliver. Builds a new list; the caller's
        # `segments` (and the worker's transcript_data behind it) is never
        # mutated, matching clean_words' convention.
        self._separator = "" if (language or "en").lower() != 'en' else " "

        # Token counts run in lockstep with `normalised`. The pre-pass has to
        # tokenize every segment anyway to decide whether it is over-long, so
        # the count is carried forward rather than recomputed by the
        # accumulation loop — one encode per segment across the episode, the
        # same budget as before this list existed.
        normalised: list[dict] = []
        token_counts: list[int] = []

        for segment in segments:
            segment_tokens = self._count(segment.get("text", ""))

            # Same OR as the accumulation gate below: either max alone makes a
            # segment over-long. Token-first is the point of the check — a
            # segment can hold more than max_tokens while sitting well inside
            # max_duration, and the accumulation loop has no way to correct
            # that, since a segment is atomic to it.
            if (
                segment_tokens <= max_tokens
                and segment["end"] - segment["start"] <= max_duration
            ):
                normalised.append(segment)
                token_counts.append(segment_tokens)
                continue

            for piece in self._split_oversized_segment(
                segment,
                segment_tokens=segment_tokens,
                max_tokens=max_tokens,
                max_duration=max_duration,
                language=language,
            ):
                normalised.append(piece)
                token_counts.append(self._count(piece["text"]))

        chunks: list[ChunkSpan] = []
        # Gate arithmetic runs on raw segment boundaries, not the
        # word-derived ones _make_span produces — the two differ only by
        # leading/trailing silence, and segment boundaries keep the bound
        # checks working whether or not word-level alignment is present.
        buffer: list[dict] = []
        buffer_tokens = 0

        for segment, segment_tokens in zip(normalised, token_counts):
            if not buffer:
                buffer, buffer_tokens = [segment], segment_tokens
                continue

            # Would appending breach a max? OR across the two dimensions:
            # either bound alone forces a close attempt. The token max is
            # the one expected to fire in practice; the duration max stays
            # active as a low-token-density safety net.
            next_tokens = buffer_tokens + segment_tokens
            breaches_max = (
                segment["end"] - buffer[0]["start"] > max_duration
                or next_tokens > max_tokens
            )

            # Is the buffer legally closeable? AND across the two: both mins
            # must clear. Token-min alone is not enough, since a token-dense
            # but two-second chunk is still choppy to listen to.
            closeable = (
                buffer[-1]["end"] - buffer[0]["start"] >= min_duration
                and buffer_tokens >= min_tokens
            )

            if closeable:
                # Nearest-target stopping, inside the legal zone. A one-step
                # lookahead is enough rather than a full scan: the cumulative
                # token count only grows as segments are appended, so the
                # error is V-shaped in position — the first segment where
                # "one more makes it worse" is exactly the local optimum.
                overshoots = abs(desired_tokens - buffer_tokens) < abs(
                    desired_tokens - next_tokens
                )
                if breaches_max or overshoots:
                    chunks.append(self._make_span(buffer, language))
                    buffer, buffer_tokens = [segment], segment_tokens
                    continue

            # Falls through in two cases: a min has not cleared yet, where
            # contiguity and the min floors beat the soft max bound so we
            # append even though a max is breached; or extending actually
            # brings the buffer closer to desired_tokens.
            buffer.append(segment)
            buffer_tokens = next_tokens

        if buffer:
            # The last chunk may fall short of both mins; nothing is left to
            # merge it with.
            chunks.append(self._make_span(buffer, language))

        return chunks

    def _make_span(self, segments: list[dict], language: str | None = 'en') -> ChunkSpan:
        """Build a ``ChunkSpan`` from the segments it covers.

        ``start``/``end`` come from the first/last *word* actually included,
        not the raw segment boundary (design.md step 5) — WhisperX segment
        boundaries can sit in silence, and the audio-jump accuracy bar in
        prd.md is measured against where speech actually starts. Segments
        carrying no word-level data fall back to their own boundary.

        ``text`` is the concatenation of the segments' own ``text`` fields,
        never rebuilt from a word list (design.md step 6).
        """
        if not segments:
            raise ChunkingError("cannot build a ChunkSpan from zero segments")

        words = [w for seg in segments for w in seg.get("words", [])]
        first_word_start = next((w["start"] for w in words if "start" in w), None)
        last_word_end = next((w["end"] for w in reversed(words) if "end" in w), None)

        start = first_word_start if first_word_start is not None else segments[0]["start"]
        end = last_word_end if last_word_end is not None else segments[-1]["end"]

        # separator = "" if language.lower() != 'en' else " "
        text = self._separator.join(seg["text"].strip() for seg in segments if seg.get("text"))

        return ChunkSpan(text=text, start=start, end=end, segments=segments)

    def _split_oversized_segment(
        self,
        segment: dict,
        *,
        segment_tokens: int,
        max_tokens: int,
        max_duration: float,
        language: str | None = 'en',
    ) -> list[dict]:
        """Split one over-long segment into roughly equal word-level pieces.

        *segment_tokens* is passed in rather than recomputed: the caller has
        already tokenized this segment to decide it needed splitting.

        The piece count satisfies **both** bounds — the larger of what each
        one demands on its own::

            max(ceil(tokens / max_tokens), ceil(duration / max_duration))

        Taking the max, not either alone, is what makes every share land under
        both ceilings. Tokens usually dominate; the duration term stays as the
        low-density safety net, for a stretch of music or silence that holds
        almost no tokens but runs long.

        Equal shares rather than greedy max-width windows, which leave a
        remainder: a 361s segment against a 360s bound gives 360s + 1s instead
        of 2 x 180.5s. Each cut is then nudged onto a sentence end if one is
        within ``PUNCTUATION_SNAP_RATIO`` of a share.

        Each piece is ONE synthetic segment dict wrapping its word
        sub-range, so the contiguity invariant still holds when callers
        flatten it back into words. ``avg_logprob`` is deliberately dropped:
        it described the original whole segment and no longer applies.
        """
        words = segment.get("words") or []
        duration = segment["end"] - segment["start"]
        n_pieces = min(
            max(
                math.ceil(segment_tokens / max_tokens),
                math.ceil(duration / max_duration),
            ),
            len(words),
        )

        if n_pieces < 2:
            # A segment with no (or one) word-level entry has nothing to cut
            # at. Emitting it whole keeps the transcript complete; the
            # alternative is dropping audio content.
            logger.warning(
                "segment at %.1fs holds %d token(s) over %.1fs (max_tokens %d, "
                "max_duration %.1fs) but has %d word(s) — emitting unsplit",
                segment["start"],
                segment_tokens,
                duration,
                max_tokens,
                max_duration,
                len(words),
            )
            return [segment]

        # gap_time[i] is the wall-clock time of the gap *before* words[i].
        # Recorded before advancing, so a word WhisperX could not align (bare
        # {"word": ...}, no timestamps) inherits the last known time instead
        # of having none. Seeded with the segment's own bounds at both ends,
        # which makes gap_time[0] and gap_time[len(words)] valid too — the
        # trailing piece below can index it without a bounds check.
        #
        # This no longer places the cuts; it only supplies _piece's fallback
        # start/end for a sub-range whose words carry no timestamps at all.
        # A bare zero there would read as "the start of the episode" and
        # corrupt every duration the accumulation loop computes downstream.
        gap_time: list[float] = []
        last_end = segment["start"]
        for word in words:
            gap_time.append(last_end)
            if "end" in word:
                last_end = word["end"]
        gap_time.append(segment["end"])

        # cum[i] is the token count of words[:i], so a cut at index i splits
        # between words[i - 1] and words[i] — the same indexing gap_time uses,
        # measured on the axis the bound is actually expressed in.
        #
        # Shares come from cum[-1], NOT from segment_tokens. The two differ
        # whenever the segment's own text carries characters the word list
        # does not (FunASR writes punctuation into text only), and a target
        # derived from the larger number could sit past the end of cum, where
        # the sweep would never bottom out on it. n_pieces above is free to
        # use segment_tokens: it is a count, not a position.
        cum = self._token_prefix(words)
        piece_tokens = cum[-1] / n_pieces
        tolerance = piece_tokens * PUNCTUATION_SNAP_RATIO
        targets = [i * piece_tokens for i in range(1, n_pieces)]

        # One forward sweep, taking targets in order. Cumulative token counts
        # are non-decreasing, so distance to a fixed target is V-shaped in
        # position: the first place where advancing one word would move away
        # from the target is that target's best boundary — the same
        # one-step-lookahead argument build_chunks runs on tokens. The
        # pointer only moves forward, so two cuts can never land on one
        # boundary and no piece can come out empty.
        #
        # `snap` remembers the most recent sentence end seen inside the
        # current target's tolerance window, and wins over the arithmetic
        # boundary when the sweep bottoms out: a slightly early cut at a
        # sentence end reads better than an exactly even one mid-clause.
        # Errors shrink monotonically on the way down to the bottom, so the
        # last sentence end recorded is also the closest one — no comparison
        # needed, just overwrite.
        pieces: list[dict] = []
        left = 0
        pending = 0
        snap = 0
        for cut in range(1, len(words) - 1):
            if pending >= len(targets):
                break
            target = targets[pending]
            error = abs(cum[cut] - target)

            previous = words[cut - 1].get("word", "").strip()
            if error <= tolerance and previous[-1:] in SENTENCE_FINAL_PUNCTUATION:
                snap = cut

            if abs(cum[cut + 1] - target) > error:
                at = snap if snap > left else cut
                pieces.append(self._piece(words[left:at], language, gap_time[left], gap_time[at]))
                left, pending, snap = at, pending + 1, 0

        pieces.append(
            self._piece(words[left:], language, gap_time[left], gap_time[len(words)])
        )
        return pieces

    
    def _piece(self, words: list[dict], language: str, start: float, end: float) -> dict:
        """Wrap a word sub-range from a split in one synthetic segment dict.

        This is the one place text is rebuilt from words, so it goes through
        ``join_words`` — a naive ``" ".join(...)`` reintroduces the CJK
        spacing bug on Chinese episodes.

        *start* / *end* are the surrounding gap times, used only when the
        whole range is unaligned and carries no timestamps of its own. They
        keep such a piece inside its parent segment's span; a bare zero would
        read as "the start of the episode" and corrupt every duration
        ``build_chunks`` computes downstream.
        """
        starts = [w["start"] for w in words if "start" in w]
        ends = [w["end"] for w in words if "end" in w]
        text = self._separator.join([w['word'] for w in words if 'word' in w])

        return {
            "text": text,
            "start": starts[0] if starts else start,
            "end": ends[-1] if ends else end,
            "words": words,
        }
