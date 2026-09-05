# Token-driven oversized segment split in SizeBasedChunker

## Goal

Make the oversized-segment pre-pass in `SizeBasedChunker` a **token-driven** rule
with duration kept only as a low-density safety net, matching the token-first
posture the main accumulation gate already has. Along the way, make token
counting consistent across the whole chunker.

Today the pre-pass is duration-only: it triggers on `duration > max_duration`,
cuts into `ceil(duration / max_duration)` equal *time* shares, and snaps to
punctuation inside a window measured in *seconds*. A token-dense segment that
sits under `max_duration` is never split, and a split piece's token count is
never considered — so a piece can exceed `max_tokens` on its own and the main
loop has no way to correct it (a single segment is atomic to the accumulation
loop).

## Requirements

### R1 — Token-primary split trigger, duration as fallback

- The pre-pass splits a segment when **either** `segment_tokens > max_tokens`
  **or** `duration > max_duration` (OR, mirroring the main gate's `breaches_max`).
- Piece count is whatever satisfies both bounds:
  `n_pieces = max(ceil(segment_tokens / max_tokens), ceil(duration / max_duration))`,
  still capped at `len(words)`.

### R2 — Cuts placed in token space

- The `n_pieces - 1` cut targets are equal **token** shares of the segment, not
  equal time shares.
- The punctuation snap window is measured in tokens
  (`PUNCTUATION_SNAP_RATIO × piece_tokens`), preserving the existing property
  that adjacent targets' snap windows are provably disjoint.
- Word timestamps are still what a piece's `start`/`end` come from; only the
  *targeting* moves from time to tokens.

### R3 — One token-counting convention

- Every token count in the chunker excludes the tokenizer's special tokens
  (`add_special_tokens=False`). Verified: `Tokenizer.encode()` on BGE-M3 wraps
  every call in `<s>`/`</s>`, so today's per-segment counts are inflated by
  exactly 2 per segment (measured on `funasr_aligned_test_segments.json`:
  4942 vs 4164 tokens across 389 segments), i.e. a 20-segment chunk overshoots
  a 275-token target by ~15%.
- Per-word cumulative counts used for cut placement must be measured in the same
  units as the per-segment counts the accumulation loop sums — one encode of the
  joined text, mapped back to words via token offsets. Summing per-word encodes
  is explicitly rejected: measured ~2× inflation on Chinese (28 vs 15 tokens for
  one real segment), because per-word encoding breaks subword merges and adds a
  `▁` marker per word.

### R4 — No behavioural regressions

- Contiguity holds: flattening every emitted chunk's `segments` back into words
  reproduces the input word sequence exactly — nothing skipped, nothing
  duplicated (the existing iron-law tests must keep passing).
- Input `segments` are never mutated.
- Pieces still re-enter the ordinary accumulation gate (pre-pass, not in-loop
  fallback), so an undersized end piece still merges with its neighbour.
- A segment with fewer than two words is still logged and emitted whole.

## Constraints

- `build_chunks`' public signature and `ChunkSpan` shape do not change; no
  downstream module is touched.
- Token counting stays O(n) over the episode — no re-tokenizing a growing buffer,
  and at most one extra encode per *oversized* segment (not per segment).
- The unit suite must keep running without network access; the fake tokenizer in
  `tests/test_size_based_chunker.py` grows an `offsets` attribute and an
  `add_special_tokens` parameter rather than being replaced by the real one.

## Out of scope — recorded, separate tasks

1. **FunASR Chinese punctuation (confirmed defect, own task).** FunASR writes
   punctuation only into `segment["text"]`; its `words` entries are bare
   characters. Two consequences, both pre-existing and *unchanged* by this task:
   the punctuation snap can never fire on the FunASR path
   (`SizeBasedChunker.py:264` reads `words[cut-1]["word"]`), and `_piece` rebuilds
   piece text from words, dropping every mark. This task must not make either
   worse, and must not silently rely on punctuation being present.
2. **`_separator` language rule is inverted** vs `cleaning/text_join.join_words`:
   `SizeBasedChunker.py:75` strips spaces for *every* non-`en` language, while
   `join_words` strips only for `zh`/`ja`. design.md says pieces should be joined
   via `join_words`. Left as-is here.
3. **Keyword-only `*` is commented out** at `SizeBasedChunker.py:56`, so the
   signature no longer matches `BaseChunkerInterface.Chunker`. `scripts/test_chunker.py`
   depends on the positional form. Left as-is here.
4. **Tier re-calibration.** R3 lowers every measured token count by
   `2 × segment_count` per chunk, which shifts real chunk boundaries. Whether
   `settings.audio_tier`'s numbers should be re-anchored against
   `tokenizer_playground.py` is a follow-up, not this task.

5. **Cross-segment merge gap (found during step 2, zh only, not fixed).** Even
   with special tokens excluded, the gate's summed per-segment count exceeds the
   token count of the chunk's actual joined text. Measured on
   `funasr_aligned_test_segments.json`: gate 276 vs text 257 for a 26-segment
   chunk (~8%); English measures 0. Cause: `_separator` is empty for Chinese, so
   subword merges span segment boundaries — two tokens in two adjacent segments
   become one token once joined. The gate therefore closes chunks slightly
   earlier than the token budget intends on CJK.

   Not fixed here, and not cheaply fixable: an exact count means re-tokenizing
   the joined buffer on every append, which is the O(n²) behaviour this module's
   header comment explicitly rejects. The error is one-directional (never
   under-counts, so `max_tokens` is never actually breached) and small. Worth
   folding into the tier re-calibration in (4) rather than chasing separately.

## Acceptance Criteria

- [x] A segment over `max_tokens` but under `max_duration` is split; the number
      of pieces is `ceil(segment_tokens / max_tokens)`.
- [x] A low-token-density segment over `max_duration` but under `max_tokens` is
      still split (duration fallback intact) — the existing duration test cases
      keep passing.
- [x] A segment breaching both bounds yields `max(...)` pieces, satisfying both.
- [x] Cuts land on equal token shares: on a segment with deliberately uneven
      token density per word, pieces are near-equal in *tokens*, not in seconds.
- [x] Punctuation snapping fires when a sentence end falls within
      `PUNCTUATION_SNAP_RATIO × piece_tokens` of a target, and is ignored beyond it.
- [x] No token count anywhere in the chunker includes `<s>`/`</s>`; a direct
      assertion pins this against the real BGE-M3 tokenizer.
- [x] Contiguity, non-mutation, ordering, and undersized-piece-merging tests all
      pass unchanged.
- [x] `uv run pytest tests/test_size_based_chunker.py` is green — `44 passed,
      3 skipped` offline, `47 passed` with `RAG_PODCAST_REAL_TOKENIZER=1`.

Still open before the task closes: the two real-data smoke checks in
implement.md step 7.
