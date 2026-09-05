# Design: token-driven oversized-segment split

Scope: `backend/src/rag_podcast/indexing/SizeBasedChunker.py` and its unit suite.
No other module changes. Amends the "`SizeBasedChunker.py` (Phase 1)" section of
`.trellis/tasks/07-22-chunking/design.md` (step 3.3 folds this back there).

## What stays

The overall shape is unchanged and deliberately so:

- Pre-pass, not in-loop fallback. Pieces are ordinary segments that re-enter the
  accumulation gate, so an undersized end piece merges with its neighbour.
- Equal shares, not greedy max-width windows.
- One forward sweep with a one-step lookahead per target, plus a
  most-recent-sentence-end snap.
- `ChunkSpan.start`/`.end` derived from word timestamps; `_piece` still carries
  the word sub-range.

Only the **axis the sweep targets** changes: cumulative time → cumulative tokens.
The correctness argument transfers unchanged, because cumulative token counts are
non-decreasing in word position exactly as word end times are — so the distance
to a fixed target is still V-shaped, the first "one more step moves away" is
still the local optimum, the pointer still only advances, and no piece can come
out empty.

## Change 1 — one counting convention (`_count`)

```python
def _count(self, text: str) -> int:
    return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
```

Replaces the inline `len(self._tokenizer.encode(...).ids)` at line 97. Rationale
and measurements: prd.md R3. Every count in the file goes through this one place
so the split path and the accumulation path can never drift apart.

## Change 2 — per-word cumulative token counts (`_token_prefix`)

The sweep needs `cum[i]` = tokens contained in `words[:i]`, with
`len(cum) == len(words) + 1`. Built with **one encode per oversized segment**:

1. Rebuild the joined text the same way `_piece` does (`self._separator.join`),
   recording each word's character start offset while joining.
2. `enc = tokenizer.encode(text, add_special_tokens=False)`; walk `enc.offsets`
   with a monotone word pointer, attributing each token to the word whose char
   range contains the token's **start** offset:
   `while w + 1 < len(words) and word_start[w + 1] <= token_start + len(self._separator): w += 1`.
3. Prefix-sum the per-word counts.

The `+ len(self._separator)` term is not cosmetic. SentencePiece emits the
word-initial marker as part of the token and reports the **preceding space** in
its offset — `▁exactly` in `"between exactly …"` comes back as `(7, 15)` while
the word itself starts at char 8. Without the term every word boundary is
credited one token early, a constant one-token skew across the whole prefix
(measured: `between` scoring 2 tokens instead of 1). With it, attribution is
exact on both paths; the term is 0 for CJK, where the separator is empty and
this problem does not arise.

Properties this buys:

- **Total is exact**: every token is attributed to exactly one word, so
  `cum[-1]` equals `_count(joined_text)` — the same number the accumulation loop
  would compute for that text. This is the invariant the whole design rests on:
  cut placement and buffer accumulation measure in identical units. Verified on
  every segment of both real fixtures (`en_aligned_test.json` 426 +
  `funasr_aligned_test_segments.json` 389): 0 mismatches out of 815.
- **Monotone**: `cum` is non-decreasing, which is all the sweep requires.
- **Residual boundary ambiguity is at most one word, and that is fine.** With the
  separator term above, the only remaining ambiguous case is a CJK token spanning
  two adjacent single-character "words" (`成了` covering two word entries) — it is
  credited to the word its start offset falls in. That can shift a cut by one
  word against a hypothetical exact accounting: far inside the snap tolerance,
  and it breaks neither monotonicity nor the total.
- Words with no `"word"` key contribute an empty string and zero tokens, so
  WhisperX's unaligned bare entries survive rather than being dropped.

## Change 3 — trigger and piece count

In `build_chunks`' normalisation loop, the pre-pass is now fed both bounds:

```python
segment_tokens = self._count(segment.get("text", ""))
duration = segment["end"] - segment["start"]
if segment_tokens <= max_tokens and duration <= max_duration:
    normalised.append(segment); counts.append(segment_tokens)
else:
    for piece in self._split_oversized_segment(segment, max_tokens=..., max_duration=...):
        normalised.append(piece); counts.append(self._count(piece["text"]))
```

`n_pieces = min(max(ceil(tokens / max_tokens), ceil(duration / max_duration)), len(words))`.
Taking the max (not the sum, not either alone) is what makes both bounds hold:
each equal share is then under both ceilings by construction, modulo word
granularity — the same best-effort caveat the duration version already carried
(a single 400-token word cannot be split).

**Counts are carried forward, not recomputed.** The normalisation loop already
has to tokenize each segment to decide whether to split, so it emits a parallel
`counts` list and the accumulation loop zips over `(segment, tokens)` instead of
calling `encode` again. Net effect: exactly one encode per segment (as today),
plus one extra per *oversized* segment for its word prefix.

## Change 4 — the sweep, in token space

```python
piece_tokens = cum[-1] / n_pieces
tolerance   = piece_tokens * PUNCTUATION_SNAP_RATIO
targets     = [i * piece_tokens for i in range(1, n_pieces)]
...
error = abs(cum[cut] - target)
if abs(cum[cut + 1] - target) > error: ...
```

`gap_time` is **kept**, unchanged, but demoted: it no longer supplies targets,
only the fallback `start`/`end` handed to `_piece` for a sub-range whose words
carry no timestamps at all. The seeded ends (`gap_time[0]`, `gap_time[len(words)]`)
still make the trailing piece indexable without a bounds check.

`PUNCTUATION_SNAP_RATIO` keeps its value (0.15) and its disjointness property:
targets are `piece_tokens` apart and windows are `±0.15 × piece_tokens`, so two
targets can never be pulled onto the same mark. Its docstring's *justification*
changes though — the old one argued time was the right unit precisely because a
word is not a fixed unit of time across languages. In token space that argument
is not needed: tokens are the unit the bound is expressed in, which is strictly
more direct. The comment must be rewritten, not left contradicting the code.

## Interaction with the known FunASR punctuation defect

The snap is a *preference*, never a requirement: when no candidate qualifies the
arithmetic boundary wins. On the FunASR Chinese path no candidate ever qualifies
(prd.md, out-of-scope 1), so those splits fall back to pure equal-token shares —
which is exactly the behaviour they have today under equal-time shares. This task
therefore neither fixes nor worsens that defect, and must not introduce any code
path that assumes punctuation is reachable from the word list.

## Test strategy

`FakeTokenizer` in `tests/test_size_based_chunker.py` grows two things:

- `encode(text, add_special_tokens=False)` — accepts and ignores the flag (the
  fake never emitted specials, so counts are unchanged and every existing
  assertion holds).
- `_Encoding.offsets` — from `re.finditer` spans of the same pattern, so the
  offsets are real character ranges into the same text.

Most existing pre-pass tests survive untouched: they use one-token-per-word
segments with words spread evenly in time, so equal token shares and equal time
shares coincide. The two token-density tests are new, and need a fake-tokenizer
segment where words differ in token weight (a multi-ideograph "word" counts as
several tokens under the existing `FakeTokenizer` pattern).

## Compatibility and rollback

Confined to one file plus its tests. `build_chunks`' signature, `ChunkSpan`, and
every caller are untouched. Rollback is `git revert` of the single commit; there
is no persisted artefact of chunk boundaries to migrate (`store.replace_chunks`
is delete-then-insert by design, and is not implemented yet regardless).
