# Implementation plan

All work is in `backend/`. Run commands from `backend/`.
Baseline first: `uv run pytest tests/test_size_based_chunker.py -q` must be green
*before* any edit, so a later failure is unambiguously ours.

## Step 1 — test scaffolding (before the source change)

`tests/test_size_based_chunker.py`:

- [x] `FakeTokenizer.encode(self, text, add_special_tokens=True)` — accept and
      ignore the flag.
- [x] `_Encoding` gains `offsets`, built from `_PATTERN.finditer(text)` spans
      alongside the existing ids.
- [x] Offsets fold in the preceding space, matching SentencePiece
      (`"Hello world."` → `[(0, 5), (5, 12)]`, not `[(0, 5), (6, 12)]`).
      Added beyond the original plan: without it the fake cannot tell a correct
      `_token_prefix` from one missing the `+ len(separator)` term, which would
      leave that regression detectable only by the network-gated opt-in test.
      CJK spans are unaffected (no whitespace to fold).
- [x] Confirm the suite is still green with only this change — 32 passed,
      1 skipped, identical to the pre-edit baseline.

Review gate: the fake must not start emitting special tokens — every existing
count assertion depends on its 1-token-per-word/ideograph model. Held: `ids` is
unchanged, only `offsets` is new.

## Step 2 — `_count` and the counting convention

`SizeBasedChunker.py`:

- [x] Add `_count(self, text: str) -> int` using `add_special_tokens=False`.
- [x] Replace the inline `len(self._tokenizer.encode(...).ids)` at line 97.
      `grep -n 'encode('` over `src/` now returns exactly one hit, inside
      `_count` — no other call site can reintroduce the default.
- [x] Run the suite — 32 passed, 1 skipped, unchanged as predicted.
- [x] Confirmed against the real BGE-M3 tokenizer on both fixtures: the gap
      between the old and new gate is exactly `2 × segment_count` in every
      chunk of both episodes.
- [x] Found while verifying: a second, unrelated gap (summed per-segment count
      vs the joined text's count) of ~8% on zh, 0 on en. Recorded as prd.md
      out-of-scope (5); not fixed here.

## Step 3 — `_token_prefix`

- [x] Add `_token_prefix(self, words: list[dict]) -> list[int]` per design.md
      Change 2: join with `self._separator`, record per-word char starts, one
      `encode(..., add_special_tokens=False)`, monotone pointer over
      `enc.offsets`, prefix-sum. Returns `len(words) + 1` entries.
- [x] The pointer condition is `word_start[w + 1] <= token_start + len(self._separator)`.
      Do not drop the separator term — without it SentencePiece's leading-space
      offsets skew every English word boundary by one token (design.md Change 2).
- [x] Unit-test it directly (`chunker._token_prefix(words)`): non-decreasing,
      `cum[0] == 0`, `cum[-1] == chunker._count(joined_text)`, and words with no
      `"word"` key contribute zero without shifting the total.
- [x] Include a case with a word that tokenizes to several pieces so the
      separator term is actually exercised. Under `FakeTokenizer` a Latin word
      is always 1 token, so the offline fixture uses a mixed-script word
      (`"你好a"` → 3); the real-tokenizer opt-in test uses `"we're"` → 3.
- [x] `self._separator` seeded in `__init__` so the helper is callable on a
      fresh instance instead of raising `AttributeError`. Does not touch the
      inverted language rule, which stays out of scope (prd.md item 2).

Review gate — **passed**:
- `prefix[-1] == _count(joined)` and monotonicity hold on **all 815 segments**
  of both fixtures under the real BGE-M3 tokenizer, run through the shipped
  implementation (0 mismatches, 0 non-monotone, correct length everywhere).
- Mutation check: deleting the `+ len(self._separator)` term fails 3 tests,
  2 of which run without network. The guard has teeth; file restored after.
- `41 passed` with `RAG_PODCAST_REAL_TOKENIZER=1`; `38 passed, 3 skipped`
  without.

## Step 4 — trigger, piece count, and carried-forward counts

- [x] `build_chunks`: compute `segment_tokens` in the normalisation loop; split
      when `segment_tokens > max_tokens or duration > max_duration`; build a
      parallel `token_counts` list; make the accumulation loop consume
      `zip(normalised, token_counts)` instead of re-encoding.
- [x] `_split_oversized_segment` takes `max_tokens` and `segment_tokens` (passed
      in, not recomputed — the caller already tokenized to reach the decision);
      `n_pieces = min(max(ceil(tokens / max_tokens), ceil(duration / max_duration)), len(words))`.
- [x] Update the docstring: its current "trigger is duration-only by design"
      paragraph now states the opposite of the code.
- [x] Widen the unsplittable-segment warning to name both bounds and the token
      count, not just duration.
- [x] `41 passed` with the real tokenizer enabled — no existing behaviour moved.
      Expected: the fixtures use `max_tokens=1000`, so nothing in the suite
      reaches the new trigger yet. Step 6 adds the cases that do.

Known intermediate state, resolved by step 5: the trigger and the piece count
are token-driven, but cut *placement* is still equal-time. Measured with a
skewed fixture (120 tokens in 60s, 110 of them inside the first 10s,
`max_tokens=50`): the split correctly asks for 3 pieces, but equal-time cuts
yield `[112, 8]` tokens, so the first piece is still over the bound. A uniform
fixture gives `[40, 40, 40]` because equal-time and equal-token coincide there.
Do not stop between steps 4 and 5 — this state fires the split without
achieving what it fires for.

## Step 5 — the sweep in token space

- [x] Replace `gap_time[cut]` / `gap_time[cut + 1]` in the sweep with
      `cum[cut]` / `cum[cut + 1]`; targets and tolerance from `piece_tokens`.
- [x] Keep `gap_time` for `_piece`'s fallback `start`/`end` only.
- [x] Rewrite `PUNCTUATION_SNAP_RATIO`'s module-level comment: the old
      "expressed in time, not in words" justification is now wrong; state the
      token rationale and keep the disjoint-window argument.
- [x] Shares derive from `cum[-1]`, not `segment_tokens`. The two differ
      whenever the segment's text carries characters the word list does not
      (FunASR punctuation), and a target from the larger number can sit past
      the end of `cum`, where the sweep never bottoms out. `n_pieces` still
      uses `segment_tokens` — it is a count, not a position.
- [x] Targets lost their origin offset: `gap_time` starts at
      `segment["start"]`, `cum` starts at 0, so `targets = [i * piece_tokens]`.

Verification:
- `41 passed` with the real tokenizer. Every existing duration-split test still
  passes untouched — their fixtures are uniform-density, where equal-time and
  equal-token cuts coincide.
- The step-4 skew fixture now yields `[40, 40, 40]` tokens over durations
  `[3.6s, 3.6s, 52.7s]`: equal in tokens, deliberately unequal in seconds. Was
  `[112, 8]` before this step.
- Real-tokenizer smoke over both episodes at four bound settings, including
  `max_tokens=25` (forces the token path hard: 244/412 chunks) and
  `max_duration=8.0` (forces the duration fallback): **0 chunks over
  `max_tokens` in every configuration, contiguity holds in every
  configuration**.

## Step 6 — new tests

All six live in a new "Token-driven split" section of
`tests/test_size_based_chunker.py`, between the oversized-segment pre-pass and
the contiguity section. Every expected value was hand-computed from the fixture
before the first run; all six passed on that run.

- [x] Token-only trigger: 200 tokens in 30s against `max_tokens=50`,
      `max_duration=90.0` → 4 pieces of 50, every piece far inside the duration
      bound. Needs `min_duration=1.0` in the fixture: at the default 20.0 the
      min-veto merges the four pieces straight back into one chunk and the
      split becomes unobservable through `build_chunks`.
- [x] Duration-only trigger: 60 tokens over 200s with `max_tokens=1000` → 3
      pieces of 20. The token term demands 1 piece, so the count is entirely
      the duration term's doing — guards against the token bound *replacing*
      duration rather than joining it.
- [x] Both bounds breached: 300 tokens over 180s, `max_tokens=60`
      (→ 5), `max_duration=50.0` (→ 4) → 5 pieces, each under both. `min()`
      or either term alone gives 4 pieces of 75 tokens, breaching `max_tokens`.
- [x] Uneven token density: new `_dense_then_sparse_segment` fixture — forty
      3-token words inside 0–10s, thirty 1-token words out to 60s. Yields
      `[51, 48, 51]` tokens over `[4.25s, 4.0s, 51.75s]`. The residual token
      spread is word granularity (a heavy word is an indivisible 3 tokens),
      not placement error.
- [x] Snap inside / outside the window. Both cases moved onto the uneven
      fixture rather than a uniform one — see the review gate below.
- [x] Real-tokenizer `_count` assertion: already shipped in step 2 as
      `test_count_excludes_special_tokens_on_the_real_tokenizer`. No new code.

Review gate — **passed**. Three mutations, each reverted after:

1. Trigger back to duration-only → **4 failures** (token-only trigger, uneven
   density, both snap tests).
2. `max(...)` → `min(...)` in `n_pieces` → **12 failures**, including all three
   new trigger tests.
3. Sweep back to `gap_time` / equal-time targets (i.e. undo step 5) → **1
   failure** on first attempt: only the uneven-density test caught it. The two
   snap tests as first written used a uniform 200-word fixture, where a window
   of `0.15` shares is the same set of words whether measured in tokens or in
   seconds — they pinned *where* the window sits but not *what it is measured
   in*, which is the entire content of step 5.

   Both were rewritten onto the uneven fixture so the units conflict directly:
   the "fires" case snaps at cumulative token 45 (5 inside the 7.5-token
   window) while sitting at 3.75s, nowhere near the 20s equal-time boundary;
   the "ignored" case puts a sentence end at exactly 20.0s — dead on the
   equal-time boundary — but 76 tokens from the 50-token target, so it must be
   ignored. A seconds-measured window snaps there and yields a 126-token first
   piece. Re-running mutation 3 after the rewrite: **3 failures**.

Note for step 7: the pre-existing snap tests
(`test_oversized_split_snaps_a_cut_onto_a_sentence_end` and friends) are the
uniform-fixture kind, so they are not evidence for step 5 either way. The two
new ones are the only guards on the window's unit.

Counts: `44 passed, 3 skipped` offline (was `38 passed, 3 skipped`);
`47 passed` with `RAG_PODCAST_REAL_TOKENIZER=1`.

## Step 7 — validation

- [x] `uv run pytest tests/test_size_based_chunker.py -q` — `44 passed,
      3 skipped`; `47 passed` with the real tokenizer.
- [x] `uv run pytest -q` (whole suite — nothing else should be affected).
      Nothing this task touched moved, but the suite is **not clean on its
      own**, both causes pre-existing and unrelated:
      - `tests/test_transcriber.py` fails to *collect*:
        `ModuleNotFoundError: rag_podcast.transcription.transcriber`. The
        module was renamed `transcriber.py` → `Transcriber.py` in the earlier
        ASR-interface refactor and the test's import was never updated (git
        still shows the old casing on Windows). Belongs to that task, not this
        one.
      - `tests/ingestion/test_service.py`: 7 failed / 7 teardown errors,
        `socket.gaierror: getaddrinfo failed` — needs a live Postgres this
        environment has no network for.
      With those two modules set aside: `106 passed, 3 skipped`.
- [ ] Smoke against real data, both languages, and eyeball chunk sizes:
      `uv run python scripts/test_chunker.py` (Chinese, FunASR) — note it calls
      `build_chunks` positionally; if the call breaks, that is a signal the
      signature changed, which this task said it would not do.
- [ ] Ad-hoc check on `en_aligned_test.json` that chunk token counts moved down
      by ~`2 × segments_per_chunk` (the R3 correction landing), and that no chunk
      exceeds `max_tokens` except where a min-veto legitimately forces it.

## Rollback points

Each step is independently revertible; steps 1–3 are additive and cannot change
behaviour on their own. The behavioural switch is step 4 — if the smoke test in
step 7 looks wrong, revert steps 4–5 and keep 1–3 (the counting fix stands on its
own merit).

## Not in this task

See prd.md "Out of scope": FunASR punctuation, the inverted `_separator` rule,
the commented-out keyword-only `*`, and tier re-calibration. Do not fix them in
passing — each needs its own verification.
