# Redesign: `LocalFunASR._funasr_to_segments`

Status: implemented. Kept as the rationale for why `sentence_info` is ignored —
the reasoning is not recoverable from the code alone.
Goal unchanged — produce the WhisperX-shaped `segments`/`words` contract with
word-level timestamps. What changes is where the data comes from.

Measured after implementation, on the same 108-minute episode: 1008 segments
(median 5.5s, mean 6.3s, max 54.6s, none over the 360s chunk bound), 34138 words
reproducing `result["timestamp"]` entry for entry, and
`join_words(words, "zh")` reproducing `result["text"]` exactly. The 1008 rather
than the 1000 predicted in section 6 are the half-width `.` sentence ends
(`CP.`), which section 6's measurement did not yet count as segment-final; the
implementation does.

---

## 1. Why

`sentence_info` — the field the current adapter reads — is misaligned by a bug
in funasr 1.3.30. Verified end to end on a 108-minute episode (`2.m4a`,
半拿铁·周刊).

### The bug

FunASR drives sentence splitting off `punc_array`, CT-Transformer's per-token
punctuation prediction. Two different tokenizers index into it:

| | count on the probe clip | used by |
|---|---|---|
| `split_words(punc_input_text)` | 233 | `punc_array` — the index basis |
| `punc_input_text.split()` | 230 | `timestamp_sentence()` |

`_join_vad_texts` (`funasr/auto/auto_model.py:56-68`) joins two VAD chunks with
**no separator** when the previous chunk ends on a CJK char and the next starts
on one. That glues tokens across the boundary into a single whitespace-delimited
unit:

```
VAD chunk 0: '您 的 半 拿 铁 周 刊 请 查 收'
VAD chunk 1: '嗯'
VAD chunk 2: '只 要 你 有 一 颗 少 女 心 ...'

punc_input_text: '您 的 半 拿 铁 周 刊 请 查 收嗯只 要 你 有 ...'
                                         ^^^^^^ three chunks, one token
```

`split_words()` expands multi-char CJK tokens back to one token per character,
so `punc_array` still counts 3. `text.split()` in `timestamp_sentence()`
(`funasr/utils/timestamp_tools.py:169-173`) counts 1, then `zip_longest`s the
two positionally. **The text stream falls one slot behind the punctuation
stream at every CJK-CJK VAD boundary, cumulatively.**

Full episode: deficit 1118 over 34138 tokens. Punctuation drift grows
monotonically to +1177 characters, then `texts` runs out and the last 113
`sentence_info` entries carry **no text at all** — just a punctuation mark over
1118 timestamp slots (the final ~6 minutes of the episode).

### Why the top-level `text` is fine

funasr 1.3.30 has a newer surface-preserving path
(`_punctuate_surface_text` / `_timestamp_sentences_from_surface`) that uses
`split_words()` on both sides and does not drift. Its entry condition is
`result.get("words")` (`auto_model.py:1016-1026`) — and **paraformer-zh does not
emit a `words` key**, so the whole path is skipped. Probe output:

```
result has 'words' key            : False
punctuated_surface returned       : None
surface_sentences returned        : None
legacy timestamp_sentence() used  : True
```

`result["text"]` therefore falls back to `punc_res[0]["text"]`, CT-Transformer's
own renderer (`funasr/models/ct_transformer/model.py:381-408`), which walks the
`split_words` token sequence directly and is index-consistent by construction.

### What is still trustworthy

`len(punc_array) == len(result["timestamp"]) == 34138`, and
`sum(len(s["timestamp"]) for s in sentence_info) == 34138`. The *sentence
boundaries in token-index space are correct*; only the step that fills text into
those slots is broken. So the fix needs no re-transcription.

---

## 2. Principle

Stop reading `sentence_info`. Derive everything from two fields that share one
tokenization:

| field | content | trust |
|---|---|---|
| `result["text"]` | CT-Transformer's punctuated rendering | correct |
| `result["timestamp"]` | one `[start_ms, end_ms]` per token | correct |
| `result["sentence_info"]` | — | discard |

This also drops the dependency on funasr's private splitting helpers, so a
funasr upgrade cannot silently change segment shape.

---

## 3. Tokenizer — must mirror `split_words`

Two rules (`funasr/models/ct_transformer/utils.py:74-92`):

- a non-ASCII character (CJK) is its own token
- a run of ASCII characters (`CP`, `driver`, `AI`) is one token, terminated by
  whitespace or by a CJK character

One token consumes exactly one `timestamp` entry.

## 4. Punctuation classification

CT-Transformer's `punc_list` is `_ ， 。 ？ 、`, but `model.py:401-407` converts
a mark to **half-width when the preceding token is ASCII** — so `CP.` and `AI,`
contain ASCII `.` / `,` that rule 2 above would otherwise swallow into the
token.

```
ch is punctuation  ⟺  ch ∈ {，。？！、；：…}
                    ∨  (ch ∈ {, . ? ! ; :} ∧ next char is not [A-Za-z0-9])
```

The lookahead is sound because CT-Transformer guarantees a space between two
adjacent ASCII tokens (`model.py:392-397`):

- `3. 5` (model-inserted period) → next char is a space → punctuation ✓
- `3.5` (a decimal the ASR emitted as one token) → next char is `5` → token
  content ✓

Validated: round-trip over the full 108-minute episode reproduces
`result["text"]` exactly.

## 5. Alignment invariant

```
len(tokens) == len(result["timestamp"])
```

On mismatch, **raise `TranscribeError`** carrying both counts and a text
excerpt. Rationale: the failure is deterministic (a retry cannot help) and a
misaligned timestamp stream silently poisons every chunk and citation
downstream — which is exactly how the current bug survived to production. A loud
ingest failure is cheaper than a plausible-looking bad transcript.

## 6. Segment boundaries: sentence-final marks only

Split on `。？！`. Keep `，、` inside the segment text. Measured over the full
episode:

| | A. every mark (current `sentence_info` behaviour) | B. `。？！` only |
|---|---|---|
| segments | 3238 | **1000** |
| duration median / mean | 1.58s / 1.86s | 5.57s / 6.32s |
| p90 / p99 | 3.52s / 6.41s | 10.80s / 18.33s |
| max | 28.9s | 90.6s |
| chars mean | 11.7 | 37.9 |
| segments over `chunk_max_duration` (360s) | 0 | **0** |

Choose **B**. Chunks are built by merging whole segments, so chunk boundaries
always land on segment boundaries: under A a chunk can end at a comma and cut a
sentence in half before embedding; under B every chunk ends on a complete
sentence. B's longest segment (90.6s) stays well under the 360s bound, so
`SizeBasedChunker._split_oversized` is not newly stressed.

`！` never actually appears (it is not in CT-Transformer's `punc_list`) — it is
in the set defensively.

## 7. Word shape: punctuation stays attached to the token

Change from today's behaviour, which strips punctuation out of `words`
(`LocalFunASR.py:122`). Emit `{"word": "对，", ...}`, the way WhisperX does for
English. Three call sites already assume this:

1. `cleaning/filler_words.py:36-46` — `clean_words` **rebuilds `text` from the
   surviving `words`**. With punctuation-free words, every cleaned Chinese
   segment loses all punctuation before it reaches the embedder.
2. `indexing/SizeBasedChunker.py:405-406` — the sentence-end snap reads
   `words[cut - 1]["word"][-1:]` against `SENTENCE_FINAL_PUNCTUATION`. It can
   never fire on `zh` today; the branch is dead code for Chinese episodes.
3. `indexing/SizeBasedChunker.py:368-372` — the comment "the segment's own text
   carries characters the word list does not (FunASR writes punctuation into
   text only)" describes precisely this asymmetry. Once fixed,
   `join_words(words, "zh") == segment["text"]` holds exactly.

**Coupled change, must ship in the same batch:** `filler_words.py:13`'s
`_STRIP_CHARS = string.punctuation + string.whitespace` is ASCII-only, so
`_normalize("嗯，")` returns `"嗯，"` and no longer matches `ZH_FILLER_WORDS`'s
`嗯`. Chinese filler-word removal would silently regress. Add the CJK marks to
`_STRIP_CHARS`.

## 8. Drop `rich_transcription_postprocess`

`LocalFunASR.py:71` applies a SenseVoice-oriented helper. For paraformer it is
effectively `s.strip()` (`<|NEUTRAL|>` maps to the empty string), but on any
`<|...|>` tag it **injects emoji into the text**
(`funasr/utils/postprocess_utils.py:338-357`) — characters with no corresponding
timestamp, which would break the section 5 invariant on the spot. Remove the
call. If it is kept for some reason, the invariant turns the hazard into a loud
failure rather than a silent misalignment.

---

## 9. Shape of the new implementation

No `_segment()` helper — the segment dict is built inline at the two places a
segment closes.

```python
def _funasr_to_segments(self, result: dict) -> dict:
    text = result.get("text", "")
    stamps = result.get("timestamp") or []

    tokens = self._tokenize(text)        # [(token, trailing_punct), ...]
    if len(tokens) != len(stamps):
        raise TranscribeError(...)       # both counts + a text excerpt

    segments: list[dict] = []
    words: list[dict] = []
    for (token, punct), (start_ms, end_ms) in zip(tokens, stamps):
        words.append(
            {"word": token + punct, "start": start_ms / 1000, "end": end_ms / 1000}
        )
        if punct and punct[-1] in SEGMENT_FINAL_PUNCTUATION:
            segments.append(
                {
                    "text": "".join(w["word"] for w in words),
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "words": words,
                }
            )
            words = []

    if words:                            # tail with no closing punctuation
        segments.append(
            {
                "text": "".join(w["word"] for w in words),
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "words": words,
            }
        )

    return {"language": "zh", "segments": segments}
```

Unchanged parts of the contract: no per-segment `avg_logprob` and no per-word
`score` (FunASR produces neither), no top-level `word_segments`, `language`
stays hard-coded `"zh"` since the caller already routes by language.

## 10. Edge cases

| case | handling |
|---|---|
| `text` empty or whitespace-only | return `{"language": "zh", "segments": []}`, no error |
| `timestamp` key missing | treated as `[]` → length mismatch → `TranscribeError` |
| text ends without punctuation | trailing segment emitted by the `if words` tail |
| long silence inside a segment | `start`/`end` come from first/last word, silence sits inside the span — same as WhisperX |

## 11. Tests

`_tokenize` and `_funasr_to_segments` are pure `(text, timestamps) -> dict`
functions, testable without loading a model. `data/funasr_raw_ep2.json` is a
ready golden fixture (excerpt a slice into `tests/`).

1. `len(tokens) == len(timestamps)`
2. `join_words(all_words, "zh") == "".join(result["text"].split())` — lossless
   round-trip
3. concatenating every segment's word timestamps reproduces
   `result["timestamp"]` exactly, in order — the assertion that would have
   caught the old implementation dropping its last 1118 slots

## 12. Change surface

- `LocalFunASR._funasr_to_segments` — rewritten; `_FUNASR_PUNCTUATION`
  strip-and-zip removed; new `_tokenize`
- `LocalFunASR._transcribe_sync` — drop `rich_transcription_postprocess`
- `cleaning/filler_words.py` — extend `_STRIP_CHARS` with CJK punctuation
- `indexing/SizeBasedChunker.py:368-372` — stale comment, its premise is gone

## 13. Reproduction artifacts

- `backend/scripts/funasr_raw_dump.py` — dumps native FunASR output, no
  WhisperX adaptation
- `backend/scripts/funasr_punc_probe.py` — monkeypatches the punctuation path
  and prints which branch ran plus both token counts
- `data/funasr_raw_ep2.json`, `data/funasr_raw_ep2_preview.txt` — full-episode
  dump used for every number above
