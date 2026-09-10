# Design: chunking (indexing module)

## Architecture

Mirrors the transcription pipeline's shape (status column + poll/claim worker + thin CLI) — already established, already crash-safe. The chunking step itself is a **swappable-strategy interface**, mirroring `transcription/`'s `Transcriber` Protocol (WhisperX vs FunASR) exactly: one interface file, one file per concrete strategy, one factory.

```
episode.transcript_status = DONE, index_status = PENDING
  → indexing worker claims (index_status → PROCESSING)
  → cleaning.clean_words(segments, language) → whole-transcript cleaned segments  [cleaning/, runs once, before chunking]
  → resolve token-tier (min/max/desired tokens) from episode's total duration via settings.audio_tier
  → chunker.build_chunks(cleaned_segments, min_tokens, max_tokens, desired_tokens, min_duration, max_duration, language)
       [chunker: Chunker Protocol instance built once at worker startup by Chunker.py::create_chunker()]
       → ChunkSpan{text (cleaned), start, end, segments (cleaned segment-shaped dicts for this span)}
  → indexing.embedder.embed_batch([chunk.text for chunk in chunks])  [indexing/, httpx → DeepInfra]
  → indexing.store.replace_chunks(episode, chunks, embeddings)  [indexing/, delete-then-insert]
  → index_status → DONE (or FAILED on any step's exception)
```

Cleaning runs once, upfront, on the whole transcript — *before* chunking, not per-chunk after (**revised 2026-08-12, user**: was previously per-chunk-after; reversed since the chunker's input is meant to be cleaned segments, per `Chunker` Protocol's own docstring). `chunk.text` is the cleaned text (filler words removed) — the same text used for storage, display/jump-to-audio, and the embedding call; there is no separate ephemeral string. `ChunkSpan.text` is built directly from the (already-cleaned) segments — no rebuild inside the chunker itself, no risk of the CJK "H e l l o" word-rebuild bug there (that risk lives entirely inside `clean_words`, which already handles it — see `cleaning/filler_words.py`). `embed_batch` sends `chunk.text` straight to the embedding API with no further clean/join step.

## `indexing/` module layout

One file per chunking strategy, all implementing the same `Chunker` Protocol and returning the same `list[ChunkSpan]` shape — downstream code (`embedder.py`, `store.py`, `worker.py`) imports only `BaseChunkerInterface.py`'s types, never a concrete class, so it contains zero branching on which strategy ran.

```
indexing/
├── BaseChunkerInterface.py     # ChunkSpan + ChunkingError + Chunker Protocol
├── SizeBasedChunker.py         # Phase 1 concrete implementation
├── SimilarityBiasedChunker.py  # Phase 2 slot — reserved, raises NotImplementedError for now
├── Chunker.py                  # create_chunker(strategy) factory — the only file that imports both concrete classes
├── tokenizer.py                # get_tokenizer() — BGE-M3 tokenizer loader, called once by create_chunker()
├── embedder.py                 # embed_batch() + EmbeddingError — single implementation, not a Protocol
├── store.py                    # replace_chunks()
├── worker.py                   # poll/claim/index orchestration loop
└── cli.py                      # thin entrypoint
```

Naming mirrors `transcription/`'s convention deliberately, not uniformly: PascalCase marks the swappable-strategy axis (`BaseTranscriberInterface.py` → `BaseTranscriberInterface`-style here too); snake_case marks orchestration/entrypoint/single-implementation files. Chunking has exactly one swappable axis (strategy), so only `BaseChunkerInterface.py`/`SizeBasedChunker.py`/`SimilarityBiasedChunker.py`/`Chunker.py` get PascalCase.

**No per-episode routing wrapper** (unlike transcription's `LanguageRoutingTranscriber`/`LocalLanguageDetect.py`): transcription auto-routes *per episode* at runtime based on detected language. Chunking strategy is one global choice for the whole indexing run (a config value), so `Chunker.py`'s factory does the whole job — no third "router" class needed unless a future requirement asks for per-episode strategy selection.

### `BaseChunkerInterface.py`
```python
class ChunkingError(Exception): ...

@dataclass
class ChunkSpan:
    text: str             # cleaned — concatenation of included (already-cleaned) segments' own text
    start: float
    end: float
    segments: list[dict]  # cleaned segment-shaped dicts spanned by this chunk — NOT a flat word list

class Chunker(Protocol):
    def build_chunks(
        self, segments: list[dict], *,
        min_tokens: int, max_tokens: int, desired_tokens: int,
        min_duration: float, max_duration: float,
        language: str | None = None,
    ) -> list[ChunkSpan]: ...
```

**`ChunkSpan.segments`, not `.words`**: carries the (already-cleaned) segment-shaped dicts (`{"text", "start", "end", "words": [...]}`) spanned by the chunk. Segment shape (not a flat word list) is kept because word-level timestamps are needed to derive `ChunkSpan.start`/`.end`, and the oversized-segment pre-pass needs per-word timing to find a split point. For whole-segment chunks this is just the matching sublist of the (cleaned) input `segments`; for a piece produced by the pre-pass, each word sub-span is wrapped in a single synthetic segment dict (`text` rebuilt via `join_words`, `avg_logprob` omitted — it described the original whole segment and no longer applies). Anywhere a flat word list is actually needed (contiguity checks) it's derived on demand: `[w for seg in chunk.segments for w in seg["words"]]`.

### `SizeBasedChunker.py` (Phase 1)
```python
class SizeBasedChunker:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer   # loaded once, held for the instance's lifetime

    def build_chunks(self, segments, *, min_tokens, max_tokens, desired_tokens,
                      min_duration, max_duration, language=None) -> list[ChunkSpan]: ...
```
Tokenizer is constructor state, not a per-call argument — mirrors `LocalWhisperX.__init__`'s eager-load-once precedent (model/tokenizer loaded once at construction, held for the object's lifetime, never reloaded per-chunk/per-episode). `min_tokens`/`max_tokens`/`desired_tokens` arrive pre-resolved from the caller (`worker.py`, picking a tier from the episode's total runtime) — `build_chunks` itself has no tiering logic, keeping it a pure, easily-unit-testable function of plain numeric bounds.

**Algorithm** — greedy accumulation over segments in order, extending the old duration-only loop with a parallel running token count:
1. **Pre-pass — normalise oversized segments (`_split_long_segments`)**, before the accumulation loop runs at all. Any segment whose own duration exceeds `max_duration` is split at the word level into `ceil(duration / max_duration)` roughly equal pieces; everything else passes through untouched. Returns a new list, never mutating the caller's `segments` (matching `clean_words`' convention — the worker still holds the original `transcript_data`). **Trigger is duration-only, no token check** (BGE-M3's 8192-token ceiling gives enormous headroom over these budgets-in-the-hundreds, so a token-dense-but-short single segment isn't a real overflow risk). See "Oversized-segment pre-pass" below for why this is a pre-pass rather than an in-loop fallback.
2. Start a new chunk buffer at the next unconsumed segment. Track both its wall-clock span and a running `buffer_token_count`.
3. Each segment's token count is computed once — the loop visits each segment exactly once and carries the result forward in `buffer_token_count` (O(n) over the episode, not O(n²) — no re-tokenizing the buffer's joined text every iteration).
4. **Closing gate, symmetric across both dimensions:**
   - **Hard bounds (OR on max, AND on min)**: close once appending the next segment would exceed *either* `max_duration` or `max_tokens` (token-max expected to fire in practice; duration-max stays independently active as a low-token-density safety net). Only allow that close if **both** `min_duration` and `min_tokens` are already cleared — token-min alone isn't sufficient (a token-dense-but-very-short chunk is still choppy to listen to). If a max would be exceeded but a min hasn't cleared, append anyway — contiguity and the min floors win over the soft max target.
   - **Greedy nearest-target stopping, within the legal zone**: if neither max would be breached by the next segment *and* both mins are already cleared, compare `error_current = |desired_tokens - buffer_token_count|` against `error_next = |desired_tokens - (buffer_token_count + next_segment_token_count)|`; close here if `error_current < error_next`. Valid one-step lookahead (not a full scan): cumulative token count only grows as segments are appended, so the error is V-shaped in segment position — the first segment where "one more makes it worse" is exactly the local-optimum stop. Scoped to tokens only; duration has no target/lookahead of its own, staying a pure hard OR-max/AND-min bound.
5. `ChunkSpan.start`/`.end` = first word's start / last word's end of the words actually included (derived from `chunk.segments`) — never the raw segment boundary.
6. `ChunkSpan.text` = concatenation of included segments' own `text` fields — never rebuilt from any word list.
7. No overlap: flattening every emitted chunk's `segments` back into words must reconstruct the full original word sequence with nothing skipped or duplicated.

**Oversized-segment pre-pass** (**revised 2026-08-21, user** — previously an in-loop fallback, `_split_oversized_segment`, that emitted finished `ChunkSpan`s directly):

The earlier shape had two problems that the pre-pass framing removes together. It closed pieces greedily at max width, so a 361s segment against a 360s bound produced 360s + 1s; and because the fallback emitted *finished chunks* while the accumulation buffer restarted at the following segment, that 1s remainder could never merge with anything — it stood as its own chunk. Splitting into equal shares fixes the first (2 x 180.5s); splitting into plain *segments* before the loop fixes the second, since the pieces then re-enter the ordinary gate and an end piece falling under the mins merges with its neighbour. It also deletes the in-loop special case entirely, along with the question of whether an open buffer should be flushed at the seam.

Cut placement, in `_split_segment` / `_cut_index`:
- Each of the `n_pieces - 1` equal-share boundary *times* independently picks the word boundary nearest to it.
- That index is then snapped to sentence-final punctuation within `PUNCTUATION_SNAP_WINDOW` (5) words, when a candidate qualifies — both the CJK standalone-token and Latin fused-onto-word shapes. A slightly uneven split at a sentence end reads better than an exactly even one mid-clause.
- Cuts are **deduplicated** (`sorted(set(...) - {0, len(words)})`) rather than constrained to disjoint windows. Two adjacent targets can select the same word when words are sparse relative to the segment's length (long silence, music, a stretch WhisperX barely aligned), and a collision means there is no honest boundary separating those shares — emitting fewer, better-placed pieces beats forcing a cut at an arbitrary word far from its target. Nothing downstream depends on the piece count equalling `n_pieces`. Dropping `0` and `len(words)` rules out an empty leading/trailing piece.
- Words missing `start`/`end` are not a special case: WhisperX emits bare `{"word": ...}` entries for characters it could not align, so boundary time falls back to the last known timestamp, keeping those words in the transcript rather than dropping them.
- A segment that is over-long but carries fewer than two words cannot be cut; it is logged and emitted whole. Dropping audio content is worse than an over-long chunk, and the main loop handles an over-max segment without special-casing.
- Pieces are **not** guaranteed to fall under `max_duration` — word granularity is the floor (a single 400s word cannot be split), so the bound is best-effort by construction.

**Reasons this lives in the chunker, not in `cleaning`**: it needs `max_duration`, a chunking parameter, so threading it into `clean_words` would couple filler-word removal to chunk sizing; `cleaning/` is owned by the archived `07-28-filler-word-cleaning` task and is only consumed here; and the normalisation is strategy-specific — `SimilarityBiasedChunker` may want a different pre-split (e.g. on sentence boundaries regardless of duration). Merging it into `clean_words`' loop would save one O(n) pass over a few thousand segments, i.e. nothing.

### `SimilarityBiasedChunker.py` (Phase 2 — reserved, not blocking Phase 1)
Skeleton implementing `Chunker`, `build_chunks` raises `NotImplementedError` for now. When built: within the same min/max bounds, bias *which* candidate boundary a chunk closes at toward the point of highest embedding distance between adjacent units (punctuation-delimited sentences, not raw segments), instead of Phase 1's nearest-token-target criterion. Uses the existing hosted embedding API (`embedder.py`), optionally a cheaper model since these vectors are ephemeral/never persisted. No eval harness exists yet to validate quality (spec.md M4) — built as a drop-in `Chunker` once that harness lands, not gated on it existing first.

### `Chunker.py` (factory)
```python
def create_chunker(strategy: str = "size_based") -> Chunker:
    tokenizer = get_tokenizer()
    if strategy == "size_based":
        return SizeBasedChunker(tokenizer)
    if strategy == "similarity_biased":
        raise NotImplementedError("Phase 2 not yet implemented")
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
```
Called once at worker startup (mirrors `Transcriber.py::create_transcriber`). Swapping strategies is a one-argument change here — no downstream file ever needs to change.

### `tokenizer.py`
```python
def get_tokenizer() -> Tokenizer:
    path = hf_hub_download(
        repo_id="BAAI/bge-m3", filename="tokenizer.json",
        cache_dir=Path(settings.data_dir) / "hf_cache", revision=<pinned_sha>,
    )
    return Tokenizer.from_file(path)
```
Plain loader, no internal singleton/caching — the "once per process" guarantee comes from `create_chunker` only calling it once, same as `LocalWhisperX` has no module-level guard of its own. Explicit `cache_dir` (not `HF_HOME`) scopes the cache to this project's `data_dir`, avoiding leakage into other repos' HF caches on the same machine. Pin `revision=<commit_sha>` so the vocab can't silently change later; combine with `HF_HUB_OFFLINE=1` (set narrowly for the indexing worker's process env) once cached, so restarts don't need network reachability. First run needs network access to huggingface.co (~17MB download) — the indexing worker's first real network dependency, slightly undercutting the "no GPU/extra needed" framing below.

### Token threshold tiers — **Status: Done**, see `config.py:53-70`
`settings.audio_tier` — a dict keyed `"Short"`/`"Medium"`/`"Long"`, each holding `max_episode_duration_seconds` (single boundary, ascending, `"Long"` uses `float("inf")`), `min_chunk_tokens`, `desired_chunk_tokens`, `max_chunk_tokens`. Lookup is first-match-wins iterating tiers in written order (`Short → Medium → Long`) — no per-tier `min` field, since duplicating both edges of each boundary risked drift between adjacent tiers if edited independently; the single shared boundary can't fall out of sync with itself.

Numbers pinned 2026-07-29 `[assumption]`, anchored via `backend/scripts/tokenizer_playground.py` against `en_aligned_test.json` (real 426-segment, 32.4min English episode, ~248.5 tokens/min): Medium tier's `max_chunk_tokens=400` matches what the old 90s duration bound implied in token terms for a typical-paced episode; Short/Long are extrapolated (~0.6x/~1.5x), not independently measured against real short-news or long-form audio. Tier boundaries (15min/45min) are a reasoned guess. Revisit once real episodes of those genres are indexed.

**Single table for all languages, not split by language**: a controlled test (3 translation-equivalent sentence pairs) tokenized to 39 tokens English vs. 39 tokens Chinese — exactly equal, no directional bias — confirms BGE-M3 subword tokenization doesn't need per-language thresholds. An aggregate ~25% gap seen between two real episodes is a speaking-pace/genre artifact (already absorbed by the duration-tier axis), not a tokenizer effect.

**Duration bounds: flat, widened, not tiered — Status: Done**, `chunk_min_duration_seconds=20.0` (unchanged), `chunk_max_duration_seconds=360.0` (up from 90.0) — sized so even a half-rate talker reaches the Long tier's 600-token ceiling (~290s) with headroom before duration_max would cut it off first. Duration stays a flat safety-net bound, not tiered, since it's now secondary to the token-count signal.

### `indexing/embedder.py`
```python
class EmbeddingError(Exception): ...

async def embed_batch(texts: list[str]) -> list[list[float]]:
    """POST {embedding_base_url}/embeddings, model=settings.embedding_model,
    input=texts. One batched call per episode's full chunk list, not one
    call per chunk."""
```
`httpx.AsyncClient` (already a dependency). Not a Protocol — only one embedding backend is configured; swapping providers is a settings change (`embedding_base_url`/`key`/`model`), not a code change. Raises `EmbeddingError` on non-2xx/malformed response, caught by the worker the same way `TranscribeError` is caught today.

Caller (`worker.py`) passes `chunk.text` straight through — no per-chunk `clean_words`/`join_words` call, since cleaning already ran once on the whole transcript before chunking (see Architecture above):
```python
embeddings = await embed_batch([chunk.text for chunk in chunks])
```
`embed_batch` itself takes plain strings and has no opinion on how they were built.

### `indexing/store.py`
```python
async def replace_chunks(
    session: AsyncSession, episode: Episode,
    chunks: list[ChunkSpan], embeddings: list[list[float]],
) -> None:
    """Delete existing chunk rows for episode.id, insert the new set, commit.
    Delete+insert (not diff) — chunk boundaries can change between runs
    (different clean/chunk params), so there's no stable identity to diff
    against. Writes `text`, `start`, `end` straight from each `ChunkSpan`
    (start/end already word-level-derived by the chunker, never recomputed
    here) plus the caller-supplied `embedding`. search_vector is DB-generated
    (see below) — no app code needs to populate it."""
```

### `chunk.search_vector` (full-text search column) — **Status: Done**
Generated `TSVECTOR` column + GIN index, see `models/chunk.py` and migration `0004_indexing_fields.py`. `GENERATED ALWAYS AS to_tsvector('english', text)` — always derived from `text`, no app code needs to populate it. Ranking/query use (`ts_rank`, RRF merge) deferred to the future retrieval task.

**Known limitation**: fixed `'english'` text-search config — can't vary per row by `episode.language` (Postgres generated columns need a constant expression). Acceptable while target podcasts are English; revisit if a non-English podcast needs full-text search (would need `'simple'` config or a non-generated, app-populated column).

### `indexing/worker.py`
Same shape as `transcription/worker.py`, retargeted at `index_status`, plus one addition with no transcription equivalent:
- `reset_stale_processing` — `index_status: PROCESSING → PENDING` on startup.
- `poll_ready` — next episode where `transcript_status = DONE AND index_status = PENDING`, ordered by id.
- `claim_episode` — atomic `UPDATE ... WHERE index_status = PENDING RETURNING *`.
- `resolve_token_tier(episode_duration_seconds) -> dict` — first-match-wins lookup into `settings.audio_tier` (see above). Episode duration = `segments[-1]["end"] - segments[0]["start"]`.
- `index_episode` — orchestrates: `clean_words(segments, episode.language)` (once, on the whole transcript) → resolve tier → `chunker.build_chunks(cleaned_segments, ...)` (depends only on the `Chunker` Protocol, never a concrete class — same dependency direction as today's worker depending only on `Transcriber`) → `embedder.embed_batch([c.text for c in chunks])` → `store.replace_chunks(...)` → status update. Any exception at any step sets `index_status = FAILED` and commits (per-episode failure isolation, matches transcription).
- `run_worker` — same poll/sleep loop shape.

### `indexing/cli.py`
Thin argparse wrapper mirroring `transcription/cli.py`. No `[extra]`-guarded import block needed (unlike transcription's whisperx guard) — `tokenizers`/`httpx` are core deps, not optional. Entry point: `uv run python -m rag_podcast.indexing.cli`. Calls `create_chunker()` + `run_worker(...)`.

## Prerequisite fixes — Status: Done

- **`transcription/transcriber.py`**: language captured from pre-align `model.transcribe()` result, merged back into the post-align dict (`transcriber.py:161-187`). `worker.py:130` sets `episode.language` alongside `transcript_data`/`transcript_status`.
- **`cleaning/filler_words.py` + `cleaning/text_join.py`**: implemented, owned by the separate archived `07-28-filler-word-cleaning` task. This task only consumes `clean_words(segments, language=None)` / `join_words(words, language=None)`.
- **Migration `0004_indexing_fields.py`**: applied — `episode.index_status` enum (mirrors `0002`'s pattern), `episode.language` (nullable, no default), `chunk.search_vector` (generated + GIN index). Has a symmetric `downgrade()`.

## Compatibility

- `chunk` table gets one additive column (`search_vector`); `episode` gets two (`index_status`, `language`). No existing column changes.
- `settings.embedding_*` already existed; new config is additive (`chunk_min/max_duration_seconds`, `audio_tier`).

## Rollback

- Migration `0004` has a symmetric `downgrade()` — safe to revert independently of code changes.
- Each new `indexing/*` module is additive; no existing module is modified beyond the transcriber fix. Reverting is "delete the new files + revert the transcriber fix + revert the migration." `cleaning/*` is out of scope for this task entirely.
