# Design: embed chunks and persist to pgvector

## Architecture

A third long-running process alongside `api` and the transcription `worker`,
using the same database-as-queue pattern already established. No new
infrastructure, no message broker, no schema change.

```
episode.transcript_status = done AND index_status = pending
  → claim (index_status → processing, committed immediately)
  → clean_words(transcript_data["segments"], episode.language)      [cleaning/]
  → chunker.build_chunks(cleaned, **tier_bounds)                    [indexing/, Chunker Protocol]
  → embed_batch([span.text for span in spans])                      [embedding/embedder, openai → DeepInfra]
  → replace_chunks(session, episode, spans, vectors)                [embedding/store, no commit]
  → episode.index_status = done
  → ONE commit  ────────────────────────────────────────────────────┐
                                                                    │ R2 atomicity
  any exception → rollback that transaction, then set failed ───────┘
```

The single-commit boundary is the load-bearing part of this design: the DELETE of
the old rows, the INSERT of the new rows, and the `done` status transition either
all land or none do. `claim` commits before this, and the `failed` transition
commits after a rollback — both are deliberately outside it.

## Module layout

```
indexing/                      # chunking: split cleaned text into ChunkSpans
├── BaseChunkerInterface.py   # exists — ChunkSpan, Chunker Protocol, ChunkingError
├── SizeBasedChunker.py       # exists — Phase 1 strategy
├── SimilarityBiasedChunker.py# exists — Phase 2 stub, NotImplementedError
├── tokenizer.py              # exists — get_tokenizer()
└── Chunker.py                # NEW — create_chunker() factory

embedding/                     # embed → persist → orchestrate
├── embedder.py               # NEW — embed_batch() + EmbeddingError
├── store.py                  # NEW — replace_chunks()
├── worker.py                 # NEW — poll / claim / index / loop
└── cli.py                    # NEW — entry point
```

`embedder`, `store`, and `worker` import only `BaseChunkerInterface`'s types —
never a concrete chunker class. Swapping chunking strategy is a one-argument
change at `create_chunker()` and nowhere else (AC9). `cli.py` is the one
`embedding/` module that imports `indexing` — it calls `create_chunker()` once at
startup and passes the resulting chunker into `run_worker`.

## Contracts

### `indexing/Chunker.py`

```python
def create_chunker(strategy: str = "size_based") -> Chunker
```

The only module importing both concrete strategies. Loads the tokenizer exactly
once (`get_tokenizer()`) and hands it to `SizeBasedChunker(tokenizer)`, mirroring
`Transcriber.py::create_transcriber` and `LocalWhisperX`'s eager-load-once
precedent. Called once, by `cli.py`, at startup — never per episode.
Unknown strategy → `ValueError`.

### `embedding/embedder.py`

```python
EMBEDDING_DIM = 1024

class EmbeddingError(Exception): ...

async def embed_batch(
    texts: list[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    batch_size: int = 32,
    timeout: float = 60.0,
) -> list[list[float]]
```

- Config arrives **pre-resolved** from the caller rather than being read from
  `settings` inside — same convention as `build_chunks`' token bounds, and what
  keeps the function callable from a throwaway probe script (implement step 1).
- Builds one `AsyncOpenAI` client per call (`api_key`, `base_url`, `timeout`) and
  sends `ceil(len(texts)/batch_size)` `.embeddings.create(...)` calls
  **sequentially** (R3), each with `model=model`, `input=batch`,
  `encoding_format="float"`.
- Response: `resp.data` is a list of `Embedding` objects, each carrying
  `.embedding` (list[float]) and `.index` (int). Sort each batch's `resp.data` by
  `.index` before extending the result — the OpenAI-compatible contract returns
  the index precisely because array order is not promised.
- Raises `EmbeddingError` (never a bare `openai`/`httpx` exception) on: API
  errors, timeout/connection failure, a response whose item count differs from
  the batch's, or any vector whose length is not `EMBEDDING_DIM`. **No retry**
  (R3).
- Returns vectors in the caller's input order, which is what lets `store` zip them
  positionally against the spans.

**BGE-M3 needs no instruction prefix.** Unlike the E5 family (`"query: "` /
`"passage: "`), BGE-M3 embeds raw text on both sides. Store `span.text` verbatim,
and the future retrieval code must embed the user's question the same way — any
asymmetry introduced later silently degrades recall rather than erroring.

### `embedding/store.py`

```python
async def replace_chunks(
    session: AsyncSession,
    episode: Episode,
    spans: list[ChunkSpan],
    embeddings: list[list[float]],
) -> int
```

The persistence half of the single-transaction boundary (R2). It mutates the
session (DELETE + INSERT) but **never commits** — the caller
(`worker.index_episode`) owns the transaction and the one `commit()` that makes
`done` a trustworthy claim.

**Preconditions** (guaranteed by the caller, not re-checked here):
- `spans` is non-empty — `index_episode` treats an empty chunk list as `failed`
  *before* reaching this function, so a vacuous "delete everything, insert
  nothing" can never masquerade as `done`.
- every vector in `embeddings` is already 1024-dim (`embed_batch` validated it).
- `span.text` is cleaned and `span.start`/`span.end` are word-level-derived —
  both are persisted **verbatim**, never recomputed here.

**Step 1 — length guard.** `len(spans) != len(embeddings)` → raise `ValueError`.
The two lists are zipped positionally below; a mismatch would silently attach
the wrong vector to the wrong text, and nothing downstream could ever detect
it. `ValueError` (not a custom stage error) because this is a programmer-error
precondition, not a runtime/external failure — error-handling.md's "one class
per stage" rule applies to external-integration/parse stages, which an internal
persistence helper is not.

**Step 2 — delete old rows.** `await session.execute(delete(Chunk).where(
Chunk.episode_id == episode.id).execution_options(synchronize_session=False))`.
Core `delete`, no ORM cascade involved (`chunk` has FKs but no relationships).
`synchronize_session=False` is the idiomatic bulk-delete form: the session's
identity map holds no `Chunk` objects here, so there is nothing to synchronize.

**Step 3 — bulk insert new rows.** Build one `Chunk(...)` per span and
`session.add_all([...])` so they flush in one statement batch. Sizing is a few
hundred rows × 1024 float32 (≈8KB/row → ~2.4MB at 600 chunks), so no Core
bulk-insert or streaming optimisation is warranted; `add_all` keeps
`Computed`-column and default handling correct.

**Column mapping** (from `models/chunk.py`, migration 0004):

| column | value |
|---|---|
| `text` | `span.text` (already cleaned upstream) |
| `start`, `end` | `span.start`, `span.end` — verbatim |
| `embedding` | the positionally-paired vector (`list[float]` → `Vector(1024)`) |
| `episode_id` | `episode.id` |
| `podcast_id` | `episode.podcast_id` (denormalised) |
| `speaker` | omitted → `NULL` (diarization not in this pipeline) |
| `id` | omitted → DB-generated |
| `search_vector` | omitted → `Computed(to_tsvector('english', text), persisted=True)`, DB-generated |

`search_vector` is a persisted `Computed` column, so SQLAlchemy never includes
it in the INSERT — the DB populates it on write. The ORM does not read it back
(no `RETURNING`/eager refresh needed), and nothing in this task consumes it.

**Return.** `len(spans)` — the number of rows written, used by the worker's
progress log (AC2). The delete's rowcount is deliberately *not* returned: how
many stale rows existed is uninteresting, and exposing it would invite callers
to reason about it.

**Error handling.** No dedicated error class. A constraint/dimension failure at
the DB (unreachable given upstream validation, but possible) propagates as a
SQLAlchemy exception; `worker.index_episode`'s broad `except` rolls back and
marks `failed`. The only store-raised exception is the `ValueError` length guard.

**No commit — the load-bearing rule.** Everything above is uncommitted on
return. R2's atomicity depends on the DELETE, the INSERT, and the
`index_status = done` transition sharing one commit in the worker, so this
function must not introduce its own commit or flush.

### `embedding/worker.py`

Mirrors `transcription/worker.py` function-for-function.

```python
async def reset_stale_processing(session) -> int          # processing → pending, startup
async def reset_failed(session) -> int                    # failed → pending, only with --retry-failed
async def poll_ready(session, episode_id: int | None) -> Episode | None
async def claim_episode(session, episode_id: int) -> Episode | None
def resolve_token_tier(duration_seconds: float) -> tuple[int, int, int]
async def index_episode(session, episode, chunker) -> None
async def run_worker(chunker, session_factory, *, poll_interval, once, episode_id, retry_failed) -> None
```

- `poll_ready` selects `transcript_status == DONE AND index_status == PENDING`,
  ordered by id, limit 1; `episode_id` narrows it to one episode for debugging.
- `claim_episode` is the same atomic status-guarded update the transcription worker
  uses: `UPDATE ... WHERE id = :id AND index_status = 'pending' ... RETURNING`, so
  two workers can never both claim one episode.
- `resolve_token_tier` reads `settings.audio_tier` first-match-wins in
  `Short → Medium → Long` order on `max_episode_duration_seconds`, returning
  `(min_chunk_tokens, max_chunk_tokens, desired_chunk_tokens)`. Duration is derived
  from the cleaned segments (`segments[-1]["end"] - segments[0]["start"]`), not from
  `episode.duration_seconds`, which can be `None`.
- `index_episode` treats missing or empty `transcript_data["segments"]`, and an
  empty chunk list, as `failed` — not as a vacuous `done`. An episode that yields
  no chunks means the upstream data is broken, and a silent `done` would hide it
  forever. Mirrors the transcription worker's missing-audio → `failed` handling.
- `once=True` drains the ready queue and returns instead of sleeping — so a debug
  run terminates on its own.

### `embedding/cli.py`

Shape copied from `transcription/cli.py` (argparse + logging setup + build
dependencies from `settings` + `run_worker`). No `[extra]`-guarded import: every
dependency is in the base set.

Flags: `--debug`, `--poll-interval`, `--strategy` (default `size_based`),
`--once`, `--episode-id`, `--retry-failed`.

## Config additions

Two fields on `Settings`, defaults inline (no `.env` change required):

```python
embedding_batch_size: int = 32
embedding_timeout_seconds: float = 60.0
```

Poll interval reuses the existing `worker_poll_interval`.

## Compatibility and rollback

- **No migration.** Every column this task writes already exists (`0004`). The
  task is purely additive code plus two settings fields and one compose service.
- Rollback is "delete the four new `embedding/*` files plus `indexing/Chunker.py`,
  revert two lines of `config.py`, drop the `indexer` service block". Nothing
  existing is modified.
- The transcription pipeline is untouched; the two workers share only the
  `episode` table and never write the same columns.
- Data-level rollback of a bad index run is `DELETE FROM chunk WHERE episode_id =
  ...` plus resetting `index_status` — no state lives outside Postgres.

## Trade-offs taken

| Decision | Bought | Paid |
|---|---|---|
| Whole-episode transaction (R2) | `done` is a trustworthy claim; re-runs idempotent by construction | a failure re-embeds the whole episode |
| No retry (R3) | far less code; no backoff/jitter/partial-batch logic | a transient blip costs the episode's embedding work |
| `--retry-failed` flag (R4) | recovery is one command, no schema change | recovery is manual, not automatic |
| Reuse api image for `indexer` (R5) | no third Dockerfile | the indexer image carries FastAPI/uvicorn it never uses |
| No unit tests (R6) | matches the project's actual test surface; less scaffolding | embedder error branches ship unverified — probed once by hand (AC5) |
