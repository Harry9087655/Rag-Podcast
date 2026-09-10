# Embed chunks and persist to pgvector

## Goal

Turn the `list[ChunkSpan]` that `SizeBasedChunker` already produces into persisted
`chunk` rows carrying a 1024-dim BGE-M3 vector, driven by a background worker, so
retrieval has something to query. This closes the second half of the indexing
pipeline (`clean → chunk` is done; `embed → store → orchestrate` is not).

## Background: relationship to `07-22-chunking`

`07-22-chunking` (still `in_progress`) planned the whole indexing pipeline in one
task, and its `implement.md` steps 8–12 cover exactly this scope. The user chose
(2026-09-05) to plan and ship the embed/store/worker half as an **independent
task** instead of continuing inside `07-22-chunking`. Consequence: `07-22-chunking`
should be narrowed to the chunker at its own finish step, and its
embedding/persistence acceptance criteria are superseded by this task's.
Contracts already pinned in `07-22-chunking/design.md` are reused rather than
re-derived (see Confirmed Facts).

## Confirmed Facts

Established by inspection — not to be re-asked.

**Embedding provider**
- Hosted, OpenAI-compatible `/embeddings` endpoint: DeepInfra, `BAAI/bge-m3`
  ([.env.example:6-8](.env.example)), read via `settings.embedding_base_url` /
  `embedding_api_key` / `embedding_model` ([config.py:33-35](backend/src/rag_podcast/config.py#L33-L35)).
- The `openai` SDK is a dependency ([pyproject.toml](backend/pyproject.toml));
  it provides the OpenAI-compatible `/embeddings` client used against DeepInfra.
  No additional heavy dep is needed.
- BGE-M3 output dim is 1024, matching the column below. The embedding model is
  effectively locked for the corpus' lifetime — changing it invalidates every
  stored vector.

**Schema — already migrated, no table work needed**
- `chunk`: `id, episode_id, podcast_id, text, embedding Vector(1024), start, end,
  speaker (nullable), search_vector` ([models/chunk.py](backend/src/rag_podcast/models/chunk.py)).
- `search_vector` is a DB-generated `TSVECTOR` (`to_tsvector('english', text)`,
  persisted) with a GIN index — populated by Postgres, never by app code
  ([0004_indexing_fields.py:41-50](backend/alembic/versions/0004_indexing_fields.py#L41-L50)).
- `episode.index_status` enum (`pending/processing/done/failed`, default `pending`)
  exists ([models/episode.py:19-46](backend/src/rag_podcast/models/episode.py#L19-L46)).
- The `vector` extension is created in `0001_init`, but **no ANN index
  (ivfflat/hnsw) exists on `chunk.embedding`** — only the GIN index on
  `search_vector`.

**Upstream input**
- `SizeBasedChunker.build_chunks(...) -> list[ChunkSpan{text, start, end, segments}]`
  is complete ([SizeBasedChunker.py](backend/src/rag_podcast/indexing/SizeBasedChunker.py),
  442 lines) and takes **already-cleaned** segments; the caller runs
  `clean_words` once on the whole transcript before chunking
  ([BaseChunkerInterface.py:11-33](backend/src/rag_podcast/indexing/BaseChunkerInterface.py#L11-L33)).
- `ChunkSpan.start/.end` are already word-level-derived — persist them as-is,
  never recompute downstream.
- `indexing/tokenizer.py::get_tokenizer()` loads the BGE-M3 tokenizer at a pinned
  revision into `settings.hf_cache_dir`.
- Token tiers live in `settings.audio_tier` (`Short`/`Medium`/`Long`, first-match
  on episode runtime) ([config.py:62-74](backend/src/rag_podcast/config.py#L62-L74)).

**Worker precedent to mirror**
- `transcription/worker.py`: `reset_stale_processing` (startup crash recovery),
  `poll_*`, `claim_episode` (atomic `UPDATE ... WHERE status = <expected>
  RETURNING`), per-episode try/except → `DONE`/`FAILED`, commit after every state
  change ([worker.py:23-139](backend/src/rag_podcast/transcription/worker.py#L23-L139)).
- `transcription/cli.py`: argparse (`--debug`, `--poll-interval`) + logging setup +
  build dependencies from `settings` + `run_worker(...)`.
- DB access is async throughout: `async_session` / asyncpg
  ([db.py](backend/src/rag_podcast/db.py)).

**Current state of the files this task owns** — all empty (0 bytes):
`embedding/embedder.py`, `embedding/store.py`, `embedding/worker.py`,
`embedding/cli.py`, and `indexing/Chunker.py` (the `create_chunker` factory that
`07-22-chunking`'s design assumed exists — it does not).

## Requirements

### R1 — Deliverable is the full vertical slice

Decided 2026-09-05 (user). This task ships a pipeline that can be verified from
the database, not a library function:

- `embedding/embedder.py` — `embed_batch` + `EmbeddingError`.
- `embedding/store.py` — `replace_chunks`.
- `embedding/worker.py` — poll / claim / index / status transitions.
- `embedding/cli.py` — thin entry point, mirroring `transcription/cli.py`.
- `indexing/Chunker.py` — `create_chunker()` factory. Included because it is
  currently a 0-byte file that `07-22-chunking`'s design assumed already existed;
  the worker cannot construct a chunker (or load the tokenizer once at startup)
  without it.

Rejected: an embedder-only or embedder+store-only slice — neither leaves the
project in a state where "indexing works" can be checked by querying `chunk`,
and both defer the integration risks (batching, failure granularity, re-run
de-duplication) that this task exists to settle.

### R2 — Whole-episode atomicity, delete-then-insert

Decided 2026-09-05 (user). Per episode:

1. Chunk, then embed **every** span (internally batched) and hold the vectors in
   memory. Sizing: a few hundred chunks × 1024 float32 ≈ 2.4 MB at 600 chunks, so
   memory is not a constraint and no streaming design is warranted.
2. In a **single transaction**: `DELETE FROM chunk WHERE episode_id = ?`, then
   bulk-INSERT the new rows, then set `index_status = DONE`, then commit.
3. Any exception anywhere → roll back the transaction, set `index_status = FAILED`
   (committed separately), leave the previously stored rows untouched.

Consequences this buys: `chunk` never holds a partially-indexed episode, so
retrieval can never silently under-recall; re-indexing is idempotent by
construction (no row-level UPDATE path is needed, and new/old chunk boundaries
need not correspond); `index_status = DONE` is a trustworthy assertion that the
episode's rows are complete.

Rejected: per-batch commit (leaves readable half-episodes while `FAILED`, and
needs orphan-cleanup logic); delete-first-then-stream-insert (the only option
that can net-lose data on failure).

Accepted cost: a retry re-embeds the whole episode. At DeepInfra BGE-M3 pricing
that is cents per episode.

### R3 — Sequential embedding calls, no retry

Decided 2026-09-05 (user).

- `embed_batch(texts) -> list[list[float]]` sends requests **sequentially**, batch
  size from config (default 32), with a client timeout.
- **No retry layer.** Any HTTP error — including a transient 429/5xx or a dropped
  connection — raises `EmbeddingError` immediately, which R2 turns into a rolled-back
  transaction and `index_status = FAILED`.
- Returned vectors must be returned in request order and validated for length
  1024 before they reach `store` (a silently wrong dimension would fail at the
  pgvector insert anyway, but with a far less obvious error).

Accepted cost, stated plainly: because R2 makes the episode atomic and there is
no retry, a single network blip discards all embedding work done for that
episode. This is a deliberate simplicity choice; if it proves annoying in
practice, a retry belongs inside `embed_batch` (cheap — retries one HTTP
request) rather than in the worker (expensive — re-embeds the episode).

Unvalidated assumption for the implementation step: DeepInfra's `/embeddings`
per-request limits (max inputs per array, max total tokens) are not documented
in this repo. Step 1 of implementation is a single real small-input call to
confirm request/response shape and those limits before wiring anything else.

### R4 — `failed` episodes are recoverable by flag, not by hand-written SQL

Decided 2026-09-05 (user). The poll query looks for `index_status = pending`, so
`failed` is terminal and never re-picked up on its own. Combined with R3 (no
retry), a transient network blip parks an episode permanently.

- `cli.py` takes `--retry-failed`: when passed, reset every
  `index_status = failed` row to `pending` before entering the main loop.
- Off by default, so a genuinely broken episode is not re-run on every restart.
- No schema change. Rejected: automatic retry inside the poll query, which needs
  a new attempt-counter column (extra migration) to stop a poison episode from
  spinning the worker forever.

### R5 — Wire the indexing worker into `docker-compose.yml`

Decided 2026-09-05 (user).

- New `indexer` service reusing the **api image** (`build: ./backend`) with
  `command: python -m rag_podcast.embedding.cli`. No new Dockerfile: this worker
  needs only `openai` / `tokenizers` / `sqlalchemy`, all in the base dependency
  set — unlike the transcription worker, whose `Dockerfile.worker` exists for
  CUDA + whisperx/funasr + ffmpeg (`--extra transcription`).
- Same conventions as the existing services: `env_file: .env`, container
  `DATABASE_URL` pointing at `db:5432`, `./backend/src:/app/src` bind mount for
  live reload, and the `${HOST_CACHE_DIR}/huggingface` mount so the BGE-M3
  tokenizer is not re-downloaded per container rebuild. No GPU reservation.
- Sequencing: this is the **last** implementation step. The pipeline is first
  validated by running `python -m rag_podcast.embedding.cli` on the host, so that
  pipeline bugs and container/network bugs are never being debugged at once.

Corrects a stale fact in `07-22-chunking`'s PRD ("transcription's worker isn't
wired into docker-compose either") — commit `92aa882` containerised it as the
GPU-enabled `worker` service ([docker-compose.yml:28-59](docker-compose.yml#L28-L59)).

### R6 — Validation is end-to-end against the real DeepInfra API, no unit tests

Decided 2026-09-05 (user: "这个功能没有这么庞大"). Consistent with the existing
test surface — `backend/tests/` holds only pure-function tests
([test_cleaning.py](backend/tests/test_cleaning.py)) and there is no database
fixture in the project.

- No new automated tests. Correctness is established by running the worker
  against a really-transcribed episode and checking the database.
- To make that loop practical, `cli.py` takes `--episode-id N` (process only that
  episode) and `--once` (process the ready queue and exit instead of looping).

Accepted cost, stated plainly: the embedder's error branches (timeout, 4xx,
wrong-dimension response) will not be exercised by the happy-path run, so they
are written-but-unverified. They are the code most likely to be wrong at the
moment it first matters. Mitigation is one cheap manual probe — run once with a
deliberately broken `EMBEDDING_API_KEY` and confirm the episode lands in `failed`
with no rows written (this is AC5, not a test file).

## Acceptance Criteria

Checked by running the pipeline and querying the database — see R6.

- [x] **AC1** — For an episode with `transcript_status = done` and
  `index_status = pending`, running
  `python -m rag_podcast.embedding.cli --once --episode-id <N>` ends with that
  episode at `index_status = done`.
- [x] **AC2** — `SELECT count(*), vector_dims(embedding) FROM chunk WHERE
  episode_id = <N> GROUP BY 2` returns exactly one row, with `vector_dims = 1024`
  and a count equal to the chunk count the worker logged.
- [x] **AC3** — Spot-checking 3 stored rows: `text` matches the corresponding
  `ChunkSpan.text` exactly, `start`/`end` match the chunker's output exactly (they
  are persisted verbatim, never recomputed), and those timestamps line up with the
  actual audio within ±2s.
- [x] **AC4** — Re-running the same episode twice leaves the row count unchanged
  with no duplicates, and the rows' `id`s change (proving delete-then-insert, R2).
- [x] **AC5** — With a deliberately invalid `EMBEDDING_API_KEY`, the episode ends
  at `index_status = failed` and `chunk` holds either zero rows for it or exactly
  the previous successful set — never a partial one (R2).
- [x] **AC6** — Re-running that failed episode with `--retry-failed` brings it to
  `done` (R4).
- [x] **AC7** — `SELECT search_vector FROM chunk WHERE episode_id = <N> LIMIT 1`
  is non-empty (DB-generated; no app code populates it).
- [x] **AC8** — `docker compose up indexer` runs the same pipeline to `done`
  against the containerised database (R5).
- [x] **AC9** — By inspection: `embedder.py`, `store.py`, and `worker.py` contain
  no conditional branching on which chunking strategy produced a `ChunkSpan`;
  they import only `BaseChunkerInterface`'s types.

## Out of Scope

Recorded deliberately (user, 2026-09-05: "这些点都要记录，只是暂时不实现") —
these are known gaps with owners, not oversights.

### ANN index on `chunk.embedding`

The `vector` extension is created but the embedding column carries **no index**,
so vector search will be a sequential scan. Deferred because:

- The index's operator class must match the distance operator the retrieval code
  uses (`vector_cosine_ops` vs `vector_l2_ops`); picking it now would be guessing
  on behalf of a module that does not exist yet, and a mismatch means the index
  is silently never used.
- At the current corpus size (single-digit episodes, hundreds of rows) a
  sequential scan beats HNSW anyway.

Owner: the future retrieval/qa task. Trigger to revisit: first real
vector-similarity query, or corpus growth past a few thousand chunks.

### `search_vector` is useless for Chinese episodes

`chunk.search_vector` is `to_tsvector('english', text)`
([0004_indexing_fields.py:41-49](backend/alembic/versions/0004_indexing_fields.py#L41-L49)).
The `english` configuration does not segment Chinese, so a Chinese chunk
collapses into a near-useless token stream and BM25/full-text search over it will
not work. This is a real defect, not a theoretical one — the corpus already has
Chinese episodes (hence the FunASR backend).

Deferred because the fix is not a parameter change: Postgres ships no Chinese
text-search configuration, so it needs a `zhparser`/`pg_jieba` extension, which
the current `pgvector/pgvector:pg16` image does not carry — meaning a custom
image plus a rebuild of the generated column. Out of proportion to this task.

Owner: the future retrieval task that actually implements BM25/hybrid search
(that task cannot ship without resolving it). This task still populates the
column, since it is DB-generated on insert regardless.

### Also out of scope

- Retrieval, ranking, hybrid search — this task only writes rows.
- `SimilarityBiasedChunker` (Phase 2 chunking) — still a `NotImplementedError`
  stub; unaffected by this task, which must not branch on chunking strategy.
- Chunk-size/strategy quality tuning and any retrieval eval harness.
- `chunk.speaker` — written as `NULL`; diarization is not in this pipeline.

## Open Questions

None blocking. One item is unknown but resolved by doing rather than by asking:
DeepInfra's per-request `/embeddings` limits (max inputs per array, max total
tokens), which the first implementation step probes with a real call (see R3).
