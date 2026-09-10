# Implementation plan: embed chunks and persist to pgvector

## Ordered checklist

1. **Probe the DeepInfra API first, before writing anything.** Throwaway script in
   `backend/scripts/` (or an inline `uv run python -c`): one real `/embeddings`
   call with 2–3 short strings against `settings.embedding_base_url` /
   `embedding_api_key` / `embedding_model`. Confirm: response JSON shape, that
   `data[*].index` is present, that vectors are length 1024, and — by pushing a
   larger array — where the per-request input-count/token limit sits. This is the
   only unvalidated external assumption in the plan (PRD Open Questions);
   everything else is built on top of what it returns. **Review gate.**

2. **Config** — add `embedding_batch_size: int = 32` and
   `embedding_timeout_seconds: float = 60.0` to `Settings`
   ([config.py](backend/src/rag_podcast/config.py)), in the indexing block. Adjust
   the batch default if step 1 found a lower ceiling. No `.env` change needed.

3. **`embedding/embedder.py`** — `EMBEDDING_DIM = 1024`, `EmbeddingError`,
   `async def embed_batch(texts, *, base_url, api_key, model, batch_size, timeout)`.
   Sequential batches, sort each response's `data` by `index`, validate item count
   and every vector's length, wrap every client failure in `EmbeddingError`, no
   retry. See design.md for the full contract. Re-run step 1's probe through
   `embed_batch` itself to confirm it returns 1024-dim vectors in input order.

4. **`indexing/Chunker.py`** — `create_chunker(strategy="size_based") -> Chunker`.
   Calls `get_tokenizer()` once and constructs `SizeBasedChunker(tokenizer)`;
   `"similarity_biased"` returns the stub; anything else raises `ValueError`. First
   run downloads BGE-M3's `tokenizer.json` (~17MB) into `settings.hf_cache_dir` —
   confirm a second run hits the cache with no re-download.

5. **`embedding/store.py`** — `replace_chunks(session, episode, spans, embeddings)`.
   Length-guard, `DELETE ... WHERE episode_id`, bulk INSERT, **no commit**. `start`
   /`end` persisted verbatim from the span; `speaker=NULL`; `search_vector`
   untouched (DB-generated).

6. **`embedding/worker.py`** — `reset_stale_processing`, `reset_failed`,
   `poll_ready`, `claim_episode`, `resolve_token_tier`, `index_episode`,
   `run_worker`, modelled on
   [transcription/worker.py](backend/src/rag_podcast/transcription/worker.py).
   `index_episode` runs `clean_words` once on the whole transcript before chunking,
   holds the whole episode's writes in one transaction, and commits once (R2).
   Failure path: rollback, then a separate `update(Episode).values(index_status=
   FAILED)` + commit.

7. **`embedding/cli.py`** — modelled on
   [transcription/cli.py](backend/src/rag_podcast/transcription/cli.py). Flags:
   `--debug`, `--poll-interval`, `--strategy`, `--once`, `--episode-id`,
   `--retry-failed`. Builds the chunker via `create_chunker()` at startup, then
   `run_worker(...)`.

8. **End-to-end happy path on the host** — pick an episode with
   `transcript_status = done`, run with `--once --episode-id <N>`. Verifies
   **AC1, AC2, AC3, AC7**. Includes the ±2s audio spot-check on 3 rows.

9. **Idempotency** — run the same episode again; row count unchanged, ids changed,
   no duplicates. Verifies **AC4**.

10. **Failure probe** — re-run with a deliberately invalid `EMBEDDING_API_KEY`;
    confirm `index_status = failed` and that `chunk` holds no partial set. Then
    restore the key and run with `--retry-failed`. Verifies **AC5, AC6** — and is
    the only exercise the embedder's error branches get (R6).

11. **`docker-compose.yml`** — add the `indexer` service reusing `build: ./backend`
    with `command: python -m rag_podcast.embedding.cli`, `env_file: .env`, container
    `DATABASE_URL`, the `./backend/src` bind mount, the
    `${HOST_CACHE_DIR}/huggingface` mount, and `depends_on: db (healthy)`. No GPU.
    Verifies **AC8**. Deliberately last — see R5.

12. **Inspection pass** — confirm `embedder.py`, `store.py`, `worker.py` import
    only `BaseChunkerInterface` types and branch on no chunking strategy.
    Verifies **AC9**.

## Validation commands

```bash
cd backend
uv run alembic upgrade head                                        # should be a no-op; 0004 already applied
uv run python -m rag_podcast.embedding.cli --debug --once --episode-id <N>
uv run python -m rag_podcast.embedding.cli --debug --retry-failed --once
uv run pytest                                                      # existing suite must stay green
docker compose up indexer                                          # step 11
```

Database checks (via `psql` or [scripts/inspect_db.py](backend/scripts/inspect_db.py)):

```sql
SELECT index_status FROM episode WHERE id = <N>;                                    -- AC1
SELECT count(*), vector_dims(embedding) FROM chunk WHERE episode_id = <N> GROUP BY 2;  -- AC2
SELECT id, start, "end", left(text, 60) FROM chunk WHERE episode_id = <N> ORDER BY start LIMIT 3;  -- AC3
SELECT search_vector FROM chunk WHERE episode_id = <N> LIMIT 1;                     -- AC7
```

## Review gates

- **After step 1** — do not write `embedder.py` until one real DeepInfra call has
  confirmed the response shape and the per-request limits. Everything downstream
  assumes them.
- **After step 4** — confirm the tokenizer download caches (second run offline-safe)
  before building the worker on top of it.
- **After step 8** — the ±2s audio spot-check is the M1 quality bar; do it before
  moving on, not at the end.
- **Before step 11** — the whole pipeline must pass on the host first, so a
  container failure can only be a container problem.

## Risky files and rollback points

- `backend/src/rag_podcast/config.py` — the only existing source file modified
  (two added fields). Everything else is new: four files under `embedding/`, one
  under `indexing/` (`Chunker.py`).
- `docker-compose.yml` — additive service block; removing it restores the current
  topology exactly.
- No migration, so there is no schema rollback to plan. A bad indexing run is
  undone with `DELETE FROM chunk WHERE episode_id = ...; UPDATE episode SET
  index_status = 'pending' WHERE id = ...;`.
- The four new `embedding/*` files plus `indexing/Chunker.py` are self-contained:
  rollback is deleting them.
