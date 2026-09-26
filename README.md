# rag-podcast

A local, multi-podcast knowledge base. Give it a podcast RSS link and it downloads the episodes, transcribes them, and turns the audio into a **cross-episode, semantically searchable, timestamped corpus** that you can query in natural language — every answer cites the exact podcast + timestamp, and clicking a source jumps straight to that moment in the audio.

Built for individuals who want to deeply consume podcast content ("I heard it somewhere, but I can't find it / can't remember when").

---

## What it does

```
RSS link
  → feedparser parses episodes + audio URLs
  → download audio to local disk
  → transcribe (word-level timestamps)   ── WhisperX (non-Chinese) / FunASR (Chinese)
  → clean filler words on the word list   ── keeps every remaining word's timestamp intact
  → chunk into time-contiguous spans      ── leaf chunks (embedded) + parent chunks (merged)
  → embed with BGE-M3 (hosted API)        ── 1024-d vectors
  → store in PostgreSQL + pgvector
  ──────────────────────────────────────────────
  user asks a question
  → retrieve relevant chunks (optionally scoped to a podcast)
  → assemble prompt → LLM answers with inline [n] citations
  → answer + sources[] (text, podcast, episode, start/end)
  → frontend renders sources with a play button → <audio> jumps to chunk.start
```

### Features

- **Ingest any podcast** by RSS URL (also resolves Apple Podcasts share links to their feed + episode).
- **Bilingual transcription** — language detection routes each episode to WhisperX (word-aligned) or FunASR/Paraformer (Chinese), both producing word-level timestamps.
- **Two-level chunking** — small *leaf* chunks are what get embedded and retrieved; contiguous leaves are grouped into larger *parent* chunks that are merged back into results when several siblings match, giving longer, more listenable citations.
- **Semantic search** over the whole library, a single podcast, or a single episode (pgvector cosine distance, HNSW-indexed).
- **Cited Q&A** — two arms: a search arm that cites numbered chunks `[1][2]`, and an agentic full-context arm for a single episode that reverse-looks-up cited passages to exact timestamps via a tool call.
- **Audio jump** — every source carries `start`/`end`, and the API serves audio with HTTP range requests so the browser can seek.

---

## Repository layout

```
rag-podcast/
├── docker-compose.yml            # db + api + worker (transcription) + indexer (embedding)
├── .env.example                  # all required config, copy to .env
├── db/                           # Postgres 16 + pgvector + pg_jieba (Chinese FTS) image
├── backend/
│   ├── pyproject.toml            # package + deps (transcription is an optional extra)
│   ├── alembic/                  # migrations 0001–0007
│   ├── scripts/                  # throwaway probes / debug scripts
│   ├── tests/                    # pytest (needs a live Postgres test DB)
│   └── src/rag_podcast/
│       ├── config.py             # pydantic-settings, reads .env
│       ├── db.py                 # async engine + session factory
│       ├── main.py               # FastAPI app + router wiring
│       ├── models/               # SQLAlchemy: podcast, episode, chunk, parent_chunk
│       ├── ingestion/            # RSS parse → store → download  (POST /podcasts)
│       ├── transcription/        # worker + WhisperX/FunASR/language routing
│       ├── cleaning/             # filler-word removal + CJK-aware word joining
│       ├── indexing/             # chunkers (size-based / similarity-biased slot) + parent grouper
│       ├── embedding/            # BGE-M3 client + atomic chunk persistence + worker
│       ├── retrieval/            # pgvector recall + auto-merge + reranker slot
│       ├── qa/                   # answer() core + prompts + token budgeting
│       └── api/                  # query, audio, health routers
└── frontend/                     # Vite + TypeScript (no framework), consumes JSON
```

---

## Data model

```
podcast (1) ──── (N) episode (1) ──── (N) chunk (leaf)
                                      │
                                      └── (N) parent_chunk
```

| Table | Purpose | Notable columns |
|---|---|---|
| `podcast` | One RSS feed | `rss_url` (unique), `name`, `author`, `cover_url` |
| `episode` | One episode | `guid` (dedup), `audio_local_path`, `transcript_status`, `index_status`, `transcript_data` (JSONB), `language`, `duration_seconds` |
| `chunk` | Retrieval unit (leaf) | `text`, `embedding` `vector(1024)`, `start`/`end`, `speaker`, `language`, `search_vector` (`tsvector`), `parent_id` |
| `parent_chunk` | Time-contiguous leaf merge | `text`, `start`/`end` (no embedding, no search vector) |

**State machines** (two independent status columns on `episode`):

- `transcript_status`: `pending → downloaded → processing → done | failed`
- `index_status`: `pending → processing → done | failed`

Both workers are crash-safe: on startup they reset any rows stuck in `PROCESSING` from an unclean shutdown.

---

## Pipeline stages

| Stage | Module | Input → Output |
|---|---|---|
| Ingest | `ingestion/` | RSS URL → podcast/episode rows + audio on disk (`downloaded`) |
| Transcribe | `transcription/` | downloaded audio → word-level segments in `transcript_data` (`done`) |
| Clean | `cleaning/` | raw segments → filler-word-free segments (timestamps preserved) |
| Chunk | `indexing/` | cleaned segments → leaf `ChunkSpan`s + parent groups |
| Embed | `embedding/` | leaf texts → 1024-d BGE-M3 vectors |
| Store | `embedding/store.py` | replace episode's chunks + parents **in one transaction** |
| Search | `retrieval/` | query → ranked `SearchResult`s (recall → auto-merge → trim) |
| Answer | `qa/` | question + context → `{answer, sources[]}` |
| Serve | `api/` | JSON endpoints + audio streaming |

The transcription and embedding steps run as **background workers** that poll the DB and claim one episode at a time (status-guarded, so multiple workers can't double-claim).

---

## Tech stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy (async) + asyncpg, Alembic
- **Storage / search**: PostgreSQL 16 + **pgvector** (HNSW, cosine) + **pg_jieba** (Chinese full-text, reserved for hybrid search)
- **Transcription**: WhisperX (word-level alignment) + FunASR/Paraformer (Chinese), GPU-accelerated
- **Embeddings**: BGE-M3 via a hosted OpenAI-compatible API (default DeepInfra), 1024 dims
- **LLM**: DeepSeek (default) or any OpenAI-compatible endpoint
- **Frontend**: Vite + TypeScript (framework-free, HTML + `<audio>`)
- **Orchestration**: Docker Compose

---

## Quick start

### Prerequisites

- Docker (with an NVIDIA GPU available if you want to run the transcription worker)
- API keys for the embedding provider (DeepInfra) and the LLM (DeepSeek or OpenAI)
- On Windows/WSL2: raise WSL2's memory limit (FunASR loads three models in one process) — see `.env.example` / `spec.md` §7

### 1. Configure

```bash
cp .env.example .env
```

Fill in at minimum:

```ini
EMBEDDING_API_KEY=...          # DeepInfra
LLM_API_KEY=...                # DeepSeek / OpenAI
HOST_CACHE_DIR=...             # absolute path to your model-cache root (worker bind mounts)
```

See [.env.example](.env.example) for the full list (database, embedding model, LLM model, transcription model choices, chunking/retrieval/QA tuning).

### 2. Run

```bash
docker compose up --build
```

Four services come up:

| Service | Role | Needs GPU |
|---|---|---|
| `db` | Postgres 16 + pgvector + pg_jieba | no |
| `api` | FastAPI server on `:8000` | no |
| `worker` | transcription (WhisperX/FunASR) | **yes** |
| `indexer` | embedding + chunk persistence | no |

### 3. Ingest, wait, ask

```bash
# import a podcast (latest 1 episode by default; pass max_episodes to import more)
curl -X POST http://localhost:8000/podcasts \
  -H 'Content-Type: application/json' \
  -d '{"rss_url": "https://example.com/feed.xml", "max_episodes": 3}'

# once the workers finish (watch the logs), ask a question
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "what did they say about ...", "podcast_id": 1}'
```

---

## API

| Method | Path | Body / params | Returns |
|---|---|---|---|
| `POST` | `/podcasts` | `{rss_url, max_episodes?}` | podcast + created episodes + counts |
| `POST` | `/query` | `{question, podcast_id?, episode_id?}` | `{answer, sources[{text, podcast_name, episode_title, start, audio_url}]}` |
| `GET` | `/audio/{episode_id}` | — | audio file (supports `Range` / HTTP 206) |
| `GET` | `/health` | — | `{status, db}` |

`/query` scope semantics: `episode_id` → single episode (agentic full-context when the transcript fits the model's context window); `podcast_id` only → cross-episode search within that podcast; neither → whole-library search.

---

## Running locally (without Docker)

```bash
cd backend
uv sync                     # base deps only
# for transcription: uv sync --extra transcription

# migrate the database (point DATABASE_URL at your Postgres)
uv run alembic upgrade head

# run the API
uv run uvicorn rag_podcast.main:app --reload
```

### CLI entry points (useful for debugging)

```bash
# transcribe worker (needs the transcription extra + a GPU)
uv run --extra transcription python -m rag_podcast.transcription.cli

# embedding worker
uv run python -m rag_podcast.embedding.cli \
    --once --episode-id 1          # drain queue once / index one episode
    --retry-failed                 # re-queue FAILED
    --reindex                      # re-queue DONE (full rebuild)

# semantic search
uv run python -m rag_podcast.retrieval.cli "your query" --podcast-id 1 --top-k 5

# Q&A
uv run python -m rag_podcast.qa.cli "your question" --episode-id 1
```

---

## Configuration

All settings live in `.env` and are loaded by [`config.py`](backend/src/rag_podcast/config.py) (`pydantic-settings`, `extra="ignore"`). Highlights:

- **Embedding**: `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` (default `BAAI/bge-m3`), `EMBEDDING_BATCH_SIZE`
- **LLM**: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (default `deepseek-chat`), context/response budgets
- **Transcription**: `WHISPERX_MODEL`, `FUNASR_MODEL` (default `paraformer-zh`), `LANG_DETECT_WINDOW_SECONDS`
- **Chunking**: `AUDIO_TIER` (per-duration-token budgets for Short/Medium/Long episodes), `CHUNK_MIN/MAX_DURATION_SECONDS`
- **Retrieval**: `RETRIEVAL_TOP_K`, `RETRIEVAL_SCORE_MODE` (`distance`/`similarity`), `AUTO_MERGE_THRESHOLD`

---

## Testing

```bash
cd backend
uv run pytest
```

Tests are async (`pytest-asyncio`) and require a live Postgres test database (default `ragpodcast_test`, overridable via `TEST_DATABASE_URL`), reachable over the docker-compose network as host `db`.

---

## Roadmap / reserved slots

A few pipeline axes are intentionally abstracted behind `Protocol`s or left as placeholders for later phases:

- **`SimilarityBiasedChunker`** (Phase 2 chunker) — will bias chunk boundaries toward points of highest embedding distance; currently a `NotImplementedError` placeholder. Use `size_based` today.
- **`Reranker`** — the retrieval pipeline already over-fetches candidates (`retrieval_candidate_multiplier`) and carries a commented slot for an M4 cross-encoder reranker.
- **`HybridRecaller`** — the `search_vector` (`tsvector`, GIN-indexed, jieba-aware for Chinese) is generated and indexed now, reserved for BM25 / RRF hybrid recall.
- **`eval/`** — reserved for Recall@k / MRR harness (spec M4).
- **`speaker`** on `chunk` — reserved for future diarization.

See [`spec.md`](spec.md) for the full project specification, milestones (M1–M4), and known risks/open questions.
