# WhisperX Transcription — Technical Design

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│ Docker (docker-compose)                                  │
│  ┌──────────────┐  ┌──────────────────┐                 │
│  │ FastAPI API  │  │ pgvector/pg16    │                 │
│  │ (no GPU)     │  │ (5432)           │                 │
│  └──────┬───────┘  └────────▲─────────┘                 │
└─────────┼───────────────────┼───────────────────────────┘
          │                   │
          │ HTTP (health only)│ DB connection (asyncpg)
          │                   │
┌─────────┴───────────────────┴───────────────────────────┐
│ Host (Windows)                                           │
│  ┌──────────────────────────────────────┐                │
│  │ run_transcription_worker.py          │                │
│  │  ┌────────────────────────────────┐  │                │
│  │  │ while True:                     │  │                │
│  │  │   episode = poll_downloaded()   │  │                │
│  │  │   if episode:                   │  │                │
│  │  │     set_status(PROCESSING)       │  │                │
│  │  │     result = await transcriber.transcribe(path) │  │
│  │  │     episode.transcript_data = result │  │          │
│  │  │     set_status(DONE)             │  │                │
│  │  │   sleep(poll_interval)           │  │                │
│  │  └────────────────────────────────┘  │                │
│  │            │                          │                │
│  │            ▼                          │                │
│  │  ┌──────────────────────────────┐    │                │
│  │  │ Transcriber (Protocol)       │    │                │
│  │  │  └─ LocalWhisperX (MVP)      │    │                │
│  │  │  └─ WhisperXAPI (future)     │    │                │
│  │  └──────────────────────────────┘    │                │
│  └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────┘
```

Worker runs on the host because:
1. GPU/CUDA is only available on the host
2. API container has no `whisperx`/`torch` installed (excluded from Dockerfile per `scaffold-plan.md:96`)
3. Matches `spec.md:40`: "simple status table + background worker"

## Module Boundaries

```
backend/src/rag_podcast/transcription/
├── __init__.py           # exports Transcriber, LocalWhisperX, transcribe_episode
├── transcriber.py        # Transcriber Protocol + LocalWhisperX implementation
└── worker.py             # poll loop, status transitions (library code, importable)

backend/scripts/
└── run_transcription_worker.py  # CLI entry point: configures logging, runs worker
```

### `transcriber.py` — Contract

```python
from typing import Protocol
from pathlib import Path

class Transcriber(Protocol):
    """Abstract transcription interface.

    LocalWhisperX wraps the whisperx library (sync, GPU-bound).
    Future WhisperXAPI will make HTTP calls (async, I/O-bound).
    The protocol is async so the worker doesn't change when we swap.
    """

    async def transcribe(self, audio_path: Path) -> dict:
        """Transcribe audio. Returns WhisperX result dict:

        {"model": "small", "language": "en", "segments": [...]}
        """
        ...

class LocalWhisperX:
    """Wraps whisperx.load_model() + model.transcribe() + align().

    whisperx is synchronous, so transcribe() runs in asyncio.to_thread()
    to avoid blocking the worker's event loop entirely.
    """

    def __init__(self, model: str, device: str, compute_type: str) -> None: ...
    async def transcribe(self, audio_path: Path) -> dict: ...
```

### `worker.py` — Poll Loop

```python
async def transcribe_episode(
    session: AsyncSession,
    episode: Episode,
    transcriber: Transcriber,
    data_dir: Path,
) -> None:
    """Transcribe one episode: PROCESSING → transcribe → DONE or FAILED."""
    ...

async def run_worker(
    transcriber: Transcriber,
    poll_interval: int = 10,
) -> None:
    """Main loop: poll for DOWNLOADED episodes, transcribe one at a time."""
    ...
```

### `run_transcription_worker.py` — Entry Point

- Reads DB URL from `.env`
- Creates `LocalWhisperX` from `Settings`
- Calls `run_worker()`
- Handles SIGINT/SIGTERM for graceful shutdown

## Data Model

### Episode Table — New Column

```python
# models/episode.py
from sqlalchemy.dialects.postgresql import JSONB

transcript_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
```

Requires Alembic migration adding `transcript_data JSONB DEFAULT NULL` to `episode`.

### JSONB Structure

```json
{
  "model": "small",
  "language": "en",
  "duration": 523.14,
  "segments": [
    {
      "text": "Welcome to the indicator from Planet Money.",
      "start": 0.0,
      "end": 3.2,
      "words": [
        {"word": "Welcome", "start": 0.0, "end": 0.5},
        {"word": "to", "start": 0.5, "end": 0.7}
      ]
    }
  ]
}
```

- `segments[].words` are WhisperX's word-level timestamps — preserved for jump-to-audio
- `segments[].text` / `start` / `end` are the segment-level view used by the indexing module
- The indexing module reads this JSONB → chunks text → writes `Chunk` rows with embeddings

### Why JSONB vs. Separate Table

| | JSONB on Episode | Separate transcript_segments table |
|---|---|---|
| **Complexity** | One column, one migration | New model, new migration, new queries |
| **Query** | `episode.transcript_data` | JOIN + aggregate |
| **Write** | Single UPDATE | INSERT many rows in transaction |
| **Chunking flexibility** | Indexing module reads blob → chunks freely | Indexing module reads rows → chunks freely |
| **Trade-off** | Large episodes (~200 KB JSONB) are fine; can't query individual segments in SQL | Can query individual segments but adds complexity not needed at this layer |

JSONB wins for simplicity without loss of flexibility.

## Config

```python
# config.py additions
whisperx_model: str = "small"
whisperx_device: str = "cuda"
whisperx_compute_type: str = "float16"
worker_poll_interval: int = 10  # seconds between polls when no episodes are ready
```

All have sensible defaults. `worker_poll_interval` only applies when the queue is empty — when episodes are available, the worker processes them back-to-back with no sleep.

## Worker Flow

```
┌──────────────────┐
│ poll_downloaded() │◄──────────────┐
│ SELECT ... WHERE  │               │
│ status=downloaded │               │ sleep(poll_interval)
│ LIMIT 1           │               │ (only if no episode found)
└────────┬─────────┘               │
         │ found                    │
         ▼                          │
┌──────────────────┐               │
│ claim_episode()   │               │
│ UPDATE SET        │               │
│ status=processing │               │
│ WHERE id=? AND    │               │
│ status=downloaded │               │
│ RETURNING *       │               │
└────────┬─────────┘               │
         │ claimed                  │
         ▼                          │
┌──────────────────┐               │
│ transcriber.      │               │
│ transcribe(path)  │               │
└──┬──────────┬────┘               │
   │ success  │ failure             │
   ▼          ▼                     │
┌──────┐  ┌────────┐               │
│ DONE │  │ FAILED │               │
│ SET  │  │ SET    │               │
│ tran-│  │ status │               │
│script│  │=failed │               │
│_data │  │ log    │               │
│ =    │  │ error  │               │
│result│  └───┬────┘               │
└──┬───┘      │                    │
   │          │                    │
   └──────────┴────────────────────┘
```

### Atomic Claim

```sql
UPDATE episode
SET transcript_status = 'processing'
WHERE id = :id AND transcript_status = 'downloaded'
RETURNING *
```

The `WHERE transcript_status = 'downloaded'` guard prevents two workers (if ever run concurrently) from claiming the same episode. If `RETURNING` returns zero rows, the episode was already claimed — skip and poll again.

## Error Handling

| Failure Mode | Behavior |
|---|---|
| **WhisperX model crash / OOM** | `LocalWhisperX.transcribe()` raises `TranscribeError` → worker marks FAILED, continues |
| **Audio file missing** | `FileNotFoundError` → worker marks FAILED, continues |
| **DB connection lost** | `sqlalchemy` exception → worker crashes (let it die; Docker/systemd restarts it) |
| **JSONB serialization failure** | `TypeError` → worker marks FAILED, continues |

Per-episode isolation: each episode is processed in its own try/except. One bad episode never stops the worker.

## Compatibility & Future

| Change | Impact on this design |
|---|---|
| **WhisperX API** | New `WhisperXAPI` class implementing `Transcriber` protocol. Change `run_transcription_worker.py` config only — worker loop unchanged. |
| **Concurrent workers** | Atomic claim (status guard) already handles this. Set `worker_poll_interval` = 1, run N instances if GPU allows. |
| **Speaker diarization** | Add to `LocalWhisperX.transcribe()` pipeline. JSONB schema gains `segments[].speaker` field. No migration needed (JSONB is schema-flexible). |
| **Different ASR model** | Implement `Transcriber` protocol — e.g. `FasterWhisper`, `OpenAIWhisper`. Worker loop unchanged. |

## Rollback

- Remove `transcript_data` column (Alembic downgrade)
- Worker is a standalone script — stop it, no API code depends on it
- `transcript_status` values `PROCESSING`/`DONE`/`FAILED` already exist in the enum — no DB downgrade needed there
