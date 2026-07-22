# WhisperX Transcription — Implementation Plan

## Ordering Rationale

Bottom-up: config → model → abstraction → worker → entry point. Each step is independently
testable before the next depends on it.

## Checklist

### 1. Config

- [ ] Add `whisperx_model`, `whisperx_device`, `whisperx_compute_type`, `worker_poll_interval` to `backend/src/rag_podcast/config.py`
- [ ] Defaults: `"small"`, `"cuda"`, `"float16"`, `10`
- **Validate:** `python -c "from rag_podcast.config import settings; print(settings.whisperx_model)"`

### 2. Database Migration

- [ ] Generate Alembic migration: add `transcript_data JSONB DEFAULT NULL` to `episode`
- [ ] Apply migration: `alembic upgrade head`
- **Validate:** `psql -c "\d episode"` shows `transcript_data` column

### 3. Episode Model

- [ ] Add `transcript_data` column to `backend/src/rag_podcast/models/episode.py`
- [ ] Import `JSONB` from `sqlalchemy.dialects.postgresql`
- [ ] Type: `Mapped[dict | None] = mapped_column(JSONB, nullable=True)`
- **Validate:** model loads without import errors

### 4. Transcriber Abstraction

- [ ] Create `backend/src/rag_podcast/transcription/transcriber.py`
- [ ] Define `Transcriber` Protocol with `async def transcribe(self, audio_path: Path) -> dict`
- [ ] Implement `LocalWhisperX`:
  - `__init__(self, model: str, device: str, compute_type: str)` — loads model via `whisperx.load_model()`
  - `async def transcribe(self, audio_path: Path) -> dict` — runs `model.transcribe()` + `align()` in `asyncio.to_thread()`
  - Return dict with `{"model": ..., "language": ..., "segments": [...]}`
- [ ] Define `TranscribeError` exception
- **Validate:** import works, no runtime errors on model load (model loads lazily or at init)

### 5. Worker Logic

- [ ] Create `backend/src/rag_podcast/transcription/worker.py`
- [ ] Implement `poll_downloaded(session: AsyncSession) -> Episode | None` — SELECT + LIMIT 1
- [ ] Implement `claim_episode(session: AsyncSession, episode_id: int) -> Episode | None` — atomic UPDATE with status guard, RETURNING *
- [ ] Implement `transcribe_episode(session, episode, transcriber, data_dir) -> None`:
  - Set PROCESSING (already done by claim)
  - Resolve full audio path from `episode.audio_local_path` + `data_dir`
  - Call `transcriber.transcribe(path)` inside try/except
  - On success: set `transcript_data`, status = DONE, commit
  - On failure: set status = FAILED, log exception, commit
- [ ] Implement `run_worker(transcriber, poll_interval) -> None`:
  - Infinite loop: poll → if found, transcribe → else sleep
  - Handle KeyboardInterrupt for graceful shutdown
  - Handle DB connection errors by crashing (let process manager restart)
- **Validate:** import `worker` module, no syntax errors

### 6. CLI Entry Point

- [ ] Create `backend/scripts/run_transcription_worker.py`
- [ ] Read settings from `config.Settings`
- [ ] Create `LocalWhisperX` from settings
- [ ] Create async DB session factory
- [ ] Call `run_worker(transcriber, poll_interval)`
- [ ] Configure logging (INFO level, show episode IDs and status transitions)
- **Validate:** `python backend/scripts/run_transcription_worker.py --help` (add argparse if desired)

### 7. Integration Test

- [ ] Ensure at least one episode has `transcript_status = 'downloaded'` in the DB
- [ ] Run worker: `uv run --extra transcription python backend/scripts/run_transcription_worker.py`
- [ ] Verify: episode status advances to `done`
- [ ] Verify: `episode.transcript_data` contains valid JSON with `segments` array
- [ ] Verify: each segment has `text`, `start`, `end` fields
- [ ] Verify: worker loops back to polling (logs "no episodes ready" after processing all)

### 8. Edge Cases

- [ ] Start worker with no DOWNLOADED episodes → polls with sleep, no crash
- [ ] Start worker with a FAILED episode → skips it (status != downloaded)
- [ ] Audio file missing on disk → marks FAILED, continues
- [ ] Ctrl+C during transcription → graceful shutdown (current episode may be left in PROCESSING; next run re-claims via status guard? No — PROCESSING != downloaded, so it's stuck. Fix: on startup, reset any PROCESSING episodes back to DOWNLOADED)
- [ ] Run worker against test database (`ragpodcast_test`) first

## Risky Files

| File | Risk | Mitigation |
|------|------|------------|
| `models/episode.py` | Adding column — migration must be applied first or model import crashes | Run migration before touching model |
| `transcription/transcriber.py` | `whisperx.load_model()` can OOM if model doesn't fit GPU | Default to "small" (2.5 GB VRAM); config allows "tiny" |
| `worker.py` | Infinite loop — CPU spin if poll_interval=0 | Minimum poll_interval = 1s; enforce in config validator |
| `run_transcription_worker.py` | Runs outside Docker — DB must be reachable from host | DB port 5432 already exposed in docker-compose.yml |

## Rollback Points

1. After migration: `alembic downgrade -1`
2. After worker start: stop worker (Ctrl+C), no DB changes to revert for un-processed episodes
3. After DONE episode: `UPDATE episode SET transcript_data = NULL, transcript_status = 'downloaded' WHERE id = ?`
