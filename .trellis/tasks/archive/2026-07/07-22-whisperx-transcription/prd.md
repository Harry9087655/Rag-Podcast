# WhisperX Transcription

## Goal

Add WhisperX-based audio transcription to the RAG podcast pipeline — a separate polling
worker that runs on the host (GPU access), transcribes `DOWNLOADED` episodes into
word-level timestamped segments, and stores the result in a JSONB column on the Episode
table.

## Background

Ingestion is fully implemented and tested. Downloaded episodes sit in `data_dir/` with
`transcript_status = DOWNLOADED`. WhisperX has been verified to work on this machine
(`backend/scripts/test_whisperx.py`, output at `backend/aligned_segments.txt`).

The pipeline handoff contract is documented in `backend/src/rag_podcast/ingestion/PLAN.md`
§7: "The transcription worker polls for episodes where transcript_status = 'downloaded',
transcribes them, and advances status to PROCESSING → DONE."

## Confirmed Facts (from codebase & specs)

| Fact | Source |
|------|--------|
| WhisperX output: **word-level timestamps** — foundation for jump-to-audio | `spec.md:18` |
| WhisperX already works on host GPU (model: "small", compute: float16, device: cuda) | `backend/scripts/test_whisperx.py` |
| `TranscriptStatus` lifecycle: `PENDING` → `DOWNLOADED` → `PROCESSING` → `DONE` / `FAILED` | `models/episode.py:10-16` |
| `Chunk` model ready: `text`, `start`, `end`, `speaker`, `embedding` (Vector(1024)) | `models/chunk.py:13-18` |
| Audio path: `data_dir/{podcast_title}/{podcast_id}/{episode_id}.{ext}` | `downloader.py:50` |
| Audio path stored relative to `data_dir` in `episode.audio_local_path` | `service.py:130` |
| Download is synchronous (requests), blocking the async event loop | `downloader.py:33` |
| No worker infrastructure exists — no Celery, ARQ, `asyncio.create_task`, or polling loops | full codebase search |
| Dependencies (`whisperx`, `torch`, `torchaudio`, `ctranslate2`, `faster-whisper`) defined as `[transcription]` optional extra | `pyproject.toml:21-31` |
| Worker runs on **host** (not Docker) to access GPU/CUDA | `scaffold-plan.md:95-96` |
| Planned worker script `backend/scripts/run_transcription_worker.py` was never created | `scaffold-plan.md:93` |
| No speaker diarization in current scope (Chunk.speaker exists but is nullable) | `spec.md`, `chunk.py:18` |

## Requirements

- **R1**: Transcribe `DOWNLOADED` episodes using WhisperX with word-level timestamps
- **R2**: Store output as JSONB on `Episode.transcript_data`:
  `{"model": "...", "language": "...", "segments": [{"text": "...", "start": 0.0, "end": 3.2}, ...]}`
- **R3**: Advance `transcript_status` through `PROCESSING` → `DONE` (or `FAILED`)
- **R4**: Separate polling worker (`backend/scripts/run_transcription_worker.py`), runs on host
- **R5**: Abstract `Transcriber` protocol so local WhisperX can be swapped for a WhisperX API in the future
- **R6**: Worker polls DB in a loop, serial execution (one episode at a time)
- **R7**: WhisperX config (`model`, `device`, `compute_type`) configurable via `config.py` with defaults: `small` / `cuda` / `float16`
- **R8**: Resumable — worker crash + restart picks up where it left off (DB is source of truth)
- **R9**: Per-episode error isolation — one failure doesn't stop the worker

## Acceptance Criteria

- [ ] `Episode` model gains `transcript_data` JSONB column (Alembic migration)
- [ ] `config.py` gains WhisperX settings (`whisperx_model`, `whisperx_device`, `whisperx_compute_type`) with defaults
- [ ] `Transcriber` abstract protocol defined in `transcription/` module
- [ ] `LocalWhisperX` implementation passes through to `whisperx` library
- [ ] `backend/scripts/run_transcription_worker.py` polls DB, transcribes one episode at a time
- [ ] Worker correctly cycles status: `DOWNLOADED` → `PROCESSING` → `DONE` (or `FAILED`)
- [ ] Worker stores segments in `episode.transcript_data` as valid JSONB
- [ ] Worker runs successfully on host with `uv run --extra transcription`
- [ ] Failed transcription sets status `FAILED` and logs the error; worker continues to next episode
- [ ] Manual test: transcribe an existing `DOWNLOADED` episode and verify `transcript_data` contains valid segments

## Out of Scope

- Speaker diarization
- Embedding generation (indexing module)
- Re-transcription of already-DONE episodes
- WhisperX API implementation (abstract protocol defined, but only `LocalWhisperX` built)
- Chunking/slicing strategy (indexing module reads `transcript_data` → produces `Chunk` rows)
