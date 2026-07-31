# Wire transcription worker into ingestion pipeline with GPU Docker service

## Goal

Today, downloading (`POST /podcasts` → `ingest_podcast()`) and transcription
(`run_worker()`) are disconnected: the worker only runs if a human manually
executes `backend/scripts/run_transcription_worker.py` on the host — that
script and the rest of `backend/scripts/` are exploratory/test-only, not
production wiring. Replace the manual step with a real, always-running
transcription worker service, containerized separately from `api` so it can
access the host GPU, so episodes transcribe automatically once downloaded.

## Background (confirmed facts from repo inspection)

- **Download path**: `ingestion/service.py:_download_episode()` (lines
  119-135) sets `episode.audio_local_path` and
  `episode.transcript_status = DOWNLOADED` (or `FAILED`), commits, inside the
  `POST /podcasts` request handler (`ingestion/router.py:44`).
- **Worker path**: `transcription/worker.py:run_worker()` polls for
  `transcript_status == DOWNLOADED` episodes and transcribes them; only
  entry point today is `backend/scripts/run_transcription_worker.py` →
  `transcription/cli.py:main()`, both docstring'd "Run on the host (not
  inside Docker) to access GPU/CUDA."
- **No orchestration exists today**: `main.py` mounts only `ingestion_router`
  + `health_router`; `docker-compose.yml` defines only `db` and `api`; no
  Procfile; no `[project.scripts]` entry.
- **Coordination pattern already chosen at the spec level**: `spec.md`
  (row "转录任务编排", §7 open question #4) explicitly leans toward
  "status table + background worker" over a task queue (RQ/Celery) to avoid
  over-engineering. The DB-poll pattern in `worker.py` already implements
  this; `worker_poll_interval` defaults to 10s (`config.py:37`).
- **GPU dependency chain already pinned and host-validated** (`spec.md` §7):
  `torch~=2.8.0` from `download.pytorch.org/whl/cu128`, `ctranslate2==4.8.1`
  (post-fix for the sm_120/Blackwell int8 crash), `whisperx==3.8.6`. Host
  GPU is an RTX 5070 Ti Laptop (Blackwell, sm_120, 12GB). These are isolated
  in the `transcription` optional-dependency group in
  `backend/pyproject.toml:21-22` — the `api` image does not install them
  today (its `Dockerfile` runs plain `uv sync --frozen --no-dev`, explicit
  `COPY` list, no `COPY . .`).
- **Settings already assume GPU**: `config.py` has `whisperx_device: str =
  "cuda"`, `funasr_device: str = "cuda"`, plus all model/compute-type knobs
  `transcription/cli.py` already wires into `create_transcriber()`.
- **Networking/env pattern**: `docker-compose.yml`'s `api` service overrides
  `DATABASE_URL` to `db:5432` (compose service name) and mounts
  `./data:/data` (`DATA_DIR=/data`); a worker container needs the same
  overrides to reach Postgres and read downloaded audio files.
- **GPU passthrough verified working in this exact dev environment**:
  `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
  succeeds and shows the RTX 5070 Ti Laptop from `spec.md`; `docker info`
  lists an `nvidia` container runtime. No separate host setup step needed.
- **Full spike already executed against the real `backend/Dockerfile.worker`**
  (built, not just designed) — see `design.md`'s "Validation plan" for the
  complete list. Headline results: image builds clean on Linux (torch
  2.8.0+cu128, ctranslate2 4.8.1, whisperx 3.8.6, funasr all resolve);
  GPU + cuDNN work (fp16 `Conv2d` forward pass succeeds, `cudnn.version()`
  → 91002); the real `transcriber.py` code path (`load_audio` →
  `transcribe` → `align`, and FunASR's `AutoModel.generate()`) runs
  end-to-end inside the container with no CUDA/cuDNN errors.
- **Two real gaps found via testing, both now fixed**:
  1. `whisperx.load_audio()` shells out to the `ffmpeg` CLI binary
     (separate from `torchcodec`'s bundled libs) — `python:3.12-slim`
     doesn't ship it. Fixed: `apt-get install ffmpeg` added to
     `backend/Dockerfile.worker`.
  2. WSL2's default memory cap (50% of host RAM when `.wslconfig` has no
     `memory=` override — was ~7.6GB on this 15.2GB host) is too small for
     FunASR's `AutoModel`, which loads three models (paraformer-large, VAD,
     punc) in one process; it OOM-killed mid-load (confirmed via `docker
     events --filter event=oom`). Fixed: added `memory=12GB` to
     `C:\Users\harry\.wslconfig`, applied via `wsl --shutdown` + Docker
     Desktop restart (`docker info` now reports 11.68GB). This is a real
     Windows+WSL2 deployment prerequisite, not just a session fix — worth
     a line in setup docs.
- **Model cache reuse verified**: bind-mounting the host's existing
  `~/.cache/{huggingface,modelscope,torch}` (5.6GB + 3.0GB + 361MB already
  on disk) into the container avoided redownloading weights FunASR/WhisperX
  already have — confirmed by inspecting the mounted ModelScope cache for
  the exact model directory FunASR requested.

## Requirements

1. New `worker` compose service, always started with `docker compose up`
   alongside `db`/`api` (not an opt-in profile — explicitly decided: this is
   a single-user/personal dev setup, distribution-friendliness for
   GPU-less contributors is out of scope for now).
2. New Dockerfile for the worker (`backend/Dockerfile.worker`, already
   created and build-validated), built from the `backend/` context,
   installing the `transcription` extra
   (`uv sync --frozen --no-dev --extra transcription`), plus `ffmpeg` via
   `apt-get` (required by `whisperx.load_audio()` — found missing during
   testing), `CMD` running `python -m rag_podcast.transcription.cli`.
3. GPU access via compose's `deploy.resources.reservations.devices`
   (`driver: nvidia`, `capabilities: [gpu]`) — the standard mechanism for
   Docker Compose v2, verified compatible with this host's Docker
   Desktop/WSL2 setup.
4. Worker container reuses the same env overrides as `api`
   (`DATABASE_URL` → `db:5432`, `DATA_DIR` → `/data`) and mounts the same
   `./data:/data` volume so it can read audio files ingestion downloaded.
5. Coordination stays DB-poll only (existing `worker_poll_interval`,
   default 10s) — no new HTTP trigger/queue between `api` and `worker`.
   Matches `spec.md`'s explicit preference against a task queue.
6. Model weight caches persist across container restarts/rebuilds by
   bind-mounting the host's existing `~/.cache/{huggingface,modelscope,
   torch}` directories (not a fresh named Docker volume — verified a named
   volume would start empty and redownload the ~9GB already on this host).
7. Worker container has `restart: unless-stopped` and `depends_on: db
   (service_healthy)`, matching the `api` service pattern.
   `worker.py:reset_stale_processing()` already resets orphaned PROCESSING
   episodes back to DOWNLOADED on startup, so container restarts mid-job are
   already safe — no code change needed there.
8. No new required env vars: all transcription settings in `config.py`
   already have defaults (`whisperx_device="cuda"`, etc.).
9. Host prerequisite (Windows+WSL2 only): WSL2 must be given enough memory
   headroom for FunASR's combined model load (paraformer-large + VAD +
   punc in one process) — this machine needed raising `.wslconfig`'s
   `memory=` from WSL2's ~7.6GB default to 12GB. Document this in setup
   docs during implementation so it isn't a mystery OOM later.

## Acceptance Criteria

Already validated via a standalone `docker run` spike against the real
`backend/Dockerfile.worker` (not yet through `docker-compose.yml` — see
`design.md`'s Validation Plan for the full list):

- [x] Image builds cleanly on Linux with the `transcription` extra.
- [x] GPU is visible and cuDNN works inside the container (fp16 `Conv2d`
      forward pass, `ctranslate2.get_cuda_device_count()` → 1).
- [x] WhisperX's real code path (`load_audio → transcribe → align`) and
      FunASR's `AutoModel(...).generate()` both run end-to-end with no
      CUDA/cuDNN/OOM errors.
- [x] Bind-mounting the host's existing model caches avoids re-downloading
      already-present weights.

Still pending — require the actual `docker-compose.yml` wiring:

- [ ] `docker compose up` (no profile flag) starts `db`, `api`, and
      `worker`; `docker compose ps` shows `worker` running.
- [ ] End-to-end smoke test through the running compose stack: `POST
      /podcasts` to ingest+download a short test episode → worker picks
      it up within one poll interval → `episode.transcript_status`
      transitions `DOWNLOADED → PROCESSING → DONE`, with
      `transcript_data`/`language` populated.
- [ ] Restarting the worker container (`docker compose restart worker`)
      does not re-download model weights, via the compose-managed bind
      mounts specifically (not the ad-hoc `docker run` mounts used in the
      spike).

## Out of Scope

- Making the worker optional / GPU-less fallback (deferred — see
  Requirement 1's rationale).
- Explicit push/HTTP trigger between `api` and `worker` (DB-poll kept).
- Migrating off the DB-poll pattern to a task queue (RQ/Celery) —
  `spec.md` already leans against this.
- Multiple worker replicas / concurrency control beyond the existing
  single-worker claim-by-update-where pattern in `worker.py`.
- Non-Windows/non-WSL2 host documentation (this environment's GPU
  passthrough is already verified working).
