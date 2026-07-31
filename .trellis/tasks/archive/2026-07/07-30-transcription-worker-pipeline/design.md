# Design: transcription worker as a GPU docker-compose service

## Architecture

Three always-on compose services, coordinated only through Postgres
(`episode.transcript_status`) — no new network calls between `api` and
`worker`:

```
docker compose up
├── db      (existing, unchanged)
├── api     (existing, unchanged) ── POST /podcasts → downloads audio,
│                                     sets transcript_status=DOWNLOADED
└── worker  (NEW) ── polls transcript_status==DOWNLOADED every
                      worker_poll_interval (10s), transcribes via
                      WhisperX/FunASR on host GPU, sets DONE/FAILED
```

`worker` is a straight containerization of the existing
`transcription/cli.py:main()` → `worker.py:run_worker()` code path — no
application code changes needed, only packaging/orchestration.

## Worker Dockerfile (`backend/Dockerfile.worker`)

Mirrors the existing `backend/Dockerfile` (same `python:3.12-slim` base,
same `uv`-via-copy pattern, same explicit `COPY` list — no `COPY . .`), with
two differences: installs the `transcription` extra, and runs the
transcription CLI instead of uvicorn.

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra transcription

# whisperx.load_audio() shells out to the ffmpeg CLI (not just torchcodec's
# bundled libs) — python:3.12-slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "rag_podcast.transcription.cli"]
```

No `alembic`/`alembic.ini` copy — the worker never runs migrations, only
`api` does (unchanged). This is the actual content of
`backend/Dockerfile.worker`, already created and build-validated (see
"Validated" below) — not just a proposal.

**Why not an `nvidia/cuda` base image**: verified via `uv.lock` that
`torch` and `ctranslate2`'s Linux wheels each vendor their own CUDA/cuDNN
runtime libraries as pip dependencies/bundled `.so`s
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc. resolve as transitive
deps of `torch`; `ctranslate2`'s wheel has no CUDA pip deps at all — it
statically bundles its own). Only the NVIDIA *driver* interface
(`libcuda.so`, `libnvidia-ml.so`) needs to come from the host via the
`nvidia` container runtime, which `docker info` on this machine already
lists. `python:3.12-slim` + pip-installed GPU wheels is therefore
sufficient and keeps the worker image consistent with the `api` image's
existing pattern. Confirmed by the build below — no missing CUDA runtime
library errors.

**ffmpeg gap (found, fixed)**: `whisperx.load_audio()` (called directly by
`transcriber.py`) runs the `ffmpeg` binary via `subprocess`, separately
from `torchcodec`'s bundled libs. `python:3.12-slim` doesn't include it —
the first real audio-loading smoke test failed with `ffmpeg: command not
found` until the `apt-get install ffmpeg` layer above was added.

## docker-compose.yml changes

```yaml
  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.worker
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
      DATA_DIR: /data
    volumes:
      - ./data:/data
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/.cache/modelscope:/root/.cache/modelscope
      - ${HOME}/.cache/torch:/root/.cache/torch
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy

volumes:
  pgdata:
```

Mirrors `api`'s existing `DATABASE_URL`/`DATA_DIR` overrides and
`./data:/data` mount exactly (Requirement 4). `deploy.resources.reservations
.devices` is the standard Compose v2 GPU-request syntax and needs no Swarm
mode for `docker compose up`.

**`${HOME}` interpolation caveat (to finalize during implementation)**:
docker compose resolves `${HOME}` from the shell environment active when
`docker compose up` is invoked. This session's Git Bash sets `HOME`
correctly, but native `cmd.exe`/PowerShell use `USERPROFILE` instead and
may leave `HOME` unset. Implementation should either confirm `${HOME}`
resolves correctly from whatever shell will actually run `docker compose
up` day-to-day, or fall back to `${USERPROFILE}` / an explicit path in
`.env`. Not resolved yet — deliberately left as an implementation step
rather than guessed here.

## Model cache: bind-mount host cache, not a named volume

**Original plan (superseded)**: a fresh named Docker volume
(`model_cache:/root/.cache`). Rejected after testing — it starts empty,
so the containerized worker would redownload the ~9GB of weights already
sitting on this host (5.6GB Hugging Face cache including
`Systran/faster-whisper-small`, 3.0GB ModelScope cache including the
FunASR paraformer/VAD/punc models, 361MB torch cache) instead of reusing
them.

**Decision**: bind-mount the existing host cache directories directly
(`~/.cache/huggingface`, `~/.cache/modelscope`, `~/.cache/torch` →
matching container paths under `/root/.cache`, since the worker runs as
root). Verified working: FunASR's `AutoModel` recognized the already
-cached `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common
-vocab8404-pytorch` directory on the mounted host path and didn't
re-download the multi-GB weights (only did fast manifest/checksum
checks). Same principle applies to WhisperX's Hugging Face cache.

## Host prerequisite: WSL2 memory limit (found, fixed on this machine)

Discovered via testing, not analysis: FunASR's `AutoModel` loads three
models at once in a single process (paraformer-**large**, VAD,
punc-transformer). With no `memory=` override in `.wslconfig`, WSL2
defaults to capping itself at 50% of host physical RAM — on this 15.2GB
host that's ~7.6GB, matching what `docker info` reported
(`Total Memory: 7.358GiB`). The combined FunASR load OOM-killed the
container (confirmed via `docker events --filter event=oom`) partway
through loading the punc model.

**Fix applied**: added `memory=12GB` under `[wsl2]` in
`C:\Users\harry\.wslconfig`, then `wsl --shutdown` + Docker Desktop
restart. `docker info` now reports `Total Memory: 11.68GiB`; the same
FunASR load/generate test that OOM-killed before now completes cleanly.

**This is a real deployment prerequisite, not just a one-off fix for this
session** — anyone else running this worker service needs the same WSL2
memory headroom (Windows+WSL2 hosts only; not applicable on native Linux).
Worth a line in the repo's setup docs during implementation so a future
`docker compose up` failure here isn't a mystery.

## Compatibility / migration

No DB schema changes. No changes to `worker.py`, `transcriber.py`, or
`cli.py` — this task is packaging/orchestration only. `api` and `db`
services are untouched.

## Rollback

`worker` is additive and isolated: `docker compose stop worker` (or
removing the service block) disables it without affecting `api`/`db`.
`reset_stale_processing()` (already in `worker.py`) means an interrupted
worker leaves no stuck `PROCESSING` rows — the next `worker` boot (or a
manual UPDATE) recovers cleanly.

## Validation plan

### Already executed as a pre-implementation spike (this session)

Built `rag-podcast-worker:risktest` directly from `backend/Dockerfile.worker`
(the real file, not a throwaway) and ran it standalone with `docker run
--gpus all` + host cache bind-mounts (no compose file involved yet):

1. **Image build** — `uv sync --frozen --no-dev --extra transcription`
   resolved and installed cleanly on the Linux target (torch 2.8.0+cu128,
   ctranslate2 4.8.1, whisperx 3.8.6, funasr). Confirms the win32-marker
   dependency-set risk noted above does not break the build.
2. **GPU + cuDNN isolation tests** — `torch.cuda.is_available()` →
   `True`, device name matches the RTX 5070 Ti Laptop; `ctranslate2.
   get_cuda_device_count()` → 1; explicit fp16 `Conv2d` forward pass
   (exercises cuDNN specifically, not just cuBLAS) succeeded
   (`torch.backends.cudnn.version()` → 91002).
3. **Real code-path end-to-end (WhisperX)** — `whisperx.load_audio()` →
   `model.transcribe()` → `whisperx.align()` on a synthetic 3s audio
   clip, using the exact call sequence `transcriber.py` makes. Required
   adding the `ffmpeg` apt package (see above) — failed without it, passed
   after. Output: `FULL PIPELINE OK`.
4. **Real code-path end-to-end (FunASR)** — `funasr.AutoModel(...)` +
   `.generate()` on the same clip. First attempt OOM-killed (see WSL2
   memory section above); passed cleanly after raising the WSL2 memory
   limit to 12GB.
5. **Cache reuse** — bind-mounting the host's existing `~/.cache/{
   huggingface,modelscope,torch}` avoided redownloading the ~9GB of
   weights already present; confirmed by inspecting the mounted
   ModelScope cache for the exact model directory FunASR requested.

### Still pending (real implementation, not yet done)

6. Add the `worker` service block to `docker-compose.yml` (resolve the
   `${HOME}` interpolation caveat above first).
7. `docker compose up -d` (all three services) — confirm `worker` starts
   and stays up (`restart: unless-stopped`, `depends_on: db`).
8. End-to-end through the *actual* pipeline: `POST /podcasts` to
   ingest+download a real short episode, watch `episode.transcript_status`
   transition `DOWNLOADED → PROCESSING → DONE` via the running compose
   stack (not a standalone `docker run`).
9. `docker compose restart worker` — confirm fast restart with no
   model re-download, this time through the compose-managed bind mounts.
