# Implementation plan: worker GPU docker-compose service

## Already done (pre-implementation validation spike, this session)

- [x] `backend/Dockerfile.worker` created — `python:3.12-slim` base,
      `uv sync --frozen --no-dev --extra transcription`, `apt-get install
      ffmpeg` (found necessary via testing — `whisperx.load_audio()` shells
      out to the CLI binary), `CMD ["python", "-m",
      "rag_podcast.transcription.cli"]`.
- [x] Built the image standalone (`docker build -f backend/Dockerfile.worker
      -t rag-podcast-worker:risktest backend/`) — confirmed the
      `transcription` extra resolves cleanly on Linux.
- [x] GPU + cuDNN isolation tests passed (`torch.cuda.is_available()`,
      fp16 `Conv2d` forward pass, `ctranslate2.get_cuda_device_count()`).
- [x] Real code-path end-to-end tests passed: WhisperX
      (`load_audio → transcribe → align`) and FunASR (`AutoModel(...)
      .generate()`), both via standalone `docker run --gpus all` with host
      cache bind-mounts (`~/.cache/{huggingface,modelscope,torch}`).
- [x] Found + fixed WSL2 memory ceiling: raised `.wslconfig`'s `memory=` to
      12GB (was defaulting to ~7.6GB), applied via `wsl --shutdown` +
      Docker Desktop restart. `docker info` now reports 11.68GB.

## Ordered checklist (remaining — not yet done)

1. Add `worker` service to `docker-compose.yml` (see `design.md` for the
   block): build context `./backend` with `dockerfile: Dockerfile.worker`,
   `DATABASE_URL`/`DATA_DIR` env overrides matching `api`, `./data:/data`
   mount, **plus bind-mounts of the host's existing model caches**
   (`~/.cache/huggingface`, `~/.cache/modelscope`, `~/.cache/torch` — not a
   fresh named volume, see design.md's "Model cache" section for why),
   `deploy.resources.reservations.devices` GPU block, `restart:
   unless-stopped`, `depends_on: db (service_healthy)`.
   - Resolve the `${HOME}` interpolation caveat first (design.md flags
     this): confirm it resolves correctly from whatever shell will
     actually run `docker compose up`, or use `${USERPROFILE}` / an
     explicit `.env` variable instead.
2. `docker compose config` — sanity-check the compose file parses.
3. `docker compose up -d` (all three services) — confirm `worker` starts
   and stays up. GPU/model-load correctness is already validated by the
   spike above; this step is about compose wiring specifically (env vars,
   mounts, `depends_on`), not re-proving the GPU chain works.
4. End-to-end smoke test through the actual running stack: `POST
   /podcasts` with a short/cheap test episode, poll the DB and confirm
   `transcript_status: DOWNLOADED → PROCESSING → DONE` with
   `transcript_data`/`language` populated within one `worker_poll_interval`
   after download completes.
5. `docker compose restart worker` — confirm fast restart via the
   compose-managed bind mounts (no re-download).
6. Update `backend/scripts/run_transcription_worker.py`'s docstring (and
   `transcription/cli.py`'s, which has the same "Run on the host (not
   inside Docker)" line) — no longer accurate once the containerized path
   works. Keep the script itself as a manual/local-debugging fallback,
   just correct the comment.
7. Add a short note to repo setup docs (README or spec.md) about the WSL2
   memory prerequisite (`memory=12GB` in `.wslconfig`) so a future
   Windows+WSL2 setup doesn't hit the same OOM mystery.

## Validation commands

- `docker compose config`
- `docker compose up -d`
- `docker compose logs -f worker`
- `docker compose exec worker python -c "import torch; print(torch.cuda.is_available())"`
- `docker compose restart worker` (cache-persistence check)

## Rollback point

Everything here is additive (`backend/Dockerfile.worker` already exists
and is validated; `docker-compose.yml`'s `worker` block is the only
remaining new addition). If compose wiring hits issues, `db`/`api` are
unaffected — `worker` can be dropped from `docker-compose.yml` and
iterated on independently. The GPU/model-loading risk that would have been
the expensive thing to debug post-hoc is already retired.

## Follow-up checks before `task.py start`

- [x] `prd.md` acceptance criteria map 1:1 to the checklist above, split
      into already-validated vs. still-pending.
- [x] `design.md` covers architecture, Dockerfile (with ffmpeg fix), compose
      block, cache bind-mount decision (with rationale for rejecting the
      original named-volume plan), WSL2 memory prerequisite, compatibility,
      rollback, and a full validation-plan writeup of what was already
      tested.
- [x] `implement.jsonl` / `check.jsonl` have real entries (spec.md,
      backend/pyproject.toml, docker-compose.yml), seed line removed.
