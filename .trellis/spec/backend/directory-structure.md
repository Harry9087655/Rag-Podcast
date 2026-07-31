# Directory Structure

> How backend code is organized in this project.

---

## Overview

<!--
Document your project's backend directory structure here.

Questions to answer:
- How are modules/packages organized?
- Where does business logic live?
- Where are API endpoints defined?
- How are utilities and helpers organized?
-->

(To be filled by the team)

---

## Directory Layout

```
<!-- Replace with your actual structure -->
src/
├── ...
└── ...
```

---

## Module Organization

The `ingestion/` package (`backend/src/rag_podcast/ingestion/`) is organized by pipeline stage, one module per external concern, each independently testable:

- `parser.py` — turns a fetched RSS/Atom document into `ParsedPodcast`/`ParsedEpisode`.
- `downloader.py` — fetches episode audio bytes.
- `apple_podcasts.py` — resolves an Apple Podcasts URL to a real feed URL (+ optional target episode guid) via the iTunes Lookup API, upstream of `parser.py`.
- `service.py` — orchestrates the stages plus DB writes (`ingest_podcast`).
- `router.py` — the FastAPI surface; wires request fields to `service.py` calls and maps stage-specific exceptions to HTTP responses.

The `transcription/` package (`backend/src/rag_podcast/transcription/`) follows the same stage-module pattern:

- `transcriber.py` — `Transcriber` Protocol (async, swappable backends) + `LocalWhisperX` implementation wrapping the `whisperx` library via `asyncio.to_thread()`.
- `worker.py` — orchestrates the poll→claim→transcribe→store loop (`transcribe_episode`, `run_worker`). Imports `models/episode.py` and `transcriber.py` but not `whisperx` directly.
- `__init__.py` — guarded re-exports with `try/except ImportError` so the module can be imported in the API container (Docker, no GPU/whisperx) without crashing.

The worker entry point (`backend/scripts/run_transcription_worker.py` → `transcription/cli.py:main()`) runs as its own `worker` container (`docker-compose.yml`, built from `backend/Dockerfile.worker`) with GPU passthrough — it is a separate process/image from the `api` container, not imported by the API process. Running the script directly on the host still works as a manual/local-debugging fallback.

When adding a new external integration (a new source, a new lookup/resolution step, etc.), add a new stage module rather than growing `service.py` or `router.py` directly — each module owns its own error type (see `.trellis/spec/backend/error-handling.md`) and, if it makes HTTP calls, its own `requests.Session` with retry/backoff (mirror `downloader.py`'s `_build_session`).

---

## Configuration (`config.py` / `.env`)

`Settings` (`config.py`) loads from a single repo-root `.env` shared by every compose service (`db`, `api`, `worker`) plus host-side runs (`pytest`, `uv run uvicorn`). Because `docker-compose.yml` also reads that same `.env` file directly for its own variable interpolation (e.g. `HOST_CACHE_DIR`, used only to build the worker's model-cache bind-mount paths — see `.env.example`), `.env` legitimately contains keys that are not, and should never become, `Settings` fields.

> **Gotcha**: `SettingsConfigDict` must set `extra="ignore"`. `pydantic-settings` defaults to `extra="forbid"`; with that default, any key present in `.env` but not declared on `Settings` raises `ValidationError` at import time. This only breaks host-side use (`pytest`, `uv run uvicorn`) — containers never see this because they get env vars from compose's `environment:`/`env_file:` merge, not by reading `.env` through `Settings`' own loader, so the bug can hide behind a green `docker compose up` and still fail every host-side test run. If you add a new compose-only interpolation variable to `.env`, it does not need a matching `Settings` field, but `extra="ignore"` must stay set.

**Cross-platform bind-mount paths**: don't reference `$HOME`/`%USERPROFILE%` inside `docker-compose.yml` for host paths — which one resolves depends on whichever shell happens to invoke `docker compose up` (Git Bash vs. native PowerShell on Windows), so the same file behaves differently across dev shells. Instead, define an explicit variable in `.env` (e.g. `HOST_CACHE_DIR`) with the real host path; compose reads `.env` directly regardless of invoking shell, so this sidesteps the ambiguity entirely.

---

## Naming Conventions

<!-- File and folder naming rules -->

(To be filled by the team)

---

## Examples

<!-- Link to well-organized modules as examples -->

(To be filled by the team)
