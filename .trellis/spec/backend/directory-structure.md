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

When adding a new external integration (a new source, a new lookup/resolution step, etc.), add a new stage module rather than growing `service.py` or `router.py` directly — each module owns its own error type (see `.trellis/spec/backend/error-handling.md`) and, if it makes HTTP calls, its own `requests.Session` with retry/backoff (mirror `downloader.py`'s `_build_session`).

---

## Naming Conventions

<!-- File and folder naming rules -->

(To be filled by the team)

---

## Examples

<!-- Link to well-organized modules as examples -->

(To be filled by the team)
