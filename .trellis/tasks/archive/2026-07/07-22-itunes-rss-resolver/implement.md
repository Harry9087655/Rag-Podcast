# Implementation Plan: iTunes Podcast URL to RSS Feed Resolver

Reference: `design.md` for the technical design (already verified against the live iTunes Lookup API — no further API spike needed).

## Checklist

1. **`backend/src/rag_podcast/ingestion/apple_podcasts.py`** (new file)
   - `AppleResolutionError` exception.
   - `ResolvedApplePodcast` dataclass (`feed_url: str`, `target_guid: str | None`).
   - `is_apple_podcasts_url(url: str) -> bool`.
   - `resolve_apple_podcasts_url(url: str) -> ResolvedApplePodcast`, implementing the collection-id and episode-id lookups per `design.md`.
   - Reuse the `requests.Session` + `Retry` pattern from `downloader.py:67` (own private session builder, same retry policy on 429/5xx).
   - Timeout on lookup calls (e.g. 10s) — do not let a slow Apple API hang the request indefinitely.

2. **`backend/src/rag_podcast/ingestion/parser.py`**
   - Add `target_guid: str | None = None` param to `parse_feed`.
   - When set: filter `episodes` to guid match, raise `FeedParseError` if empty, skip the `max_episodes` truncation.
   - When unset: existing behavior unchanged.

3. **`backend/src/rag_podcast/ingestion/service.py`**
   - Add `target_guid: str | None = None` param to `ingest_podcast`, pass through to `parse_feed`.

4. **`backend/src/rag_podcast/ingestion/router.py`**
   - Import `is_apple_podcasts_url`, `resolve_apple_podcasts_url`, `AppleResolutionError`.
   - In `create_podcast`: detect + resolve before calling `ingest_podcast`; catch `AppleResolutionError` → HTTP 400 (mirror the existing `FeedParseError` handling at `router.py:49`).
   - `IngestRequest`/`IngestResponse` schemas unchanged.

5. **Tests** — new `backend/tests/ingestion/test_apple_podcasts.py`:
   - `is_apple_podcasts_url`: true for `podcasts.apple.com`/`itunes.apple.com` URLs with `/id<digits>`, false for arbitrary RSS URLs.
   - `resolve_apple_podcasts_url`: podcast-level success (mock `requests` to return a `feedUrl`), episode-level success (mock both lookup calls, verify `target_guid` extraction via `trackId` match), no-`feedUrl` → `AppleResolutionError`, no `trackId` match → `AppleResolutionError`, HTTP/network failure → `AppleResolutionError`.

   Update existing tests:
   - `backend/tests/ingestion/test_parser.py`: add cases for `target_guid` match found (returns 1 episode, ignores `max_episodes`) and not found (`FeedParseError`).
   - `backend/tests/ingestion/test_service.py`: verify `target_guid` is passed through to `parse_feed`.
   - `backend/tests/ingestion/test_router.py`: add a case where `rss_url` is an Apple Podcasts URL (monkeypatch `resolve_apple_podcasts_url` in `router_module`) and verify `ingest_podcast` is called with the resolved feed URL/guid; add a case where resolution raises `AppleResolutionError` → 400; confirm existing non-Apple-URL tests still pass unmodified (no-regression check).

## Validation Commands

```bash
cd backend
uv run pytest tests/ingestion/ -v
```

## Rollback Point

All changes are additive (new optional params with defaults preserving old behavior, new module, router branch gated by `is_apple_podcasts_url`). If something regresses, reverting the router's new branch alone restores prior behavior without touching `parser.py`/`service.py` call sites used elsewhere.
