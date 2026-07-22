# Error Handling

> How errors are handled in this project.

---

## Overview

Each ingestion sub-step (feed fetch/parse, audio download, external ID resolution) defines its own narrow exception type. Routers catch these specific types and translate them to HTTP responses — they never catch bare `Exception`. Failures that are per-item (e.g. one episode's insert or download) are isolated with `try/except` + logging so they don't abort the whole request; failures that are request-level (bad input URL, unresolvable feed) are allowed to propagate out of the service layer straight to the router.

---

## Error Types

One exception class per ingestion sub-step, always a direct `Exception` subclass with only a docstring (no custom fields) — see `FeedParseError` (`ingestion/parser.py`), `DownloadError` (`ingestion/downloader.py`), `AppleResolutionError` (`ingestion/apple_podcasts.py`). Adding a new external integration or parsing stage should follow this same one-class-per-stage pattern rather than reusing an existing error type across unrelated stages.

---

## Error Handling Patterns

- **Request-level failure → let it propagate.** `parse_feed`/`resolve_apple_podcasts_url` raise directly; the router's `except` clauses are the only place these are caught (see API Error Responses below). Service-layer functions do not swallow these.
- **Per-item failure → isolate and continue.** `service._insert_episode` wraps a single episode insert in `session.begin_nested()` and catches broadly, logging via `logger.exception` and returning `None` so the caller marks it `skipped` instead of aborting the batch. `service._download_episode` similarly catches only `DownloadError` around a single episode's download and marks that episode `TranscriptStatus.FAILED` rather than failing the request.
- **External HTTP calls → dedicated session with retry, translated errors.** Any module calling out to a third-party HTTP API builds its own `requests.Session` with `HTTPAdapter(max_retries=Retry(...))` for transient 429/5xx (see `downloader.py:_build_session`, mirrored in `apple_podcasts.py`), and wraps the call in `try/except requests.RequestException` to re-raise as that module's own error type — callers should never see a raw `requests` exception.

---

## API Error Responses

`POST /podcasts` (`ingestion/router.py:create_podcast`) maps every ingestion-stage exception to `HTTPException(status_code=400, detail=str(exc))` — one `except` clause per exception type, checked in resolution order (Apple resolution happens before feed parsing, so `AppleResolutionError` is caught first):

```python
if is_apple_podcasts_url(rss_url):
    try:
        resolved = resolve_apple_podcasts_url(rss_url)
    except AppleResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ...
try:
    result = await ingest_podcast(session, rss_url, max_episodes=..., target_guid=...)
except FeedParseError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
```

All client-caused ingestion failures are `400`, never `500` — a bad/unreachable/unresolvable URL is a client input problem, not a server fault. Error `detail` messages should name which stage failed (e.g. "no feedUrl for collection id", "episode not found in feed") so the client can tell input-format problems from transient network problems.

### Validation & Error Matrix — Apple Podcasts URL resolution

| Condition | Result |
|---|---|
| `rss_url` is a plain RSS/Atom URL (not `podcasts.apple.com`/`itunes.apple.com`) | Passes through unchanged to `ingest_podcast`; zero behavior change |
| Apple URL, collection id resolves, no `?i=` episode param | `feed_url` resolved via `entity=podcast` lookup; normal `max_episodes` ingestion |
| Apple URL, collection id has no `feedUrl` in iTunes Lookup response | `AppleResolutionError` → HTTP 400 |
| Apple URL with `?i=<episodeId>`, episode found in the podcast's recent-episode list (`entity=podcastEpisode&limit=200`, matched by `trackId`) | `target_guid` resolved from `episodeGuid`; `parse_feed` ingests exactly that one episode, `max_episodes` ignored |
| Apple URL with `?i=<episodeId>`, episode not found (incl. episode older than Apple's ~40-50 episode lookup window — no paging exists) | `AppleResolutionError` → HTTP 400 |
| `target_guid` set but no RSS entry in the resolved feed has a matching `.guid` | `FeedParseError` → HTTP 400 |
| iTunes Lookup API network/HTTP failure (timeout, 5xx after retries exhausted) | `AppleResolutionError` → HTTP 400 (same class as a bad ID — not a 500) |

---

## Common Mistakes

- **Assuming Apple's iTunes Lookup API supports direct episode-ID lookup.** `GET /lookup?id=<episodeId>&entity=podcastEpisode` returns `resultCount: 0` even for a valid, currently-published episode — verified live. The only working call is `GET /lookup?id=<collectionId>&entity=podcastEpisode&limit=200`, scanning the returned list for `trackId == episodeId`. This list is capped to roughly the most recent 40-50 episodes regardless of the requested `limit`, with no documented offset/paging — treat "episode not found" for an old episode as an expected, user-facing 400, not a bug to chase. See `.trellis/tasks/07-22-itunes-rss-resolver/research/itunes-lookup-api.md` for the full verification trace.
- **Matching iTunes episodes to RSS entries by enclosure/audio URL.** Don't — Apple's `episodeGuid` field is an exact string match for the RSS `<guid>` element (`feedparser`'s `entry.get("id")`, i.e. `ParsedEpisode.guid`). Match on that field directly; comparing audio URLs is unnecessary and more fragile (redirects, query-param variations).
