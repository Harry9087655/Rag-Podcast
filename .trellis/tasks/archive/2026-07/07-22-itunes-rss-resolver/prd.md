# iTunes Podcast URL to RSS Feed Resolver

## Goal

When the user submits an Apple Podcasts URL to the existing podcast ingestion endpoint, the system transparently resolves it to the podcast's real RSS feed URL (via Apple's iTunes Lookup API) and hands that off to the existing RSS parsing/ingestion workflow — without the user having to manually find the RSS feed themselves.

## Background / Confirmed Facts

- Current ingestion entry point: `POST /podcasts` (`backend/src/rag_podcast/ingestion/router.py:43`) accepts `{rss_url, max_episodes}` and calls `ingest_podcast(session, rss_url, max_episodes)` (`backend/src/rag_podcast/ingestion/service.py:28`), which calls `parse_feed(rss_url, max_episodes)` (`backend/src/rag_podcast/ingestion/parser.py:32`). `parse_feed` truncates to the newest `max_episodes` entries after extracting all audio-bearing entries.
- `parse_feed` requires a working RSS/Atom URL parseable by `feedparser`; it raises `FeedParseError` on failure, which the router (`router.py:49`) turns into an HTTP 400.
- No existing code touches Apple Podcasts / iTunes (confirmed via repo-wide search).
- Frontend (`frontend/src/main.ts`) is currently just a health-check stub — no podcast submission form exists yet, so this feature is backend-scoped for now.
- `requests` is already a backend dependency; `backend/src/rag_podcast/ingestion/downloader.py:67` has an established pattern for a `requests.Session` with retry/backoff via `urllib3.util.retry.Retry`, reusable for calling Apple's iTunes Lookup API.
- Apple Podcasts URLs look like `https://podcasts.apple.com/<locale>/podcast/<slug>/id<collectionId>` and, for episode-specific links, add a `?i=<episodeId>` query param.
- Apple's iTunes Lookup API (`https://itunes.apple.com/lookup?id=<id>&entity=<entity>`) is a public, unauthenticated, unkeyed HTTP endpoint — no new settings/env vars needed. It is a direct ID lookup (the Apple URL already contains the numeric id), not a fuzzy keyword search. **Verified live against a real podcast (2026-07-22)**:
  - `entity=podcast` with the collection id → returns a `feedUrl` field for the podcast.
  - Episode ids are **not** directly look-up-able (`lookup?id=<episodeId>&entity=podcastEpisode` returns `resultCount: 0`). Instead, `lookup?id=<collectionId>&entity=podcastEpisode&limit=200` returns a list of the podcast's most recent episodes; the target is found by scanning for `trackId == <episodeId>`, yielding an `episodeGuid` field.
  - `episodeGuid` is an exact string match for the RSS `<guid>` element (`feedparser`'s `entry.get("id")`, i.e. `ParsedEpisode.guid`) — confirmed byte-for-byte on a live example. Matching is a plain equality check, no enclosure-URL comparison needed.
  - **Known limitation**: despite requesting `limit=200`, Apple's API only returns ~40-50 of the most recent episodes regardless of the requested limit, with no documented paging. Episode-level resolution therefore only works for a podcast's recent episodes — an older episode link will not match and falls into the same hard-error path as any other resolution failure (requirement #4). This is an accepted v1 limitation, not a bug.
- Test convention: one test file per module under `backend/tests/ingestion/`; service-layer functions are monkeypatched in router tests (see `backend/tests/ingestion/test_router.py`).

## Requirements

1. **Detection**: `POST /podcasts` must detect when `rss_url` is an Apple Podcasts URL (host `podcasts.apple.com` or `itunes.apple.com`, path containing `/id<digits>`) versus a normal RSS URL, and only apply resolution in the former case. Non-Apple URLs go through the existing flow unchanged.
2. **Podcast-level resolution**: For an Apple Podcasts URL with only a collection id (no `?i=`), resolve the `feedUrl` via iTunes Lookup (`entity=podcast`) and pass it into the existing `ingest_podcast(session, feed_url, max_episodes)` unchanged — the normal "latest N episodes" ingestion semantics apply.
3. **Episode-level resolution**: For an Apple Podcasts URL that also has `?i=<episodeId>`, resolve:
   - the podcast's `feedUrl` via iTunes Lookup (`entity=podcast`), and
   - the target episode's RSS `guid` via iTunes Lookup (`entity=podcastEpisode&limit=200`, matched by `trackId`).
   Then ingest **only that one episode**: parse the resolved feed, find the entry whose guid matches, and ingest exactly that entry — ignoring/overriding `max_episodes` for this request.
4. **Resolution failures are hard errors** (v1 scope): if iTunes Lookup returns no `feedUrl` for a collection id, or no guid match can be found for a requested episode id (including because it's older than Apple's ~40-50 episode lookup window), the request fails with a clear 400 error that names which step failed. No "direct audio link, no RSS" fallback is implemented in v1 — the whole ingestion pipeline assumes full podcast/episode metadata from an RSS feed, and a bare audio URL doesn't fit that model.
5. **Out of scope for v1**: Apple short links (`apple.co/...`), non-podcast iTunes entity types, and any frontend UI changes (no submission form exists yet to change).

## Acceptance Criteria

- [ ] Submitting a podcast-level Apple Podcasts URL (e.g. `https://podcasts.apple.com/us/podcast/some-show/id123456789`) to `POST /podcasts` ingests the podcast using its real RSS feed, respecting `max_episodes` as today.
- [ ] Submitting an episode-level Apple Podcasts URL (e.g. `.../id123456789?i=1000456789012`) for a **recent** episode to `POST /podcasts` ingests exactly that one episode, regardless of `max_episodes`.
- [ ] Submitting a plain RSS URL to `POST /podcasts` behaves exactly as it does today (no regression).
- [ ] An Apple Podcasts URL whose collection id has no `feedUrl` in iTunes Lookup, or whose episode id can't be matched (including because it's outside Apple's recent-episode lookup window), returns HTTP 400 with a message identifying which resolution step failed.
- [ ] Network/HTTP failures calling the iTunes Lookup API surface as the same class of 400 error as other resolution failures (not a 500).
- [ ] New unit tests cover: URL detection, podcast-level resolution success, episode-level resolution success, no-feedUrl failure, episode-match failure, and the router's pass-through behavior for non-Apple URLs.

## Notes

- Complex task — see `design.md` for technical design and `implement.md` for the execution checklist.
