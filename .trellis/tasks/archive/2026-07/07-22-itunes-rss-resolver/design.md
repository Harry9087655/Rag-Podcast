# Design: iTunes Podcast URL to RSS Feed Resolver

## Overview

Add a new `apple_podcasts.py` module in `backend/src/rag_podcast/ingestion/` that detects and resolves Apple Podcasts URLs via the iTunes Lookup API. Wire it into `router.py` as a pre-processing step before the existing `ingest_podcast` call. `parser.py` and `service.py` gain an optional "restrict to one specific entry" capability to support episode-level ingestion, without changing their behavior for existing callers.

## New Module: `ingestion/apple_podcasts.py`

Mirrors the existing `downloader.py` pattern (own `requests.Session` with retry/backoff).

```python
class AppleResolutionError(Exception):
    """Raised when an Apple Podcasts URL can't be resolved to feed/episode data."""

@dataclass
class ResolvedApplePodcast:
    feed_url: str
    target_guid: str | None  # set only for episode-level URLs

EPISODE_LOOKUP_LIMIT = 200  # see "Verified against live API" below re: actual cap

def is_apple_podcasts_url(url: str) -> bool: ...

def resolve_apple_podcasts_url(url: str) -> ResolvedApplePodcast:
    """
    1. Parse host + path to extract collection_id (required) and episode_id (optional, from ?i=).
    2. Call https://itunes.apple.com/lookup?id={collection_id}&entity=podcast
       -> feed_url = results[0]["feedUrl"]; raise AppleResolutionError if missing/empty results.
    3. If episode_id present, call
       https://itunes.apple.com/lookup?id={collection_id}&entity=podcastEpisode&limit={EPISODE_LOOKUP_LIMIT}
       -> find the result whose trackId == episode_id -> target_guid = result["episodeGuid"]
       -> raise AppleResolutionError if no result matches (episode outside Apple's lookup window, see risk note).
    4. Return ResolvedApplePodcast(feed_url, target_guid).
    """
```

- Detection regex: host in `{"podcasts.apple.com", "itunes.apple.com"}` and path matches `/id(\d+)`; episode id from query param `i` (digits only).
- HTTP errors / timeouts from `requests` are caught and re-raised as `AppleResolutionError` (same shape as `DownloadError` in `downloader.py`), so the router can treat all resolution failures uniformly.
- No new settings/env vars — the Lookup API is public and unauthenticated.

### Verified against live API (2026-07-22)

Confirmed by directly calling `itunes.apple.com/lookup` against a real podcast (NYT "The Daily", collection id 1200361736) and cross-checking its live RSS feed:

- `entity=podcast&id={collectionId}` reliably returns `feedUrl`.
- **Episode ids cannot be looked up directly** — `lookup?id={episodeId}&entity=podcastEpisode` returns `resultCount: 0` even for a valid, recently-published episode id. This rules out the design sketched in the PRD's background notes.
- The working call is `lookup?id={collectionId}&entity=podcastEpisode&limit=200`, which returns a *list* of the podcast's most recent episodes (each a `wrapperType: "podcastEpisode"` object with `trackId`, `episodeGuid`, `episodeUrl`). The target episode is found by scanning this list for `trackId == episode_id` from the URL's `?i=` param.
- `episodeGuid` in that response is an **exact string match** for the RSS `<guid>` element of the corresponding feed entry (verified byte-for-byte on a live example) — which is exactly what `feedparser` already surfaces as `entry.get("id")`, i.e. `ParsedEpisode.guid`'s primary source. So matching is a simple `episode.guid == target_guid` equality check, no enclosure-URL comparison needed.
- **Limitation**: despite requesting `limit=200`, the API only returned 41 episodes for a show with 2666 total episodes — Apple's Lookup API appears to cap `entity=podcastEpisode` results to roughly its most recent ~40-50, regardless of the requested limit, and there is no documented paging/offset mechanism. **Episode-level resolution therefore only works for a podcast's recent episodes**; pasting a link to an older episode will not find a match. Per the PRD's decided v1 scope (hard error on no-match, no fallback), this surfaces as the same 400 "episode not found" error as any other match failure — it is a known and accepted limitation, not a bug, and should be called out in the endpoint's error message (e.g. mention that only recent episodes are supported) so it's not confused with a broken feature.

## `parser.py` change

`parse_feed` gains an optional `target_guid: str | None = None` parameter:

```python
def parse_feed(rss_url: str, max_episodes: int = 1, target_guid: str | None = None) -> ParsedPodcast:
    ...
    if target_guid is not None:
        matches = [e for e in episodes if e.guid == target_guid]
        if not matches:
            raise FeedParseError(f"Episode with guid {target_guid} not found in feed: {rss_url}")
        episodes = matches
    else:
        episodes = episodes[:max_episodes]
```

Reuses `FeedParseError` (not a new exception) since this is still "the feed didn't contain what we were told to expect" — same failure class the router already maps to 400. `max_episodes` is ignored when `target_guid` is set (requirement #3: episode-level ingestion returns exactly one episode). `target_guid` is compared against `ParsedEpisode.guid`, which already prioritizes the RSS `<guid>` (`entry.get("id")`) — the same field Apple's `episodeGuid` matches against (see "Verified against live API" above).

## `service.py` change

`ingest_podcast` gains an optional `target_guid: str | None = None`, passed straight through to `parse_feed`. No other logic changes — the rest of the insert/download flow is agnostic to how many episodes were parsed.

## `router.py` change

```python
@router.post("/podcasts", response_model=IngestResponse)
async def create_podcast(body: IngestRequest, session=Depends(get_session)) -> IngestResponse:
    rss_url = body.rss_url
    target_guid = None

    if is_apple_podcasts_url(rss_url):
        try:
            resolved = resolve_apple_podcasts_url(rss_url)
        except AppleResolutionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rss_url = resolved.feed_url
        target_guid = resolved.target_guid

    try:
        result = await ingest_podcast(session, rss_url, max_episodes=body.max_episodes,
                                       target_guid=target_guid)
    except FeedParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ...
```

`IngestRequest` schema is unchanged (`rss_url` accepts either a real RSS URL or an Apple Podcasts URL — same field, detection happens server-side).

## Data Flow

```
podcast-level:  Apple URL -> extract collection_id -> lookup(entity=podcast) -> feed_url
                -> parse_feed(feed_url, max_episodes) -> ingest_podcast (unchanged behavior)

episode-level:  Apple URL -> extract collection_id + episode_id
                -> lookup(entity=podcast) -> feed_url
                -> lookup(entity=podcastEpisode, limit=200) -> scan for trackId == episode_id -> target_guid
                -> parse_feed(feed_url, target_guid=...) -> exactly 1 episode (matched by RSS <guid>)
                -> ingest_podcast (max_episodes ignored)
```

## Error Handling / Compatibility

- All resolution failures (`AppleResolutionError`) and feed-content failures (`FeedParseError`) map to HTTP 400 in the router — same external contract, no new error shape for API consumers.
- Plain RSS URLs bypass `apple_podcasts.py` entirely (`is_apple_podcasts_url` returns False) — zero behavior change for existing callers, satisfying the no-regression acceptance criterion.
- No DB schema changes; `Podcast`/`Episode` models are untouched.

## Out of Scope (confirmed in prd.md)

- `apple.co` short links.
- Falling back to a bare audio URL when feed/episode resolution fails.
- Any frontend changes.

## Risk

- **Accepted limitation**: episode-level resolution only works for a podcast's most recent ~40-50 episodes (Apple Lookup API's undocumented cap on `entity=podcastEpisode` results, confirmed live — see above). Older episode links hit the same 400 "not found" path as any other resolution failure; the error message should mention this so it reads as an explained limitation rather than a bug.
