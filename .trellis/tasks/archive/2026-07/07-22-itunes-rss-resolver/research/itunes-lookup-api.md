# Research: iTunes Lookup API behavior (verified live, 2026-07-22)

Verified by calling `https://itunes.apple.com/lookup` directly against a real podcast (NYT "The Daily", collection id `1200361736`) and cross-checking its live RSS feed (`https://feeds.simplecast.com/Sl5CSM3S`).

## Podcast-level lookup

`GET https://itunes.apple.com/lookup?id={collectionId}&entity=podcast`

Returns `results[0].feedUrl` — a direct, real RSS feed URL. No auth/API key required.

## Episode-level lookup

`GET https://itunes.apple.com/lookup?id={episodeId}&entity=podcastEpisode` returns `resultCount: 0` — **episode ids are not independently look-up-able**, contrary to a naive reading of the API.

Working approach: `GET https://itunes.apple.com/lookup?id={collectionId}&entity=podcastEpisode&limit=200` returns a list of the podcast's most recent episodes as `wrapperType: "podcastEpisode"` objects, each with `trackId`, `episodeGuid`, `episodeUrl`. Scan this list for `trackId == episodeId` (the id extracted from the Apple Podcasts URL's `?i=` query param).

`episodeGuid` from a matched result is an **exact string match** for the target RSS `<guid>` element — verified byte-for-byte:

- iTunes `episodeGuid`: `bda544d7-1aea-49d5-b6bf-c51ebca0a188`
- RSS `<guid isPermaLink="false">bda544d7-1aea-49d5-b6bf-c51ebca0a188</guid>`

`feedparser` already surfaces `<guid>` as `entry.get("id")`, which is the primary source for `ParsedEpisode.guid` in `backend/src/rag_podcast/ingestion/parser.py`. So matching a target episode is a plain `episode.guid == target_guid` equality check — no need to compare enclosure/audio URLs.

## Known limitation

Despite requesting `limit=200`, the API returned only 41 episodes for a show with 2666 total episodes. Apple's Lookup API appears to cap `entity=podcastEpisode` results to roughly the most recent ~40-50 regardless of the requested `limit`, with no documented offset/paging mechanism to reach older episodes.

**Consequence**: episode-level resolution (matching an Apple Podcasts episode link to its RSS entry) only works for a podcast's recent episodes. A link to an older episode will not be found via this lookup and must be treated as a resolution failure (see `prd.md` requirement #4 — hard error, no fallback). This is a platform limitation, not an implementation bug — worth surfacing in the 400 error message so it reads as an explained limitation.

## Implication for implementation

- `apple_podcasts.py`'s `resolve_apple_podcasts_url` must call the collection-level `entity=podcastEpisode&limit=200` endpoint (not attempt a direct episode-id lookup) to get `target_guid`.
- `parser.parse_feed`'s new `target_guid` param should filter on `ParsedEpisode.guid`, matching this exact field semantics.
