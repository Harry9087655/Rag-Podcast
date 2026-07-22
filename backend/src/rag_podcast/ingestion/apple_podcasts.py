from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOOKUP_URL = "https://itunes.apple.com/lookup"
EPISODE_LOOKUP_LIMIT = 200  # Apple's API appears to cap results well below this (see research notes)
REQUEST_TIMEOUT = 10

APPLE_HOSTS = {"podcasts.apple.com", "itunes.apple.com"}
_COLLECTION_ID_RE = re.compile(r"/id(\d+)")


class AppleResolutionError(Exception):
    """Raised when an Apple Podcasts URL can't be resolved to feed/episode data."""


@dataclass
class ResolvedApplePodcast:
    feed_url: str
    target_guid: str | None


def is_apple_podcasts_url(url: str) -> bool:
    """True if `url` is an Apple/iTunes Podcasts link with a collection id."""
    parsed = urlparse(url)
    if parsed.hostname not in APPLE_HOSTS:
        return False
    return _COLLECTION_ID_RE.search(parsed.path) is not None


def resolve_apple_podcasts_url(url: str) -> ResolvedApplePodcast:
    """Resolve an Apple Podcasts URL to its RSS feed URL and, if the URL
    points at a specific episode, that episode's RSS guid.

    Raises AppleResolutionError if the collection has no feedUrl, if a
    requested episode can't be matched (including because it's outside
    Apple's lookup window of recent episodes), or on any network/HTTP
    failure calling the iTunes Lookup API.
    """
    collection_id, episode_id = _extract_ids(url)

    session = _build_session()
    feed_url = _lookup_feed_url(session, collection_id)

    target_guid = None
    if episode_id is not None:
        target_guid = _lookup_episode_guid(session, collection_id, episode_id)

    return ResolvedApplePodcast(feed_url=feed_url, target_guid=target_guid)


def _extract_ids(url: str) -> tuple[str, str | None]:
    parsed = urlparse(url)
    match = _COLLECTION_ID_RE.search(parsed.path)
    if match is None:
        raise AppleResolutionError(f"Could not extract a collection id from Apple Podcasts URL: {url}")
    collection_id = match.group(1)

    episode_id = None
    query = parse_qs(parsed.query)
    episode_values = query.get("i")
    if episode_values and episode_values[0].isdigit():
        episode_id = episode_values[0]

    return collection_id, episode_id


def _lookup_feed_url(session: requests.Session, collection_id: str) -> str:
    data = _call_lookup(session, {"id": collection_id, "entity": "podcast"})
    results = data.get("results") or []
    if not results:
        raise AppleResolutionError(
            f"iTunes Lookup returned no podcast results for collection id {collection_id}"
        )

    feed_url = results[0].get("feedUrl")
    if not feed_url:
        raise AppleResolutionError(
            f"iTunes Lookup result for collection id {collection_id} has no feedUrl"
        )

    return feed_url


def _lookup_episode_guid(session: requests.Session, collection_id: str, episode_id: str) -> str:
    data = _call_lookup(
        session,
        {"id": collection_id, "entity": "podcastEpisode", "limit": EPISODE_LOOKUP_LIMIT},
    )
    results = data.get("results") or []

    for result in results:
        if str(result.get("trackId")) == episode_id:
            episode_guid = result.get("episodeGuid")
            if not episode_guid:
                raise AppleResolutionError(
                    f"iTunes Lookup matched episode id {episode_id} but it has no episodeGuid"
                )
            return episode_guid

    raise AppleResolutionError(
        f"Could not find episode id {episode_id} in iTunes Lookup's recent-episode results for "
        f"collection id {collection_id} (Apple's Lookup API only returns a podcast's most recent "
        "episodes, so links to older episodes cannot be resolved)"
    )


def _call_lookup(session: requests.Session, params: dict) -> dict:
    try:
        response = session.get(LOOKUP_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise AppleResolutionError(f"iTunes Lookup request failed: {exc}") from exc


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
