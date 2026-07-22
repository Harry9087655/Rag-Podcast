from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import feedparser

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".aac")

# Some RSS hosts (e.g. xyzfm.space / 小宇宙) blacklist Python's default
# urllib User-Agent and return 403. A browser-like UA keeps us off the list.
FEEDPARSER_USER_AGENT = "Mozilla/5.0 (compatible; rag-podcast/1.0)"


class FeedParseError(Exception):
    """Raised when a feed URL can't be fetched or contains no usable data."""


@dataclass
class ParsedEpisode:
    guid: str
    title: str
    published_date: datetime | None
    enclosure_url: str
    duration_seconds: int | None


@dataclass
class ParsedPodcast:
    name: str
    author: str | None
    cover_url: str | None
    episodes: list[ParsedEpisode]


def parse_feed(rss_url: str, max_episodes: int = 1, target_guid: str | None = None) -> ParsedPodcast:
    """Fetch and parse an RSS feed into podcast + episode metadata.

    Extracts every episode with a recognizable audio enclosure, then
    truncates to `max_episodes` — see PLAN.md §3.5 for why truncation
    happens after full extraction rather than during the feed loop.

    If `target_guid` is given, `max_episodes` is ignored and the result
    contains exactly the one episode whose guid matches (raising
    FeedParseError if none does) — used for Apple Podcasts episode-level
    resolution, where the feed's own guid is authoritative.
    """
    feed = feedparser.parse(rss_url, agent=FEEDPARSER_USER_AGENT)

    if not feed.entries and not feed.feed:
        raise FeedParseError(f"Could not fetch or parse feed: {rss_url}")

    feed_info = feed.feed
    name = feed_info.get("title", "")
    author = feed_info.get("author") or feed_info.get("itunes_author")
    cover_url = _extract_cover_url(feed_info)

    episodes: list[ParsedEpisode] = []
    for entry in feed.entries:
        enclosure_url = _extract_enclosure(entry)
        if enclosure_url is None:
            # Not every RSS item is an episode — some feeds mix in blog
            # posts, ads, or bonus text-only content with no audio.
            continue

        episodes.append(
            ParsedEpisode(
                guid=entry.get("id") or entry.get("link") or enclosure_url,
                title=entry.get("title", ""),
                published_date=_parse_published_date(entry),
                enclosure_url=enclosure_url,
                duration_seconds=_parse_duration(entry),
            )
        )

    if target_guid is not None:
        matches = [e for e in episodes if e.guid == target_guid]
        if not matches:
            raise FeedParseError(f"Episode with guid {target_guid} not found in feed: {rss_url}")
        episodes = matches
    else:
        episodes = episodes[:max_episodes]

    return ParsedPodcast(
        name=name,
        author=author,
        cover_url=cover_url,
        episodes=episodes,
    )


def _extract_cover_url(feed_info) -> str | None:
    image = feed_info.get("image")
    if image:
        return image.get("href") or image.get("url")
    return None


def _extract_enclosure(entry) -> str | None:
    """Check the three RSS/Atom locations audio can live in, in order.

    See PLAN.md §3.3 — feeds are inconsistent about where the enclosure
    lives, so each location is tried in turn until one yields an audio URL.
    """
    for enclosure in entry.get("enclosures", []):
        url = enclosure.get("href") or enclosure.get("url")
        if url and _looks_like_audio(url, enclosure.get("type")):
            return url

    for media in entry.get("media_content", []):
        url = media.get("url")
        if url and _looks_like_audio(url, media.get("type")):
            return url

    for link in entry.get("links", []):
        if link.get("rel") == "enclosure":
            url = link.get("href")
            if url and _looks_like_audio(url, link.get("type")):
                return url

    return None


def _looks_like_audio(url: str, mime_type: str | None) -> bool:
    if mime_type and mime_type.startswith("audio/"):
        return True
    # Some feeds omit the type or use a generic application/octet-stream —
    # fall back to sniffing the file extension in the URL path.
    return url.lower().split("?", 1)[0].endswith(AUDIO_EXTENSIONS)


def _parse_duration(entry) -> int | None:
    raw = entry.get("itunes_duration")
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)

    parts = raw.split(":")
    if not parts or not all(p.isdigit() for p in parts):
        return None

    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def _parse_published_date(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    if parsed is None:
        return None
    return datetime(*parsed[:6])
