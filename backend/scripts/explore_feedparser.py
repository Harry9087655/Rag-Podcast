"""Explore feedparser: parse a podcast RSS feed and inspect its structure.

Usage:
    uv run python scripts/explore_feedparser.py <RSS_URL>

Example:
    uv run python scripts/explore_feedparser.py https://feeds.megaphone.fm/vergecast
"""

import sys
from pprint import pprint

import feedparser


def explore_feed(url: str) -> None:
    print(f"Fetching: {url}\n")
    feed = feedparser.parse(url)

    # --- Did it work? ---
    if feed.bozo:
        print(f"⚠  Parse warning: {feed.bozo_exception}")
    if not feed.feed:
        print("❌ No feed data found — not a valid RSS/Atom feed")
        return

    f = feed.feed

    # ============================================================
    # 1. PODCAST-LEVEL METADATA
    # ============================================================
    print("=" * 65)
    print("PODCAST")
    print("=" * 65)
    fields = [
        ("title", "Title"),
        ("subtitle", "Subtitle"),
        ("link", "Website"),
        ("author", "Author"),
        ("language", "Language"),
        ("itunes_author", "iTunes Author"),
        ("itunes_explicit", "Explicit"),
        ("itunes_type", "Type (episodic/serial)"),
        ("itunes_image", "iTunes Image"),
        ("image", "Feed Image"),
    ]
    for attr, label in fields:
        val = f.get(attr)
        if val:
            if isinstance(val, dict):
                print(f"  {label}: {val.get('href', val)}")
            else:
                print(f"  {label}: {val}")

    # iTunes categories
    cats = f.get("itunes_category") or f.get("tags", [])
    if cats:
        print(f"  Categories: {cats}")

    print()

    # ============================================================
    # 2. EPISODES — first 5 only
    # ============================================================
    print("=" * 65)
    print(f"EPISODES ({len(feed.entries)} total, showing first 5)")
    print("=" * 65)

    for i, entry in enumerate(feed.entries[:5]):
        print(f"\n--- Episode {i + 1} ---")
        print(f"  title:           {entry.get('title')}")
        print(f"  guid:            {entry.get('id') or entry.get('guid')}")
        print(f"  link:            {entry.get('link')}")
        print(f"  published:       {entry.get('published')}")
        print(f"  itunes_duration: {entry.get('itunes_duration')}")
        print(f"  itunes_episode:  {entry.get('itunes_episode')}")
        print(f"  itunes_season:   {entry.get('itunes_season')}")

        # --- Enclosure (audio file) ---
        encs = entry.get("enclosures", [])
        print(f"  enclosures ({len(encs)}):")
        for enc in encs:
            print(f"    href:  {enc.get('href')}")
            print(f"    type:  {enc.get('type')}")
            print(f"    length:{enc.get('length')}")

        # --- Media content ---
        media = entry.get("media_content", [])
        if media:
            print(f"  media_content ({len(media)}):")
            for m in media:
                print(f"    url:   {m.get('url')}")
                print(f"    type:  {m.get('type')}")

        # --- Links ---
        links = entry.get("links", [])
        enclosure_links = [l for l in links if l.get("rel") == "enclosure"]
        if enclosure_links:
            print(f"  links[rel=enclosure] ({len(enclosure_links)}):")
            for l in enclosure_links:
                print(f"    href:  {l.get('href')}")
                print(f"    type:  {l.get('type')}")

    # ============================================================
    # 3. RAW KEYS (debug: see ALL available fields on one episode)
    # ============================================================
    if feed.entries:
        print(f"\n{'=' * 65}")
        print("RAW KEYS on first episode (everything feedparser exposes)")
        print("=" * 65)
        pprint(list(feed.entries[0].keys()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/explore_feedparser.py <RSS_URL>")
        print("Example: uv run python scripts/explore_feedparser.py https://feeds.megaphone.fm/vergecast")
        sys.exit(1)
    explore_feed(sys.argv[1])
