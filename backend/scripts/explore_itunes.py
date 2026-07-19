"""Explore iTunes Search API: find podcasts by name and get their RSS feed URL.

The iTunes Search API is free, public, and requires no authentication.
Docs: https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/

Usage:
    uv run python scripts/explore_itunes.py "podcast name"

Examples:
    uv run python scripts/explore_itunes.py "the vergecast"
    uv run python scripts/explore_itunes.py "hardcore history"
    uv run python scripts/explore_itunes.py "lex fridman"
"""

import json
import sys
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def search_podcasts(term: str, limit: int = 10) -> list[dict]:
    """Search iTunes for podcasts matching the given term.

    Returns a list of result dicts with keys like:
        collectionName, artistName, feedUrl, trackCount, genres,
        artworkUrl100/600, primaryGenreName
    """
    params = urlencode({
        "term": term,
        "entity": "podcast",
        "limit": limit,
    })
    url = f"{ITUNES_SEARCH_URL}?{params}"

    req = Request(url, headers={"User-Agent": "PodcastRAG/1.0"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    return data.get("results", [])


def display_results(results: list[dict]) -> None:
    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s):\n")

    for i, r in enumerate(results):
        print(f"{'─' * 65}")
        print(f"  #{i + 1}: {r.get('collectionName', 'Unknown')}")
        print(f"{'─' * 65}")
        print(f"  Artist:       {r.get('artistName')}")
        print(f"  Genre:        {r.get('primaryGenreName')}")
        print(f"  Episodes:     {r.get('trackCount')}")
        print(f"  Feed URL:     {r.get('feedUrl')}")
        print(f"  Artwork 600:  {r.get('artworkUrl600')}")
        print(f"  iTunes ID:    {r.get('collectionId')}")
        print()

    # Quick summary for picking one
    print("=" * 65)
    print("FEED URLS (copy-paste one into explore_feedparser.py):")
    print("=" * 65)
    for i, r in enumerate(results):
        feed_url = r.get("feedUrl", "N/A")
        name = r.get("collectionName", "Unknown")
        print(f"  [{i + 1}] {name}")
        print(f"      {feed_url}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: uv run python scripts/explore_itunes.py "podcast name"')
        print('Example: uv run python scripts/explore_itunes.py "the vergecast"')
        sys.exit(1)

    term = " ".join(sys.argv[1:])
    print(f'Searching iTunes for: "{term}"...\n')
    results = search_podcasts(term)
    display_results(results)
