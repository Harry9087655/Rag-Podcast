"""Explore requests.Session: understand retry logic, headers, and connection reuse.

Usage:
    uv run python scripts/explore_session.py <URL>

Examples:
    uv run python scripts/explore_session.py https://lexfridman.com/feed/podcast/
    uv run python scripts/explore_session.py https://httpstat.us/500  (always fails → triggers retries)
    uv run python scripts/explore_session.py https://httpbin.org/status/500  (same idea)
"""

import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session(retries: int = 3, backoff: float = 1.0) -> requests.Session:
    """Create a requests.Session with retry logic for transient failures.

    A Session is like a reusable HTTP client object. Instead of calling
    requests.get() once and throwing it away, a Session holds settings
    (headers, retry rules, connection pools) that persist across many
    requests.

    The Retry adapter lives inside the Session and intercepts every
    request that goes through it. When a request fails with a status
    code listed in status_forcelist, the adapter waits and retries
    automatically — no try/except needed at the call site.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=retries,                              # max total retries
        backoff_factor=backoff,                     # wait 1s, then 2s, then 4s...
        status_forcelist=[429, 500, 502, 503, 504], # only retry on these
        allowed_methods=["GET", "HEAD"],            # only retry safe methods
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # User-Agent identifies your app to the server. Some servers block
    # requests that have no User-Agent (thinking they're bots/scrapers).
    # The default Python requests User-Agent is "python-requests/2.x" —
    # setting a custom one is good practice.
    session.headers.update({"User-Agent": "PodcastRAG-M1-Explorer/1.0"})

    return session


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/explore_session.py <URL>")
        print("Example: uv run python scripts/explore_session.py https://lexfridman.com/feed/podcast/")
        sys.exit(1)

    url = sys.argv[1]

    # ================================================================
    # 1. Create a session with retry logic
    # ================================================================
    print(f"Creating session (3 retries, 1s backoff)...")
    session = create_session()

    print(f"  Session object:       {session}")
    print(f"  Mounted adapters:     {list(session.adapters.keys())}")
    print(f"  Default headers:      {dict(session.headers)}")
    print(f"  Retry config:         3 attempts on [429, 500, 502, 503, 504]")

    # ================================================================
    # 2. Fetch using the session
    # ================================================================
    print(f"\nFetching: {url}")
    start = time.time()

    try:
        resp = session.get(url, timeout=30)

        elapsed = time.time() - start
        print(f"  Status:     {resp.status_code} {resp.reason}")
        print(f"  Elapsed:    {elapsed:.2f}s")
        print(f"  Content-Type: {resp.headers.get('content-type', 'unknown')}")
        print(f"  Size:       {len(resp.content):,} bytes")

        # Show redirect chain if any (http → https, etc.)
        if resp.history:
            print(f"  Redirects:  {len(resp.history)}")
            for h in resp.history:
                print(f"    {h.status_code} → {h.url}")

        print(f"\n  First 200 chars of body:")
        print(f"  {resp.text[:200]}...")

    except requests.exceptions.RetryError as e:
        elapsed = time.time() - start
        print(f"  ❌ All retries exhausted after {elapsed:.2f}s")
        print(f"  {e}")

    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start
        print(f"  ❌ Connection failed after {elapsed:.2f}s")
        print(f"  {e}")

    except requests.exceptions.Timeout as e:
        elapsed = time.time() - start
        print(f"  ❌ Request timed out after {elapsed:.2f}s")
        print(f"  {e}")

    finally:
        # ================================================================
        # 3. Clean up (good practice, not strictly required for scripts)
        # ================================================================
        session.close()
        print(f"\nSession closed.")


if __name__ == "__main__":
    main()
