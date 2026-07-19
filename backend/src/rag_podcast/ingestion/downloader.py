from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CHUNK_SIZE = 1024 * 1024  # 1 MiB

KNOWN_EXTENSIONS = (".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".wav", ".aac")
EXTENSION_BY_MIME = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/aac": ".aac",
}
DEFAULT_EXTENSION = ".mp3"


class DownloadError(Exception):
    """Raised when an audio file can't be downloaded after retries."""



def download_audio(url: str, podcast_id: int, episode_id: int, podcast_title: str, data_dir: Path) -> Path:
    """Stream an episode's audio to DATA_DIR/{podcast_id}/{episode_id}.{ext}.

    Retries transient failures (429/5xx) at the HTTP layer via urllib3 so a
    blip during a long download doesn't fail the whole request — see
    PLAN.md §4.2/§4.3 for why retry lives here and IDs (not names) key the
    file path.
    """
    session = _build_session()

    try:
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    extension = _detect_extension(url, response.headers.get("Content-Type"))
    podcast_dir = data_dir / podcast_title / str(podcast_id)
    podcast_dir.mkdir(parents=True, exist_ok=True)
    dest_path = podcast_dir / f"{episode_id}{extension}"

    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    except (requests.RequestException, OSError) as exc:
        dest_path.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download {url}: {exc}") from exc
    finally:
        response.close()

    return dest_path


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,  # sleeps 1s, 2s, 4s between the 3 attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _detect_extension(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    for ext in KNOWN_EXTENSIONS:
        if path.endswith(ext):
            return ext

    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime in EXTENSION_BY_MIME:
            return EXTENSION_BY_MIME[mime]

    return DEFAULT_EXTENSION
