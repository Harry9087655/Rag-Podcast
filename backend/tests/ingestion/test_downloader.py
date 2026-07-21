from __future__ import annotations

import requests
import pytest

from rag_podcast.ingestion import downloader
from rag_podcast.ingestion.downloader import DownloadError, download_audio


class FakeResponse:
    def __init__(self, chunks, status_code=200, headers=None, http_error=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks
        self._http_error = http_error
        self.closed = False

    def raise_for_status(self):
        if self._http_error:
            raise self._http_error

    def iter_content(self, chunk_size):
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def patch_get(monkeypatch, response):
    def fake_get(self, url, stream=True, timeout=30):
        return response

    monkeypatch.setattr(requests.Session, "get", fake_get)


def test_download_audio_success_extension_from_url(tmp_path, monkeypatch):
    response = FakeResponse([b"hello ", b"world"], headers={"Content-Type": "audio/mpeg"})
    patch_get(monkeypatch, response)

    dest = download_audio(
        "http://example.com/ep1.mp3",
        podcast_id=1,
        episode_id=2,
        podcast_title="vergecast",
        data_dir=tmp_path,
    )

    assert dest == tmp_path / "vergecast" / "1" / "2.mp3"
    assert dest.read_bytes() == b"hello world"
    assert response.closed


def test_download_audio_extension_from_content_type_when_url_has_none(tmp_path, monkeypatch):
    response = FakeResponse([b"data"], headers={"Content-Type": "audio/mp4"})
    patch_get(monkeypatch, response)

    dest = download_audio(
        "http://example.com/stream?id=abc",
        podcast_id=1,
        episode_id=2,
        podcast_title="vergecast",
        data_dir=tmp_path,
    )

    assert dest.suffix == ".m4a"


def test_download_audio_falls_back_to_default_extension(tmp_path, monkeypatch):
    response = FakeResponse([b"data"], headers={})
    patch_get(monkeypatch, response)

    dest = download_audio(
        "http://example.com/stream?id=abc",
        podcast_id=1,
        episode_id=2,
        podcast_title="vergecast",
        data_dir=tmp_path,
    )

    assert dest.suffix == downloader.DEFAULT_EXTENSION


def test_download_audio_creates_podcast_subdirectory(tmp_path, monkeypatch):
    response = FakeResponse([b"data"])
    patch_get(monkeypatch, response)

    download_audio(
        "http://example.com/ep1.mp3",
        podcast_id=42,
        episode_id=1,
        podcast_title="vergecast",
        data_dir=tmp_path,
    )

    assert (tmp_path / "vergecast" / "42").is_dir()


def test_download_audio_raises_on_http_error(tmp_path, monkeypatch):
    response = FakeResponse([b"data"], http_error=requests.HTTPError("404"))
    patch_get(monkeypatch, response)

    with pytest.raises(DownloadError):
        download_audio(
            "http://example.com/ep1.mp3",
            podcast_id=1,
            episode_id=2,
            podcast_title="vergecast",
            data_dir=tmp_path,
        )

    assert not (tmp_path / "vergecast" / "1").exists()


def test_download_audio_raises_on_connection_error(tmp_path, monkeypatch):
    def fake_get(self, url, stream=True, timeout=30):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests.Session, "get", fake_get)

    with pytest.raises(DownloadError):
        download_audio(
            "http://example.com/ep1.mp3",
            podcast_id=1,
            episode_id=2,
            podcast_title="vergecast",
            data_dir=tmp_path,
        )


def test_download_audio_cleans_up_partial_file_on_write_error(tmp_path, monkeypatch):
    response = FakeResponse([b"partial-chunk", OSError("disk full")])
    patch_get(monkeypatch, response)

    with pytest.raises(DownloadError):
        download_audio(
            "http://example.com/ep1.mp3",
            podcast_id=1,
            episode_id=2,
            podcast_title="vergecast",
            data_dir=tmp_path,
        )

    assert not (tmp_path / "vergecast" / "1" / "2.mp3").exists()
