from __future__ import annotations

import requests
import pytest

from rag_podcast.ingestion.apple_podcasts import (
    AppleResolutionError,
    is_apple_podcasts_url,
    resolve_apple_podcasts_url,
)


class FakeJsonResponse:
    def __init__(self, payload, status_code=200, http_error=None):
        self.status_code = status_code
        self._payload = payload
        self._http_error = http_error

    def raise_for_status(self):
        if self._http_error:
            raise self._http_error

    def json(self):
        return self._payload


def patch_get(monkeypatch, responses):
    """responses: a list consumed in call order, or a single response reused for every call."""
    calls = []

    def fake_get(self, url, params=None, timeout=10):
        calls.append(params)
        if isinstance(responses, list):
            return responses[len(calls) - 1]
        return responses

    monkeypatch.setattr(requests.Session, "get", fake_get)
    return calls


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://podcasts.apple.com/us/podcast/some-show/id123456789", True),
        ("https://podcasts.apple.com/us/podcast/some-show/id123456789?i=1000456789012", True),
        ("https://itunes.apple.com/us/podcast/some-show/id123456789", True),
        ("http://feed.example.com/rss", False),
        ("https://podcasts.apple.com/us/podcast/some-show", False),
        ("https://example.com/id123456789", False),
    ],
)
def test_is_apple_podcasts_url(url, expected):
    assert is_apple_podcasts_url(url) is expected


def test_resolve_podcast_level_url_success(monkeypatch):
    calls = patch_get(
        monkeypatch,
        FakeJsonResponse({"resultCount": 1, "results": [{"feedUrl": "http://feed.example.com/rss"}]}),
    )

    resolved = resolve_apple_podcasts_url("https://podcasts.apple.com/us/podcast/some-show/id123456789")

    assert resolved.feed_url == "http://feed.example.com/rss"
    assert resolved.target_guid is None
    assert len(calls) == 1
    assert calls[0]["id"] == "123456789"
    assert calls[0]["entity"] == "podcast"


def test_resolve_episode_level_url_success(monkeypatch):
    podcast_response = FakeJsonResponse(
        {"resultCount": 1, "results": [{"feedUrl": "http://feed.example.com/rss"}]}
    )
    episode_response = FakeJsonResponse(
        {
            "resultCount": 2,
            "results": [
                {"wrapperType": "podcastEpisode", "trackId": 1000111111111, "episodeGuid": "other-guid"},
                {"wrapperType": "podcastEpisode", "trackId": 1000456789012, "episodeGuid": "target-guid-abc"},
            ],
        }
    )
    calls = patch_get(monkeypatch, [podcast_response, episode_response])

    url = "https://podcasts.apple.com/us/podcast/some-show/id123456789?i=1000456789012"
    resolved = resolve_apple_podcasts_url(url)

    assert resolved.feed_url == "http://feed.example.com/rss"
    assert resolved.target_guid == "target-guid-abc"
    assert len(calls) == 2
    assert calls[0]["entity"] == "podcast"
    assert calls[1]["entity"] == "podcastEpisode"
    assert calls[1]["id"] == "123456789"
    assert calls[1]["limit"] == 200


def test_resolve_raises_when_no_feed_url(monkeypatch):
    patch_get(monkeypatch, FakeJsonResponse({"resultCount": 0, "results": []}))

    with pytest.raises(AppleResolutionError):
        resolve_apple_podcasts_url("https://podcasts.apple.com/us/podcast/some-show/id123456789")


def test_resolve_raises_when_no_track_id_match(monkeypatch):
    podcast_response = FakeJsonResponse(
        {"resultCount": 1, "results": [{"feedUrl": "http://feed.example.com/rss"}]}
    )
    episode_response = FakeJsonResponse(
        {
            "resultCount": 1,
            "results": [
                {"wrapperType": "podcastEpisode", "trackId": 999999999999, "episodeGuid": "other-guid"},
            ],
        }
    )
    patch_get(monkeypatch, [podcast_response, episode_response])

    url = "https://podcasts.apple.com/us/podcast/some-show/id123456789?i=1000456789012"
    with pytest.raises(AppleResolutionError):
        resolve_apple_podcasts_url(url)


def test_resolve_raises_on_http_error(monkeypatch):
    patch_get(monkeypatch, FakeJsonResponse({}, http_error=requests.HTTPError("500")))

    with pytest.raises(AppleResolutionError):
        resolve_apple_podcasts_url("https://podcasts.apple.com/us/podcast/some-show/id123456789")


def test_resolve_raises_on_connection_error(monkeypatch):
    def fake_get(self, url, params=None, timeout=10):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests.Session, "get", fake_get)

    with pytest.raises(AppleResolutionError):
        resolve_apple_podcasts_url("https://podcasts.apple.com/us/podcast/some-show/id123456789")
