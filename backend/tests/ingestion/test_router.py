from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_podcast.db import get_session
from rag_podcast.ingestion import router as router_module
from rag_podcast.ingestion.parser import FeedParseError
from rag_podcast.ingestion.service import IngestResult
from rag_podcast.main import app
from rag_podcast.models.episode import Episode, TranscriptStatus
from rag_podcast.models.podcast import Podcast


async def _fake_get_session():
    yield None


@pytest.fixture
def client():
    app.dependency_overrides[get_session] = _fake_get_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def make_result(new_episodes=1, skipped=0, already_imported=False):
    podcast = Podcast(rss_url="http://feed.example.com/rss", name="Test Podcast", author="Jane", cover_url=None)
    podcast.id = 1
    episode = Episode(
        podcast_id=1,
        guid="ep-1",
        title="Episode 1",
        published_date=None,
        enclosure_url="http://example.com/ep1.mp3",
        duration_seconds=None,
        transcript_status=TranscriptStatus.DOWNLOADED,
    )
    episode.id = 10
    return IngestResult(
        podcast=podcast,
        episodes=[episode],
        new_episodes=new_episodes,
        skipped=skipped,
        already_imported=already_imported,
    )


def test_create_podcast_success(client, monkeypatch):
    result = make_result()

    async def fake_ingest(session, rss_url, max_episodes):
        return result

    monkeypatch.setattr(router_module, "ingest_podcast", fake_ingest)

    response = client.post("/podcasts", json={"rss_url": "http://feed.example.com/rss"})

    assert response.status_code == 200
    body = response.json()
    assert body["podcast"]["id"] == 1
    assert body["podcast"]["name"] == "Test Podcast"
    assert body["episodes"] == [{"id": 10, "title": "Episode 1", "transcript_status": "downloaded"}]
    assert body["new_episodes"] == 1
    assert body["skipped"] == 0
    assert body["already_imported"] is False


def test_create_podcast_feed_parse_error_returns_400(client, monkeypatch):
    async def raise_parse_error(session, rss_url, max_episodes):
        raise FeedParseError(f"Could not fetch or parse feed: {rss_url}")

    monkeypatch.setattr(router_module, "ingest_podcast", raise_parse_error)

    response = client.post("/podcasts", json={"rss_url": "http://bad.example.com/rss"})

    assert response.status_code == 400
    assert "bad.example.com" in response.json()["detail"]


def test_create_podcast_defaults_max_episodes_to_one(client, monkeypatch):
    captured = {}

    async def fake_ingest(session, rss_url, max_episodes):
        captured["max_episodes"] = max_episodes
        return make_result()

    monkeypatch.setattr(router_module, "ingest_podcast", fake_ingest)

    client.post("/podcasts", json={"rss_url": "http://feed.example.com/rss"})

    assert captured["max_episodes"] == 1


def test_create_podcast_passes_through_max_episodes(client, monkeypatch):
    captured = {}

    async def fake_ingest(session, rss_url, max_episodes):
        captured["max_episodes"] = max_episodes
        return make_result()

    monkeypatch.setattr(router_module, "ingest_podcast", fake_ingest)

    client.post("/podcasts", json={"rss_url": "http://feed.example.com/rss", "max_episodes": 20})

    assert captured["max_episodes"] == 20
