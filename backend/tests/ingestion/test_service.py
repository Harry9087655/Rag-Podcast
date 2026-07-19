from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from rag_podcast.ingestion import service
from rag_podcast.ingestion.downloader import DownloadError
from rag_podcast.ingestion.parser import FeedParseError, ParsedEpisode, ParsedPodcast
from rag_podcast.ingestion.service import ingest_podcast, _insert_episode
from rag_podcast.models.episode import Episode, TranscriptStatus
from rag_podcast.models.podcast import Podcast


def make_parsed_episode(guid="ep-1", title="Episode 1", url="http://example.com/ep1.mp3"):
    return ParsedEpisode(
        guid=guid,
        title=title,
        published_date=None,
        enclosure_url=url,
        duration_seconds=None,
    )


def make_parsed_podcast(episodes, name="Test Podcast"):
    return ParsedPodcast(name=name, author="Jane Doe", cover_url=None, episodes=episodes)


async def test_ingest_podcast_creates_podcast_and_downloads_episodes(db_session, monkeypatch, tmp_path):
    parsed = make_parsed_podcast([make_parsed_episode("ep-1"), make_parsed_episode("ep-2")])
    monkeypatch.setattr(service, "parse_feed", lambda url, max_episodes: parsed)

    downloaded = tmp_path / "audio.mp3"
    downloaded.write_bytes(b"data")
    monkeypatch.setattr(service, "download_audio", lambda *a, **kw: downloaded)

    result = await ingest_podcast(db_session, "http://feed.example.com/rss", max_episodes=10)

    assert result.already_imported is False
    assert result.new_episodes == 2
    assert result.skipped == 0
    assert {ep.guid for ep in result.episodes} == {"ep-1", "ep-2"}
    for ep in result.episodes:
        assert ep.transcript_status == TranscriptStatus.DOWNLOADED
        assert ep.audio_local_path == str(downloaded)

    stored_podcast = await db_session.scalar(select(Podcast).where(Podcast.rss_url == "http://feed.example.com/rss"))
    assert stored_podcast is not None
    assert stored_podcast.name == "Test Podcast"


async def test_ingest_podcast_already_imported_skips_reinsertion(db_session, monkeypatch):
    first_parsed = make_parsed_podcast([make_parsed_episode("ep-1")])
    monkeypatch.setattr(service, "parse_feed", lambda url, max_episodes: first_parsed)
    monkeypatch.setattr(service, "download_audio", lambda *a, **kw: Path("/tmp/fake.mp3"))

    first = await ingest_podcast(db_session, "http://feed.example.com/rss", max_episodes=10)
    assert first.already_imported is False
    assert first.new_episodes == 1

    second_parsed = make_parsed_podcast([make_parsed_episode("ep-1"), make_parsed_episode("ep-2")])
    monkeypatch.setattr(service, "parse_feed", lambda url, max_episodes: second_parsed)

    second = await ingest_podcast(db_session, "http://feed.example.com/rss", max_episodes=10)

    assert second.already_imported is True
    assert second.new_episodes == 0
    assert second.skipped == 0
    # returns the podcast's existing episodes untouched, not a re-parse of the feed
    assert len(second.episodes) == 1
    assert second.episodes[0].guid == "ep-1"


async def test_ingest_podcast_duplicate_guid_within_same_feed_is_skipped(db_session, monkeypatch):
    parsed = make_parsed_podcast([make_parsed_episode("dup-guid"), make_parsed_episode("dup-guid")])
    monkeypatch.setattr(service, "parse_feed", lambda url, max_episodes: parsed)
    monkeypatch.setattr(service, "download_audio", lambda *a, **kw: Path("/tmp/fake.mp3"))

    result = await ingest_podcast(db_session, "http://feed.example.com/rss", max_episodes=10)

    assert result.new_episodes == 1
    assert result.skipped == 1

    rows = (
        await db_session.scalars(
            select(Episode).where(Episode.podcast_id == result.podcast.id)
        )
    ).all()
    assert len(rows) == 1


async def test_ingest_podcast_marks_episode_failed_on_download_error(db_session, monkeypatch):
    parsed = make_parsed_podcast([make_parsed_episode("ep-1")])
    monkeypatch.setattr(service, "parse_feed", lambda url, max_episodes: parsed)

    def failing_download(*a, **kw):
        raise DownloadError("network blew up")

    monkeypatch.setattr(service, "download_audio", failing_download)

    result = await ingest_podcast(db_session, "http://feed.example.com/rss", max_episodes=10)

    assert result.new_episodes == 1
    ep = result.episodes[0]
    assert ep.transcript_status == TranscriptStatus.FAILED
    assert ep.audio_local_path is None

    stored = await db_session.get(Episode, ep.id)
    assert stored.transcript_status == TranscriptStatus.FAILED


async def test_ingest_podcast_propagates_feed_parse_error(db_session, monkeypatch):
    def raise_parse_error(url, max_episodes):
        raise FeedParseError("unreachable feed")

    monkeypatch.setattr(service, "parse_feed", raise_parse_error)

    with pytest.raises(FeedParseError):
        await ingest_podcast(db_session, "http://bad.example.com/rss", max_episodes=10)

    stored_podcast = await db_session.scalar(
        select(Podcast).where(Podcast.rss_url == "http://bad.example.com/rss")
    )
    assert stored_podcast is None


async def test_insert_episode_isolates_failure_via_savepoint(db_session):
    podcast = Podcast(rss_url="http://feed.example.com/rss", name="P", author=None, cover_url=None)
    db_session.add(podcast)
    await db_session.flush()

    first = await _insert_episode(db_session, podcast.id, make_parsed_episode("dup-guid"))
    assert first is not None

    # same guid + podcast_id violates the unique constraint at the DB level
    second = await _insert_episode(db_session, podcast.id, make_parsed_episode("dup-guid"))
    assert second is None

    # the session must still be usable after the rolled-back savepoint
    third = await _insert_episode(db_session, podcast.id, make_parsed_episode("ep-other"))
    assert third is not None
