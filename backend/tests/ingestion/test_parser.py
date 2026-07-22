from __future__ import annotations

from datetime import datetime

import pytest

from rag_podcast.ingestion.parser import FeedParseError, parse_feed

FEED_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:media="http://search.yahoo.com/mrss/"
     xmlns:atom="http://www.w3.org/2005/Atom">
<channel>
<title>Test Podcast</title>
<itunes:author>Jane Doe</itunes:author>
<image><url>http://example.com/cover.jpg</url></image>
"""
FEED_FOOTER = "</channel>\n</rss>\n"


def make_feed(items_xml: str) -> str:
    return FEED_HEADER + items_xml + FEED_FOOTER


def test_parse_feed_extracts_podcast_metadata():
    xml = make_feed(
        """
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
          <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg" length="123"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert result.name == "Test Podcast"
    assert result.author == "Jane Doe"
    assert result.cover_url == "http://example.com/cover.jpg"
    assert len(result.episodes) == 1
    ep = result.episodes[0]
    assert ep.guid == "ep-1-guid"
    assert ep.title == "Episode 1"
    assert ep.enclosure_url == "http://example.com/ep1.mp3"
    assert ep.published_date == datetime(2024, 1, 1, 0, 0, 0)


def test_parse_feed_skips_items_without_audio_enclosure():
    xml = make_feed(
        """
        <item>
          <title>Blog post, no audio</title>
          <guid>ep-text-only</guid>
        </item>
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert len(result.episodes) == 1
    assert result.episodes[0].guid == "ep-1-guid"


def test_parse_feed_finds_enclosure_via_media_content():
    xml = make_feed(
        """
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <media:content url="http://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert len(result.episodes) == 1
    assert result.episodes[0].enclosure_url == "http://example.com/ep1.mp3"


def test_parse_feed_finds_enclosure_via_atom_link_rel():
    xml = make_feed(
        """
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <atom:link rel="enclosure" href="http://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert len(result.episodes) == 1
    assert result.episodes[0].enclosure_url == "http://example.com/ep1.mp3"


def test_parse_feed_sniffs_extension_when_mime_type_missing():
    xml = make_feed(
        """
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <enclosure url="http://example.com/ep1.mp3?token=abc" type="application/octet-stream"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert len(result.episodes) == 1


def test_parse_feed_rejects_non_audio_enclosure():
    xml = make_feed(
        """
        <item>
          <title>Show notes PDF</title>
          <guid>ep-pdf</guid>
          <enclosure url="http://example.com/notes.pdf" type="application/pdf"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert result.episodes == []


def test_parse_feed_truncates_to_max_episodes():
    items = "\n".join(
        f"""
        <item>
          <title>Episode {i}</title>
          <guid>ep-{i}</guid>
          <enclosure url="http://example.com/ep{i}.mp3" type="audio/mpeg"/>
        </item>
        """
        for i in range(5)
    )
    xml = make_feed(items)

    result = parse_feed(xml, max_episodes=2)

    assert len(result.episodes) == 2
    assert [ep.guid for ep in result.episodes] == ["ep-0", "ep-1"]


@pytest.mark.parametrize(
    "guid_xml, link_xml, expected",
    [
        ("<guid>real-guid</guid>", "<link>http://example.com/page</link>", "real-guid"),
        ("", "<link>http://example.com/page</link>", "http://example.com/page"),
        ("", "", "http://example.com/ep1.mp3"),
    ],
)
def test_parse_feed_guid_fallback_chain(guid_xml, link_xml, expected):
    xml = make_feed(
        f"""
        <item>
          <title>Episode 1</title>
          {guid_xml}
          {link_xml}
          <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert result.episodes[0].guid == expected


@pytest.mark.parametrize(
    "duration_xml, expected",
    [
        ("<itunes:duration>125</itunes:duration>", 125),
        ("<itunes:duration>1:02:03</itunes:duration>", 3723),
        ("<itunes:duration>02:03</itunes:duration>", 123),
        ("<itunes:duration>N/A</itunes:duration>", None),
        ("", None),
    ],
)
def test_parse_feed_duration_parsing(duration_xml, expected):
    xml = make_feed(
        f"""
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg"/>
          {duration_xml}
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert result.episodes[0].duration_seconds == expected


def test_parse_feed_missing_published_date_is_none():
    xml = make_feed(
        """
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
        """
    )

    result = parse_feed(xml, max_episodes=10)

    assert result.episodes[0].published_date is None


def test_parse_feed_raises_on_unparseable_input():
    with pytest.raises(FeedParseError):
        parse_feed("", max_episodes=10)


def test_parse_feed_target_guid_match_returns_only_that_episode_ignoring_max_episodes():
    items = "\n".join(
        f"""
        <item>
          <title>Episode {i}</title>
          <guid>ep-{i}</guid>
          <enclosure url="http://example.com/ep{i}.mp3" type="audio/mpeg"/>
        </item>
        """
        for i in range(5)
    )
    xml = make_feed(items)

    result = parse_feed(xml, max_episodes=10, target_guid="ep-3")

    assert len(result.episodes) == 1
    assert result.episodes[0].guid == "ep-3"


def test_parse_feed_target_guid_no_match_raises():
    xml = make_feed(
        """
        <item>
          <title>Episode 1</title>
          <guid>ep-1-guid</guid>
          <enclosure url="http://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
        """
    )

    with pytest.raises(FeedParseError):
        parse_feed(xml, max_episodes=10, target_guid="does-not-exist")
