from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models.episode import TranscriptStatus
from .apple_podcasts import AppleResolutionError, is_apple_podcasts_url, resolve_apple_podcasts_url
from .parser import FeedParseError
from .service import ingest_podcast

router = APIRouter()


class IngestRequest(BaseModel):
    rss_url: str
    max_episodes: int = 1


class PodcastOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    author: str | None
    cover_url: str | None


class EpisodeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    transcript_status: TranscriptStatus


class IngestResponse(BaseModel):
    podcast: PodcastOut
    episodes: list[EpisodeOut]
    new_episodes: int
    skipped: int
    already_imported: bool


@router.post("/podcasts", response_model=IngestResponse)
async def create_podcast(
    body: IngestRequest, session: AsyncSession = Depends(get_session)
) -> IngestResponse:
    rss_url = body.rss_url
    target_guid = None

    if is_apple_podcasts_url(rss_url):
        try:
            resolved = resolve_apple_podcasts_url(rss_url)
        except AppleResolutionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rss_url = resolved.feed_url
        target_guid = resolved.target_guid

    try:
        result = await ingest_podcast(
            session, rss_url, max_episodes=body.max_episodes, target_guid=target_guid
        )
    except FeedParseError as exc:
        # Feed fetch/parse failure aborts the whole ingestion (PLAN.md §5.4)
        # — surfaced as a 400 since the client passed a bad/unreachable URL.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        podcast=PodcastOut.model_validate(result.podcast),
        episodes=[EpisodeOut.model_validate(ep) for ep in result.episodes],
        new_episodes=result.new_episodes,
        skipped=result.skipped,
        already_imported=result.already_imported,
    )
