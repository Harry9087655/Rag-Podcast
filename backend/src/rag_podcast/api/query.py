"""QA query endpoint: the thin HTTP translation over ``qa.core.answer()``.

No business logic here.  The api layer's only transform is adding the
``audio_url`` (``/audio/{episode_id}``) each source needs for the frontend's
audio jump — routing, citation parsing, and prompt assembly all live in ``qa``.

Design contract: .trellis/tasks/09-16-api/design.md
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..qa.core import QAError, SourceRef, answer

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    podcast_id: int | None = None  # whole-podcast scope (search only)
    episode_id: int | None = None  # single-episode scope (full-context eligible)


class SourceOut(BaseModel):
    text: str
    podcast_name: str
    episode_title: str
    start: float
    audio_url: str


class QueryResponse(BaseModel):
    answer: str  # LLM verbatim, inline [n] markers retained
    sources: list[SourceOut]


def _to_source_out(source: SourceRef) -> SourceOut:
    """Adapt ``qa.SourceRef`` → wire ``SourceOut``; the api layer's only transform."""
    return SourceOut(
        text=source.text,
        podcast_name=source.podcast_name,
        episode_title=source.episode_title,
        start=source.start,
        audio_url=f"/audio/{source.episode_id}",
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest, session: AsyncSession = Depends(get_session)
) -> QueryResponse:
    try:
        result = await answer(
            body.question,
            session=session,
            episode_id=body.episode_id,
            podcast_id=body.podcast_id,
        )
    except QAError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[_to_source_out(s) for s in result.sources],
    )
