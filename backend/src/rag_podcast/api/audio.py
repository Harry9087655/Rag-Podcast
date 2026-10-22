"""Episode audio endpoint: serve a downloaded episode's audio file.

``FileResponse`` handles ``Range`` requests (HTTP 206) natively, which is what
the browser ``<audio>`` element needs to seek to a source's ``start`` timestamp.
No manual range parsing, no transcoding — just resolve the stored relative path
against ``settings.data_dir`` and stream the file.

Design contract: .trellis/tasks/09-16-api/design.md
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import get_session
from ..models.episode import Episode

router = APIRouter()


@router.get("/audio/{episode_id}")
async def get_audio(
    episode_id: int, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    episode = await session.get(Episode, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="episode not found")
    if not episode.audio_local_path:
        raise HTTPException(status_code=404, detail="audio not available")

    path = Path(settings.data_dir) / episode.audio_local_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="audio file missing")

    return FileResponse(path)
