from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Register every model on Base.metadata so that any entry point — the API,
# the transcription/embedding workers, their CLIs, or alembic — gets the full
# table set regardless of which module it imports. Without this, a table only
# exists if some module happened to import its model; e.g. the embedding worker
# imports Episode and Chunk but not Podcast, so the chunk.podcast_id foreign
# key could not resolve at flush time ("could not find table 'podcast'").
from .podcast import Podcast  # noqa: F401,E402
from .episode import Episode  # noqa: F401,E402
from .chunk import Chunk  # noqa: F401,E402
