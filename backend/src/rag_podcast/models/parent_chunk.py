from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from . import Base


class ParentChunk(Base):
    __tablename__ = "parent_chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episode.id"))
    podcast_id: Mapped[int] = mapped_column(ForeignKey("podcast.id"))
    text: Mapped[str] = mapped_column(Text)
    start: Mapped[float]
    end: Mapped[float]
