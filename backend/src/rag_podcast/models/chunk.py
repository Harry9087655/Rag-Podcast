from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from . import Base


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    episode_id: Mapped[int] = mapped_column(ForeignKey("episode.id"))
    podcast_id: Mapped[int] = mapped_column(ForeignKey("podcast.id"))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))
    start: Mapped[float]
    end: Mapped[float]
    speaker: Mapped[str | None]
    language: Mapped[str]
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector(CASE WHEN language = 'zh' THEN 'jiebacfg'::regconfig "
            "ELSE 'english'::regconfig END, text)",
            persisted=True,
        ),
    )
