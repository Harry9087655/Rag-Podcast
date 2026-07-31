import enum
from datetime import datetime

from sqlalchemy import Enum, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from . import Base


class TranscriptStatus(str, enum.Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class IndexStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Episode(Base):
    __tablename__ = "episode"
    __table_args__ = (UniqueConstraint("guid", "podcast_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    guid: Mapped[str]
    podcast_id: Mapped[int] = mapped_column(ForeignKey("podcast.id"))
    title: Mapped[str]
    published_date: Mapped[datetime | None]
    enclosure_url: Mapped[str]
    duration_seconds: Mapped[int | None]
    audio_local_path: Mapped[str | None]
    transcript_status: Mapped[TranscriptStatus] = mapped_column(
        Enum(TranscriptStatus, values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        default=TranscriptStatus.PENDING,
    )
    transcript_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    index_status: Mapped[IndexStatus] = mapped_column(
        Enum(IndexStatus, values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        default=IndexStatus.PENDING,
    )
    language: Mapped[str | None]
