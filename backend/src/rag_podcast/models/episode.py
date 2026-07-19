import enum
from datetime import datetime

from sqlalchemy import Enum, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from . import Base


class TranscriptStatus(str, enum.Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
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
