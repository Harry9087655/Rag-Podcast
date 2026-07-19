from sqlalchemy.orm import Mapped, mapped_column

from . import Base


class Podcast(Base):
    __tablename__ = "podcast"

    id: Mapped[int] = mapped_column(primary_key=True)
    rss_url: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    author: Mapped[str | None]
    cover_url: Mapped[str | None]
