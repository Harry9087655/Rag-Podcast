"""init

Revision ID: 0001
Revises:
Create Date: 2026-07-10

"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

transcript_status = sa.Enum(
    "pending", "processing", "done", "failed", name="transcriptstatus"
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "podcast",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("rss_url", sa.String(), nullable=False, unique=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("author", sa.String(), nullable=True),
        sa.Column("cover_url", sa.String(), nullable=True),
    )

    op.create_table(
        "episode",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "podcast_id",
            sa.Integer(),
            sa.ForeignKey("podcast.id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("published_date", sa.DateTime(), nullable=True),
        sa.Column("audio_local_path", sa.String(), nullable=True),
        sa.Column(
            "transcript_status",
            transcript_status,
            nullable=False,
            server_default="pending",
        ),
    )

    op.create_table(
        "chunk",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "episode_id",
            sa.Integer(),
            sa.ForeignKey("episode.id"),
            nullable=False,
        ),
        sa.Column(
            "podcast_id",
            sa.Integer(),
            sa.ForeignKey("podcast.id"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("speaker", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("chunk")
    op.drop_table("episode")
    op.drop_table("podcast")
    transcript_status.drop(op.get_bind(), checkfirst=True)
