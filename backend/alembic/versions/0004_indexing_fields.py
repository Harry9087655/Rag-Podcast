"""Add indexing fields: episode.index_status, episode.language, chunk.search_vector.

Adds the indexing pipeline's status column (mirrors transcript_status),
episode.language (populated by the fixed transcriber), and a generated
tsvector column + GIN index on chunk.text for future full-text search.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

index_status = sa.Enum(
    "pending", "processing", "done", "failed", name="indexstatus", create_type=False
)


def upgrade() -> None:
    index_status.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "episode",
        sa.Column(
            "index_status",
            index_status,
            nullable=False,
            server_default="pending",
        ),
    )
    op.alter_column("episode", "index_status", server_default=None)

    op.add_column(
        "episode",
        sa.Column("language", sa.String(), nullable=True),
    )

    op.add_column(
        "chunk",
        sa.Column(
            "search_vector",
            TSVECTOR,
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
    )
    op.execute("CREATE INDEX ix_chunk_search_vector ON chunk USING GIN (search_vector)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunk_search_vector")
    op.drop_column("chunk", "search_vector")
    op.drop_column("episode", "language")
    op.drop_column("episode", "index_status")
    index_status.drop(op.get_bind(), checkfirst=True)
