"""Add Chinese full-text search: denormalize language + language-aware search_vector.

Registers pg_jieba (compiled into the db image), adds chunk.language denormalized
from episode.language, and rebuilds the persisted generated search_vector to pick
the tokenizer config by language (jiebacfg for 'zh', english otherwise).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

SEARCH_VECTOR_CASE_EXPR = (
    "to_tsvector(CASE WHEN language = 'zh' THEN 'jiebacfg'::regconfig "
    "ELSE 'english'::regconfig END, text)"
)
ENGLISH_SEARCH_VECTOR_EXPR = "to_tsvector('english', text)"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_jieba")

    op.add_column("chunk", sa.Column("language", sa.String(), nullable=True))
    op.execute(
        "UPDATE chunk SET language = episode.language "
        "FROM episode WHERE chunk.episode_id = episode.id"
    )
    op.execute("UPDATE chunk SET language = 'en' WHERE language IS NULL")
    op.alter_column("chunk", "language", nullable=False)

    # Postgres can't ALTER a generated column's expression in place: drop and
    # re-add. Dropping the column also drops its GIN index (re-created below).
    op.drop_column("chunk", "search_vector")
    op.add_column(
        "chunk",
        sa.Column(
            "search_vector",
            TSVECTOR,
            sa.Computed(SEARCH_VECTOR_CASE_EXPR, persisted=True),
            nullable=False,
        ),
    )
    op.execute("CREATE INDEX ix_chunk_search_vector ON chunk USING GIN (search_vector)")


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunk_search_vector")
    op.drop_column("chunk", "search_vector")
    op.add_column(
        "chunk",
        sa.Column(
            "search_vector",
            TSVECTOR,
            sa.Computed(ENGLISH_SEARCH_VECTOR_EXPR, persisted=True),
            nullable=False,
        ),
    )
    op.execute("CREATE INDEX ix_chunk_search_vector ON chunk USING GIN (search_vector)")
    op.drop_column("chunk", "language")
    op.execute("DROP EXTENSION IF EXISTS pg_jieba")
