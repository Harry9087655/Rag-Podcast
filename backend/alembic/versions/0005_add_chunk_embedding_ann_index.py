"""Add HNSW ANN index on chunk.embedding for cosine similarity search.

Adds ix_chunk_embedding, an HNSW index over chunk.embedding using the
vector_cosine_ops operator class, so `<=>` nearest-neighbour queries use an
index scan instead of a sequential scan.
"""

from alembic import op


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_chunk_embedding ON chunk "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX ix_chunk_embedding")
