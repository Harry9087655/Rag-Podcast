"""Add parent_chunk table and chunk.parent_id.

Parent chunks are time-contiguous merges of leaf chunks (the only embedded and
retrieved unit). They store text + start/end for auto-merge after leaf recall,
with no embedding and no search_vector. chunk.parent_id points a leaf at its
parent, and chunk converges to the leaf layer.
"""

from alembic import op
import sqlalchemy as sa


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "parent_chunk",
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
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
    )
    op.add_column(
        "chunk",
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("parent_chunk.id"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("chunk", "parent_id")
    op.drop_table("parent_chunk")
