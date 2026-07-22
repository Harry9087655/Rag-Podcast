"""Add transcript_data JSONB column to episode table.

Allows the transcription worker to store WhisperX output
(segments with word-level timestamps) directly on the Episode row.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "episode",
        sa.Column(
            "transcript_data",
            postgresql.JSONB,
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("episode", "transcript_data")
