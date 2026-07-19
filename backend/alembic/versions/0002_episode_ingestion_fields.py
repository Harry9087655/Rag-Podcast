from alembic import op
import sqlalchemy as sa



revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column(
    "episode",
    sa.Column("guid", sa.String(), nullable=False, server_default=""),
)
    op.alter_column("episode", "guid", server_default=None)

    op.add_column(
    "episode",
    sa.Column("enclosure_url", sa.String(), nullable=False, server_default="NOT GIVEN"),
)
    op.alter_column("episode", "enclosure_url", server_default=None)

    op.add_column(
    "episode",
    sa.Column("duration_seconds", sa.Integer(), nullable=True),
)
    op.create_unique_constraint("uq_episode_guid_podcast_id", "episode", ["guid", "podcast_id"])

    op.execute("ALTER TYPE transcriptstatus ADD VALUE 'downloaded' AFTER 'pending'")



def downgrade() -> None:
    op.drop_constraint("uq_episode_guid_podcast_id", "episode", type_="unique")
    op.drop_column("episode", "guid")
    op.drop_column("episode", "enclosure_url")
    op.drop_column("episode", "duration_seconds")
    op.execute("UPDATE episode SET transcript_status = 'pending' WHERE transcript_status = 'downloaded'")
    op.execute("ALTER TYPE transcriptstatus RENAME TO transcriptstatus_old_1")
    op.execute("""CREATE TYPE transcriptstatus AS ENUM ('pending', 
               'processing', 'done', 'failed');""")
    op.execute("ALTER TABLE episode ALTER COLUMN transcript_status DROP DEFAULT")

    op.execute("ALTER TABLE episode ALTER COLUMN transcript_status TYPE transcriptstatus USING transcript_status::text::transcriptstatus;")
    op.execute("ALTER TABLE episode ALTER COLUMN transcript_status SET DEFAULT 'pending'::transcriptstatus")
    op.execute("DROP TYPE transcriptstatus_old_1")
    