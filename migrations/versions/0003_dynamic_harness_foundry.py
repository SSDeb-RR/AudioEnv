"""Dynamic environment and model harness build records."""

from alembic import op
import sqlalchemy as sa


revision = "0003_dynamic_harness_foundry"
down_revision = "0002_product_workflow"
branch_labels = None
depends_on = None


def upgrade():
    # Migration 0002 used ``Base.metadata.create_all`` for legacy backfill. On
    # a fresh install that metadata may already include this newer table, while
    # a real 0.3 database will not. Support both paths deterministically.
    if "harness_builds" not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(
            "harness_builds",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(), sa.ForeignKey("projects.id"), nullable=False, index=True),
            sa.Column("kind", sa.String(), nullable=False, index=True),
            sa.Column("source_uri", sa.Text(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, index=True),
            sa.Column("report_json", sa.Text(), nullable=False),
            sa.Column("generated_artifact_ids_json", sa.Text(), nullable=False, server_default="[]"),
            sa.Column("environment_id", sa.String(), sa.ForeignKey("environments.id"), nullable=True),
            sa.Column("model_id", sa.String(), sa.ForeignKey("models.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade():
    op.drop_table("harness_builds")
