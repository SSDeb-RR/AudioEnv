"""Separate dependency availability from model-driver conformance."""

from alembic import op
import sqlalchemy as sa


revision = "0004_model_registration_state"
down_revision = "0003_dynamic_harness_foundry"
branch_labels = None
depends_on = None


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("models")}
    if "registration_state" not in columns:
        op.add_column(
            "models",
            sa.Column(
                "registration_state", sa.String(), nullable=False,
                server_default="generated_needs_review",
            ),
        )
    op.execute(
        "UPDATE models SET registration_state = 'certified' "
        "WHERE adapter_type = 'small-audio-tool-policy'"
    )
    op.execute(
        "UPDATE models SET available = 0 "
        "WHERE registration_state NOT IN ('certified', 'conformant')"
    )


def downgrade():
    op.drop_column("models", "registration_state")
