"""Projects, protocol-safe evaluations, logs, and provider resources."""

from alembic import op
import sqlalchemy as sa

from audioenv.platform.db import Base

revision = "0002_product_workflow"
down_revision = "0001_v1_platform"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)
    inspector = sa.inspect(bind)
    additions = {
        "models": [("project_id", sa.String())],
        "environments": [("project_id", sa.String())],
        "experiments": [
            ("project_id", sa.String()),
            ("config_hash", sa.String(length=64)),
            ("preflight_json", sa.Text()),
        ],
        "artifacts": [("project_id", sa.String())],
        "evaluations": [
            ("project_id", sa.String()),
            ("protocol_hash", sa.String(length=64)),
        ],
    }
    for table, columns in additions.items():
        existing = {column["name"] for column in inspector.get_columns(table)}
        for name, column_type in columns:
            if name not in existing:
                op.add_column(table, sa.Column(name, column_type, nullable=True))
    connection = bind
    connection.execute(sa.text(
        "INSERT OR IGNORE INTO projects (id, name, description, created_at) "
        "VALUES ('imported-local-workspace', 'Imported local workspace', "
        "'Default project for local and legacy records.', CURRENT_TIMESTAMP)"
    ))
    for table in ("models", "environments", "experiments", "artifacts", "evaluations"):
        connection.execute(sa.text(
            f"UPDATE {table} SET project_id='imported-local-workspace' WHERE project_id IS NULL"
        ))


def downgrade():
    # Deliberately retain data-bearing tables. V1 migrations are forward-only.
    pass
