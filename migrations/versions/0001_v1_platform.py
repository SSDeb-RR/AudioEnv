"""Initial AudioEnv V1 platform schema."""

from alembic import op

from audioenv.platform.db import Base

revision = "0001_v1_platform"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    Base.metadata.create_all(bind=op.get_bind())


def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())
