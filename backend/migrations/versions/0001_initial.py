"""Initial camera, sighting, alert and health schema.

Revision ID: 0001_initial
Revises:
"""

from alembic import op

from app.db import Base
from app import models  # noqa: F401

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    Base.metadata.create_all(bind=bind)


def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())

