"""Recorded-footage traceability on sightings.

Revision ID: 0002_recorded_sightings
Revises: 0001_initial
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_recorded_sightings"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

COLUMNS = (("session_id", sa.String(64)), ("time_basis", sa.String(24)),
           ("processed_at", sa.DateTime(timezone=True)))


def upgrade():
    # 0001 creates tables from current metadata, so a fresh database already
    # has these columns; only an existing 0001 database needs them added.
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("sightings")}
    for name, kind in COLUMNS:
        if name not in existing:
            op.add_column("sightings", sa.Column(name, kind, nullable=True))


def downgrade():
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("sightings")}
    for name, _ in COLUMNS:
        if name in existing:
            op.drop_column("sightings", name)
