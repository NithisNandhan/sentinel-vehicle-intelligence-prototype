"""Masked watchlist displays, monotonic revisions and mutation audits.

Revision ID: 0003_watchlist_management
Revises: 0002_recorded_sightings
"""
import sqlalchemy as sa
from alembic import op

from app.models import WatchlistAudit, WatchlistEntry, WatchlistRevision

revision = "0003_watchlist_management"
down_revision = "0002_recorded_sightings"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    # Some historical/repair databases contain only the sightings subset.
    # Creating the absent watchlist preserves those existing events.
    WatchlistEntry.__table__.create(bind=bind, checkfirst=True)
    columns = {column["name"] for column in sa.inspect(bind).get_columns("watchlist_entries")}
    if "masked_plate" not in columns:
        op.add_column("watchlist_entries", sa.Column("masked_plate", sa.String(32), nullable=True))
    WatchlistRevision.__table__.create(bind=bind, checkfirst=True)
    WatchlistAudit.__table__.create(bind=bind, checkfirst=True)
    if not bind.execute(sa.text("SELECT id FROM watchlist_revision WHERE id = 1")).first():
        initial = bind.execute(sa.text("SELECT COALESCE(MAX(version), 1) FROM watchlist_entries")).scalar_one()
        bind.execute(sa.text("INSERT INTO watchlist_revision (id, version) VALUES (1, :version)"), {"version": initial})


def downgrade():
    bind = op.get_bind()
    WatchlistAudit.__table__.drop(bind=bind, checkfirst=True)
    WatchlistRevision.__table__.drop(bind=bind, checkfirst=True)
    with op.batch_alter_table("watchlist_entries") as batch:
        batch.drop_column("masked_plate")
