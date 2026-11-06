"""Result-state events and the review audit (schema 2.0 recognition results).

Revision ID: 0004_result_states
Revises: 0003_watchlist_management

Additive only: existing sightings, alerts and watchlist tables are unchanged, so
1.0 sighting producers and consumers keep working. Downgrade drops only the two
new tables.
"""
from app.models import RecognitionResult, ReviewDecision

revision = "0004_result_states"
down_revision = "0003_watchlist_management"
branch_labels = None
depends_on = None


def upgrade():
    from alembic import op
    bind = op.get_bind()
    RecognitionResult.__table__.create(bind=bind, checkfirst=True)
    ReviewDecision.__table__.create(bind=bind, checkfirst=True)


def downgrade():
    from alembic import op
    bind = op.get_bind()
    ReviewDecision.__table__.drop(bind=bind, checkfirst=True)
    RecognitionResult.__table__.drop(bind=bind, checkfirst=True)
