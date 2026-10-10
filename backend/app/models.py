from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    region: Mapped[str | None] = mapped_column(String(120), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinates_approximate: Mapped[bool] = mapped_column(Boolean, default=True)
    coordinate_source: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="UNSCHEDULED")
    source_mode: Mapped[str] = mapped_column(String(16), default="live")
    preview_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Sighting(Base):
    __tablename__ = "sightings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    schema_version: Mapped[str] = mapped_column(String(12))
    edge_id: Mapped[str] = mapped_column(String(80))
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    stream_pts_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    track_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    masked_plate: Mapped[str] = mapped_column(String(32))
    plate_hmac: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    source_mode: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    time_basis: Mapped[str | None] = mapped_column(String(24), nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    alert: Mapped["Alert | None"] = relationship(back_populates="sighting", uselist=False)


class WatchlistEntry(Base):
    __tablename__ = "watchlist_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_hmac: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    reason: Mapped[str] = mapped_column(String(240))
    severity: Mapped[str] = mapped_column(String(16), default="high")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"), unique=True)
    reason: Mapped[str] = mapped_column(String(240))
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24), default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    sighting: Mapped[Sighting] = relationship(back_populates="alert")
    audit: Mapped[list["AlertAudit"]] = relationship(back_populates="alert", order_by="AlertAudit.id")


class AlertAudit(Base):
    __tablename__ = "alert_audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), index=True)
    from_status: Mapped[str] = mapped_column(String(24))
    to_status: Mapped[str] = mapped_column(String(24))
    actor: Mapped[str] = mapped_column(String(100), default="local-operator")
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    alert: Mapped[Alert] = relationship(back_populates="audit")


class HealthReport(Base):
    __tablename__ = "health_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    edge_id: Mapped[str] = mapped_column(String(80))
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(24))
    decoded_frames: Mapped[int] = mapped_column(Integer, default=0)
    inference_frames: Mapped[int] = mapped_column(Integer, default=0)
    reconnects: Mapped[int] = mapped_column(Integer, default=0)
    outbox_depth: Mapped[int] = mapped_column(Integer, default=0)
    source_mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

