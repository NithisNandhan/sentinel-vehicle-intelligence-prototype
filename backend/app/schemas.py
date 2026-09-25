from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


SourceMode = Literal["live", "synthetic"]
CameraStatus = Literal["ONLINE", "DEGRADED", "OFFLINE", "UNSCHEDULED"]


def utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC timezone")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    return value


class SightingIn(BaseModel):
    schema_version: Literal["1.0"]
    event_id: UUID
    edge_id: str = Field(min_length=1, max_length=80)
    camera_id: str = Field(min_length=1, max_length=80)
    observed_at: datetime
    stream_pts_ms: int | None = Field(default=None, ge=0)
    track_id: str | None = Field(default=None, max_length=100)
    masked_plate: str = Field(min_length=3, max_length=32)
    plate_hmac: str = Field(pattern=r"^[a-f0-9]{64}$")
    confidence: float = Field(ge=0, le=1)
    source_mode: SourceMode
    model_version: str | None = Field(default=None, max_length=100)

    _validate_utc = field_validator("observed_at")(utc_datetime)


class HealthIn(BaseModel):
    event_id: UUID
    edge_id: str = Field(min_length=1, max_length=80)
    camera_id: str = Field(min_length=1, max_length=80)
    observed_at: datetime
    status: CameraStatus
    decoded_frames: int = Field(ge=0)
    inference_frames: int = Field(ge=0)
    reconnects: int = Field(ge=0)
    outbox_depth: int = Field(ge=0)
    source_mode: SourceMode

    _validate_utc = field_validator("observed_at")(utc_datetime)


class CameraIn(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(default="", max_length=200)
    region: str | None = Field(default=None, max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    coordinates_approximate: bool = True
    coordinate_source: str | None = Field(default=None, max_length=80)
    status: CameraStatus = "UNSCHEDULED"
    source_mode: SourceMode = "live"
    preview_url: str | None = None

    @field_validator("preview_url")
    @classmethod
    def local_preview_only(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not (value.startswith("/streams/") or value.startswith("http://localhost:") or value.startswith("http://127.0.0.1:")):
            raise ValueError("preview_url must be a local credential-free URL")
        if "@" in value:
            raise ValueError("preview_url must not contain credentials")
        return value


class CameraSyncIn(BaseModel):
    cameras: list[CameraIn] = Field(min_length=1, max_length=10000)


class CatalogueCameraOut(BaseModel):
    id: str
    name: str
    location: str = ""
    codec: str = ""
    live: bool


class CatalogueSearchOut(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[CatalogueCameraOut]


class PreviewOut(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    camera_id: str = Field(min_length=1, max_length=80)
    status: Literal["starting", "ready", "failed"]
    preview_url: str = Field(pattern=r"^/media/preview-[a-f0-9]{32}/index\.m3u8$")


class AlertPatch(BaseModel):
    status: Literal["acknowledged", "resolved", "false_positive"]
    actor: str = Field(default="local-operator", min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=500)


class RouteSearch(BaseModel):
    plate: str = Field(min_length=5, max_length=20)


class DemoInject(BaseModel):
    camera_id: str = "cam01"
    event_id: UUID | None = None


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    from_status: str
    to_status: str
    actor: str
    note: str | None
    changed_at: datetime


class AlertOut(BaseModel):
    id: int
    sighting_id: int
    camera_id: str
    observed_at: datetime
    masked_plate: str
    reason: str
    severity: str
    status: str
    source_mode: SourceMode
    created_at: datetime
    updated_at: datetime
    audit: list[AuditOut] = []

