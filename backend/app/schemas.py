from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# "recorded" is permitted downloaded footage replayed locally; never label it "live".
SourceMode = Literal["live", "synthetic", "recorded"]
# Whether observed_at comes from the recording's own stated clock or is only
# the time the file was processed. Unknown recording times are never invented.
TimeBasis = Literal["source_clock", "processing_time"]
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
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    time_basis: TimeBasis | None = None
    # When the edge analysed the frame. Kept separately because observed_at
    # may instead be the recording's own stated clock.
    processed_at: datetime | None = None

    _validate_utc = field_validator("observed_at")(utc_datetime)

    @field_validator("processed_at")
    @classmethod
    def processed_at_is_utc(cls, value: datetime | None) -> datetime | None:
        return value if value is None else utc_datetime(value)

    @model_validator(mode="after")
    def recorded_events_are_traceable(self):
        if self.source_mode == "recorded" and (self.session_id is None or self.time_basis is None
                                               or self.stream_pts_ms is None or self.processed_at is None):
            raise ValueError("recorded sightings need session_id, time_basis, stream_pts_ms and processed_at")
        return self


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
    stream_pts_ms: int | None = None
    session_id: str | None = None
    time_basis: TimeBasis | None = None
    processed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    audit: list[AuditOut] = []



# ----- Result states (schema 2.0) and review --------------------------------

ResultState = Literal["confirmed", "partial", "unreadable"]
_MASKED_PLATE = r"^[A-Z0-9]{2}\*{1,16}[A-Z0-9]{2}$"
_MASKED_PATTERN = r"^[A-Z0-9?]{2}\*{1,16}[A-Z0-9?]{2}$"
_REASON = r"^[a-z_]{3,48}$"
_HMAC = r"^[a-f0-9]{64}$"


class QualityIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    height_px: int | None = Field(default=None, ge=0, le=10000)
    width_px: int | None = Field(default=None, ge=0, le=10000)
    sharpness: float | None = Field(default=None, ge=0)
    contrast: float | None = Field(default=None, ge=0)
    glare_fraction: float | None = Field(default=None, ge=0, le=1)
    dark_fraction: float | None = Field(default=None, ge=0, le=1)
    mean_brightness: float | None = Field(default=None, ge=0, le=255)
    angle_deg: float | None = Field(default=None, ge=-90, le=90)
    blurred: bool | None = None
    low_contrast: bool | None = None
    glare: bool | None = None
    low_light: bool | None = None
    tilted: bool | None = None


class WatchlistLeadIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_hmacs: list[str] = Field(min_length=1, max_length=8)
    candidates_considered: int = Field(ge=1, le=200000)

    @field_validator("candidate_hmacs")
    @classmethod
    def fingerprints(cls, value: list[str]) -> list[str]:
        import re
        if any(not re.fullmatch(_HMAC, item) for item in value):
            raise ValueError("lead candidates are fingerprints")
        return value


class RecognitionResultIn(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal["2.0"]
    event_type: Literal["recognition_result"]
    event_id: UUID
    edge_id: str = Field(min_length=1, max_length=80)
    camera_id: str = Field(min_length=1, max_length=80)
    passage_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    result_state: ResultState
    observed_at: datetime
    last_observed_at: datetime
    stream_pts_ms: int | None = Field(default=None, ge=0)
    end_pts_ms: int | None = Field(default=None, ge=0)
    track_ids: list[str] = Field(default_factory=list, max_length=16)
    observations: int = Field(ge=0, le=1000)
    layout: Literal["one_row", "two_row", "conflict", "undetermined", "unknown"] = "unknown"
    review_required: bool = False
    source_mode: SourceMode
    model_version: str | None = Field(default=None, max_length=100)
    profile: str | None = Field(default=None, max_length=24)
    policy_version: str | None = Field(default=None, max_length=40)
    quality: QualityIn | None = None
    masked_plate: str | None = Field(default=None, max_length=32)
    plate_hmac: str | None = Field(default=None, pattern=_HMAC)
    plate_length: int | None = Field(default=None, ge=5, le=20)
    unknown_positions: list[int] | None = Field(default=None, max_length=20)
    reason: str | None = Field(default=None, pattern=_REASON)
    length_certain: bool | None = None
    format_status: Literal["valid", "lookalike", "ambiguous", "invalid"] | None = None
    evidence_ref: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{16,64}$")
    sighting_event_id: UUID | None = None
    watchlist_lead: WatchlistLeadIn | None = None
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    time_basis: TimeBasis | None = None
    processed_at: datetime | None = None

    _validate_utc = field_validator("observed_at", "last_observed_at")(utc_datetime)

    @field_validator("processed_at")
    @classmethod
    def processed_at_is_utc(cls, value: datetime | None) -> datetime | None:
        return value if value is None else utc_datetime(value)

    @field_validator("track_ids")
    @classmethod
    def short_track_ids(cls, value: list[str]) -> list[str]:
        if any(not 1 <= len(item) <= 100 for item in value):
            raise ValueError("track IDs are 1 to 100 characters")
        return value

    @model_validator(mode="after")
    def state_specific(self):
        import re
        if self.last_observed_at < self.observed_at:
            raise ValueError("last_observed_at precedes observed_at")
        if self.source_mode == "recorded" and (self.session_id is None or self.time_basis is None
                                               or self.stream_pts_ms is None or self.processed_at is None):
            raise ValueError("recorded results need session_id, time_basis, stream_pts_ms and processed_at")
        if self.result_state == "confirmed":
            if (not self.plate_hmac or not self.masked_plate or not re.fullmatch(_MASKED_PLATE, self.masked_plate)
                    or self.reason or self.unknown_positions or self.watchlist_lead or self.plate_length is None):
                raise ValueError("a confirmed result carries a fingerprint and a masked plate only")
        elif self.result_state == "partial":
            if (self.plate_hmac or not self.masked_plate or not re.fullmatch(_MASKED_PATTERN, self.masked_plate)
                    or self.plate_length is None or len(self.masked_plate) != self.plate_length
                    or self.unknown_positions is None or not self.reason):
                raise ValueError("a partial result carries a masked pattern, its length and positions only")
            if any(not 0 <= index < self.plate_length for index in self.unknown_positions):
                raise ValueError("unknown positions lie outside the plate")
            if self.watchlist_lead is not None and self.length_certain is False:
                raise ValueError("no watchlist lead from an uncertain length")
        else:
            if self.plate_hmac or self.masked_plate or self.watchlist_lead or not self.reason:
                raise ValueError("an unreadable result carries a reason and no plate data")
        return self


class ReviewDecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["confirm", "correct", "reject", "unreadable"]
    plate: str | None = Field(default=None, min_length=5, max_length=20)
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def plate_only_when_confirming(self):
        if (self.action in ("confirm", "correct")) != (self.plate is not None):
            raise ValueError("confirm and correct need the plate; reject and unreadable must not include one")
        return self
