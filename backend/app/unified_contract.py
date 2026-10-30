"""Authoritative Pydantic/OpenAPI DTOs for the unified local dashboard.

Generate browser types with tools/generate_unified_contract.py. Images are raw
bounded base64 PNG/JPEG, never URLs. These objects contain no service secrets.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LocalSessionOut(ContractModel):
    csrf_token: str
    expires_at: datetime


class WatchlistItem(ContractModel):
    id: int
    masked_plate: str | None = None
    reason: str
    severity: Literal["low", "medium", "high", "critical"]
    active: bool
    version: int
    created_at: datetime


class WatchlistPage(ContractModel):
    items: list[WatchlistItem]
    total: int
    offset: int
    limit: int
    version: int


class WatchlistCreate(ContractModel):
    plate: str = Field(min_length=1, max_length=40)
    reason: str = Field(min_length=1, max_length=240)
    severity: Literal["low", "medium", "high", "critical"] = "high"


class WatchlistPatch(ContractModel):
    expected_version: int = Field(ge=1)
    reason: str | None = Field(default=None, min_length=1, max_length=240)
    severity: Literal["low", "medium", "high", "critical"] | None = None
    active: bool | None = None


class WatchlistSearch(ContractModel):
    query: str = Field(default="", max_length=240)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=25, ge=1, le=100)


class CharacterAlternative(ContractModel):
    character: str
    score: float


class CharacterEvidence(CharacterAlternative):
    alternatives: list[CharacterAlternative] = Field(default_factory=list)


class PlateCandidate(ContractModel):
    plate: str
    score: float
    minimum_character_score: float | None = None
    log_score: float | None = None
    observations: int | None = None


class FrameEvidence(ContractModel):
    pts_ms: int
    width: int
    height: int
    quality: float | None = None
    text: str = ""
    characters: list[CharacterEvidence] = Field(default_factory=list)
    image: str | None = None


class ReconstructionEvidence(ContractModel):
    status: str
    reason: str = ""
    image: str | None = None
    text: str | None = None
    width: int | None = None
    height: int | None = None
    contributing_originals: int = 0
    elapsed_ms: float | None = None
    characters: list[CharacterEvidence] = Field(default_factory=list)
    candidates: list[PlateCandidate] = Field(default_factory=list)


class TrackDiagnostics(ContractModel):
    track_id: str
    vehicle_box: list[int] = Field(default_factory=list)
    vehicle_class: str | int = "vehicle"
    seen_pts_ms: int | None = None
    observations: int = 0
    recognition_state: str = "reading"
    masked_plate: str | None = None
    best_reading: str | None = None
    model_score: float | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    frames: list[FrameEvidence] = Field(default_factory=list)
    candidates: list[PlateCandidate] = Field(default_factory=list)
    reconstruction: ReconstructionEvidence | None = None


class DiagnosticsSnapshot(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    source_mode: Literal["recorded"] = "recorded"
    session_id: str
    camera_id: str
    video_position_ms: int | None = None
    models: dict[str, str | None] = Field(default_factory=dict)
    stage_ms: dict[str, float] = Field(default_factory=dict)
    track: TrackDiagnostics
