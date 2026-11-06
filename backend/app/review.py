"""Result-state ingestion (schema 2.0) and the access-restricted review workflow.

Safety rules enforced here, independent of the edge:
- partial and unreadable results never create sightings or alerts;
- a watchlist lead from a partial reading only opens a *pending review*;
- only a named reviewer's confirm/correct decision can create a (human-reviewed)
  sighting, and only that sighting may then match the watchlist and alert;
- no plaintext plate, image or full partial pattern is persisted centrally.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import RecognitionResult, ReviewDecision, Sighting, WatchlistEntry, utc_now
from .schemas import RecognitionResultIn, ReviewDecisionIn
from .security import check_edge_key, check_operator_key, normalize_plate, plate_token, reviewer_identity

REVIEW_EVENTS = uuid5(NAMESPACE_URL, "https://sentinel.invalid/review-sightings")
OPEN = ("pending", "not_required")
results_router = APIRouter(tags=["recognition-results"])
review_router = APIRouter(prefix="/v1/review", tags=["review"], dependencies=[Depends(check_operator_key)])


def as_utc(value):
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def retention_days(state: str) -> int:
    settings = get_settings()
    return {"confirmed": settings.retention_confirmed_days, "partial": settings.retention_partial_days,
            "unreadable": settings.retention_unreadable_days}[state]


def require_reviewer(request: Request, x_reviewer_token: str | None = Header(default=None)) -> str:
    """Operator access (dependency of the router) plus a named reviewer token."""
    reviewer = reviewer_identity(x_reviewer_token)
    if reviewer is None:
        raise HTTPException(status_code=403, detail="Reviewer access required")
    return reviewer


def public_result(row: RecognitionResult, *, detail=False) -> dict:
    data = {
        "id": row.id, "event_id": row.event_id, "camera_id": row.camera_id, "passage_id": row.passage_id,
        "result_state": row.result_state, "observed_at": as_utc(row.observed_at).isoformat(),
        "last_observed_at": as_utc(row.last_observed_at).isoformat(), "masked_plate": row.masked_plate,
        "plate_length": row.plate_length, "unknown_positions": json.loads(row.unknown_positions or "[]"),
        "reason": row.reason, "review_required": row.review_required, "review_status": row.review_status,
        "length_certain": row.length_certain, "observations": row.observations, "layout": row.layout,
        "format_status": row.format_status, "source_mode": row.source_mode, "stream_pts_ms": row.stream_pts_ms,
        "end_pts_ms": row.end_pts_ms, "session_id": row.session_id, "time_basis": row.time_basis,
        "profile": row.profile, "policy_version": row.policy_version,
        "watchlist_lead": bool(row.lead_hmacs), "evidence_available": bool(row.evidence_ref),
        "expires_at": as_utc(row.expires_at).isoformat(),
    }
    if detail:
        data.update(quality=json.loads(row.quality) if row.quality else None, edge_id=row.edge_id,
                    model_version=row.model_version, track_ids=json.loads(row.track_ids or "[]"),
                    lead_candidates=len(json.loads(row.lead_hmacs or "[]")),
                    lead_candidates_considered=row.lead_candidates_considered,
                    sighting_event_id=row.sighting_event_id)
    return data


@results_router.post("/v1/recognition-results", dependencies=[Depends(check_edge_key)])
async def post_result(payload: RecognitionResultIn, session: Session = Depends(get_db)) -> dict:
    from .main import ensure_camera, require_demo_for_synthetic, sockets
    require_demo_for_synthetic(payload.source_mode)
    event_id = str(payload.event_id)
    existing = session.scalar(select(RecognitionResult).where(RecognitionResult.event_id == event_id))
    if existing:
        return {"accepted": True, "duplicate": True, "result_id": existing.id, "review_status": existing.review_status}
    leads = []
    if payload.watchlist_lead is not None:
        # Keep only fingerprints that are on the active watchlist now. Never an alert.
        active = set(session.scalars(select(WatchlistEntry.plate_hmac).where(
            WatchlistEntry.active.is_(True), WatchlistEntry.plate_hmac.in_(payload.watchlist_lead.candidate_hmacs))).all())
        leads = sorted(active)
    pending = payload.result_state == "partial" and (payload.review_required or bool(leads))
    if payload.result_state == "confirmed" and payload.review_required:
        pending = True  # Later evidence disagreed with an earlier confirmation.
    now = utc_now()
    ensure_camera(session, payload.camera_id, payload.source_mode)
    row = RecognitionResult(
        event_id=event_id, schema_version=payload.schema_version, edge_id=payload.edge_id,
        camera_id=payload.camera_id, passage_id=payload.passage_id, result_state=payload.result_state,
        observed_at=payload.observed_at, last_observed_at=payload.last_observed_at,
        stream_pts_ms=payload.stream_pts_ms, end_pts_ms=payload.end_pts_ms, track_ids=json.dumps(payload.track_ids),
        masked_plate=payload.masked_plate, plate_hmac=payload.plate_hmac, plate_length=payload.plate_length,
        unknown_positions=json.dumps(sorted(set(payload.unknown_positions))) if payload.unknown_positions is not None else None,
        reason=payload.reason, review_required=payload.review_required, length_certain=payload.length_certain,
        evidence_ref=payload.evidence_ref, observations=payload.observations, layout=payload.layout,
        format_status=payload.format_status,
        quality=payload.quality.model_dump_json(exclude_none=True) if payload.quality else None,
        sighting_event_id=str(payload.sighting_event_id) if payload.sighting_event_id else None,
        lead_hmacs=json.dumps(leads) if leads else None,
        lead_candidates_considered=payload.watchlist_lead.candidates_considered if payload.watchlist_lead else None,
        source_mode=payload.source_mode, session_id=payload.session_id, time_basis=payload.time_basis,
        processed_at=payload.processed_at, model_version=payload.model_version, profile=payload.profile,
        policy_version=payload.policy_version, review_status="pending" if pending else "not_required",
        created_at=now, expires_at=now + timedelta(days=retention_days(payload.result_state)))
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(select(RecognitionResult).where(RecognitionResult.event_id == event_id))
        if existing is None:
            raise
        return {"accepted": True, "duplicate": True, "result_id": existing.id, "review_status": existing.review_status}
    if pending:
        await sockets.broadcast({"type": "review_pending", "result": public_result(row)})
    return {"accepted": True, "duplicate": False, "result_id": row.id, "review_status": row.review_status}


@results_router.get("/v1/recognition-results", dependencies=[Depends(check_operator_key)])
def list_results(response: Response, state: str | None = Query(default=None, pattern="^(confirmed|partial|unreadable)$"),
                 limit: int = Query(100, ge=1, le=500), session: Session = Depends(get_db)) -> dict:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    query = select(RecognitionResult).order_by(RecognitionResult.observed_at.desc(), RecognitionResult.id.desc()).limit(limit)
    if state:
        query = query.where(RecognitionResult.result_state == state)
    counts = dict(session.execute(select(RecognitionResult.result_state, func.count())
                                  .group_by(RecognitionResult.result_state)).all())
    return {"counts": {key: counts.get(key, 0) for key in ("confirmed", "partial", "unreadable")},
            "items": [public_result(row) for row in session.scalars(query).all()]}


@review_router.get("/queue")
def review_queue(response: Response, status: str = Query("pending", pattern="^(pending|decided|all)$"),
                 limit: int = Query(50, ge=1, le=200), reviewer: str = Depends(require_reviewer),
                 session: Session = Depends(get_db)) -> dict:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    query = select(RecognitionResult).order_by(RecognitionResult.lead_hmacs.is_(None), RecognitionResult.observed_at.desc())
    if status == "pending":
        query = query.where(RecognitionResult.review_status == "pending")
    elif status == "decided":
        query = query.where(RecognitionResult.review_status.not_in(OPEN))
    pending = session.scalar(select(func.count()).where(RecognitionResult.review_status == "pending")) or 0
    return {"reviewer": reviewer, "pending": pending,
            "items": [public_result(row) for row in session.scalars(query.limit(limit)).all()]}


def _row(session, result_id) -> RecognitionResult:
    row = session.get(RecognitionResult, result_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Result not found or past retention")
    return row


@review_router.get("/{result_id}")
def review_detail(result_id: int, response: Response, reviewer: str = Depends(require_reviewer),
                  session: Session = Depends(get_db)) -> dict:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    row = _row(session, result_id)
    decisions = session.scalars(select(ReviewDecision).where(ReviewDecision.result_event_id == row.event_id)
                                .order_by(ReviewDecision.id)).all()
    data = public_result(row, detail=True)
    data["decisions"] = [{"reviewer": d.reviewer, "action": d.action, "from_status": d.from_status, "note": d.note,
                          "masked_plate": d.masked_plate, "alert_id": d.alert_id,
                          "decided_at": as_utc(d.decided_at).isoformat()} for d in decisions]
    return data


@review_router.get("/{result_id}/evidence")
def review_evidence(result_id: int, response: Response, reviewer: str = Depends(require_reviewer),
                    session: Session = Depends(get_db)) -> dict:
    """Edge-held evidence, proxied without storage. Opt-in; expires on the edge."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    if not get_settings().review_evidence_enabled:
        raise HTTPException(status_code=404, detail="Review evidence is not enabled")
    row = _row(session, result_id)
    if not row.evidence_ref:
        raise HTTPException(status_code=404, detail="No evidence was retained for this result")
    from .recognition import request_edge
    from urllib.parse import quote
    data = request_edge("GET", "/internal/review-evidence?ref=" + quote(row.evidence_ref, safe=""))
    if not isinstance(data, dict) or data.get("passage_id") != row.passage_id:
        raise HTTPException(status_code=404, detail="Evidence expired or unavailable")
    return data


def _agrees_with_mask(plate: str, row: RecognitionResult) -> bool:
    """A 'confirm' must agree with every position the edge reported as known and visible."""
    mask = row.masked_plate or ""
    if row.plate_length is not None and len(plate) != row.plate_length:
        return False
    if len(mask) != len(plate):
        return False
    return all(m in "*?" or m == c for m, c in zip(mask, plate))


@review_router.post("/{result_id}/decision")
async def review_decision(result_id: int, payload: ReviewDecisionIn, reviewer: str = Depends(require_reviewer),
                          session: Session = Depends(get_db)) -> dict:
    from .main import alert_dict, ingest_sighting, sockets
    from .schemas import SightingIn
    row = _row(session, result_id)
    if row.review_status not in OPEN:
        raise HTTPException(status_code=409, detail="This result has already been decided")
    plate = token = masked = None
    if payload.plate is not None:
        try:
            plate = normalize_plate(payload.plate)
        except ValueError:
            raise HTTPException(status_code=422, detail="Plate must contain 5 to 20 letters or digits") from None
        if payload.action == "confirm":
            if row.result_state == "confirmed":
                if plate_token(plate) != row.plate_hmac:
                    raise HTTPException(status_code=422, detail="Confirm must repeat the model reading; use correct")
            elif not _agrees_with_mask(plate, row):
                raise HTTPException(status_code=422, detail="Confirm must agree with the known characters; use correct")
        token, masked = plate_token(plate), plate[:2] + "*" * (len(plate) - 4) + plate[-2:]
    sighting_id = alert = None
    if token is not None and not (row.result_state == "confirmed" and token == row.plate_hmac):
        # A human-reviewed identity: an ordinary 1.0 sighting, so the existing watchlist
        # and alert workflow applies. Its model_version records that a person decided.
        observed = as_utc(row.observed_at)
        data = SightingIn(schema_version="1.0", event_id=uuid5(REVIEW_EVENTS, row.event_id + "|" + token),
                          edge_id=row.edge_id[:80], camera_id=row.camera_id, observed_at=observed,
                          stream_pts_ms=row.stream_pts_ms, track_id=None, masked_plate=masked, plate_hmac=token,
                          confidence=1.0, source_mode=row.source_mode, model_version="human-review",
                          session_id=row.session_id, time_basis=row.time_basis,
                          processed_at=as_utc(row.processed_at) if row.processed_at else (
                              observed if row.source_mode == "recorded" else None))
        sighting, alert, duplicate = ingest_sighting(session, data)
        sighting_id = sighting.id
        if alert is not None and not duplicate:
            alert.reason = ("Human-reviewed · " + alert.reason)[:240]
            session.commit()
    previous = row.review_status
    row.review_status = {"confirm": "confirmed", "correct": "corrected", "reject": "rejected",
                         "unreadable": "unreadable"}[payload.action]
    session.add(ReviewDecision(result_id=row.id, result_event_id=row.event_id, reviewer=reviewer,
                               action=payload.action, from_status=previous, note=payload.note,
                               masked_plate=masked, plate_hmac=token, sighting_id=sighting_id,
                               alert_id=alert.id if alert is not None else None))
    session.commit()
    serialized = alert_dict(alert) if alert is not None else None
    if serialized:
        await sockets.broadcast({"type": "alert_created", "alert": serialized})
    return {"result": public_result(row), "sighting_id": sighting_id, "alert": serialized}


@review_router.get("/export/decisions")
def export_decisions(response: Response, reviewer: str = Depends(require_reviewer),
                     session: Session = Depends(get_db)) -> dict:
    """Masked review outcomes for development use. Never eligible for a sealed set."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    rows = session.scalars(select(ReviewDecision).order_by(ReviewDecision.id)).all()
    return {"label_source": "review", "sealed_eligible": False, "exported_by": reviewer,
            "notice": "Masked review outcomes only. Review-derived labels must never enter a sealed test set.",
            "decisions": [{"result_event_id": d.result_event_id, "action": d.action, "masked_plate": d.masked_plate,
                           "reviewer": d.reviewer, "decided_at": as_utc(d.decided_at).isoformat()} for d in rows]}


def purge_expired(session: Session, now: datetime | None = None) -> dict:
    """Delete results past retention and audit rows past the audit retention."""
    now = now or utc_now()
    results = session.execute(delete(RecognitionResult).where(RecognitionResult.expires_at < now)
                              .execution_options(synchronize_session=False)).rowcount or 0
    audit_cutoff = now - timedelta(days=get_settings().retention_review_audit_days)
    audits = session.execute(delete(ReviewDecision).where(ReviewDecision.decided_at < audit_cutoff)
                             .execution_options(synchronize_session=False)).rowcount or 0
    session.commit()
    return {"results_deleted": results, "review_audit_deleted": audits}


@results_router.post("/v1/admin/retention/run", dependencies=[Depends(check_operator_key)])
def run_retention(session: Session = Depends(get_db)) -> dict:
    return purge_expired(session)
