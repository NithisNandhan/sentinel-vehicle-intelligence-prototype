import asyncio
import hmac
from datetime import datetime, timedelta, timezone
from html import escape
from urllib.parse import quote, urlencode
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .catalogue import gateway_request
from .recognition import router as recognition_router
from .local_access import router as local_router
from .watchlist import router as watchlist_router, revision as watchlist_revision
from .db import Base, SessionLocal, engine, get_db
from .models import Alert, AlertAudit, Camera, HealthReport, Sighting, WatchlistEntry, utc_now
from .schemas import (AlertOut, AlertPatch, AuditOut, CameraSyncIn, CatalogueSearchOut,
                      DemoInject, HealthIn, PreviewOut, RouteSearch, SightingIn)
from .security import check_edge_key, check_operator_key, plate_token, local_session


settings = get_settings()
app = FastAPI(title="Sentinel Vehicle Intelligence API", version="1.0.0")
app.include_router(recognition_router)
app.include_router(local_router)
app.include_router(watchlist_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.local_dashboard_mode,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["X-Edge-Key", "X-Operator-Key", "Content-Type", "X-CSRF-Token"],
)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request: Request, error: RequestValidationError):
    # Pydantic normally echoes rejected inputs, including raw watchlist plates.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": list(item["loc"]), "type": item["type"], "msg": "Request field is invalid"}
        for item in error.errors()
    ]})


class AlertSockets:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def broadcast(self, data: dict) -> None:
        dead = []
        for client in tuple(self.clients):
            try:
                await client.send_json(data)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)


sockets = AlertSockets()


def as_utc(value: datetime) -> datetime:
    """SQLite returns naive timestamps even for timezone-aware columns."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def require_demo_for_synthetic(source_mode: str) -> None:
    if source_mode == "synthetic" and not settings.demo_mode:
        raise HTTPException(status_code=403, detail="synthetic events are disabled")


@app.on_event("startup")
def startup() -> None:
    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(engine)
    with SessionLocal() as session:
        demo_token = plate_token("GJ01AB1234")
        existing = session.scalar(select(WatchlistEntry).where(WatchlistEntry.plate_hmac == demo_token))
        if not existing:
            session.add(WatchlistEntry(plate_hmac=demo_token, reason="Synthetic watchlist demonstration", severity="high"))
        for camera_id, name, lat, lon in (
            ("cam01", "Chiman bhai Bridge", 23.046, 72.57),
            ("cam04", "Paldi Circle", 23.012, 72.566),
        ):
            camera = session.get(Camera, camera_id)
            if camera is None:
                session.add(Camera(id=camera_id, name=name, region="Ahmedabad", latitude=lat, longitude=lon,
                                   coordinates_approximate=True, coordinate_source="demo_approximate", status="UNSCHEDULED",
                                   source_mode="synthetic"))
        session.commit()
        watchlist_revision(session)
        session.commit()


def alert_dict(alert: Alert) -> dict:
    sighting = alert.sighting
    return AlertOut(
        id=alert.id,
        sighting_id=sighting.id,
        camera_id=sighting.camera_id,
        observed_at=as_utc(sighting.observed_at),
        masked_plate=sighting.masked_plate,
        reason=alert.reason,
        severity=alert.severity,
        status=alert.status,
        source_mode=sighting.source_mode,
        stream_pts_ms=sighting.stream_pts_ms,
        session_id=sighting.session_id,
        time_basis=sighting.time_basis,
        processed_at=as_utc(sighting.processed_at) if sighting.processed_at else None,
        created_at=as_utc(alert.created_at),
        updated_at=as_utc(alert.updated_at),
        audit=[AuditOut(id=entry.id, from_status=entry.from_status, to_status=entry.to_status,
                        actor=entry.actor, note=entry.note, changed_at=as_utc(entry.changed_at))
               for entry in alert.audit],
    ).model_dump(mode="json")


def camera_dict(camera: Camera) -> dict:
    status = camera.status
    if status in ("ONLINE", "DEGRADED") and camera.updated_at:
        updated = as_utc(camera.updated_at)
        if utc_now() - updated > timedelta(seconds=180):
            status = "OFFLINE"
    return {
        "id": camera.id,
        "name": camera.name,
        "region": camera.region,
        "latitude": camera.latitude,
        "longitude": camera.longitude,
        "coordinates_approximate": camera.coordinates_approximate,
        "coordinate_source": camera.coordinate_source,
        "status": status,
        "source_mode": camera.source_mode,
        "preview_url": camera.preview_url,
        "updated_at": as_utc(camera.updated_at).isoformat() if camera.updated_at else None,
    }


def ensure_camera(session: Session, camera_id: str, source_mode: str) -> Camera:
    camera = session.get(Camera, camera_id)
    if camera is None:
        camera = Camera(id=camera_id, name=camera_id, source_mode=source_mode)
        session.add(camera)
        session.flush()
    return camera


def ingest_sighting(session: Session, payload: SightingIn) -> tuple[Sighting, Alert | None, bool]:
    event_id = str(payload.event_id)
    existing = session.scalar(select(Sighting).where(Sighting.event_id == event_id))
    if existing:
        alert = session.scalar(
            select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).where(Alert.sighting_id == existing.id)
        )
        return existing, alert, True
    try:
        ensure_camera(session, payload.camera_id, payload.source_mode)
        sighting = Sighting(
            event_id=event_id,
            schema_version=payload.schema_version,
            edge_id=payload.edge_id,
            camera_id=payload.camera_id,
            observed_at=payload.observed_at,
            stream_pts_ms=payload.stream_pts_ms,
            track_id=payload.track_id,
            masked_plate=payload.masked_plate,
            plate_hmac=payload.plate_hmac,
            confidence=payload.confidence,
            source_mode=payload.source_mode,
            model_version=payload.model_version,
            session_id=payload.session_id,
            time_basis=payload.time_basis,
            processed_at=payload.processed_at,
        )
        session.add(sighting)
        session.flush()
        match = session.scalar(
            select(WatchlistEntry).where(WatchlistEntry.plate_hmac == payload.plate_hmac, WatchlistEntry.active.is_(True))
        )
        alert = None
        if match:
            alert = Alert(sighting_id=sighting.id, reason=match.reason, severity=match.severity)
            session.add(alert)
            session.flush()
        session.commit()
        if alert:
            session.refresh(alert)
            alert = session.scalar(
                select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).where(Alert.id == alert.id)
            )
        return sighting, alert, False
    except IntegrityError:
        session.rollback()
        existing = session.scalar(select(Sighting).where(Sighting.event_id == event_id))
        if not existing:
            raise
        alert = session.scalar(
            select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).where(Alert.sighting_id == existing.id)
        )
        return existing, alert, True


@app.get("/v1/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/cameras/sync", dependencies=[Depends(check_edge_key)])
def sync_cameras(payload: CameraSyncIn, session: Session = Depends(get_db)) -> dict:
    for data in payload.cameras:
        require_demo_for_synthetic(data.source_mode)
        camera = session.get(Camera, data.id)
        if camera is None:
            camera = Camera(id=data.id)
            session.add(camera)
        for name, value in data.model_dump().items():
            if name != "id" and (name in data.model_fields_set or name not in {
                "region", "latitude", "longitude", "coordinate_source", "coordinates_approximate"
            }):
                setattr(camera, name, value)
        camera.updated_at = utc_now()
    session.commit()
    return {"synced": len(payload.cameras)}


@app.get("/v1/cameras", dependencies=[Depends(check_operator_key)])
def list_cameras(session: Session = Depends(get_db)) -> list[dict]:
    cameras = session.scalars(select(Camera).order_by(Camera.id)).all()
    return [camera_dict(camera) for camera in cameras]


@app.get("/v1/catalogue/search", dependencies=[Depends(check_operator_key)])
def search_catalogue(q: str = Query(min_length=2, max_length=120),
                     offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=25)) -> dict:
    result = gateway_request("GET", "/search?" + urlencode({"q": q, "offset": offset, "limit": limit}))
    return CatalogueSearchOut.model_validate(result).model_dump()


@app.post("/v1/catalogue/{camera_id}/preview", dependencies=[Depends(check_operator_key)])
def start_catalogue_preview(camera_id: str) -> dict:
    return PreviewOut.model_validate(gateway_request("POST", "/previews", {"camera_id": camera_id})).model_dump()


@app.get("/v1/previews/{session_id}", dependencies=[Depends(check_operator_key)])
def catalogue_preview_status(session_id: str) -> dict:
    return PreviewOut.model_validate(gateway_request("GET", "/previews/" + quote(session_id, safe=""))).model_dump()


@app.post("/v1/previews/{session_id}/heartbeat", dependencies=[Depends(check_operator_key)])
def catalogue_preview_heartbeat(session_id: str) -> dict:
    return PreviewOut.model_validate(gateway_request("POST", "/previews/" + quote(session_id, safe="") + "/heartbeat")).model_dump()


@app.delete("/v1/previews/{session_id}", dependencies=[Depends(check_operator_key)])
def stop_catalogue_preview(session_id: str) -> dict:
    return gateway_request("DELETE", "/previews/" + quote(session_id, safe=""))


@app.post("/v1/sightings", dependencies=[Depends(check_edge_key)])
async def post_sighting(payload: SightingIn, session: Session = Depends(get_db)) -> dict:
    require_demo_for_synthetic(payload.source_mode)
    sighting, alert, duplicate = ingest_sighting(session, payload)
    serialized_alert = alert_dict(alert) if alert else None
    if alert and not duplicate:
        await sockets.broadcast({"type": "alert_created", "alert": serialized_alert})
    return {"accepted": True, "duplicate": duplicate, "sighting_id": sighting.id, "alert": serialized_alert}


@app.post("/v1/health", dependencies=[Depends(check_edge_key)])
async def post_health(payload: HealthIn, session: Session = Depends(get_db)) -> dict:
    require_demo_for_synthetic(payload.source_mode)
    existing = session.scalar(select(HealthReport).where(HealthReport.event_id == str(payload.event_id)))
    if existing:
        return {"accepted": True, "duplicate": True, "report_id": existing.id}
    camera = ensure_camera(session, payload.camera_id, payload.source_mode)
    report = HealthReport(**(payload.model_dump(mode="python") | {"event_id": str(payload.event_id)}))
    session.add(report)
    camera.status = payload.status
    camera.updated_at = payload.observed_at
    session.commit()
    await sockets.broadcast({"type": "camera_health", "camera": camera_dict(camera)})
    return {"accepted": True, "duplicate": False, "report_id": report.id}


@app.get("/v1/watchlist/snapshot", dependencies=[Depends(check_edge_key)])
def watchlist_snapshot(session: Session = Depends(get_db)) -> dict:
    entries = session.scalars(select(WatchlistEntry).where(WatchlistEntry.active.is_(True))).all()
    return {
        "version": watchlist_revision(session),
        "generated_at": utc_now().isoformat(),
        "entries": [{"plate_hmac": entry.plate_hmac, "reason": entry.reason, "severity": entry.severity} for entry in entries],
    }


@app.get("/v1/alerts", dependencies=[Depends(check_operator_key)])
def list_alerts(status: str | None = None, limit: int = Query(100, ge=1, le=500), session: Session = Depends(get_db)) -> list[dict]:
    query = select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).order_by(Alert.created_at.desc(), Alert.id.desc()).limit(limit)
    if status:
        query = query.where(Alert.status == status)
    return [alert_dict(item) for item in session.scalars(query).all()]


@app.patch("/v1/alerts/{alert_id}", dependencies=[Depends(check_operator_key)])
async def update_alert(alert_id: int, payload: AlertPatch, session: Session = Depends(get_db)) -> dict:
    alert = session.scalar(
        select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).where(Alert.id == alert_id)
    )
    if not alert:
        raise HTTPException(status_code=404, detail="alert not found")
    if alert.status in ("resolved", "false_positive"):
        raise HTTPException(status_code=409, detail="alert is closed")
    if alert.status == payload.status:
        return alert_dict(alert)
    if alert.status == "new" and payload.status == "resolved":
        raise HTTPException(status_code=409, detail="acknowledge alert before resolution")
    old = alert.status
    alert.status = payload.status
    alert.updated_at = utc_now()
    session.add(AlertAudit(alert_id=alert.id, from_status=old, to_status=payload.status, actor=payload.actor, note=payload.note))
    session.commit()
    session.refresh(alert)
    updated = session.scalar(
        select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).where(Alert.id == alert_id)
    )
    data = alert_dict(updated)
    await sockets.broadcast({"type": "alert_updated", "alert": data})
    return data


@app.post("/v1/routes/search", dependencies=[Depends(check_operator_key)])
def search_route(payload: RouteSearch, session: Session = Depends(get_db)) -> dict:
    try:
        token = plate_token(payload.plate)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    rows = session.execute(
        select(Sighting, Camera)
        .join(Camera, Camera.id == Sighting.camera_id)
        .where(Sighting.plate_hmac == token)
        .order_by(Sighting.observed_at, Sighting.id)
    ).all()
    return {
        "sightings": [
            {
                "event_id": sighting.event_id,
                "camera_id": sighting.camera_id,
                "camera_name": camera.name,
                "observed_at": as_utc(sighting.observed_at).isoformat(),
                "masked_plate": sighting.masked_plate,
                "confidence": sighting.confidence,
                "source_mode": sighting.source_mode,
                "stream_pts_ms": sighting.stream_pts_ms,
                "session_id": sighting.session_id,
                "time_basis": sighting.time_basis,
                "processed_at": as_utc(sighting.processed_at).isoformat() if sighting.processed_at else None,
                "latitude": camera.latitude,
                "longitude": camera.longitude,
                "coordinates_approximate": camera.coordinates_approximate,
            }
            for sighting, camera in rows
        ]
    }


@app.get("/v1/reports/hourly", dependencies=[Depends(check_operator_key)])
def hourly_reports(hours: int = Query(24, ge=1, le=168), session: Session = Depends(get_db)) -> dict:
    cutoff = utc_now() - timedelta(hours=hours)
    sightings = session.scalars(select(Sighting).where(Sighting.observed_at >= cutoff)).all()
    reports = session.scalars(select(HealthReport).where(HealthReport.observed_at >= cutoff)).all()
    buckets: dict[str, dict] = {}
    for item in sightings:
        start = as_utc(item.observed_at).replace(minute=0, second=0, microsecond=0).isoformat()
        bucket = buckets.setdefault(start, {"window_start": start, "sightings": 0, "alerts": 0, "health_reports": 0, "decoded_frames": 0, "inference_frames": 0, "reconnects": 0})
        bucket["sightings"] += 1
        bucket["alerts"] += int(item.alert is not None)
    for item in reports:
        start = as_utc(item.observed_at).replace(minute=0, second=0, microsecond=0).isoformat()
        bucket = buckets.setdefault(start, {"window_start": start, "sightings": 0, "alerts": 0, "health_reports": 0, "decoded_frames": 0, "inference_frames": 0, "reconnects": 0})
        bucket["health_reports"] += 1
        bucket["decoded_frames"] += item.decoded_frames
        bucket["inference_frames"] += item.inference_frames
        bucket["reconnects"] += item.reconnects
    return {"hours": hours, "buckets": [buckets[key] for key in sorted(buckets)]}


@app.get("/v1/reports/incidents/{alert_id}", dependencies=[Depends(check_operator_key)])
def incident_report(alert_id: int, format: str = "json", session: Session = Depends(get_db)):
    alert = session.scalar(
        select(Alert).options(selectinload(Alert.sighting), selectinload(Alert.audit)).where(Alert.id == alert_id)
    )
    if not alert:
        raise HTTPException(status_code=404, detail="alert not found")
    data = alert_dict(alert)
    data["event_id"] = alert.sighting.event_id
    data["edge_id"] = alert.sighting.edge_id
    data["model_version"] = alert.sighting.model_version
    data["synthetic_notice"] = "SYNTHETIC DEMONSTRATION EVENT" if alert.sighting.source_mode == "synthetic" else None
    data["recorded_notice"] = None
    if alert.sighting.source_mode == "recorded":
        data["recorded_notice"] = "RECORDED FOOTAGE · NOT A LIVE CAMERA" + (
            " · TIME SHOWN IS PROCESSING TIME" if alert.sighting.time_basis == "processing_time" else "")
    if format == "html":
        rows = "".join(f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>" for key, value in data.items() if key != "audit")
        actions = "".join(f"<li>{escape(entry['changed_at'])}: {escape(entry['from_status'])} → {escape(entry['to_status'])} ({escape(entry['actor'])})</li>" for entry in data["audit"])
        return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><title>Incident {alert_id}</title></head><body><h1>Incident {alert_id}</h1><table>{rows}</table><h2>Audit trail</h2><ol>{actions}</ol></body></html>")
    if format != "json":
        raise HTTPException(status_code=422, detail="format must be json or html")
    return data


@app.post("/v1/demo/inject", dependencies=[Depends(check_operator_key)])
async def inject_demo(payload: DemoInject, session: Session = Depends(get_db)) -> dict:
    if not settings.demo_mode:
        raise HTTPException(status_code=404, detail="demo mode disabled")
    token = plate_token("GJ01AB1234")
    sighting = SightingIn(
        schema_version="1.0",
        event_id=payload.event_id or uuid4(),
        edge_id="demo-injector",
        camera_id=payload.camera_id,
        observed_at=utc_now(),
        stream_pts_ms=None,
        track_id=None,
        masked_plate="GJ******34",
        plate_hmac=token,
        confidence=0.98,
        source_mode="synthetic",
        model_version="simulator-1",
    )
    item, alert, duplicate = ingest_sighting(session, sighting)
    serialized = alert_dict(alert) if alert else None
    if alert and not duplicate:
        await sockets.broadcast({"type": "alert_created", "alert": serialized})
    return {"accepted": True, "duplicate": duplicate, "sighting_id": item.id, "alert": serialized}


@app.websocket("/v1/ws/alerts")
async def alert_websocket(websocket: WebSocket) -> None:
    protocols = websocket.scope.get("subprotocols", [])
    keyed = (len(protocols) == 2 and protocols[0] == "sentinel-operator"
             and hmac.compare_digest(protocols[1], settings.operator_api_key))
    local = not protocols and local_session(websocket, require_origin=True) is not None
    if not keyed and not local:
        await websocket.close(code=1008)
        return
    await websocket.accept(subprotocol="sentinel-operator" if keyed else None)
    sockets.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sockets.clients.discard(websocket)

