"""Local-only access, non-plaintext watchlist writes and bounded diagnostic DTOs."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from pydantic import ValidationError

_temporary = tempfile.TemporaryDirectory()
os.environ.setdefault("DATABASE_URL", "sqlite:///" + _temporary.name + "/unified.db")
os.environ.setdefault("EDGE_API_KEY", "test-edge-key")
os.environ.setdefault("OPERATOR_API_KEY", "test-operator-key")
os.environ.setdefault("WATCHLIST_HMAC_KEY", "test-hmac-key")
os.environ.setdefault("DEMO_MODE", "true")

from app.main import app, settings
from app.db import SessionLocal, engine
from app.models import WatchlistEntry, WatchlistAudit, WatchlistRevision
from app import recognition, security
from app.config import Settings
from app.unified_contract import DiagnosticsSnapshot, WatchlistPage

ORIGIN = "http://127.0.0.1:5173"
KEY = {"X-Operator-Key": "test-operator-key"}
EDGE = {"X-Edge-Key": "test-edge-key"}


@pytest.fixture(scope="session", autouse=True)
def dispose_unified_database():
    yield
    engine.dispose()
    _temporary.cleanup()


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(settings, "local_dashboard_mode", True)
    monkeypatch.setattr(settings, "dashboard_recorded_evidence", True)
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    monkeypatch.setattr(recognition, "get_settings", lambda: settings)
    with TestClient(app, base_url=ORIGIN) as client:
        with SessionLocal() as session:
            session.execute(delete(WatchlistAudit))
            session.execute(delete(WatchlistEntry))
            session.execute(delete(WatchlistRevision))
            session.add(WatchlistRevision(id=1, version=1))
            session.commit()
        yield client


def bootstrap(client):
    response = client.post("/v1/local/session", headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert "test-operator-key" not in response.text
    return {"Origin": ORIGIN, "X-CSRF-Token": response.json()["csrf_token"]}


def test_local_bootstrap_and_no_manual_key(local):
    assert local.get("/v1/watchlist").status_code == 401
    bootstrap(local)
    response = local.get("/v1/watchlist")
    assert response.status_code == 200
    WatchlistPage.model_validate(response.json())
    assert response.headers["cache-control"].startswith("no-store")


def test_origin_host_csrf_and_expired_session(local, monkeypatch):
    for headers in ({}, {"Origin": "https://evil.example"}, {"Origin": ORIGIN, "Host": "evil.example"},
                    {"Origin": ORIGIN, "Sec-Fetch-Site": "cross-site"}):
        assert local.post("/v1/local/session", headers=headers).status_code == 403
    mutation = bootstrap(local)
    body = {"plate": "GJ02AB9988", "reason": "Synthetic test"}
    assert local.post("/v1/watchlist", json=body, headers={"Origin": ORIGIN}).status_code == 403
    assert local.post("/v1/watchlist", json=body, headers={**mutation, "Origin": "https://evil.example"}).status_code == 401
    assert local.get("/v1/watchlist", headers={"Host": "evil.example"}).status_code == 401
    clock = security.time.time()
    monkeypatch.setattr(security.time, "time", lambda: clock + security.SESSION_SECONDS + 1)
    assert local.get("/v1/watchlist").status_code == 401


def test_service_key_compatibility_and_local_config(local):
    assert local.get("/v1/watchlist", headers=KEY).status_code == 200
    assert local.get("/v1/watchlist", headers={"X-Operator-Key": "wrong"}).status_code == 401
    for extra in ({"local_dashboard_origins": "http://evil.example"}, {"local_dashboard_bind_host": "0.0.0.0"}):
        with pytest.raises(ValidationError):
            Settings(edge_api_key="test", operator_api_key="test", watchlist_hmac_key="test",
                     local_dashboard_mode=True, **extra)


def test_watchlist_lifecycle_revisions_and_no_plaintext(local):
    headers = bootstrap(local)
    plate = "GJ02AB9988"
    created = local.post("/v1/watchlist", json={"plate": plate, "reason": "Synthetic test"}, headers=headers)
    assert created.status_code == 201
    item = created.json()
    assert item["masked_plate"] == "GJ******88"
    assert plate not in created.text and "hmac" not in created.text
    assert local.post("/v1/watchlist", json={"plate": plate, "reason": "Duplicate"}, headers=headers).status_code == 409
    found = local.post("/v1/watchlist/search", json={"query": plate}, headers=headers)
    assert found.json()["total"] == 1
    before = found.json()["version"]
    unchanged = local.patch(f'/v1/watchlist/{item["id"]}', json={"expected_version": item["version"], "active": True}, headers=headers)
    assert unchanged.json()["version"] == item["version"]
    inactive = local.patch(f'/v1/watchlist/{item["id"]}', json={"expected_version": item["version"], "active": False}, headers=headers)
    assert inactive.status_code == 200 and inactive.json()["version"] > before
    assert local.patch(f'/v1/watchlist/{item["id"]}', json={"expected_version": item["version"], "reason": "Stale edit"}, headers=headers).status_code == 409
    snapshot = local.get("/v1/watchlist/snapshot", headers=EDGE).json()
    assert snapshot["entries"] == [] and snapshot["version"] == inactive.json()["version"]
    active = local.patch(f'/v1/watchlist/{item["id"]}', json={"expected_version": inactive.json()["version"], "active": True}, headers=headers)
    assert active.status_code == 200 and active.json()["version"] > inactive.json()["version"]
    with SessionLocal() as session:
        audit = session.scalars(select(WatchlistAudit).where(WatchlistAudit.entry_id == item["id"])).all()
        assert [entry.action for entry in audit] == ["create", "deactivate", "reactivate"]
        assert all(plate not in entry.changes for entry in audit)


def test_watchlist_legacy_and_validation_redaction(local):
    with SessionLocal() as session:
        session.add(WatchlistEntry(plate_hmac=security.plate_token("GJ03AB1122"), reason="Legacy", severity="high"))
        session.commit()
    assert local.get("/v1/watchlist", headers=KEY).json()["items"][0]["masked_plate"] is None
    secret = "PRIVATEPLATE123"
    response = local.post("/v1/watchlist", json={"plate": secret, "reason": 123}, headers=KEY)
    assert response.status_code == 422 and secret not in response.text and '"input"' not in response.text
    assert local.post("/v1/watchlist", json={"plate": "??", "reason": "Invalid"}, headers=KEY).status_code == 422
    assert local.post("/v1/watchlist", json={"plate": "GJ01AA2211", "reason": " "}, headers=KEY).status_code == 422


def test_cookie_websocket(local):
    bootstrap(local)
    with local.websocket_connect("ws://127.0.0.1:5173/v1/ws/alerts", headers={"Origin": ORIGIN}) as socket:
        socket.send_text("keepalive")
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with local.websocket_connect("ws://127.0.0.1:5173/v1/ws/alerts", headers={"Origin": "https://evil.example"}):
            pass


def diagnostic():
    return {"schema_version": "1.0", "source_mode": "recorded", "session_id": "s-test", "camera_id": "rec-test",
            "models": {"ocr_device": "gpu", "credential": "SECRET", "ocr_model": "rtsp://SECRET"},
            "stage_ms": {"ocr": 12, "secret": 45}, "secret": "SECRET",
            "track": {"track_id": "7", "best_reading": "GJ01AB1234", "gate_reasons": ["Need more frames", "https://SECRET"],
                      "frames": [{"pts_ms": n, "width": 100, "height": 30, "text": "GJ01", "image": "A" * 80004} for n in range(12)],
                      "candidates": [{"plate": "GJ01AB1234", "score": .9}] * 210}}


def test_diagnostics_contract_sanitization_and_stale(local, monkeypatch):
    calls = []
    value = diagnostic()
    monkeypatch.setattr(recognition, "request_edge", lambda method, path: (calls.append(path), value)[1])
    response = local.get("/v1/recognition/diagnostics?session_id=s-test&track_id=7", headers=KEY)
    assert response.status_code == 200
    result = DiagnosticsSnapshot.model_validate(response.json())
    assert len(result.track.frames) == 8 and len(result.track.candidates) == 200
    assert all(frame.image is None for frame in result.track.frames)
    assert "SECRET" not in response.text and "credential" not in response.text
    assert response.headers["cache-control"].startswith("no-store")
    value["session_id"] = "s-other"
    assert local.get("/v1/recognition/diagnostics?session_id=s-test&track_id=7", headers=KEY).status_code == 409
    value["session_id"] = "s-test"
    value["source_mode"] = "live"
    assert local.get("/v1/recognition/diagnostics?session_id=s-test&track_id=7", headers=KEY).status_code == 404
    monkeypatch.setattr(settings, "dashboard_recorded_evidence", False)
    prior = len(calls)
    assert local.get("/v1/recognition/diagnostics?session_id=s-test&track_id=7", headers=KEY).status_code == 404
    assert len(calls) == prior
