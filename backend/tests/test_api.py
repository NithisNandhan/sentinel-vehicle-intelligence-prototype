import os
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select


_database_dir = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = f"sqlite:///{_database_dir.name}/test.db"
os.environ["EDGE_API_KEY"] = "test-edge-key"
os.environ["OPERATOR_API_KEY"] = "test-operator-key"
os.environ["WATCHLIST_HMAC_KEY"] = "test-hmac-key"
os.environ["DEMO_MODE"] = "true"

from app.db import SessionLocal, engine  # noqa: E402
from app.main import app, settings  # noqa: E402
from app.models import Alert, AlertAudit, HealthReport, Sighting  # noqa: E402
from app.security import plate_token  # noqa: E402


EDGE = {"X-Edge-Key": "test-edge-key"}
OPERATOR = {"X-Operator-Key": "test-operator-key"}


@pytest.fixture(scope="session", autouse=True)
def close_test_database():
    yield
    engine.dispose()
    _database_dir.cleanup()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        with SessionLocal() as session:
            session.execute(delete(AlertAudit))
            session.execute(delete(Alert))
            session.execute(delete(HealthReport))
            session.execute(delete(Sighting))
            session.commit()
        yield test_client


def sighting(camera_id="cam01", when=None, plate="GJ01AB1234", event_id=None):
    return {
        "schema_version": "1.0",
        "event_id": str(event_id or uuid4()),
        "edge_id": "edge-test",
        "camera_id": camera_id,
        "observed_at": (when or datetime.now(timezone.utc)).isoformat(),
        "stream_pts_ms": 123,
        "track_id": "track-1",
        "masked_plate": "GJ01****34",
        "plate_hmac": plate_token(plate),
        "confidence": 0.91,
        "source_mode": "synthetic",
        "model_version": "test-model",
    }


def test_sighting_idempotency_and_watchlist_alert(client):
    data = sighting()
    first = client.post("/v1/sightings", json=data, headers=EDGE)
    assert first.status_code == 200, first.text
    assert first.json()["duplicate"] is False
    assert first.json()["alert"]["status"] == "new"
    second = client.post("/v1/sightings", json=data, headers=EDGE)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    alerts = client.get("/v1/alerts", headers=OPERATOR).json()
    assert len(alerts) == 1
    with SessionLocal() as session:
        assert session.scalar(select(Sighting).where(Sighting.event_id == data["event_id"])).plate_hmac == data["plate_hmac"]


def test_alert_audit_persists_and_incident_is_printable(client):
    created = client.post("/v1/demo/inject", json={"camera_id": "cam01"}, headers=OPERATOR).json()
    alert_id = created["alert"]["id"]
    ack = client.patch(f"/v1/alerts/{alert_id}", json={"status": "acknowledged", "actor": "tester"}, headers=OPERATOR)
    assert ack.status_code == 200
    closed = client.patch(f"/v1/alerts/{alert_id}", json={"status": "resolved", "actor": "tester", "note": "verified"}, headers=OPERATOR)
    assert closed.status_code == 200
    assert [row["to_status"] for row in closed.json()["audit"]] == ["acknowledged", "resolved"]
    report = client.get(f"/v1/reports/incidents/{alert_id}", headers=OPERATOR)
    assert report.json()["synthetic_notice"] == "SYNTHETIC DEMONSTRATION EVENT"
    printable = client.get(f"/v1/reports/incidents/{alert_id}?format=html", headers=OPERATOR)
    assert printable.status_code == 200 and "Incident" in printable.text
    assert "test-hmac-key" not in printable.text


def test_route_order_and_no_plaintext_storage(client):
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    later = sighting(camera_id="cam04", when=base + timedelta(minutes=5))
    earlier = sighting(camera_id="cam01", when=base)
    assert client.post("/v1/sightings", json=later, headers=EDGE).status_code == 200
    assert client.post("/v1/sightings", json=earlier, headers=EDGE).status_code == 200
    route = client.post("/v1/routes/search", json={"plate": "GJ01 AB 1234"}, headers=OPERATOR)
    assert route.status_code == 200
    assert [item["camera_id"] for item in route.json()["sightings"]] == ["cam01", "cam04"]
    assert "GJ01AB1234" not in route.text
    assert all(item["coordinates_approximate"] for item in route.json()["sightings"])


def test_health_and_hourly_report(client):
    event_id = str(uuid4())
    payload = {
        "event_id": event_id,
        "edge_id": "edge-test",
        "camera_id": "cam01",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "ONLINE",
        "decoded_frames": 120,
        "inference_frames": 20,
        "reconnects": 1,
        "outbox_depth": 0,
        "source_mode": "synthetic",
    }
    first = client.post("/v1/health", json=payload, headers=EDGE)
    assert first.status_code == 200, first.text
    assert client.post("/v1/health", json=payload, headers=EDGE).json()["duplicate"] is True
    camera = next(row for row in client.get("/v1/cameras", headers=OPERATOR).json() if row["id"] == "cam01")
    assert camera["status"] == "ONLINE"
    buckets = client.get("/v1/reports/hourly", headers=OPERATOR).json()["buckets"]
    assert sum(row["health_reports"] for row in buckets) == 1


def test_camera_sync_preserves_approximate_demo_coordinates(client):
    before = next(row for row in client.get("/v1/cameras", headers=OPERATOR).json() if row["id"] == "cam01")
    synced = client.post("/v1/cameras/sync", headers=EDGE, json={"cameras": [{
        "id": "cam01", "name": "Verified live name", "status": "ONLINE", "source_mode": "live",
        "preview_url": "http://localhost:8888/cam01/index.m3u8",
    }]})
    assert synced.status_code == 200, synced.text
    after = next(row for row in client.get("/v1/cameras", headers=OPERATOR).json() if row["id"] == "cam01")
    assert (after["latitude"], after["longitude"]) == (before["latitude"], before["longitude"])
    assert after["coordinate_source"] == "demo_approximate"
    assert after["source_mode"] == "live"


def test_catalogue_search_requires_operator_and_strips_stream_urls(client, monkeypatch):
    calls = []
    def fake_gateway(method, path, payload=None):
        calls.append((method, path, payload))
        return {"total": 1, "offset": 0, "limit": 25, "items": [{
            "id": "cam03", "name": "Janpath North", "location": "Janpath", "codec": "h265",
            "live": True, "rtsp_url": "rtsp://secret.invalid/private",
        }]}
    monkeypatch.setattr("app.main.gateway_request", fake_gateway)
    assert client.get("/v1/catalogue/search?q=Janpath").status_code == 401
    response = client.get("/v1/catalogue/search?q=Janpath", headers=OPERATOR)
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["location"] == "Janpath"
    assert "rtsp_url" not in response.text
    assert "secret.invalid" not in response.text
    assert calls[0][0] == "GET"


def test_catalogue_preview_is_operator_protected(client, monkeypatch):
    session_id = "a" * 32
    def fake_gateway(method, path, payload=None):
        if method == "DELETE":
            return {"stopped": True}
        return {"id": session_id, "camera_id": "cam03", "status": "ready",
                "preview_url": f"/media/preview-{session_id}/index.m3u8",
                "rtsp_url": "rtsp://secret.invalid/private"}
    monkeypatch.setattr("app.main.gateway_request", fake_gateway)
    assert client.post("/v1/catalogue/cam03/preview").status_code == 401
    started = client.post("/v1/catalogue/cam03/preview", headers=OPERATOR)
    assert started.status_code == 200, started.text
    assert "rtsp_url" not in started.text
    assert client.get(f"/v1/previews/{session_id}", headers=OPERATOR).json()["status"] == "ready"
    assert client.post(f"/v1/previews/{session_id}/heartbeat", headers=OPERATOR).status_code == 200
    assert client.delete(f"/v1/previews/{session_id}", headers=OPERATOR).json() == {"stopped": True}


def test_catalogue_search_is_unavailable_without_gateway(client):
    response = client.get("/v1/catalogue/search?q=cam01", headers=OPERATOR)
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_output_timestamps_remain_utc_with_sqlite(client):
    created = client.post("/v1/sightings", json=sighting(), headers=EDGE)
    assert created.status_code == 200, created.text
    alert = created.json()["alert"]
    assert alert["observed_at"].endswith(("+00:00", "Z"))
    assert alert["created_at"].endswith(("+00:00", "Z"))
    route = client.post("/v1/routes/search", json={"plate": "GJ01AB1234"}, headers=OPERATOR).json()
    assert all(row["observed_at"].endswith(("+00:00", "Z")) for row in route["sightings"])
    buckets = client.get("/v1/reports/hourly", headers=OPERATOR).json()["buckets"]
    assert all(row["window_start"].endswith(("+00:00", "Z")) for row in buckets)


def test_synthetic_ingestion_disabled_outside_demo_mode(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    assert client.post("/v1/sightings", json=sighting(), headers=EDGE).status_code == 403
    assert client.post("/v1/cameras/sync", headers=EDGE, json={"cameras": [
        {"id": "cam01", "source_mode": "synthetic"}
    ]}).status_code == 403
    assert client.post("/v1/demo/inject", json={}, headers=OPERATOR).status_code == 404


@pytest.mark.parametrize("path,method,body", [
    ("/v1/cameras", "get", None),
    ("/v1/alerts", "get", None),
    ("/v1/reports/hourly", "get", None),
    ("/v1/routes/search", "post", {"plate": "GJ01AB1234"}),
    ("/v1/demo/inject", "post", {}),
])
def test_operator_endpoints_reject_missing_or_wrong_key(client, path, method, body):
    call = getattr(client, method)
    assert call(path, **({"json": body} if body is not None else {})).status_code == 401
    assert call(path, headers={"X-Operator-Key": "wrong"}, **({"json": body} if body is not None else {})).status_code == 401


def test_edge_endpoint_rejects_missing_key_and_invalid_timestamp(client):
    data = sighting()
    assert client.post("/v1/sightings", json=data).status_code == 401
    data["observed_at"] = "2026-01-01T10:00:00"
    assert client.post("/v1/sightings", json=data, headers=EDGE).status_code == 422


def test_websocket_operator_subprotocol(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/ws/alerts"):
            pass
    with client.websocket_connect("/v1/ws/alerts", subprotocols=["sentinel-operator", "test-operator-key"]) as websocket:
        assert websocket.accepted_subprotocol == "sentinel-operator"


def recorded_sighting(**overrides):
    data = sighting(camera_id="rec-0123456789ab")
    data.update(source_mode="recorded", session_id="s-20261010-a1b2c3", time_basis="processing_time",
                stream_pts_ms=4500, processed_at="2026-10-10T16:30:00Z")
    data.update(overrides)
    return data


def test_recorded_sighting_keeps_its_label_through_alert_route_and_report(client):
    data = recorded_sighting()
    first = client.post("/v1/sightings", json=data, headers=EDGE)
    assert first.status_code == 200, first.text
    alert = first.json()["alert"]
    assert alert["source_mode"] == "recorded"
    assert alert["session_id"] == "s-20261010-a1b2c3"
    assert alert["time_basis"] == "processing_time"
    assert alert["stream_pts_ms"] == 4500
    assert alert["processed_at"].startswith("2026-10-10T16:30:00")
    assert alert["processed_at"].endswith(("+00:00", "Z"))
    assert client.post("/v1/sightings", json=data, headers=EDGE).json()["duplicate"] is True
    assert len(client.get("/v1/alerts", headers=OPERATOR).json()) == 1
    cameras = {camera["id"]: camera for camera in client.get("/v1/cameras", headers=OPERATOR).json()}
    assert cameras["rec-0123456789ab"]["source_mode"] == "recorded"
    route = client.post("/v1/routes/search", json={"plate": "GJ01AB1234"}, headers=OPERATOR).json()["sightings"]
    assert [(row["source_mode"], row["session_id"], row["stream_pts_ms"]) for row in route] == [
        ("recorded", "s-20261010-a1b2c3", 4500)]
    report = client.get(f"/v1/reports/incidents/{alert['id']}", headers=OPERATOR).json()
    assert report["source_mode"] == "recorded"
    assert report["synthetic_notice"] is None
    assert "RECORDED FOOTAGE" in report["recorded_notice"] and "PROCESSING TIME" in report["recorded_notice"]
    assert "RECORDED FOOTAGE" in client.get(f"/v1/reports/incidents/{alert['id']}?format=html", headers=OPERATOR).text


def test_recorded_sighting_from_a_new_session_is_a_separate_traceable_event(client):
    first = client.post("/v1/sightings", json=recorded_sighting(), headers=EDGE).json()
    replay = client.post("/v1/sightings", json=recorded_sighting(session_id="s-20261010-ffff00"), headers=EDGE).json()
    assert replay["duplicate"] is False and replay["sighting_id"] != first["sighting_id"]
    with SessionLocal() as session:
        assert sorted(session.scalars(select(Sighting.session_id)).all()) == ["s-20261010-a1b2c3", "s-20261010-ffff00"]


@pytest.mark.parametrize("missing", ["session_id", "time_basis", "stream_pts_ms", "processed_at"])
def test_recorded_sighting_without_traceability_is_rejected(client, missing):
    data = recorded_sighting()
    data[missing] = None
    assert client.post("/v1/sightings", json=data, headers=EDGE).status_code == 422


@pytest.mark.parametrize("field,value", [("time_basis", "arrival_time"), ("session_id", "has space"),
                                         ("processed_at", "2026-10-10T22:00:00+05:30"),
                                         ("source_mode", "downloaded")])
def test_recorded_fields_are_strictly_validated(client, field, value):
    assert client.post("/v1/sightings", json=recorded_sighting(**{field: value}), headers=EDGE).status_code == 422


def test_existing_payloads_remain_valid_without_recorded_fields(client):
    response = client.post("/v1/sightings", json=sighting(), headers=EDGE)
    assert response.status_code == 200
    alert = response.json()["alert"]
    assert alert["session_id"] is None and alert["time_basis"] is None
    assert client.get(f"/v1/reports/incidents/{alert['id']}", headers=OPERATOR).json()["recorded_notice"] is None


def test_migration_adds_recorded_columns_to_an_existing_0001_database(tmp_path):
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path
    database = tmp_path / "old.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE sightings (id INTEGER PRIMARY KEY, event_id VARCHAR(36), source_mode VARCHAR(16))")
        connection.execute("INSERT INTO sightings (event_id, source_mode) VALUES ('kept', 'live')")
        connection.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        connection.execute("INSERT INTO alembic_version VALUES ('0001_initial')")
        connection.commit()
    finally:
        connection.close()
    environment = dict(os.environ, DATABASE_URL=f"sqlite:///{database.as_posix()}")
    for _ in range(2):  # Upgrading twice must be harmless.
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    connection = sqlite3.connect(database)
    try:
        assert {"session_id", "time_basis", "processed_at"} <= {
            row[1] for row in connection.execute("PRAGMA table_info(sightings)")}
        assert connection.execute("SELECT event_id, session_id FROM sightings").fetchall() == [("kept", None)]
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [("0003_watchlist_management",)]
    finally:
        connection.close()

