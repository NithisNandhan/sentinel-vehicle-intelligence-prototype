"""Standalone router tests avoid database creation and real network access."""
import io
import json
import os
import urllib.error
from email.message import Message

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("EDGE_API_KEY", "test-edge-key")
os.environ.setdefault("OPERATOR_API_KEY", "test-operator-key")
os.environ.setdefault("WATCHLIST_HMAC_KEY", "test-hmac-key")

from app import recognition  # noqa: E402
from app.config import Settings  # noqa: E402

OPERATOR = {"X-Operator-Key": "test-operator-key"}


@pytest.fixture
def client(monkeypatch):
    settings = Settings(edge_api_key="test-edge-key", operator_api_key="test-operator-key",
                        watchlist_hmac_key="test-hmac-key", recognition_gateway_url="http://edge:7780")
    monkeypatch.setattr(recognition, "get_settings", lambda: settings)
    import app.security
    monkeypatch.setattr(app.security, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(recognition.router)
    with TestClient(app) as test_client:
        yield test_client


def live():
    return {"status": "live", "camera_id": "cam17", "recognition_state": "uncertain",
            "vehicle_boxes": 4, "decoded_frames": 60, "analysed_frames": 5, "plate_crops": 2,
            "resolution": [1920, 1080], "model_profile": "general", "frame_age_seconds": 0.2,
            "tracks": [{"track_id": "track-1", "observations": 3,
                        "masked_plate": "MH02****37", "recognition_state": "uncertain", "candidate_count": 9}]}


@pytest.mark.parametrize("method,path,payload", [
    ("get", "status", None), ("get", "cameras", None),
    ("post", "session", {"camera_id": "cam17"}),
    ("post", "heartbeat", {"camera_id": "cam17"}),
])
def test_operator_authentication(client, monkeypatch, method, path, payload):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: pytest.fail("unauthorised proxy access"))
    for headers in ({}, {"X-Operator-Key": "wrong"}, {"X-Edge-Key": "test-edge-key"}):
        response = getattr(client, method)(f"/v1/recognition/{path}", headers=headers,
                                           **({"json": payload} if payload is not None else {}))
        assert response.status_code == 401


def test_whitelisted_metadata_rejects_plaintext_and_media(client, monkeypatch):
    value = live()
    value.update(plate="MH02EK0837", frame="SECRET-FRAME", crop="SECRET-CROP",
                 stream_url="rtsp://credential@host", rawrows=["SECRET"], debug="SECRET")
    value["tracks"][0].update(plate="MH02EK0837", candidates=["MH02EK0837"], image="SECRET")
    value["tracks"].append({"track_id": "track-2", "masked_plate": "MH02EK0837"})
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    assert response.status_code == 200
    data = response.json()
    assert data["tracks"][0]["masked_plate"] == "MH02****37"
    assert data["tracks"][1]["masked_plate"] is None
    assert data["source_mode"] == "live"
    assert "MH02EK0837" not in response.text
    assert "SECRET" not in response.text and "rtsp" not in response.text
    assert data["vehicle_boxes"] == 4


def test_tracks_bounded_and_malformed_fields_safe(client, monkeypatch):
    value = live()
    value.update(vehicle_boxes=True, frame_age_seconds=float("nan"), resolution=[-1, 5],
                 model_profile="rtsp://secret@host")
    value["tracks"] = [{"track_id": str(i), "recognition_state": [], "candidate_count": 999,
                        "masked_plate": "MH02EK0837***"} for i in range(50)]
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    data = client.get("/v1/recognition/status", headers=OPERATOR).json()
    assert len(data["tracks"]) == 32 and data["vehicle_boxes"] == 0
    assert "frame_age_seconds" not in data and "resolution" not in data
    assert data["model_profile"] is None
    assert data["tracks"][0]["candidate_count"] == 200
    assert data["tracks"][0]["masked_plate"] is None


def test_cameras_never_expose_urls(client, monkeypatch):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: {
        "available": True, "password": "SECRET", "cameras": [
            {"id": "cam17", "name": "A crossing", "stream_url": "rtsp://SECRET"},
            {"id": "cam18", "name": "https://SECRET"}, {"id": "https://SECRET", "name": "bad"}]})
    response = client.get("/v1/recognition/cameras", headers=OPERATOR)
    assert response.json() == {"available": True, "scope": "authorised_catalogue", "cameras": [
        {"id": "cam17", "name": "A crossing"}, {"id": "cam18", "name": "cam18"}]}
    assert "SECRET" not in response.text


def test_configured_relay_scope_preserved_without_urls(client, monkeypatch):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: {
        "available": True, "scope": "configured_relays", "cameras": [
            {"id": "cam01", "name": "Chiman bhai Bridge", "url": "rtsp://SECRET"},
            {"id": "cam02", "name": "Janpath", "source_url": "rtsp://SECRET"}]})
    response = client.get("/v1/recognition/cameras", headers=OPERATOR)
    assert response.status_code == 200
    assert response.json()["scope"] == "configured_relays"
    assert len(response.json()["cameras"]) == 2 and "SECRET" not in response.text


@pytest.mark.parametrize("available,scope", [(True, "fake"), (True, []),
                                           (True, "unavailable"), (False, "configured_relays")])
def test_invalid_camera_scope_unavailable(client, monkeypatch, available, scope):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: {
        "available": available, "scope": scope, "cameras": []})
    response = client.get("/v1/recognition/cameras", headers=OPERATOR)
    assert response.status_code == 503


def test_unavailable_camera_scope_never_returns_stale_rows(client, monkeypatch):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: {
        "available": False, "cameras": [{"id": "cam01", "name": "Stale"}]})
    assert client.get("/v1/recognition/cameras", headers=OPERATOR).json() == {
        "available": False, "scope": "unavailable", "cameras": []}


@pytest.mark.parametrize("path,payload", [
    ("session", {}), ("session", {"camera_id": 17}),
    ("session", {"camera_id": "https://SECRET"}),
    ("session", {"camera_id": "cam17", "url": "rtsp://SECRET"}),
    ("heartbeat", {}), ("heartbeat", {"camera_id": None}),
])
def test_invalid_mutation_payloads(client, monkeypatch, path, payload):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: pytest.fail("invalid upstream request"))
    assert client.post(f"/v1/recognition/{path}", json=payload, headers=OPERATOR).status_code == 422


@pytest.mark.parametrize("path,camera_id", [("session", "cam17"), ("session", None), ("heartbeat", "cam17")])
def test_fixed_mutation_paths(client, monkeypatch, path, camera_id):
    calls = []
    def fake(*args):
        calls.append(args)
        return live()
    monkeypatch.setattr(recognition, "request_edge", fake)
    response = client.post(f"/v1/recognition/{path}", json={"camera_id": camera_id}, headers=OPERATOR)
    assert response.status_code == 200
    assert calls == [("POST", f"/internal/{path}", {"camera_id": camera_id})]


class Response(io.BytesIO):
    def __init__(self, content, content_type="application/json"):
        super().__init__(content)
        self.headers = Message()
        self.headers["Content-Type"] = content_type


@pytest.mark.parametrize("content,content_type", [
    (b"<html>SECRET</html>", "text/html"), (b"not json SECRET", "application/json"),
    (b"[]", "application/json"), (b"x" * 1_048_577, "application/json"),
], ids=["html", "malformed", "non-object", "oversize"])  # Windows cannot hold a 1 MiB test ID
def test_html_malformed_and_oversize_unavailable(client, monkeypatch, content, content_type):
    class Opener:
        def open(self, *args, **kwargs):
            return Response(content, content_type)
    monkeypatch.setattr(recognition.urllib.request, "build_opener", lambda *args: Opener())
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    assert response.status_code == 503 and "SECRET" not in response.text


@pytest.mark.parametrize("error,expected", [
    (urllib.error.URLError("rtsp://SECRET"), 503), (TimeoutError("SECRET"), 503),
    (urllib.error.HTTPError("http://SECRET", 403, "SECRET", {}, None), 503),
    (urllib.error.HTTPError("http://SECRET", 302, "SECRET", {}, None), 503),
    (urllib.error.HTTPError("http://SECRET", 409, "SECRET", {}, None), 409),
])
def test_errors_redacted(client, monkeypatch, error, expected):
    class Opener:
        def open(self, *args, **kwargs):
            raise error
    monkeypatch.setattr(recognition.urllib.request, "build_opener", lambda *args: Opener())
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    assert response.status_code == expected and "SECRET" not in response.text


def test_transport_auth_timeouts_and_no_redirect(client, monkeypatch):
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return Response(json.dumps(live()).encode())
    def opener(handler):
        assert isinstance(handler, recognition._NoRedirect)
        assert handler.redirect_request(None, None, 302, "", {}, "http://evil") is None
        return Opener()
    monkeypatch.setattr(recognition.urllib.request, "build_opener", opener)
    assert client.get("/v1/recognition/status", headers=OPERATOR).status_code == 200
    assert client.post("/v1/recognition/session", json={"camera_id": "cam17"}, headers=OPERATOR).status_code == 200
    assert calls[0][1] == 3 and calls[1][1] == 5
    assert calls[0][0].full_url == "http://edge:7780/internal/status"
    assert calls[0][0].get_header("X-edge-key") == "test-edge-key"


def test_unconfigured_gateway(client, monkeypatch):
    recognition.get_settings().recognition_gateway_url = ""
    assert client.get("/v1/recognition/status", headers=OPERATOR).status_code == 503


@pytest.mark.parametrize("value", [{"status": []}, {"status": "live", "recognition_state": {}},
                                   {"status": "SECRET"},
                                   {"status": "processing"},  # A camera is never 'processing'.
                                   {"status": "completed", "source_mode": "live"},
                                   {"status": "live", "source_mode": "recorded"}])
def test_invalid_upstream_status(client, monkeypatch, value):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    assert response.status_code == 503 and "SECRET" not in response.text
