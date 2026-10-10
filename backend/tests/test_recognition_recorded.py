"""Operator proxy for a recorded session: labels pass, media and paths never do."""
import os

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


def processing():
    return {"status": "processing", "camera_id": "rec-0123456789ab", "source_mode": "recorded",
            "recognition_state": "reading", "session_id": "s-20261010T101500-aaaaaaaa",
            "recording_name": "Junction A", "video_position_ms": 1500, "video_duration_ms": 6000,
            "time_basis": "processing_time", "pacing": "unpaced", "stale_after_seconds": 30,
            "frame_age_seconds": 2.5, "analysed_frames": 4, "decoded_frames": 16, "plate_crops": 3,
            "vehicle_boxes": 1, "model_profile": "general", "inference_device": "cpu",
            "tracks": [{"track_id": "7", "observations": 3, "masked_plate": "AP******34",
                        "recognition_state": "model_accepted", "candidate_count": 2}]}


def test_recorded_status_is_labelled_and_allowlisted(client, monkeypatch):
    value = processing()
    value.update(frame="SECRET-FRAME", crop="SECRET-CROP", path="/recordings/SECRET.mp4",
                 debug="SECRET", plate="AP01AB1234", sha256="SECRET")
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    assert response.status_code == 200
    data = response.json()
    assert (data["source_mode"], data["status"]) == ("recorded", "processing")
    assert data["session_id"] == "s-20261010T101500-aaaaaaaa"
    assert (data["video_position_ms"], data["video_duration_ms"]) == (1500, 6000)
    assert (data["recording_name"], data["time_basis"], data["pacing"]) == ("Junction A", "processing_time", "unpaced")
    assert (data["inference_device"], data["stale_after_seconds"]) == ("cpu", 30)
    assert data["tracks"][0]["masked_plate"] == "AP******34"
    assert "SECRET" not in response.text and "AP01AB1234" not in response.text and "/recordings" not in response.text


@pytest.mark.parametrize("status", ["connecting", "processing", "paused", "completed", "unavailable"])
def test_every_recorded_state_passes(client, monkeypatch, status):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: dict(processing(), status=status))
    data = client.get("/v1/recognition/status", headers=OPERATOR).json()
    assert (data["status"], data["source_mode"]) == (status, "recorded")


def test_malformed_recorded_fields_are_dropped_not_forwarded(client, monkeypatch):
    value = processing()
    value.update(session_id="has space SECRET", recording_name="https://SECRET/clip", video_position_ms=-5,
                 video_duration_ms=True, time_basis="SECRET", pacing=["SECRET"], stale_after_seconds=10_000,
                 inference_device="SECRET")
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    data = response.json()
    assert data["session_id"] is None and data["recording_name"] == "rec-0123456789ab"
    assert data["video_position_ms"] is None and data["video_duration_ms"] is None
    assert data["time_basis"] is None and data["pacing"] is None and data["inference_device"] is None
    assert data["stale_after_seconds"] == 30
    assert "SECRET" not in response.text


def test_failure_code_is_a_bare_name_and_only_when_unavailable(client, monkeypatch):
    value = dict(processing(), status="unavailable", failure_code="ChangedRecording")
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    assert client.get("/v1/recognition/status", headers=OPERATOR).json()["failure_code"] == "ChangedRecording"
    value["failure_code"] = "open /recordings/SECRET.mp4 failed"
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    assert "failure_code" not in response.json() and "SECRET" not in response.text
    value.update(status="processing", failure_code="ChangedRecording")
    assert "failure_code" not in client.get("/v1/recognition/status", headers=OPERATOR).json()


def test_live_status_never_carries_recorded_fields(client, monkeypatch):
    value = {"status": "live", "camera_id": "cam17", "session_id": "s-leak", "recording_name": "leak",
             "video_position_ms": 5, "frame_age_seconds": 0.1}
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    data = client.get("/v1/recognition/status", headers=OPERATOR).json()
    assert data["source_mode"] == "live"
    for field in ("session_id", "recording_name", "video_position_ms", "pacing"):
        assert field not in data


def test_vehicles_seen_list_is_allowlisted_and_masked(client, monkeypatch):
    value = processing()
    value["vehicles_seen"] = [
        {"track_id": "5", "vehicle_class": "car", "confidence": 0.39, "first_ms": 1000, "last_ms": 2500,
         "observations": 3, "state": "model_accepted", "masked_plate": "TS******29", "watchlist": "possible_match",
         "reading": "TS07FS8429", "thumb": "SECRET-PIXELS", "key": "0-5"},
        {"track_id": "6", "vehicle_class": "spaceship", "confidence": 7, "first_ms": -5, "last_ms": "x",
         "observations": True, "state": "uncertain", "masked_plate": "TS07FS8429", "watchlist": "possible_match"},
        {"track_id": "https://SECRET"}, "SECRET",
    ] + [{"track_id": str(i)} for i in range(100, 180)]
    monkeypatch.setattr(recognition, "request_edge", lambda *args: value)
    response = client.get("/v1/recognition/status", headers=OPERATOR)
    rows = response.json()["vehicles_seen"]
    assert len(rows) == 48   # Only the first 50 upstream rows are considered; the two invalid ones are dropped.
    assert rows[0] == {"track_id": "5", "vehicle_class": "car", "confidence": 0.39, "first_ms": 1000, "last_ms": 2500,
                       "observations": 3, "state": "model_accepted", "masked_plate": "TS******29",
                       "watchlist": "possible_match"}
    # Unknown class, impossible numbers and an unmasked plate are replaced; an unaccepted row carries no flag.
    assert rows[1] == {"track_id": "6", "vehicle_class": "vehicle", "confidence": None, "first_ms": None,
                       "last_ms": None, "observations": 0, "state": "uncertain", "masked_plate": None, "watchlist": None}
    assert "TS07FS8429" not in response.text and "SECRET" not in response.text


def evidence_payload():
    return {"source_mode": "recorded", "session_id": "s-20261010T101500-aaaaaaaa", "video_position_ms": 1500,
            "frame": "QUJD", "path": "/recordings/SECRET.mp4",
            "vehicles": [{"track_id": "5", "vehicle_class": "car", "confidence": 0.9, "first_ms": 0, "last_ms": 1500,
                          "observations": 3, "state": "uncertain", "reading": "TS07FS8429", "thumb": "QUJD",
                          "watchlist": "no_match", "candidates": ["SECRET"]},
                         {"track_id": "6", "reading": "rtsp://SECRET", "thumb": "<script>SECRET", "state": "x"}]}


def test_evidence_view_is_off_unless_explicitly_enabled(client, monkeypatch):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: pytest.fail("evidence requested while disabled"))
    assert client.get("/v1/recognition/evidence", headers=OPERATOR).status_code == 404
    assert client.get("/v1/recognition/evidence").status_code == 401


def test_evidence_view_for_a_recording_passes_only_validated_fields(client, monkeypatch):
    recognition.get_settings().dashboard_recorded_evidence = True
    monkeypatch.setattr(recognition, "request_edge", lambda *args: evidence_payload())
    response = client.get("/v1/recognition/evidence", headers=OPERATOR)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store, max-age=0"
    data = response.json()
    assert data["frame"] == "QUJD" and data["video_position_ms"] == 1500
    assert data["vehicles"][0]["reading"] == "TS07FS8429" and data["vehicles"][0]["thumb"] == "QUJD"
    assert data["vehicles"][1] == {"track_id": "6", "vehicle_class": "vehicle", "confidence": None, "first_ms": None,
                                   "last_ms": None, "observations": 0, "state": "unreadable", "reading": None,
                                   "thumb": None, "watchlist": None}
    assert "SECRET" not in response.text and "path" not in data


def test_evidence_view_never_serves_a_live_camera(client, monkeypatch):
    recognition.get_settings().dashboard_recorded_evidence = True
    monkeypatch.setattr(recognition, "request_edge", lambda *args: dict(evidence_payload(), source_mode="live"))
    response = client.get("/v1/recognition/evidence", headers=OPERATOR)
    assert response.status_code == 404 and "QUJD" not in response.text


def test_recordings_list_has_names_and_durations_only(client, monkeypatch):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: {
        "available": False, "scope": "unavailable", "cameras": [], "recordings_rejected": [{"file": "SECRET.mp4"}],
        "recordings": [{"id": "rec-0123456789ab", "name": "Junction A", "duration_ms": 6000,
                        "time_basis": "source_clock", "path": "/recordings/SECRET.mp4", "sha256": "SECRET"},
                       {"id": "rec-ffffffffffff", "name": "file:///SECRET", "duration_ms": "long"},
                       {"id": "../SECRET", "name": "bad"}, "SECRET"]})
    response = client.get("/v1/recognition/recordings", headers=OPERATOR)
    assert response.json() == {"recordings": [
        {"id": "rec-0123456789ab", "name": "Junction A", "duration_ms": 6000, "time_basis": "source_clock"},
        {"id": "rec-ffffffffffff", "name": "rec-ffffffffffff", "duration_ms": None, "time_basis": "processing_time"}]}
    assert "SECRET" not in response.text
    monkeypatch.setattr(recognition, "request_edge", lambda *args: {"available": True, "cameras": []})
    assert client.get("/v1/recognition/recordings", headers=OPERATOR).json() == {"recordings": []}


@pytest.mark.parametrize("body,forwarded", [
    ({"action": "pause"}, {"action": "pause"}), ({"action": "resume"}, {"action": "resume"}),
    ({"action": "restart"}, {"action": "restart"}),
    ({"action": "seek", "position_ms": 4000}, {"action": "seek", "position_ms": 4000}),
])
def test_control_forwards_only_the_validated_action(client, monkeypatch, body, forwarded):
    calls = []
    def fake(*args):
        calls.append(args)
        return processing()
    monkeypatch.setattr(recognition, "request_edge", fake)
    response = client.post("/v1/recognition/control", json=body, headers=OPERATOR)
    assert response.status_code == 200 and response.json()["source_mode"] == "recorded"
    assert calls == [("POST", "/internal/control", forwarded)]


@pytest.mark.parametrize("body", [
    {}, {"action": "rewind"}, {"action": "seek"}, {"action": "pause", "position_ms": 5},
    {"action": "seek", "position_ms": -1}, {"action": "seek", "position_ms": 1.5},
    {"action": "seek", "position_ms": "4000"}, {"action": "restart", "url": "rtsp://SECRET"},
])
def test_invalid_control_is_rejected_before_reaching_the_edge(client, monkeypatch, body):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: pytest.fail("invalid upstream request"))
    assert client.post("/v1/recognition/control", json=body, headers=OPERATOR).status_code == 422


def test_control_and_recordings_need_the_operator_key(client, monkeypatch):
    monkeypatch.setattr(recognition, "request_edge", lambda *args: pytest.fail("unauthorised proxy access"))
    assert client.get("/v1/recognition/recordings").status_code == 401
    assert client.post("/v1/recognition/control", json={"action": "pause"},
                       headers={"X-Edge-Key": "test-edge-key"}).status_code == 401


def test_control_conflict_stays_visible_and_redacted(client, monkeypatch):
    import urllib.error

    class Opener:
        def open(self, *args, **kwargs):
            raise urllib.error.HTTPError("http://SECRET", 409, "SECRET", {}, None)
    monkeypatch.setattr(recognition.urllib.request, "build_opener", lambda *args: Opener())
    response = client.post("/v1/recognition/control", json={"action": "pause"}, headers=OPERATOR)
    assert response.status_code == 409 and "SECRET" not in response.text
    assert "Playback control" in response.json()["detail"]
