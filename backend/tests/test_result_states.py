"""Result-state ingestion and the review workflow. Plates are SYNTHETIC."""
import hashlib
import os
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import uuid4

_database_dir = tempfile.TemporaryDirectory()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_database_dir.name}/results.db")
os.environ.setdefault("EDGE_API_KEY", "test-edge-key")
os.environ.setdefault("OPERATOR_API_KEY", "test-operator-key")
os.environ.setdefault("WATCHLIST_HMAC_KEY", "test-hmac-key")
os.environ.setdefault("DEMO_MODE", "true")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (Alert, AlertAudit, RecognitionResult, ReviewDecision, Sighting,  # noqa: E402
                        WatchlistEntry)
from app.review import purge_expired  # noqa: E402
from app.security import plate_token  # noqa: E402

EDGE = {"X-Edge-Key": "test-edge-key"}
OPERATOR = {"X-Operator-Key": "test-operator-key"}
REVIEWER_TOKEN = "reviewer-secret-token"
REVIEWER = dict(OPERATOR, **{"X-Reviewer-Token": REVIEWER_TOKEN})
PLATE = "MH49BE5557"


@pytest.fixture
def client(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "reviewer_tokens", "asha:" + hashlib.sha256(REVIEWER_TOKEN.encode()).hexdigest())
    monkeypatch.setattr(settings, "revoked_edge_ids", "lost-edge")
    with TestClient(app) as test_client:
        with SessionLocal() as session:
            for model in (ReviewDecision, RecognitionResult, AlertAudit, Alert, Sighting):
                session.execute(delete(model))
            session.execute(delete(WatchlistEntry).where(WatchlistEntry.reason == "Result-state test"))
            session.commit()
        yield test_client


def watch(plate=PLATE):
    with SessionLocal() as session:
        session.add(WatchlistEntry(plate_hmac=plate_token(plate), masked_plate="MH******57",
                                   reason="Result-state test", severity="high"))
        session.commit()


def result(state="partial", **extra):
    now = datetime.now(timezone.utc)
    data = {"schema_version": "2.0", "event_type": "recognition_result", "event_id": str(uuid4()),
            "edge_id": "edge-1-shared", "camera_id": "cam01", "passage_id": "p" + uuid4().hex[:8],
            "result_state": state, "observed_at": now.isoformat(), "last_observed_at": (now + timedelta(seconds=2)).isoformat(),
            "stream_pts_ms": 1000, "end_pts_ms": 3000, "track_ids": ["4"], "observations": 3, "layout": "one_row",
            "review_required": False, "source_mode": "synthetic", "profile": "candidate", "policy_version": "result-policy-1",
            "quality": {"height_px": 30, "sharpness": 120.5, "blurred": False}}
    if state == "confirmed":
        data.update(masked_plate="MH******57", plate_hmac=plate_token(PLATE), plate_length=10, format_status="valid")
    elif state == "partial":
        data.update(masked_plate="MH******57", plate_length=10, unknown_positions=[5], reason="uncertain_characters",
                    length_certain=True, format_status="valid")
    else:
        data.update(reason="no_plate_found")
    data.update(extra)
    return data


def test_three_states_persist_idempotently_without_alerts(client):
    for state in ("confirmed", "partial", "unreadable"):
        body = result(state)
        first = client.post("/v1/recognition-results", json=body, headers=EDGE)
        assert first.status_code == 200, first.text
        again = client.post("/v1/recognition-results", json=body, headers=EDGE)
        assert again.json()["duplicate"] is True
    listing = client.get("/v1/recognition-results", headers=OPERATOR).json()
    assert listing["counts"] == {"confirmed": 1, "partial": 1, "unreadable": 1}
    with SessionLocal() as session:
        assert session.scalar(select(Alert)) is None, "result events never alert by themselves"
        assert session.scalar(select(Sighting)) is None
        unreadable = session.scalar(select(RecognitionResult).where(RecognitionResult.result_state == "unreadable"))
        assert unreadable.plate_hmac is None and unreadable.masked_plate is None


def test_state_specific_validation_and_no_plaintext_echo(client):
    bad = [result("partial", plate_hmac=plate_token(PLATE)),
           result("partial", masked_plate=PLATE),
           result("partial", unknown_positions=[12]),
           result("partial", masked_plate="MH*****57"),
           result("unreadable", masked_plate="MH******57"),
           result("unreadable", reason=None),
           result("confirmed", plate_hmac=None),
           result("confirmed", reason="uncertain_characters"),
           result("partial", length_certain=False,
                  watchlist_lead={"candidate_hmacs": [plate_token(PLATE)], "candidates_considered": 26}),
           result("partial", result_state="maybe")]
    for body in bad:
        response = client.post("/v1/recognition-results", json=body, headers=EDGE)
        assert response.status_code == 422, body
        assert PLATE not in response.text


def test_edge_authentication_and_revocation(client):
    assert client.post("/v1/recognition-results", json=result(), headers={"X-Edge-Key": "wrong"}).status_code == 401
    revoked = dict(EDGE, **{"X-Edge-Id": "lost-edge"})
    assert client.post("/v1/recognition-results", json=result(), headers=revoked).status_code == 403
    assert client.get("/v1/watchlist/snapshot", headers=revoked).status_code == 403
    assert client.post("/v1/recognition-results", json=result(), headers=dict(EDGE, **{"X-Edge-Id": "edge-1"})).status_code == 200


def test_snapshot_is_minimal_and_expires(client):
    watch()
    snapshot = client.get("/v1/watchlist/snapshot", headers=EDGE)
    data = snapshot.json()
    assert snapshot.headers["cache-control"].startswith("no-store")
    assert all(set(entry) == {"plate_hmac"} for entry in data["entries"])
    assert datetime.fromisoformat(data["expires_at"]) > datetime.fromisoformat(data["generated_at"])
    assert data["ttl_seconds"] == get_settings().watchlist_snapshot_ttl_seconds


def test_partial_lead_opens_review_but_never_alerts(client):
    watch()
    lead = {"candidate_hmacs": [plate_token(PLATE), plate_token("KA01AB1234")], "candidates_considered": 26}
    response = client.post("/v1/recognition-results", json=result("partial", watchlist_lead=lead), headers=EDGE)
    assert response.json()["review_status"] == "pending"
    with SessionLocal() as session:
        row = session.scalar(select(RecognitionResult))
        assert row.lead_hmacs == '["%s"]' % plate_token(PLATE), "only active watchlist fingerprints are kept"
        assert session.scalar(select(Alert)) is None, "pending review must not create a confirmed alert"
    plain = client.post("/v1/recognition-results", json=result("partial"), headers=EDGE).json()
    assert plain["review_status"] == "not_required"


def test_review_access_control(client):
    client.post("/v1/recognition-results", json=result("partial", review_required=True), headers=EDGE)
    assert client.get("/v1/review/queue").status_code == 401
    assert client.get("/v1/review/queue", headers=OPERATOR).status_code == 403
    wrong = dict(OPERATOR, **{"X-Reviewer-Token": "guess"})
    assert client.get("/v1/review/queue", headers=wrong).status_code == 403
    queue = client.get("/v1/review/queue", headers=REVIEWER).json()
    assert queue["reviewer"] == "asha" and queue["pending"] == 1
    assert PLATE not in str(queue)


def test_confirm_creates_human_reviewed_alert_with_audit(client):
    watch()
    lead = {"candidate_hmacs": [plate_token(PLATE)], "candidates_considered": 26}
    client.post("/v1/recognition-results", json=result("partial", watchlist_lead=lead), headers=EDGE)
    item = client.get("/v1/review/queue", headers=REVIEWER).json()["items"][0]
    assert item["watchlist_lead"] is True
    mismatch = client.post(f"/v1/review/{item['id']}/decision", json={"action": "confirm", "plate": "KA49BE5557"},
                           headers=REVIEWER)
    assert mismatch.status_code == 422, "confirm cannot contradict known characters"
    decided = client.post(f"/v1/review/{item['id']}/decision", json={"action": "confirm", "plate": PLATE, "note": "clear"},
                          headers=REVIEWER)
    assert decided.status_code == 200, decided.text
    body = decided.json()
    assert body["alert"]["reason"].startswith("Human-reviewed")
    assert body["result"]["review_status"] == "confirmed"
    assert PLATE not in decided.text
    again = client.post(f"/v1/review/{item['id']}/decision", json={"action": "reject"}, headers=REVIEWER)
    assert again.status_code == 409
    detail = client.get(f"/v1/review/{item['id']}", headers=REVIEWER).json()
    assert [(d["reviewer"], d["action"], d["masked_plate"]) for d in detail["decisions"]] == [("asha", "confirm", "MH******57")]
    with SessionLocal() as session:
        sighting = session.scalar(select(Sighting))
        assert sighting.model_version == "human-review"
        assert session.scalar(select(ReviewDecision)).plate_hmac == plate_token(PLATE)


def test_correct_reject_unreadable_and_export(client):
    ids = []
    for _ in range(3):
        client.post("/v1/recognition-results", json=result("partial", review_required=True), headers=EDGE)
    ids = [item["id"] for item in client.get("/v1/review/queue", headers=REVIEWER).json()["items"]]
    corrected = client.post(f"/v1/review/{ids[0]}/decision", json={"action": "correct", "plate": "KA01AB1234"}, headers=REVIEWER)
    assert corrected.json()["result"]["review_status"] == "corrected" and corrected.json()["alert"] is None
    assert client.post(f"/v1/review/{ids[1]}/decision", json={"action": "reject"}, headers=REVIEWER).status_code == 200
    assert client.post(f"/v1/review/{ids[2]}/decision", json={"action": "unreadable", "plate": PLATE},
                       headers=REVIEWER).status_code == 422
    assert client.post(f"/v1/review/{ids[2]}/decision", json={"action": "unreadable"}, headers=REVIEWER).status_code == 200
    export = client.get("/v1/review/export/decisions", headers=REVIEWER).json()
    assert export["sealed_eligible"] is False and export["label_source"] == "review"
    assert {d["action"] for d in export["decisions"]} == {"correct", "reject", "unreadable"}
    assert "KA01AB1234" not in str(export)


def test_evidence_proxy_is_opt_in(client):
    client.post("/v1/recognition-results", json=result("partial", review_required=True, evidence_ref="r" * 24), headers=EDGE)
    item = client.get("/v1/review/queue", headers=REVIEWER).json()["items"][0]
    assert item["evidence_available"] is True
    assert client.get(f"/v1/review/{item['id']}/evidence", headers=REVIEWER).status_code == 404
    assert client.get(f"/v1/review/{item['id']}/evidence", headers=OPERATOR).status_code == 403


def test_retention_purges_expired_results(client):
    client.post("/v1/recognition-results", json=result("unreadable"), headers=EDGE)
    with SessionLocal() as session:
        row = session.scalar(select(RecognitionResult))
        assert (row.expires_at.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days >= 29
        counts = purge_expired(session, now=datetime.now(timezone.utc) + timedelta(days=31))
        assert counts["results_deleted"] == 1
    assert client.post("/v1/admin/retention/run", headers=OPERATOR).json() == {"results_deleted": 0, "review_audit_deleted": 0}
