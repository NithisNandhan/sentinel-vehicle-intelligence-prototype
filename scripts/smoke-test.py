"""Run a small, synthetic end-to-end API check against a local demo stack.

This writes clearly labelled synthetic records and leaves them available for
dashboard inspection. It does not access Sentinel streams or real plates.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def setting(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    path = Path(__file__).resolve().parents[1] / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, separator, candidate = line.partition("=")
            if separator and key == name:
                return candidate
    raise RuntimeError(f"{name} is missing from environment and local .env")


def request(base_url: str, path: str, *, key_name: str, key: str,
            method: str = "GET", data: dict | None = None) -> dict:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {key_name: key}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=body,
                                 headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {path} returned HTTP {error.code}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    edge_key = setting("EDGE_API_KEY")
    operator_key = setting("OPERATOR_API_KEY")
    hmac_key = setting("WATCHLIST_HMAC_KEY")
    plate = "GJ01AB1234"  # the documented synthetic watchlist entry
    token = hmac.new(hmac_key.encode(), plate.encode(), hashlib.sha256).hexdigest()
    event_id = str(uuid.uuid4())
    sighting = {
        "schema_version": "1.0",
        "event_id": event_id,
        "edge_id": "smoke-test",
        "camera_id": "cam01",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "stream_pts_ms": None,
        "track_id": "synthetic-smoke-test",
        "masked_plate": "GJ******34",
        "plate_hmac": token,
        "confidence": 0.99,
        "source_mode": "synthetic",
        "model_version": "smoke-test-1",
    }

    request(args.url, "/v1/healthz", key_name="X-Operator-Key", key=operator_key)
    start = time.perf_counter()
    first = request(args.url, "/v1/sightings", key_name="X-Edge-Key", key=edge_key,
                    method="POST", data=sighting)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    if first.get("duplicate") or not first.get("alert"):
        raise AssertionError("New synthetic watchlist sighting did not create an alert")
    second = request(args.url, "/v1/sightings", key_name="X-Edge-Key", key=edge_key,
                     method="POST", data=sighting)
    if not second.get("duplicate") or second.get("sighting_id") != first.get("sighting_id"):
        raise AssertionError("Repeated event was not idempotent")
    alert_id = first["alert"]["id"]
    alerts = request(args.url, "/v1/alerts", key_name="X-Operator-Key", key=operator_key)
    if not any(item["id"] == alert_id for item in alerts):
        raise AssertionError("Alert is absent from the operator queue")
    acknowledged = request(args.url, f"/v1/alerts/{alert_id}", key_name="X-Operator-Key",
                           key=operator_key, method="PATCH", data={"status": "acknowledged", "actor": "smoke-test"})
    if acknowledged["status"] != "acknowledged":
        raise AssertionError("Alert acknowledgement failed")
    resolved = request(args.url, f"/v1/alerts/{alert_id}", key_name="X-Operator-Key",
                       key=operator_key, method="PATCH", data={"status": "resolved", "actor": "smoke-test"})
    if resolved["status"] != "resolved":
        raise AssertionError("Alert resolution failed")
    route = request(args.url, "/v1/routes/search", key_name="X-Operator-Key",
                    key=operator_key, method="POST", data={"plate": plate})
    if not any(item["event_id"] == event_id for item in route["sightings"]):
        raise AssertionError("Sighting is absent from route search")
    report = request(args.url, f"/v1/reports/incidents/{alert_id}", key_name="X-Operator-Key",
                     key=operator_key)
    if report.get("source_mode") != "synthetic" or len(report.get("audit", [])) < 2:
        raise AssertionError("Incident report lost synthetic provenance or audit actions")
    request(args.url, "/v1/reports/hourly", key_name="X-Operator-Key", key=operator_key)
    print(json.dumps({"result": "passed", "event_id": event_id, "alert_id": alert_id,
                      "ingest_latency_ms": latency_ms, "checks": ["watchlist", "idempotency", "queue",
                      "acknowledgement", "resolution", "route", "incident", "hourly"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
