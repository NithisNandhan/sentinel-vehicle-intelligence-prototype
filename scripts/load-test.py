"""Opt-in local event-ingestion benchmark. Use a disposable database for long runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


def read_local_env(name: str) -> str | None:
    if name in os.environ:
        return os.environ[name]
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(name + "="):
            return line.partition("=")[2]
    return None


def send_event(base_url: str, edge_key: str, plate_hash: str, index: int, cameras: int) -> tuple[bool, float, str]:
    event = {
        "schema_version": "1.0",
        "event_id": str(uuid.uuid4()),
        "edge_id": "load-test",
        "camera_id": f"load-{index % cameras + 1:03d}",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "stream_pts_ms": None,
        "track_id": f"load-{index}",
        "masked_plate": "GJ00****00",
        "plate_hmac": plate_hash,
        "confidence": 0.99,
        "source_mode": "synthetic",
        "model_version": "load-test-1",
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/sightings",
        data=json.dumps(event).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Edge-Key": edge_key},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)
            return response.status == 200 and bool(data.get("accepted")), time.perf_counter() - start, ""
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        return False, time.perf_counter() - start, type(error).__name__


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = round((len(sorted_values) - 1) * fraction)
    return round(sorted_values[position] * 1000, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--rate", type=int, default=5, help="events per second")
    parser.add_argument("--cameras", type=int, default=50)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if min(args.seconds, args.rate, args.cameras) < 1:
        parser.error("seconds, rate, and cameras must be positive")
    edge_key = read_local_env("EDGE_API_KEY")
    hmac_key = read_local_env("WATCHLIST_HMAC_KEY")
    if not edge_key or not hmac_key:
        parser.error("EDGE_API_KEY and WATCHLIST_HMAC_KEY are required in environment or local .env")

    plate_hash = hmac.new(hmac_key.encode(), b"GJ00TEST00", hashlib.sha256).hexdigest()
    futures: list[concurrent.futures.Future[tuple[bool, float, str]]] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(8, min(64, args.rate))) as pool:
        index = 0
        for second in range(args.seconds):
            target = started + second
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            for _ in range(args.rate):
                futures.append(pool.submit(send_event, args.url, edge_key, plate_hash, index, args.cameras))
                index += 1
        results = [future.result() for future in futures]

    successes = [latency for ok, latency, _ in results if ok]
    failures = [error for ok, _, error in results if not ok]
    output = {
        "configuration": {"seconds": args.seconds, "rate_per_second": args.rate, "cameras": args.cameras},
        "attempted": len(results),
        "accepted": len(successes),
        "failed": len(failures),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "latency_ms": {
            "mean": round(statistics.mean(successes) * 1000, 2) if successes else None,
            "p50": percentile(successes, 0.5),
            "p95": percentile(successes, 0.95),
            "p99": percentile(successes, 0.99),
        },
        "error_types": {name: failures.count(name) for name in sorted(set(failures))},
    }
    print(json.dumps(output, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

