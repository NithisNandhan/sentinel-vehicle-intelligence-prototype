"""Small central HTTP client with no response-body logging."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from .outbox import Outbox


# Rejections that resending can never fix. Authentication (401/403), rate
# limits and server errors stay queued and are retried with backoff.
PERMANENT_REJECTIONS = frozenset({400, 404, 409, 413, 422})


class CentralClient:
    def __init__(self, base_url: str, edge_api_key: str, outbox: Outbox, edge_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.edge_api_key = edge_api_key
        self.outbox = outbox
        self.edge_id = edge_id
        self.dead_lettered = 0

    def _headers(self, extra=None) -> dict:
        headers = {"X-Edge-Key": self.edge_api_key}
        if self.edge_id:
            # Lets the central API refuse a revoked device (docs/SECURITY_AND_PRIVACY.md).
            headers["X-Edge-Id"] = self.edge_id
        headers.update(extra or {})
        return headers

    def post(self, endpoint: str, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=body,
            method="POST",
            headers=self._headers({"Content-Type": "application/json"}),
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status not in {200, 201, 202}:
                raise OSError("Central server did not accept event")

    def flush_once(self, limit: int = 100) -> int:
        delivered = 0
        for event_id, endpoint, payload in self.outbox.next_batch(limit):
            try:
                self.post(endpoint, payload)
            except urllib.error.HTTPError as error:
                if error.code in PERMANENT_REJECTIONS:
                    self.outbox.dead_letter(event_id, error.code)
                    self.dead_lettered += 1
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                break
            self.outbox.ack(event_id)
            delivered += 1
        return delivered

    def start_retry_loop(self, stop: threading.Event) -> threading.Thread:
        def run() -> None:
            delay = 1
            while not stop.is_set():
                sent = self.flush_once()
                if sent:
                    delay = 1
                    pause = 0.1 if self.outbox.depth() else 0.25
                elif self.outbox.depth():
                    delay = min(delay * 2, 30)
                    pause = delay
                else:
                    delay = 1
                    pause = 0.25
                stop.wait(pause)
        thread = threading.Thread(target=run, daemon=True, name="outbox-delivery")
        thread.start()
        return thread

    def watchlist_snapshot(self) -> dict:
        request = urllib.request.Request(
            self.base_url + "/v1/watchlist/snapshot",
            headers=self._headers(),
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def sync_cameras(self, cameras: list[dict]) -> None:
        self.post("/v1/cameras/sync", {"cameras": cameras})
