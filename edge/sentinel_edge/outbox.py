"""SQLite event outbox. Events are committed before any network send."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Outbox:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS outbox (
                event_id TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, endpoint: str, payload: dict) -> None:
        if endpoint not in {"/v1/sightings", "/v1/health"}:
            raise ValueError("Unsupported outbox endpoint")
        event_id = payload["event_id"]
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO outbox(event_id, endpoint, payload) VALUES (?, ?, ?)",
                       (event_id, endpoint, body))

    def next_batch(self, limit: int = 100) -> list[tuple[str, str, dict]]:
        with self._connect() as db:
            rows = db.execute("SELECT event_id, endpoint, payload FROM outbox ORDER BY created_at, rowid LIMIT ?",
                              (limit,)).fetchall()
        return [(event_id, endpoint, json.loads(payload)) for event_id, endpoint, payload in rows]

    def ack(self, event_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM outbox WHERE event_id = ?", (event_id,))

    def depth(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

