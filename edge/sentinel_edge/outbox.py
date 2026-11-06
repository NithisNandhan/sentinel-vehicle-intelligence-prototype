"""SQLite event outbox. Events are committed before any network send."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


ENDPOINTS = frozenset({"/v1/sightings", "/v1/health", "/v1/recognition-results"})


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
            # Events the central API permanently rejected (e.g. 422). Kept, never
            # silently dropped, so they can be inspected and re-queued after a fix.
            db.execute("""CREATE TABLE IF NOT EXISTS dead_letters (
                event_id TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                payload TEXT NOT NULL,
                status INTEGER NOT NULL,
                failed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        if endpoint not in ENDPOINTS:
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


    def dead_letter(self, event_id: str, status: int) -> None:
        """Move a permanently rejected event aside so it cannot block later events."""
        with self._connect() as db:
            row = db.execute("SELECT endpoint, payload FROM outbox WHERE event_id = ?", (event_id,)).fetchone()
            if row is None:
                return
            db.execute("INSERT OR REPLACE INTO dead_letters(event_id, endpoint, payload, status) VALUES (?, ?, ?, ?)",
                       (event_id, row[0], row[1], int(status)))
            db.execute("DELETE FROM outbox WHERE event_id = ?", (event_id,))

    def dead_letter_count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM dead_letters").fetchone()[0])

    def requeue_dead_letters(self) -> int:
        """Put every dead letter back in the queue (after the cause has been fixed)."""
        with self._connect() as db:
            rows = db.execute("SELECT event_id, endpoint, payload FROM dead_letters").fetchall()
            for row in rows:
                db.execute("INSERT OR IGNORE INTO outbox(event_id, endpoint, payload) VALUES (?, ?, ?)", row)
            db.execute("DELETE FROM dead_letters")
        return len(rows)
