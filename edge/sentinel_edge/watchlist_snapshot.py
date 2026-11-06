"""Edge watchlist snapshot: keyed fingerprints only, in memory, with a hard expiry.

The edge already holds WATCHLIST_HMAC_KEY (it fingerprints readings) and this
snapshot. Together they allow an offline guessing attack against the watchlist
(an Indian plate space is small). That exposure exists with or without partial
matching; see docs/SECURITY_AND_PRIVACY.md. Mitigations implemented here:

- fingerprints only (no reasons, severities or masked text);
- never written to disk or logged;
- refuses to answer after `max_age_seconds` or the server's `expires_at`;
- cleared immediately when the central API rejects this device (401/403).

Partial watchlist matching (`partial_leads`) only proposes *review leads*. It
never produces a confirmed alert.
"""
from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
import urllib.error

from .plate_format import TooManyCompletions, completions
from .privacy import plate_hmac

MAX_PARTIAL_UNKNOWN = 2
MAX_PARTIAL_COMPLETIONS = 5000


class WatchlistSnapshot:
    def __init__(self, client, *, max_age_seconds=900, clock=time.time):
        if not 30 <= max_age_seconds <= 3600:
            raise ValueError('Snapshot age must be between 30 s and 1 h')
        self.client, self.max_age, self.clock = client, max_age_seconds, clock
        self._entries, self._expires, self.version = frozenset(), 0., None
        self.revoked = False
        self.failures = 0
        self._lock = threading.Lock()

    def refresh(self):
        try:
            data = self.client.watchlist_snapshot()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                self.clear(revoked=True)
            self.failures += 1
            return False
        except Exception:
            self.failures += 1
            return False
        entries = frozenset(entry['plate_hmac'] for entry in data.get('entries', [])
                            if isinstance(entry, dict) and isinstance(entry.get('plate_hmac'), str)
                            and len(entry['plate_hmac']) == 64)
        now = self.clock()
        expires = now + self.max_age
        server = data.get('expires_at')
        if isinstance(server, str):
            try:
                stamp = datetime.fromisoformat(server.replace('Z', '+00:00'))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                expires = min(expires, stamp.timestamp())
            except ValueError:
                pass
        with self._lock:
            self._entries, self._expires, self.version, self.revoked = entries, expires, data.get('version'), False
        return True

    def clear(self, *, revoked=False):
        with self._lock:
            self._entries, self._expires = frozenset(), 0.
            self.revoked = self.revoked or revoked

    def entries(self):
        """Current fingerprints, or None when the snapshot is missing, expired or revoked."""
        with self._lock:
            if self.revoked or self.clock() >= self._expires:
                return None
            return self._entries

    def status(self):
        with self._lock:
            return {'fresh': not self.revoked and self.clock() < self._expires, 'revoked': self.revoked,
                    'entries': len(self._entries), 'version': self.version,
                    'expires_in_seconds': max(0, round(self._expires - self.clock()))}


def partial_leads(pattern, hmac_key, fingerprints, *, length_certain=True, max_unknown=MAX_PARTIAL_UNKNOWN,
                  max_completions=MAX_PARTIAL_COMPLETIONS):
    """Watchlist fingerprints consistent with a partial reading, for human review only.

    Returns {'candidate_hmacs': [...], 'candidates_considered': n} or a dict with
    'skipped' giving the reason. Format rules only bound the search; every
    candidate keeps the known characters exactly as read.
    """
    if fingerprints is None:
        return {'skipped': 'snapshot_unavailable'}
    if not length_certain:
        return {'skipped': 'length_uncertain'}
    unknown = pattern.count('?')
    if unknown == 0 or unknown > max_unknown:
        return {'skipped': 'unknown_positions_out_of_bounds'}
    try:
        options = completions(pattern, limit=max_completions)
    except TooManyCompletions:
        return {'skipped': 'too_many_completions'}
    matches = sorted({token for token in (plate_hmac(text, hmac_key) for text in options) if token in fingerprints})
    return {'candidate_hmacs': matches[:8], 'candidates_considered': len(options)}
