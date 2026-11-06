# Security and privacy

What exists today, what this work changed, and what risk remains. Statements about code are backed by tests named
in `TEST_REPORT.md`.

## 1. HMAC key and watchlist snapshot on the edge

### Findings (existing exposure, inspected in the code)

| Question | Finding |
| --- | --- |
| Key distribution | `WATCHLIST_HMAC_KEY` is passed to the edge service by Docker environment (`compose.yaml`, `edge/config.py`). One key for every edge and the API. |
| Snapshot retrieval | `GET /v1/watchlist/snapshot`, authenticated by the single shared `EDGE_API_KEY` (no per-device identity). Fetched once at session start (`tracked_live_viewer.py`). |
| Snapshot contents (before) | every active entry's `plate_hmac` **plus reason and severity**. |
| Expiry / storage | none; held in process memory for the session. Not written to disk or logs by the code inspected. |
| Revocation | none: any holder of `EDGE_API_KEY` was indistinguishable. |
| Key rotation | manual (`.env`), no procedure. |

### Security property (honest)

An attacker who controls an edge device holds the HMAC key **and** the snapshot of fingerprints. Indian registrations
are a small space (state × district × series × number is far below 10^11 plausible strings; HMAC-SHA-256 is cheap to
test), so such an attacker can recover most watchlist plates **offline** by enumerating plausible plates and comparing
fingerprints. The fingerprint hides nothing from a party that has both the key and the list. This was already true
before this work and is **not removable** while the edge fingerprints readings locally with a key it also holds.

### Mitigations implemented

- **Smallest practical snapshot:** fingerprints only; reasons/severities stay central (alerts are raised centrally).
  Test: `test_snapshot_is_minimal_and_expires`.
- **Expiry:** server `expires_at` / `ttl_seconds` (default 900 s, 30–3600 enforced); the edge refuses to use a snapshot
  after expiry or revocation (`watchlist_snapshot.py`) and refreshes in the background. `Cache-Control: no-store`.
- **Memory only:** the edge never persists the snapshot; partial matching uses it transiently.
- **Device identity and revocation:** the edge sends `X-Edge-Id`; the API rejects ids listed in `REVOKED_EDGE_IDS`
  (403) and can demand an id (`REQUIRE_EDGE_ID`). A 401/403 on refresh clears the snapshot immediately.
- **Dead-lettering** of permanently rejected events, so a revoked or invalid device cannot wedge or flood the queue.
- **Candidate fingerprints are bounded:** partial leads enumerate ≤2 unknown positions, ≤5,000 format-valid completions,
  only when the frames agree on length, and return ≤8 matching fingerprints (never the candidate strings).

### Evaluated and not adopted

| Option | Verdict |
| --- | --- |
| Separate edge-only watchlist key | Would let the snapshot use a key different from the sightings key, but the same compromised device must hold both to do its job, so it is **not a security boundary**. Not implemented; adds rotation burden for no gain. Useful only if edges stop computing fingerprints locally. |
| Per-device API keys | The real fix for revocation. `X-Edge-Id` revocation is a **policy** control on a shared key: a thief who learns `EDGE_API_KEY` can omit or forge the id unless `REQUIRE_EDGE_ID` plus per-device keys exist. **Remaining risk; see below.** |
| Moving all matching to central (send fingerprints only) | The strongest option: the edge would hold no snapshot. It is what already happens for confirmed sightings. Partial leads are the only reason the snapshot is needed on the edge, so **partial matching is off by default** and can be disabled to remove the snapshot entirely. |

### Key rotation procedure (manual, documented, untested end to end)

1. Generate new `EDGE_API_KEY` and `WATCHLIST_HMAC_KEY`. 2. Re-hash the watchlist (the API stores fingerprints, not
plates; entries must be re-entered or re-derived by an operator who holds plaintext). 3. Roll the API then every edge.
4. Retire old sightings' fingerprints from route search (they no longer match). **Rotation invalidates historical
fingerprint matching**, so it is a deliberate event, not routine.

## 2. Privacy policy for partial results (decision)

| Question | Decision implemented |
| --- | --- |
| May partial text be persisted? | **Centrally only masked**: first two + last two positions with `?` for unknown inside the mask edges; plate length and unknown-position indexes. The full partial pattern never reaches the database, logs or events. |
| Edge-local? | The full pattern, best native crops (≤3) and one downscaled frame live in **bounded edge memory** (`EvidenceStore`: ≤256 items, ≤96 MB, TTL 2 h, unguessable 24-char reference), only when `SENTINEL_REVIEW_EVIDENCE=true`. Live-camera imagery additionally needs `SENTINEL_REVIEW_EVIDENCE_LIVE=true`. Lost on restart. |
| Does the backend receive partial information? | Masked pattern, length, unknown positions, reason, quality indicators, layout, and (if matching on) matching **fingerprints** of lead candidates that are on the active watchlist. |
| Who can access review records? | Operators with a **named reviewer token** (`REVIEWER_TOKENS`, stored as SHA-256 digests; compared constant-time). Operator access alone is refused (403). |
| Retention / deletion | `RETENTION_PARTIAL_DAYS` 30, `RETENTION_UNREADABLE_DAYS` 30, `RETENTION_CONFIRMED_DAYS` 90 (1–3650 enforced), review audit 365 days; purge at API start and `POST /v1/admin/retention/run`. Edge evidence expires on its own TTL. |
| Audit | every decision row: reviewer id, action, prior status, masked plate, fingerprint, resulting sighting/alert id, time. A decided result cannot be re-decided (409). |
| Export | `GET /v1/review/export/decisions`: masked plates and actions only, marked `label_source: review`, `sealed_eligible: false`. No images, no plaintext. |
| Screenshots with plate text? | **Not permitted** by policy. The review page shows the notice "Do not export or screenshot plate text"; this is policy text plus the absence of an export path, not a technical block. |
| Plaintext in API errors | Validation errors never echo input (existing handler); tests assert plaintext plates are absent from responses. |

## 3. Review safety rules enforced in code

- Partial and unreadable results never create sightings or alerts (test: `test_three_states_persist_idempotently_without_alerts`).
- A partial watchlist lead opens a **pending review**; no alert exists until a named reviewer confirms or corrects, which
  creates a `human-review` sighting that then follows the ordinary watchlist path (`test_confirm_creates_human_reviewed_alert_with_audit`).
- A reviewer's *confirm* must agree with every known position of the edge's mask and the agreed length; otherwise the
  server answers 422 and the reviewer must use *correct* (explicit, audited).
- Review-derived labels cannot enter a sealed set (label loader rejects them).

## 4. Remaining security risks (not solved)

1. Shared `EDGE_API_KEY` for all devices: revocation by id is bypassable by a key holder. Needs per-device credentials.
2. Offline plate recovery from key + snapshot on a compromised edge (Section 1). Mitigated only by disabling partial matching.
3. Reviewer tokens are static bearer tokens in browser memory; no expiry, MFA, rate limiting or lockout.
4. Local dashboard mode, CSRF and loopback binding are existing controls; this work did not change or re-audit them.
5. Edge review evidence is served on the existing internal edge gateway behind `X-Edge-Key`; the key is shared (item 1).
6. Review evidence travels API → browser as base64 over the local dashboard connection; deploy only on loopback/TLS.
7. No automated secret scanning of history was run beyond the repository's `tools/check_handoff_secrets.py` conventions.
8. Hash-chaining / tamper-evidence of the audit table is not implemented; database administrators can edit it.
9. Backup/restore of the new tables is not covered by the handoff tooling.
10. **The existing watchlist-flag path is not covered by the new expiry.** `tracked_live_viewer.fetch_watchlist` still
    loads the (now fingerprint-only) snapshot once into `PassedVehicles.watchlist` for the whole session, with no expiry
    or revocation handling. Expiry/revocation apply to the new `WatchlistSnapshot` (partial leads) only. With partial
    matching off, that session-lifetime set is the only snapshot copy on the edge. Not changed, to avoid altering
    existing dashboard behaviour; it should be migrated to `WatchlistSnapshot` in a follow-up.
