# Event contract

All edge → central events go through the durable SQLite outbox: committed before any send, retried with backoff,
idempotent on `event_id`. Authentication: `X-Edge-Key`; optional device identity `X-Edge-Id` (revocation).

## Sightings — schema 1.0 (unchanged)

`POST /v1/sightings`. Confirmed readings only; requires `plate_hmac`. Sole source of **watchlist alerts**. Unchanged for
existing consumers. Recorded sightings still require `session_id`, `time_basis`, `stream_pts_ms`, `processed_at`.

## Recognition results — schema 2.0 (new)

`POST /v1/recognition-results` → `{accepted, duplicate, result_id, review_status}`. One event per **passage**.
`event_id` for the edge is a UUIDv5 of `session|camera|passage_id`, so a resend is idempotent and a restart or seek
(new session) yields distinct, traceable events. `extra` fields are rejected (422).

Common: `schema_version "2.0"`, `event_type "recognition_result"`, `event_id`, `edge_id`, `camera_id`, `passage_id`,
`result_state`, `observed_at`, `last_observed_at` (UTC, ≥ observed_at), `stream_pts_ms`, `end_pts_ms`, `track_ids` (≤16),
`observations`, `layout` (`one_row|two_row|conflict|undetermined|unknown`), `review_required`, `source_mode`,
`model_version`, `profile`, `policy_version`, `quality{height_px,width_px,sharpness,contrast,glare_fraction,dark_fraction,mean_brightness,angle_deg,blurred,low_contrast,glare,low_light,tilted}`; recorded adds the session fields.

| State | Required | Forbidden |
| --- | --- | --- |
| `confirmed` | `plate_hmac`, `masked_plate` (`AA******AA`), `plate_length`; optional `sighting_event_id`, `format_status` | `reason`, `unknown_positions`, `watchlist_lead` |
| `partial` | `masked_plate` (may contain `?`), `plate_length` = mask length, `unknown_positions` (inside length), `reason`; optional `length_certain`, `evidence_ref`, `format_status`, `watchlist_lead{candidate_hmacs≤8, candidates_considered}` | `plate_hmac`; a lead with `length_certain=false` |
| `unreadable` | `reason` | `plate_hmac`, `masked_plate`, `watchlist_lead` |

Reasons are lowercase tokens (`uncertain_characters`, `length_uncertain`, `layout_conflict`, `insufficient_resolution`,
`secondary_reader_veto`, `format_lookalike`, `no_plate_found`, `plate_too_small`, …; see `result_state.REASONS`).

Server behaviour: partial/unreadable never create sightings or alerts. A partial with `review_required` or a lead that is
on the **active** watchlist opens a `pending` review; the stored lead keeps only fingerprints still on the watchlist. A
confirmed result with `review_required` (later evidence disagreed with an earlier confirmation) also opens a review.
Rows expire per retention settings.

## Review endpoints (operator key **and** reviewer token)

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/recognition-results[?state=]` | operator list and counts (masked) |
| `GET /v1/review/queue?status=pending|decided|all` | queue, leads first |
| `GET /v1/review/{id}` | masked detail and decision history |
| `GET /v1/review/{id}/evidence` | edge-held frame/crops/character evidence, proxied, opt-in, may be gone |
| `POST /v1/review/{id}/decision` | `{action: confirm|correct|reject|unreadable, plate?, note?}` |
| `GET /v1/review/export/decisions` | masked outcomes, `sealed_eligible:false` |
| `POST /v1/admin/retention/run` | purge expired rows |

*confirm* must repeat every known position and the length (else 422 → use *correct*). *confirm/correct* creates a 1.0
sighting with `model_version "human-review"` and event id `uuid5(review-sightings, result_event|fingerprint)`, so repeating
a decision is idempotent; only that sighting can raise an alert. A decided result returns 409.

## Watchlist snapshot (changed)

`GET /v1/watchlist/snapshot` → `{version, generated_at, expires_at, ttl_seconds, entries:[{plate_hmac}]}`; `no-store`.
**Breaking for consumers that read `reason`/`severity` from entries**: the only in-repo consumer reads `plate_hmac` only.

## Edge-local gateway

`GET /internal/review-evidence?ref=` (`X-Edge-Key`): returns the evidence record or 404.

## Backward compatibility

1.0 sightings, health, alerts, camera sync unchanged. Existing tests for them pass unmodified except the Alembic head
revision assertion (now `0004_result_states`). The `baseline` profile emits no 2.0 events.
