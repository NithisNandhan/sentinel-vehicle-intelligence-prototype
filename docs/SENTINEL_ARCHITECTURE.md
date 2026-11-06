# Sentinel architecture (as implemented)

## Components

```
recording / camera ─► decode (PyAV) ─► vehicle detector + ByteTrack ─► plate localisation ─► OCR (CTC posteriors)
                                                         │                                        │
                              (candidate) plate-level association ◄───────────────────────────────┘
                                                         ▼
                      TrackedOCR histories ─► CTCPlateReader (multi-frame fusion, unchanged gate)
                                                         ▼
                              passage ends ─► result_state.classify ─► CONFIRMED | PARTIAL | UNREADABLE
                                   │                                          │
        accepted() + blockers ─► 1.0 sighting (confirmed only)       2.0 recognition_result (every passage)
                                   ▼                                          ▼
                       durable SQLite outbox (idempotent event IDs, dead letters) ─► HTTPS ─► central API
                                                                                              │
              sightings ─► watchlist match ─► alerts            recognition_results ─► review queue ─► decisions
                                                                                              │ (human confirm/correct only)
                                                                                              └► 1.0 "human-review" sighting ─► alerts
```

## Edge (`edge/sentinel_edge`)

| Module | Role | Status |
| --- | --- | --- |
| `live_detection_viewer.py`, `tracked_live_viewer.py` | session, sampling loop, local HTTP gateway (`/internal/*`) | existing; now flushes passages at end, adaptive 250 ms sampling, review-evidence endpoint |
| `tracked_ocr.py` | per-vehicle histories | extended: passages, plate-level association, near-duplicate guard, evidence capture, result gating |
| `passages.py` | passage identity, re-association, near-duplicate signature | **new** |
| `ctc_fusion.py` | CTC fusion, unchanged acceptance gate | extended with read-only evidence accessors |
| `result_state.py` | three outcomes, versioned policy, per-position evidence, length agreement | **new** |
| `layout.py` | one-row / two-row / conflict decision with native-pixel row crops | **new** |
| `plate_format.py` | strict Indian format check, look-alike *status*, bounded partial completion | extended |
| `plate_quality.py` | focus/contrast/glare/size/angle indicators | extended |
| `plate_proposals.py` | multi-source proposal pool (vehicle crop, native tiles, text regions), budgeted | **new**, GPU profile only |
| `secondary_veto.py` | veto-only second reader and its evaluation | **new**, disabled |
| `plate_options.py` | `baseline` / `candidate` / `candidate-gpu` profiles + per-flag overrides | extended |
| `events.py`, `shared_delivery.py`, `outbox.py`, `api.py` | schema 1.0 sightings + 2.0 results, dedup, dead letters, `X-Edge-Id` | extended |
| `watchlist_snapshot.py`, `evidence_store.py` | in-memory expiring snapshot + partial leads; bounded in-memory review evidence | **new** |
| `recorded_evaluation.py`, `annotations.py`, `evaluation_stats.py` | three-state scoring, schema 1.1 labels, ledger, gates, Wilson / rule of three | extended / **new** |

## Backend (`backend/app`)

| Module | Role |
| --- | --- |
| `main.py` | sightings, alerts, health, minimal expiring snapshot, retention at startup |
| `review.py` (**new**) | `POST /v1/recognition-results`, review queue/detail/evidence/decision/export, retention purge |
| `models.py`, `migrations/versions/0004_result_states.py` | `recognition_results`, `review_decisions` (additive; sightings untouched) |
| `security.py` | edge key + optional device revocation, SHA-256 reviewer tokens |
| `schemas.py` | state-specific validation of schema 2.0 |

## Frontend (`frontend/src`)

`ReviewPage.tsx` + `review.ts`: reviewer-token gate (memory only), queue, evidence, character table, quality, actions, audit.

## Recognition flow and the three outcomes

1. Plate proposals per tracked vehicle; optional screen for non-plates.
2. Each distinct native crop → CTC posteriors → `CTCPlateReader` (≥150 ms apart, unique crop hash, optional near-duplicate guard).
3. Existing gate `accepted()` is the **safety floor**. With result states on, a reading is delivered as a sighting only if
   `classify()` also returns CONFIRMED: strict-valid format, frames agree on length, no layout conflict, plate tall enough,
   no secondary veto, not format-adjusted.
4. When the passage ends (ID gap, scene reset, end of source) exactly one outcome is reported:
   - CONFIRMED – complete reading (sighting already sent when the gate passed).
   - PARTIAL – ≥5 known positions, ≤3 unknown, agreed length, pattern format-compatible; or a complete reading blocked
     for review. Known characters must be supported by ≥2 frames with mean peak ≥0.75 and no confident competing reading.
   - UNREADABLE – with a reason (`no_plate_found`, `plate_too_small`, `non_plate_rejected`, `insufficient_observations`,
     `length_uncertain`, `insufficient_known_characters`, `format_impossible`, …).
5. Passages are counted after association and dedup: a tracker-ID switch inside 1.5 s with an unambiguous geometric match
   continues one passage; ambiguous matches never transfer evidence; fragment unreadables are suppressed at delivery.

## Profiles

| Profile | Contents |
| --- | --- |
| `baseline` (default) | exactly the previous behaviour; no result events |
| `candidate` | result states, plate-level tracking, near-duplicate guard, adaptive sampling |
| `candidate-gpu` | `candidate` + multi-source plate proposals |
| always opt-in, independent | `SENTINEL_SECONDARY_VETO`, `SENTINEL_PARTIAL_WATCHLIST`, `SENTINEL_REVIEW_EVIDENCE`, and the older plate options |

Not integrated (deliberately): two-row row-by-row OCR stays the existing opt-in `SENTINEL_TWO_LINE_PLATES`, which two
earlier attempts showed to be harmful; `layout.py` only classifies and blocks on conflict. Perspective correction and
corner annotation exist as quality indicators / label fields; no automatic perspective warp is in the acceptance path.
