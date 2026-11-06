# Test report

Executed in the build container (4-core Xeon, no GPU, no model weights, no Docker daemon), commit `b663720` plus the
documentation commit. **All recognition inputs in new tests are synthetic CTC posteriors or scripted passages**
(`edge/tests/ctc_fixtures.py`); they prove logic, safety rules and plumbing, not accuracy.

## Results

| Suite | Before (baseline) | After | Notes |
| --- | --- | --- | --- |
| Edge `edge/tests` | 505 passed, 1 skipped, 2 failed | **570 passed, 1 skipped, 2 failed** | +65 new tests; the same 2 pre-existing failures |
| Backend `backend/tests` | 105 passed | **115 passed** | +10 (`test_result_states.py`); migration-head assertion updated to `0004_result_states` |
| Frontend vitest + `tsc --noEmit` | 36 passed | **40 passed**, type-check clean, `vite build` OK | +4 (`review.test.ts`) |
| Tools `tools/test_*.py` | 26 passed | **29 passed** | +3 (real edge→HTTP→backend replay ×2, shadow comparator) |
| Backend result tests on PostgreSQL 16 | not run | **10 passed** | local server, `create_all` schema |
| Alembic 0003→0004 on PostgreSQL 16 | n/a | upgrade, repeat, downgrade, re-upgrade OK | `0001` (PostGIS) not exercised |

### Pre-existing failures (unchanged, not hidden)

- `test_svtrv2_experiment.py::…normalisation`: `ModuleNotFoundError: torchvision` (missing from the edge `test` extra).
- `test_engineering_page.py::…heartbeat`: expects the retired engineering-console text (`CPU · c2ba896d994b…`); the page
  now yields `OCR CPU / vehicles UNKNOWN / plates UNKNOWN · …`. Stale test of retired UI.

## New tests by area

| Area | File (tests) |
| --- | --- |
| Result states, format policy, partial/unknown, length, layout | `edge/tests/test_result_state.py` (20) |
| Passages, tracker-ID switch, ambiguity, near-duplicate, profiles, rollback flags | `test_passages.py` (11) |
| Event contract, delivery, evidence store, snapshot expiry/revocation, partial leads, dead letters | `test_result_delivery.py` (12) |
| Veto-only reader, proposal pool ranking and budget | `test_secondary_and_proposals.py` (6) |
| Three-state scoring, Wilson, annotation rules, ledger | `test_state_scoring.py` (12) |
| Calibration, leave-one-recording-out | `test_calibration.py` (4) |
| Backend ingestion, validation, access, decisions, alerts, retention, revocation, snapshot | `backend/tests/test_result_states.py` (10) |
| Review helpers and reviewer headers | `frontend/src/review.test.ts` (4) |
| Edge→outbox→HTTP→backend replay | `tools/test_seeded_replay.py` (3) |

## Negative-test coverage (required list)

| Case | Covered by |
| --- | --- |
| Wrong plate candidate / wrong character | `test_wrong_character_in_one_frame…`, `test_partial_with_wrong_length…` |
| Missing / extra character | `test_missing_or_extra_character_is_never_confirmed`, `test_existing_gate_deletion_path_is_blocked` |
| Uncertain plate length | `test_split_length_vote_is_unreadable`, replay case `length-disputed-watched` |
| Uncertain layout | `test_layout_conflict_and_low_resolution_block_confirmation` |
| Wrong vehicle association | `test_ambiguous_reassociation_never_mixes_vehicles` |
| Tracker ID switch | `test_tracker_id_switch_keeps_one_passage` |
| Duplicate event | backend idempotency, `test_fragments_and_compatible_partials…`, replay double delivery |
| Missing fingerprint | `test_state_specific_validation…` (confirmed without `plate_hmac` → 422) |
| Expired snapshot / revoked device | `test_expiry_revocation_and_minimal_contents`, `test_edge_authentication_and_revocation` |
| Pending review | `test_partial_lead_opens_review_but_never_alerts`; replay `reviews_pending=2, alerts=2` |
| Partial watchlist false candidate | replay: `false_review_leads = 1` measured |
| Failed event delivery | `test_permanent_rejection_is_dead_lettered…`, replay outage (11 events queued, all delivered once) |
| **Clipped plate** | **not directly tested** (only via `plate_too_small` / `insufficient_resolution`); no clipping detector exists |

## Seeded-watchlist replay (synthetic scripted passages, 8 cases, live backend)

| Metric | Matching off | Matching on |
| --- | --- | --- |
| Missed confirmed alerts | 0 / 2 | 0 / 2 |
| False confirmed alerts | 0 / 6 | 0 / 6 |
| Duplicate alerts | 0 | 0 |
| Delivery failures / dead letters | 0 / 0 | 0 / 0 |
| Events queued during central outage → delivered | 11 → 11 | same |
| Alerts changed by redelivering every event | 0 | 0 |
| Alerts from pending reviews | 0 | 0 (2 reviews pending) |
| Review recall / precision / false leads | not applicable | 1/1 · 1/2 · **1 false lead** |

These are 8 hand-scripted cases. They demonstrate integration and safety behaviour; they say nothing about real-footage
accuracy.

## Not run (blocked) — do not read as passed

Recorded-video evaluation with the real models, 200-image external regression, 37-crop regression, GPU throughput,
ONNX parity, CPU/GPU baseline-vs-candidate shadow run, Docker image build and compose start, Playwright/browser check
of the new Review page, engineering-page browser test.
