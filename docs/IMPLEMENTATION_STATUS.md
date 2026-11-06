# Implementation status

Evidence classes: **Impl** implemented · **Unit** unit tested · **Integ** integration tested (edge→outbox→HTTP→backend or
backend+DB) · **Dev** evaluated on development data · **Sealed** evaluated on sealed data · **Blocked** needs external input
· **Not validated**. Test names are in `TEST_REPORT.md`. No item is **Dev** or **Sealed** on real footage: no real-model run
and no independent human labels were available.

## Phase 0 — discovery and baseline

| Item | Status |
| --- | --- |
| Repository audit, branch/commit/tree record, feature audit (branches merged), Pipeline A/B interpretation | Done (`SENTINEL_BASELINE.md`, `FEATURE_DECISIONS.md`) |
| Baseline tests (edge/backend/frontend/tools) | Done, measured |
| Recorded-video baseline reproduction, CPU/GPU throughput, model memory | **Blocked** (no weights/GPU); prior numbers cited, labelled as cited |
| HMAC + snapshot review, partial-text privacy policy | Done; snapshot minimisation/expiry/revocation Impl+Unit+Integ |
| Hardware discovery and per-frame budget | Done (budget **defined**, not met-tested) |

## Phase 1 — benchmark, annotation, review

| Item | Status |
| --- | --- |
| Readability protocol, annotation guide, schema 1.1 (layout, conditions, corners, second check, label source) | Impl+Unit |
| Label validation, duplicate/leak/sealed rules, manifest hashes, sealed ledger | Impl+Unit |
| Evaluation command with Wilson/rule-of-three, per-layout/condition slices | Impl+Unit |
| Development benchmark, Sealed Set A, second-annotator check | **Blocked** (human labels and recordings) |
| Minimal review page (frame/crop/prediction/actions/audit/access control) | Impl+Unit+Integ (backend); browser check **Blocked/Not validated** |

## Phase 2 — localisation, tracking, evidence

| Item | Status |
| --- | --- |
| Multi-source proposal pool (dedup, rank, filter generic text, budget, top-k, margins) behind `SENTINEL_MULTI_SOURCE_PLATES` | Impl+Unit; not run with real detectors |
| GPU / CPU profiles | Impl+Unit; **no CPU/GPU benchmark** |
| Plate-level association across tracker-ID changes, ambiguity-safe | Impl+Unit |
| Near-duplicate guard, adaptive 250 ms sampling, unreadable dedup by passage | Impl+Unit/Integ |
| ByteTrack vs BoT-SORT vs OC-SORT comparison | **Not done** (blocked: no data/models) |
| Localisation recall improvement on development data | **Not validated** |

## Phase 3 — OCR input quality and layout

| Item | Status |
| --- | --- |
| One/two-row decision with conflict handling, native row crops | Impl+Unit (`layout.py`) |
| Row-by-row OCR, small plates, closer-frame recovery | row-by-row exists opt-in (earlier work, failed trials); minimum-height rule in result states Impl+Unit |
| Blur / glare / contrast / angle indicators, sharpness best-frame selection | Impl+Unit |
| Four-corner geometry estimation, perspective correction in pipeline | **Not done** (label field only) |
| Two-row improvement / single-row no-regression / slices | **Not validated** (no annotated data) |

## Phase 4 — acceptance, partial results, backend, review

| Item | Status |
| --- | --- |
| Per-position evidence, length agreement, inserted/deleted detection, disagreement → unknown | Impl+Unit |
| Strict format check, no look-alike fixing in confirmation, bounded format-compatible completion | Impl+Unit |
| Partial safety metrics (wrong known characters, useful rate, false leads) | Impl+Unit; replay measured 1 false lead on scripted case |
| Secondary veto-only reader + evaluation function | Impl+Unit; **disabled; no real-reader evaluation** |
| Calibration: risk-coverage, ECE, LOROCV, versioned stricter-only thresholds | Impl+Unit; no real development sample |
| Result-state events (edge) and `recognition_results` (backend), migration, auth, validation, retention, audit | Impl+Unit+Integ; PostgreSQL 16 verified |
| Partial watchlist matching (edge, bounded, review-only) | Impl+Unit+Integ; off by default |
| Full review workflow (queue, evidence, decisions, export) | Impl+Unit+Integ; browser **not validated** |
| Existing confirmed delivery unchanged | Unit/Integ (existing suites pass) |

## Phase 5 — sealed evaluation, replay, shadow

| Item | Status |
| --- | --- |
| Sealed protocol, ledger, gates | Impl+Unit |
| Set A evaluation, coverage and precision gates, partial gate on real data | **Blocked** |
| Seeded-watchlist replay through edge+backend | Done on **synthetic** scripted cases (Integ); real-label replay **Blocked** |
| Shadow comparison tool | Impl+Unit; real-model shadow run **Blocked** |
| Rollback tested | profile/flag level only |

## Phases 6–7

| Item | Status |
| --- | --- |
| Targeted training | Deliberately not started (no human-confirmed errors) |
| Pipeline consolidation | Deferred; Pipeline A kept |
| Performance, ONNX parity, camera-health for recordings, tracker comparison | **Not done / Blocked** |
| Operational: idempotent delivery, dead letters, audit, access control, retention, dropped-frame counters | Impl+Unit/Integ (dropped-frame counters pre-existing) |
| Licensing review | Documented; **unresolved** (`RELEASE_READINESS.md`) |
