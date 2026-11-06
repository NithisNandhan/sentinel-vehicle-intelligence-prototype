# Release readiness

## Classification

**Implementation complete (for what can be built without models and human labels), validation pending.**

Not "release candidate passed": the sealed Set A does not exist, no recognition accuracy has been measured on
independent human-labelled footage, and no real model could run in this environment.

## Gate status

| Gate | Status | Evidence / blocker |
| --- | --- | --- |
| Zero wrong auto-accepted plates on sealed set | **BLOCKED** | no sealed set; needs ≥50 blind-labelled readable passages from ≥2 unseen recordings |
| Coverage ≥60 % observed and Wilson lower ≥45 % | **BLOCKED** | same; tooling and gate function implemented and unit-tested (`release_gates`) |
| Accepted sample size and upper error bound reported | tooling done | reports include `rule_of_three_upper_95` and exact bound; no real sample yet |
| Zero wrong known characters in partial results | **BLOCKED** (measure) / safety rules unit-tested | metric implemented; no real partial results exist |
| Confirmed / partial / unreadable handling functional | **Met (synthetic, integration)** | `TEST_REPORT.md`; edge→backend verified |
| Seeded-watchlist replay functional | **Met on synthetic scripted data** | 0 missed / false / duplicate; real-label replay blocked |
| No confirmed alert from pending review | **Met** | backend and replay tests |
| Rollback tested | **Partial** | profile/flag rollback unit-tested; no Docker rollback drill |
| Long-term ≥99.9 % precision | **Not claimed; not demonstrable** | needs ~3,000 independent clean accepted passages |

## Blockers (external input required)

1. **Annotations:** development benchmark and Sealed Set A with blind human labels and a ≥20 % second check
   (estimated 15–40 h). The only labels in the repository were drafted by an AI assistant.
2. **Recordings:** ≥2 additional permitted recordings not used in development; Sets B/C only after a failure.
3. **Models and hardware:** weights are not in Git (handoff release); no GPU here. Unmeasured: GPU/CPU throughput of the
   candidate profiles, tile and text-detector cost, memory, event latency, tracker comparison.
4. **Real-model baseline vs candidate shadow run** and the 200-image / 37-crop regressions.
5. **Review page browser verification** and a reviewer dry run with real tokens.
6. **Licences (unresolved, not verified in this session):**
   - Ultralytics (YOLO, ByteTrack wrapper) is distributed under AGPL-3.0 with a separate commercial licence; the repository
     has **no LICENSE file** and uses it in a service. Needs legal decision before any distribution or hosted use.
   - SVTRv2 / OpenOCR code and model-weight licences were not re-verified here; the veto reader is off by default.
   - Awiros / PaddleOCR checkpoints and the PP-OCRv5 text detector: Apache-style per existing docs; derived adapted
     weights inherit training-data terms.
   - Datasets: the Kaggle academic set is CC BY-NC-ND 4.0 (private noncommercial evaluation only); DataCluster sample is
     CC-BY-NC-ND evaluation-only; the Pexels Mumbai clip restricts standalone redistribution (`SNAPSHOT_20261106.md`). None may
     be redistributed or used commercially without review.
7. **Security (see `SECURITY_AND_PRIVACY.md` §4):** shared edge key, HMAC key + snapshot on edge, static reviewer tokens, the
   session-lifetime snapshot in the vehicles list, no tamper-evident audit.

## Known functional gaps (not blockers of the classification, but real)

- **Phase 2:** BoT-SORT / OC-SORT were not evaluated against ByteTrack (no data/models); only an unambiguous-match
  association rule was added. No scheduled native-tile scan beyond the budgeted `ProposalPool`, which is untested against real detectors.
- **Phase 3:** two-row OCR improvement is **not shown** (earlier attempts failed; row-split remains opt-in/disabled);
  four-corner perspective estimation is a label field only; there is no automatic perspective correction in the acceptance path.
  Blur/glare/contrast/angle are indicators and slices, not decision rules. Night/low-light/perspective slices: **untested** (no annotated data).
- **Phase 4:** secondary reader integration has no real reader evaluation; calibration has no real development sample.
- **Phase 6 (targeted training):** deliberately not started; there are no human-confirmed errors to analyse.
- **Phase 7:** no ONNX export parity test, no performance numbers, recorded sessions still send no camera-health, pipeline
  A not consolidated or removed (correctly, since the replacement is unproven).
- Result-state thresholds (`result-policy-1`) are unvalidated starting values; the strict format policy never auto-confirms
  rare plate formats (military/diplomatic/temporary/trade).
- A passage already confirmed and delivered cannot be retracted automatically; later disagreement opens a review.

## What would move the status

1. Label and seal Set A; freeze commit and profile; run `evaluate_recorded.py --split sealed_a --final` for baseline and
   candidate; `shadow_compare.py`. 2. Replay the seeded watchlist on those labels. 3. If a gate fails, treat Set A as
   development data and use the next set. 4. Only then: GPU benchmark, tracker comparison, ONNX parity, licence sign-off.
