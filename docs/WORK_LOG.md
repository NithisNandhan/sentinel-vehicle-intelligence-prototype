# Work log: plate pipeline (branch `work/plate-pipeline`)

Unattended session started 2026-10-31. Nothing is pushed. Every behaviour change is
behind a setting that is off by default. The 3-frame fusion rule and every acceptance
threshold are unchanged. No accuracy claim is made beyond the labelled sample, and the
labels are AI-drafted, not verified ground truth.

## Setup decisions

- **Branch base.** `work/plate-pipeline` starts from `tracker/keep-unconfirmed-vehicles`
  (commit `de00f38`), which already holds Phase 5 (bike tracking, off by default,
  12 tests). It was finished and measured earlier today, so it is not redone.
- **Earlier frame-label change kept apart.** The compact on-frame labels approved earlier
  are not behind a setting, so they were committed on their own branch
  `ui/compact-frame-labels` (`5509db7`, from `main`) and are not part of this branch.
  The running analysis service still has them loaded; it is not restarted here.
- **Crops and labels stay out of Git.** Plate crops and draft labels contain full plate
  text from a stock clip whose page URL was not recorded, and the project keeps full
  plate text out of central storage. They live in `data/plate-review/` (ignored by
  `/data/*`). Only tools, tests and aggregate numbers are committed.
- **Backend tests.** The API image has no pytest, so backend tests run in a throwaway
  container from that image with `backend/` mounted read-only and the dev requirements
  installed. No volume is used.

## Baseline tests (before any change)

| Suite | Result |
| --- | --- |
| Edge (`edge/tests`, in the analysis container) | 397 passed, 2 skipped |
| Tools (`tools/test_handoff.py`) | 14 passed |
| Backend (`backend/tests`) | 103 passed |
| Frontend (`vitest`) | 31 passed |

## Phase 1: draft labels (done)

- 50 of the 165 probe crops from the Mumbai clip (`new video.mp4`, 15 samples at 500 ms,
  general detector profile), chosen evenly by file order, not by content.
- Labelled by Claude from enlarged contact sheets that show only an ID and size, never the
  pipeline's OCR. `UNREADABLE` was used whenever any character was uncertain.
- Result: **2 readable** (c36, c39; the same plate), **14 NO_PLATE**, **34 UNREADABLE**.
  5 crops are two-line plates (c02, c08, c09, c33, c45); all 5 are UNREADABLE.
- Files (local only, ignored by Git): `data/plate-review/mumbai/labels_draft.csv`,
  `crops/`, `sheets/`, `crops.csv`, `README.md`.
- **10 least sure** (a person should check these first): c22, c26 (looked nearly
  readable, kept UNREADABLE), c33, c45, c02, c08 (two-line, bottom line ambiguous),
  c36 (readable, medium confidence), c41, c44 (partly legible), c12 (unsure it is a plate).
- Caveat: earlier the same day a few pipeline readings for some of these vehicles were
  seen. Those crops were kept UNREADABLE unless the pixels alone were clear.
- **Consequence for later phases:** exact-match and character error rate rest on n=2
  readable crops and cannot show an improvement or a regression with any confidence. The
  NO_PLATE false-read rate (n=14) is the most informative metric available.

## Phase 2: evaluation script and baseline (done)

- `tools/eval_plates.py`: exact-match rate, character error rate (edit distance / label
  characters; an empty reading costs every character) and false-read rate on NO_PLATE
  crops. Every rate carries its denominator and is `null` at n=0. 8 unit tests in
  `tools/test_eval_plates.py`.
- `tools/run_plate_variants.py` reads crops with the real OCR. Two readings per variant:
  `raw` (greedy CTC text) and `format` (best Indian-format candidate from a one-crop
  fusion reader, which is what the pipeline could display).
- `tools/plate_crop_probe.py` re-runs the pipeline's detection steps on a clip and saves
  each crop with its plate-detector score. To get that score, `PlateDetector.box` and
  `TargetedPlateSearch` now keep `last_confidence`; which box is chosen is unchanged
  (4 tests in `edge/tests/test_plate_confidence.py`).
- The re-run reproduced the earlier probe exactly (271 vehicle boxes, 165 crops, 104 with
  no plate) and all 50 labelled crops are byte-identical, so scores attach reliably.

Baseline on the 50 draft-labelled crops:

| Reading | Exact match (n=2) | CER | False reads on NO_PLATE (n=14) | Readings on UNREADABLE (n=34) |
| --- | --- | --- | --- | --- |
| raw | 2/2 | 0/20 | 14/14 | 34/34 |
| format | 2/2 | 0/20 | 5/14 | 29/34 |

- The OCR agrees with both image-only labels, but n=2 cannot show an accuracy level.
- 5 of 14 non-plates yield a well-formed Indian plate string: invented readings that can
  reach fusion. Phase 3 targets these.
- Host note: `tools/test_handoff.py` has 4 Windows file-locking errors when run on the
  host; it passes 14/14 in the Linux container, which is where tools tests are run.

## Phase 3: reject non-plates (done)

- `edge/sentinel_edge/plate_filter.py`, setting `SENTINEL_REJECT_NON_PLATES` (off by
  default). A proposal is dropped before it can become fusion evidence when: its
  width/height is outside 0.8-8, its plate-detector score is below 0.35 (an unknown score
  is never held against it), or the OCR has no character with peak probability >= 0.6
  (the same per-character peak that acceptance already requires in two frames).
- The screen runs right after the size check, so a rejected proposal cannot move the
  plate centre or reset the reader that holds a real plate's evidence.
- Fusion safety: tests show a low-confidence invented reading is never accepted, with
  the filter off (fusion's own per-character gate) or on (it never becomes evidence).
  The 3-frame rule and the 0.85 acceptance threshold are untouched.
- Limits were chosen on the same 50 draft labels they are measured on, so the numbers
  below are optimistic (overfit) and need a held-out check.

| On 50 draft-labelled crops | Before | After |
| --- | --- | --- |
| False reads on NO_PLATE, raw (n=14) | 14 | **1** |
| False reads on NO_PLATE, format-valid (n=14) | 5 | **1** |
| Readable plates lost (n=2) | - | **0** |
| UNREADABLE crops dropped (n=34) | - | 13 |

- The remaining false read is c49, a goods-carrier sign: real, confident text that no
  shape or confidence rule separates from a plate.
- The 13 UNREADABLE crops dropped include c02, a real two-line plate whose OCR had no
  confident character. In the pipeline each dropped crop is one fewer observation, so
  some tracks may now take longer to reach 3 observations. Not measured on a full run.
- Tests: 7 in `edge/tests/test_plate_filter.py`.

## Phase 4: two-line plates (attempt 1 failed)

- `edge/sentinel_edge/two_line.py`, setting `SENTINEL_TWO_LINE_PLATES` (off by default):
  split when height/width >= 0.45 and an ink gap separates two inked rows, read each row,
  join top then bottom with a blank step (5 tests).
- Attempt 1 made readings worse. The single-line read of the two-line crops was already
  consistent with the labeller's partial notes, and splitting turned it into noise. The
  rule also split 7 single-line or non-plate crops whose padding makes them look tall.
  Format-valid readings on two-line crops went from 5 of 5 to 0 of 5. The OCR model
  appears to read stacked rows already (its training corpus included two-row plates).
- Attempt 2: keep the split only when the OCR is more confident about it (see below).
- Attempt 2 failed too. Reading both ways and keeping the split only when it has more
  confident characters left all 5 two-line crops at their single-line reading (no
  change) and split exactly one crop, c43, a single-line plate cut off at the edge.
  Exact match 2/2 and CER 0/20 unchanged; NO_PLATE false reads unchanged (format 5/14).
  No benefit on this sample and one wrong split. Logged under **Blocked**; the code stays
  off by default and should not be enabled.

## Blocked

- **Phase 4, two-line plates** (2 attempts). The OCR model already produces plausible
  single-line reads of the two-line plates here, and splitting either adds noise
  (attempt 1) or changes nothing useful (attempt 2). A real test needs readable,
  human-labelled two-line plates; all 5 in this sample are UNREADABLE.

## Phase 5: bike tracking (done earlier, re-measured on this branch)

- Code from `tracker/keep-unconfirmed-vehicles` (`de00f38`): setting
  `SENTINEL_KEEP_UNCONFIRMED_VEHICLES` (off by default) keeps detections ByteTrack hides
  until confirmed and links them by plausible motion relative to box size (the same
  discontinuity rule the OCR stage uses); 12 tests, including two touching bikes that
  must not merge or swap and a car that must not get a second ID.
- Re-measured on this branch: identical to the earlier numbers, off and on.

Mumbai clip, 15 samples, motorcycles (heuristic counts; no ground truth):

| | Off | On |
| --- | --- | --- |
| Boxes shown (detector found 92) | 11 | 92 |
| Distinct IDs | 7 | 25 |
| One-frame IDs (first/last sample excluded) | 1 | 6 |
| Likely ID switches | 0 | 0 |
| Same-frame duplicate-ID pairs, all classes | 15 | 15 |

- Trucks: likely switches 0 -> 3. On the 720p `video-project-2.mp4` bikes are shown
  (13 -> 122 boxes) but mostly not linked: 98 IDs, 80 one-frame. The switch heuristic
  uses the same motion rule as the linker, so it cannot see breaks beyond that rule.

## Phase 6: comparisons (report only, no default changed)

**Plate detectors** on the Mumbai clip (271 vehicle boxes, 15 samples):

| | general (parent, default) | plate_adapted |
| --- | --- | --- |
| Plate crops | 165 | 177 |
| Vehicle boxes with no plate found | 104 | 92 |
| Found by this detector only | 6 | 18 |

- The 18 adapted-only crops were inspected (draft judgement): about 4 look like plates and
  about 14 are windows, vehicle fronts, bars, a bus panel, a grille, an LED sign or blur.
  Its extra finds are mostly false proposals. Detector scores are on different scales.

**Enlarging crops before OCR** (cubic), on the 50 draft-labelled crops:

| | Baseline | 2x | 3x |
| --- | --- | --- | --- |
| Exact match (n=2) | 2/2 | 2/2 | 2/2 |
| CER | 0/20 | 0/20 | 0/20 |
| Format-valid false reads on NO_PLATE (n=14) | 5 | 5 | 4 |
| Format-valid readings on UNREADABLE (n=34) | 29 | 25 | 24 |
| Raw readings changed vs baseline | - | 35 | 38 |

- Every change falls on UNREADABLE or NO_PLATE crops, where nothing can be verified.
  Enlarging adds no detail. No measurable benefit; no default change.

## Phase 7: speed (code done; timing NOT measured)

- **Fusion only on a new reading:** already the case. `CTCPlateReader.candidates()` caches
  its result and the cache is cleared only by `add()` (a new reading) or `propose()`
  (`edge/sentinel_edge/ctc_fusion.py`). No code change was made, to avoid a redundant
  second cache.
- **More CPU threads:** `SENTINEL_VEHICLE_THREADS` (vehicle detector, torch) and
  `SENTINEL_PLATE_THREADS` (plate detector, ONNX). The default stays 2, the long-standing
  hard-coded value; invalid values fail loudly (3 tests). OCR threads already follow the
  existing `SENTINEL_CPU_THREADS` (default 4 in compose).
- `tools/time_tracked_pipeline.py` times the same tracked pipeline as the service per
  stage and fingerprints every output (tracks, plate boxes, candidates with scores,
  acceptances) so two runs can be checked for identical results.
- **Not measured:** the timing job (run A, defaults, then run B, 6/4/8 threads) was
  stopped by Claude Code because the computer was critically low on memory. Its leftover
  process inside the container was stopped too (container memory 2.7 GiB -> 1.4 GiB).
  It was not restarted. So there are **no before/after times and no identical-output
  check** for the thread settings; leave them at the default until measured.
- Earlier observation (service, not this tool): the analysis used about one CPU core of 16.

## Phase 8: small fixes (done; each behind an off-by-default setting)

The rules require every change to be behind a setting that is off by default, so even
these fixes do nothing until enabled. Each was chosen as the safest reading of that rule.

| Fix | Setting | Tests |
| --- | --- | --- |
| Idle panel heading "SHARED LIVE PROCESSING" -> "SHARED PROCESSING · NO SOURCE SELECTED" when no source is selected | `VITE_NEUTRAL_IDLE_LABEL` (dashboard build) | 1 frontend |
| Evidence after Stop: 404 "No recording is being analysed" instead of 503; other edge failures stay 503 | `RECOGNITION_IDLE_EVIDENCE_404` (API) | 2 backend |
| Pause/resume on a completed run: rejected (409, shown as "Playback control not available") instead of a silent no-op; restart/seek still work | `SENTINEL_STRICT_PLAYBACK` (edge) | 3 edge |

- The dashboard already treats an evidence 404 as "nothing to show", so no frontend
  change is needed for the second fix.
- The running containers were not rebuilt or restarted; none of these settings is active.
- Frontend tests and the type check were run in a throwaway container from the dashboard
  image with `frontend/src` mounted read-only, so the running dashboard was untouched.

## Final test results

| Suite | Baseline | Final |
| --- | --- | --- |
| Edge | 397 passed, 2 skipped | 420 passed, 2 skipped |
| Tools (container) | 14 passed | 22 passed |
| Backend | 103 passed | 105 passed |
| Frontend (vitest) + `tsc --noEmit` | 31 passed | 32 passed, type check clean |

## Phase 7 timing (measured 2026-10-31)

Mumbai clip (`new video.mp4`, 4K, 15 samples at 500 ms), `tools/time_tracked_pipeline.py`
in the recorded-analysis container, CPU, OCR threads 4 (`SENTINEL_CPU_THREADS`, unchanged).
One run at a time, nothing else analysing. Peak memory is the pipeline process's VmHWM,
polled every 3 s.

| Run | Vehicle / plate threads | Total analysis | Per analysed frame (mean / median) | Load | Peak memory | `outputs_sha256` |
| --- | --- | --- | --- | --- | --- | --- |
| A | 2 / 2 (default, unset) | 912.1 s | 60.8 s / 70.3 s | 21.0 s | 1.65 GB | `d072ed016a7940ab31d885133844d5ba6c8ae56f55188535db21eef7cc7f2dfa` |
| B | 4 / 4 | 891.5 s | 59.4 s / 68.2 s | 20.9 s | 1.65 GB | `d072ed016a7940ab31d885133844d5ba6c8ae56f55188535db21eef7cc7f2dfa` |

| Stage | A (s) | B (s) | Change |
| --- | --- | --- | --- |
| Vehicle detector | 7.7 | 8.1 | +5.2% |
| Plate detector | 51.7 | 42.4 | -18.0% |
| OCR | 89.6 | 86.1 | -3.9% |
| Reconstruction | 57.4 | 56.4 | -1.7% |
| Fusion | 695.1 | 687.5 | -1.1% |

- **Identical outputs:** same `outputs_sha256`, and all pipeline metrics match (271 vehicle
  boxes, 505 plate detector calls, 165 OCR calls, 48 reconstructed windows).
- **Speed:** the whole run is 2.3% faster with 4 threads, about the same size as the
  run-to-run noise in stages the setting does not touch (OCR -3.9%, fusion -1.1%). Only the
  plate detector clearly gains (-9.3 s). One run each, so no spread is known.
- **Where the time goes:** fusion is 76% of analysis time; both detectors together are
  6.5%. More detector threads cannot make this pipeline much faster.
- Default stays 2. 4 threads is safe (same results) but gains little; not tested above 4.
- Lowest free host RAM: 1.49 GB during run A, 3.39 GB during run B. No run was stopped.
- Raw per-sample timings and outputs stay local (they contain plate text).

## Labels and settings (2026-10-31, after Phase 7)

- **Smaller frame text:** vehicle tags (`CAR 28% #12 | ANPR: ...`), the plate reading and
  its status line are drawn at 60% of their former size (`LABEL_TEXT` in
  `edge/sentinel_edge/overlay.py`). The enlarged plate crop is unchanged. At 720p the
  plate reading stays legible; the vehicle tag and status line are about 6 px tall.
- **Unconfirmed vehicles on 4K only:** new `SENTINEL_KEEP_UNCONFIRMED_MIN_HEIGHT`
  (default 0 = every height, so behaviour is unchanged unless set). With 2160, frames
  shorter than 4K (e.g. `video-project-2.mp4`, 720p) use plain ByteTrack. 2 edge tests.
- **Turned on locally** in `.env` (not in Git): `SENTINEL_REJECT_NON_PLATES=true`,
  `SENTINEL_KEEP_UNCONFIRMED_VEHICLES=true`, `SENTINEL_KEEP_UNCONFIRMED_MIN_HEIGHT=2160`.
  The defaults in `compose.yaml` stay off. They apply once the recorded-analysis
  container is recreated.
- Edge tests: 420 passed, 2 skipped (422 collected).

## Plate reading: format validator and position vote (2026-10-31, afternoon)

Full write-up, earlier-work summary and what is needed next: `docs/OCR_NEXT_STEPS.md`.

- `SENTINEL_PLATE_FORMAT_VALIDATOR` (`edge/sentinel_edge/plate_format.py`, 21 tests) and
  `SENTINEL_POSITION_VOTE` (`position_vote` in `ctc_fusion.py`, 12 tests). Both off by
  default and not turned on locally. The 3-frame rule and every threshold are unchanged.
- `tools/replay_plate_tracks.py` (3 tests) replays the saved Mumbai probe through fusion with
  each setting. It took 52 min on one core.
- Result on the 50 draft labels: the validator cut greedy readings on non-plates from 14 to 0
  and on unreadable crops from 34 to 11, keeping both readable crops. In fusion every setting
  accepted the same single correct reading. The vote changed nothing. **0 human-checked
  labels, 1 readable plate: unproven.**

## Display changes (2026-10-31, afternoon; user request, not behind a setting)

- Dashboard "Recent readings" lists only tracks that have a reading; tracks showing
  "No complete reading" are left out. The run summary still counts all recent tracks.
- Frame vehicle tag is `CAR #12` (type and track number only); the detector score is no
  longer drawn. Plates with no complete reading get no call-out on the frame.
- The dashboard image was rebuilt and the analysis service restarted (after the replay).

## Follow-ups: vote diagnosis and format-adjusted mark (2026-10-31, evening)

- **Vote:** `tools/explain_position_vote.py` and `CTCPlateReader.vote_report()` log, per
  vehicle, the readings, whether the vote differed from the best reading and why it was or
  was not added. On the Mumbai clip it is **not a bug**. 15 of 33 tracks never reach 3
  readings. In 106 fusion steps: 27 had no vote (length tie), 40 votes were not a plate
  format, 38 were already in a frame's beam, and 1 was a new candidate that was not accepted.
  Details: `docs/OCR_NEXT_STEPS.md`.
- **Format-adjusted mark:** readings the validator corrected or created carry
  `format_adjusted`. They show "FORMAT-ADJUSTED, UNVERIFIED" in the frame call-out, a gate
  reason in the Inspector, and notes in the replay tool's outputs. Tests show such a reading
  is rejected by the unchanged frame gate and by the 3-frame rule.
- Both settings stay off by default. Mumbai plate text was removed from the doc and the tests
  (synthetic plates instead), keeping to this log's earlier rule.

---

# Master execution log: result states, review, evaluation (branch `claude/vibrant-shannon-v8c0p4`)

Entries below are appended to the earlier plate-pipeline log above; nothing above was edited. Dates 2026-11-06.
Nothing is pushed. Documents of record: `SENTINEL_BASELINE.md`, `IMPLEMENTATION_STATUS.md`, `RELEASE_READINESS.md`.

| Step | What was done | Verified by |
| --- | --- | --- |
| Phase 0 | Fetched remotes; fast-forwarded the feature branch to the teammate snapshot `8b8ff91` (which already merges `work/plate-pipeline`, the tracker and UI branches); audited edge, backend, frontend, contracts, compose, evaluation tools and prior documents; recorded hardware (4 vCPU, no GPU, no weights) and baseline tests | edge 505+2 failed, backend 105, frontend 36, tools 26 |
| Result states | `result_state.py`, `layout.py`, strict format status, bounded completions, quality indicators; CTC evidence accessors | 20 tests. Found and closed a hole: the unchanged `accepted()` gate confirms a candidate one character shorter than every frame read; regression test shows baseline still does, candidate does not |
| Passages and profiles | `passages.py`, `TrackedOCR` integration (one outcome per passage, re-association, near-duplicate guard, evidence capture, gating), `SENTINEL_PROFILE` baseline/candidate/candidate-gpu | 11 tests; baseline emits nothing |
| Veto and proposals | `secondary_veto.py` (disabled) with evaluation; `plate_proposals.py` budgeted pool | 6 tests |
| Edge contract | schema 2.0 events, delivery, outbox dead letters, `X-Edge-Id`, watchlist snapshot with expiry, partial leads, in-memory evidence store | 12 tests |
| Backend | `recognition_results`/`review_decisions`, migration 0004, ingestion, reviewer-gated review API, retention, revocation, minimal snapshot | 10 tests; PostgreSQL 16 run and migration up/down |
| Frontend | Review page, reviewer-token handling, helpers | 4 tests, tsc, build |
| Evaluation | three-state scoring, label schema 1.1, validation, ledger, gates, stats; existing `evaluate_recorded.py` extended (not duplicated) | 12 + existing 17 tests |
| Calibration | stricter-only versioned thresholds, risk-coverage, ECE, LOROCV | 4 tests. First draft of one test asserted 0 held-out errors; the code correctly reported 1 (the only error sat in the held-out recording), so the test was corrected, not the code |
| Replay | `seeded_watchlist_replay.py`, `shadow_compare.py` | 3 tests; reports above |
| Docs | the twelve required documents | — |

## Issues found along the way (kept for the record)

- Wilson lower bound for 30/50 is 0.4618 (not the 0.4623 I first typed); a partial with a wrong length over-counted wrong characters beyond its known ones — fixed in `score_states`.
- `PlateOptions` earlier accepted only `== 'true'`; new parser keeps that tolerance (`yes` is ignored, not an error) so existing tests and deployments behave identically.
- The pre-existing "watchlisted vehicle" flag keeps a session-lifetime snapshot copy (documented as risk 10).
- Pre-existing failing tests documented, not fixed (`TEST_REPORT.md`).

## Not done / blocked (see `RELEASE_READINESS.md`)

Real-model evaluations and shadow run, sealed Set A and human labels, GPU/CPU benchmarks, tracker comparison, ONNX parity,
perspective correction, browser verification of the Review page, licence sign-off, per-device credentials.
