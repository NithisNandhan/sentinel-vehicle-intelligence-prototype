# Wrap-up: everything done from the start

Updated 11 October 2026 on the teammate machine (AMD Ryzen 7 5700U, 13.9 GB
RAM, no NVIDIA GPU). This replaces the earlier version of this file and
includes the work done after it: the on-video call-outs, the vehicles list,
the opt-in dashboard view, the street map, and the tests on real footage.

## In one paragraph

The handoff asked for the prototype to accept permitted recorded footage
instead of live Sentinel cameras, on a CPU. That is built, running and
exercised with the real models on two openly licensed Indian traffic clips.
Both interfaces now show vehicles being boxed and listed as they pass.
**Recognition itself is still unproven.** On the footage tested, the real
models accepted zero plates. Stage tests show where it breaks: vehicles and
plates are found, but the tracker loses about half the vehicles between
samples and the OCR reads only some plates correctly.

## 1. Every change, from the start

### Central API (`backend/`)

| File | Change |
| --- | --- |
| `app/schemas.py` | New source `recorded`. Sightings accept `session_id`, `time_basis`, `processed_at`; a recorded sighting is rejected without them |
| `app/models.py`, `migrations/versions/0002_recorded_sightings.py` | Three nullable columns on `sightings`; migration is safe to run twice |
| `app/main.py` | Stores and returns the new fields on alerts, routes and incident reports; adds a "RECORDED FOOTAGE · NOT A LIVE CAMERA" notice |
| `app/recognition.py` | Recorded statuses and fields; `GET /recordings`; `POST /control`; masked `vehicles_seen` list; opt-in `GET /evidence` (frame and full readings, recordings only) |
| `app/config.py` | New setting `dashboard_recorded_evidence`, off by default |
| `tests/` | 43 new tests (53 → 96) |

### Edge pipeline (`edge/`)

| File | Change |
| --- | --- |
| `sentinel_edge/recorded_source.py` (new) | Recording library that lists only videos with a permission receipt; file decoder keeping original timestamps; seek, pause, paced/unpaced; named failures |
| `sentinel_edge/sampled_frames.py`, `crop_collector.py` | Opt-in blocking mode so a slow CPU never drops a sampled frame. Live behaviour unchanged |
| `sentinel_edge/live_detection_viewer.py` | `--device cpu|gpu` with no silent fallback; recordings selectable; pause/resume/restart/seek; new session per run; `--recordings-dir`, `--recordings-only`, `--unpaced`, `--dashboard-evidence`; reports models, hashes and device |
| `sentinel_edge/tracked_live_viewer.py` | Recorded runs; end of file is "completed"; pause holds analysis; evaluator hook; fixed a loop that could exit one sample early; draws the new overlay; keeps the vehicles list |
| `sentinel_edge/overlay.py` (new) | Corner brackets and a tag (`CAR 86% #66 | ANPR: …`) on each tracked vehicle; plate crop popped out with reading and status |
| `sentinel_edge/passed_vehicles.py` (new) | Running, bounded list of every vehicle seen in a run; full text for the protected console, masked for the operator |
| `sentinel_edge/experimental_detectors.py` | Tracks now carry the detector's score, for display only |
| `sentinel_edge/events.py`, `shared_delivery.py`, `recognition_contract.py` | Recorded label, session, video time, processing time; replay-safe event IDs; dedup on video time; failure code and vehicles list forwarded |
| `sentinel_edge/engineering_page.py` | Recordings in the selector, playback controls, video position, device and checkpoint, "Vehicles detected this run" list |
| `sentinel_edge/recorded_evaluation.py` (new) | Label loader and scorer with explicit denominators |
| `pyproject.toml`, `Dockerfile` | `live-cpu` extra and image target |
| `tests/` | 74 new tests (288 → 362 in total) |

### Frontend (`frontend/src/`)

| File | Change |
| --- | --- |
| `api.ts`, `recognition.ts` | Recorded types and helpers; recordings, control and evidence calls; optional local auto-unlock |
| `RecognitionPanel.tsx` | Cameras and recordings in one selector; RECORDED tag; video time and seek; Pause/Resume, Restart, Stop; vehicles list; analysed frame and full readings when the evidence view is on |
| `MapPanel.tsx` | Real street map (OpenStreetMap), named pins, info windows, zoom/full-screen/scale; offline outline as fallback |
| `App.tsx`, `styles.css` | RECORDED label on alerts, cameras, routes, reports; coverage count fixed |
| tests | 7 new (15 → 22) |

### Launch, tools and configuration

| File | Change |
| --- | --- |
| `compose.yaml` | `recorded-analysis` service (CPU, no GPU, no Sentinel credentials); queue volume external; weight folders mounted individually; evidence setting |
| `compose.autounlock.yaml` (new) | Optional: dashboard opens already unlocked on this PC |
| `scripts/start-recorded-analysis.ps1` (new) | One command to start recorded analysis |
| `scripts/init-env.ps1` | Writes `.env` without a byte-order mark under PowerShell 5.1 and 7 |
| `tools/run_ocr_research.py` | CPU mode with no CUDA libraries |
| `tools/register_recording.py` (new) | Writes a permission receipt for a video |
| `tools/evaluate_recorded.py` (new) | Scores the unchanged models against labels |
| `tools/diagnose_stages.py` (new) | Tests vehicle detection, plate detection and OCR separately |
| `tools/diagnose_tracking.py` (new) | Counts detected vehicles against tracked vehicles |
| `tools/repair_ocr_source.py` (new) | Restores five OCR source files the handoff archive left out |
| `tools/capture_screens.mjs` (new) | Takes real browser screenshots of both interfaces |
| `.gitignore` | Ignores `.venv*/` and `*.egg-info/` |

### Documents

`BUILD_INSTRUCTIONS.md`, `CURRENT_STATUS.md`, `WRAP_UP.md`,
`docs/RECORDED_VIDEO.md`, `task-reports/`, and a new section in `README.md`.

### On this machine only (not in Git)

`.env`; the restored model volumes; `data/` with the release assets, two
recordings and receipts, pilot labels, evaluation and diagnosis reports, and
screenshots. None of it is committed, because it holds keys, large files or
real footage.

## 2. What is complete according to the completion guide

| Guide item | State |
| --- | --- |
| Page 8: CPU option, CPU launcher, CPU Paddle and ONNX routes, diagnostics | **Done and proven.** Real models load on CPU; checkpoint hash `c2ba896d…4456d` matches |
| Page 8: backpressure, PTS sampling, video time separate from processing time | **Done.** On real 4K footage every frame was decoded and every sample analysed |
| M1: restore, new keys, central services, synthetic demo, model hashes | **Done.** Snapshot counts match the handoff. The snapshot is restored as files, not imported |
| M1: a real existing-checkpoint CPU reading | **Done.** OCR 531 ms, vehicle detector 165 ms, plate detector 126 ms per call |
| M3: recorded contracts, decoder, states, seek/replay, replay IDs | **Done**, on PostgreSQL and with real footage |
| M4: both interfaces observe one recorded session | **Done and seen.** Real browser screenshots taken of both |
| M5: baseline and failing-stage diagnosis | **Started.** Four-passage pilot plus stage and tracking tests; see sections 6 to 8 |
| M6: delivery, watchlist, operator actions, five-minute outage | **Mock only.** No real plate has been accepted to send |
| M2: permitted footage and independent labels | **Partly.** Two permitted clips; labels made by me, not an independent person |
| M7: freeze, held-out evaluation, demonstration, release | **Not done** |
| Page 8: parent versus adapted OCR on a labelled sample on CPU | **Not done** |

## 3. What is complete according to the status and next steps document

**Priority 0**

| Item | State |
| --- | --- |
| Restore and establish a reproducible baseline | Done |
| Obtain permitted recorded footage and labels | Partly: footage yes, independent labels no |
| Implement recorded input in the shared pipeline | Done and run with real models |
| Extend the contracts and both interfaces | Done and seen in a browser |
| Prove the full operator workflow with a real labelled passage | **Not done** |
| Publish measurements and record the new demonstration | **Not done** |

**Its completion table (page 15)**

| Condition | State |
| --- | --- |
| Clean-machine restoration | Largely done on this second machine |
| Recorded-video input and controls | Done |
| Shared operator and engineering session | Done |
| Readable Indian plates recognised correctly | **Not established.** Zero accepted; single-frame OCR right on some plates |
| Reliable acceptance decisions | Not measured; scores uncalibrated |
| End-to-end watchlist and incident workflow | Mock only |
| Endurance and failure recovery | Outage and replay proven with a mock; no long or two-stream run |
| Submission claims | No new measured demonstration |

Priority 1 and 2 work is not started, apart from the diagnosis in sections 7 and 8.

## 4. What should now be changed in the status documents

I have not edited the owner's documents. These statements in them are out of date:

| They say | Now true |
| --- | --- |
| "Downloaded-video dashboard input: to be built" | Built and running |
| "Current contracts include live/synthetic only" | `recorded` is a third source everywhere |
| "The shared engineering launcher and runtime still need CPU wiring" | Done |
| "Use pwsh, not Windows PowerShell 5.1" | `init-env.ps1` works in both |
| Test counts 287 / 53 / 15 | 360 passing of 362 on this Windows host (all pass on Linux) / 96 / 22 |
| The release restores a working OCR | **It does not** until `tools/repair_ocr_source.py` is run |
| Restore steps | Should add the repair step, and `-SkipCentral` for a machine that already has data |
| The operator dashboard never shows images or full plates | Still true by default. An opt-in view now shows them for permitted recordings only |
| The map is an approximate offline outline | It is a street map when online, with the outline as the offline fallback |

`CURRENT_STATUS.md` and task reports 2, 4, 6 and 7 were written before the
real-model runs and the screenshots; their scores are superseded by the table
at the end of this file.

## 5. Disadvantages, defects found, and what still needs fixing

### Found and already fixed

| Defect | Origin |
| --- | --- |
| The published release cannot load the OCR: the archiver skips folders named `token`, removing five PaddleOCR source files | Owner's archiver. Worked around; **the archiver itself is not fixed** |
| Paddle needs a writable cache folder that a read-only mount blocked | Would also affect the owner's GPU launcher on a fresh machine |
| Pause kept analysing frames already queued | My code; found by the cross-process audit |
| Operator view was not told why a recording failed | My code |
| Analysis loop could exit one sample early | Existing code |
| Recorded events did not keep the processing time | My code; found by re-reading the guide |
| "0 online" on a map showing two online cameras | Existing code counted live cameras only |
| Map canvas filled only a strip of its panel | My code; found by the screenshot |

### Disadvantages that remain

**Recognition**

- **Nothing is accepted.** Zero plates on both clips.
- **The tracker loses about half the vehicles.** On the Hyderabad clip the
  detector found 579 vehicle boxes and only 285 (49%) reached plate search.
- **Readings rarely reach three frames**, which the gate needs. A correctly
  read plate at one frame is never accepted.
- **OCR is right only some of the time.** In the stage test it read one of
  three plates exactly. Live readings included correct ones (`TS07FS8429`,
  `TG15B7412`) and wrong ones (`TS382811`, `TS175555`).
- **The plate detector boxes things that are not plates**: a motorcycle logo,
  an auto-rickshaw canopy, bus windows. On the Bangalore clip a bus window was
  "read" as `MH01651` over 8 frames. That is how a plate that does not exist
  could eventually be accepted.
- Two-row plates fail more often. You asked me not to pursue that.
- The acceptance threshold (0.85) is unchanged and uncalibrated.

**Speed**

- About 7 minutes for 56 seconds of 4K video on this CPU.

**Evidence**

- Two handheld phone clips, four labels made by me. A pilot, not a measurement.
- The watchlist and operator workflow has only been proven with a mocked plate.

**Operations and safety**

- **The evidence view crosses a privacy rule in the handoff.** With
  `DASHBOARD_RECORDED_EVIDENCE=true`, the dashboard shows the analysed frame
  and full, unverified plate text. It is off by default, limited to recordings
  and stores nothing, but it is a deliberate departure and the owner should agree.
- **Dashboard auto-unlock** (`compose.autounlock.yaml`) gives anyone at this PC
  operator access.
- **The street map needs internet** and tells OpenStreetMap which area is viewed.
- The edge client retries a rejected event forever.
- Still a local prototype: shared keys, no accounts, no retention policy.
- C: drive has about 13 GB free.
- The CPU image got OpenCV 5.0.0 through a dependency, above the project's pin.

### To detect and rectify next, in order

1. **Tracking.** Let vehicles that are detected but not tracked still get a
   plate search, then re-measure with `tools/diagnose_tracking.py`.
2. **False plates.** Reject plate boxes that are implausible for a plate before OCR.
3. **Short passages.** Decide how a plate seen for one or two frames should be treated.
4. **Proper footage and labels:** fixed-camera clips labelled by a person who
   is not me, split by vehicle and scene.
5. **Tell the owner** about the `token` omission and the evidence view.
6. Compare parent and adapted OCR on CPU with the benchmark crops.
7. Make a rejected event non-blocking in the edge client.
8. Turn off auto-unlock and the evidence view when testing is finished.

## 6. Full-pipeline evaluation on real footage

`data/evaluation/hyderabad-pilot-001.json`. Adapted OCR on CPU, `general`
profile, gate unchanged. **One clip, four labels made by me: a first
diagnostic, not an accuracy figure.**

| Measure | Result |
| --- | --- |
| Labelled passages | 4 (3 readable by eye) |
| Labelled plate box matched by a predicted box | 2 of 3 |
| **Plates read correctly and accepted** | **0 of 3** |
| Wrong plates accepted | 0 of 3 |
| Accepted precision | Undefined: nothing was accepted |
| True plate anywhere in the top 20 candidates | 0 of 3 |
| Speed | 420 s for 56 s of 4K video; peak memory 1.7 GB |

The gate is not the cause: lowering it would not have produced a correct reading.

## 7. Stage-by-stage test: vehicle, plate, OCR

`tools/diagnose_stages.py`, one frame per plate, real models.

| Plate | Vehicle detected | Plate found | OCR on that frame | Correct? |
| --- | --- | --- | --- | --- |
| White car, one row | Yes, "car" 0.90 | Yes | `TS07FS8429` | **Yes** |
| Yellow auto, two rows | Yes, as "truck" 0.50 | Yes | `TS38T812` | No, one character dropped |
| White scooter, two rows | Yes, "motorcycle" 0.63 | Yes | `AP98557` | No, mostly wrong |

Vehicle detection 3 of 3, plate detection 3 of 3, OCR 1 of 3.

## 8. Tracking test and the Bangalore clip

**Tracking (`tools/diagnose_tracking.py`, Hyderabad clip, 113 samples):**
579 vehicle boxes detected, 285 tracked (49%). Median 5 detected and 2 tracked
per sample. This is the largest single loss measured so far. A handheld moving
camera is harder on the tracker than fixed CCTV would be.

**Bangalore clip (22 s, wide rooftop view):** vehicle detection is good, with
up to 12 vehicles boxed in a frame. No real plate is large enough to read. All
20 "plate" detections I checked were bus windows. Nothing was accepted.

## 9. What the interfaces look like now

Real browser screenshots are in `data/screenshots/` (not committed).

- **Engineering console:** analysed frame with corner brackets and tags, plate
  crop popped out with its reading, and a "Vehicles detected this run" list
  that adds a row as each vehicle passes.
- **Operator dashboard:** the same list with masked plates. With the evidence
  view on, also the analysed frame and full readings for recordings.
- **Map:** street map of Ahmedabad with named pins and info windows.

Not copied from the reference video you showed: owner names, vehicle models
and "CLEAN / STOLEN / VAHAN verified" (no registry exists here, so they would
be invented); "34 Cams – Trace"; clip recording and snapshots; the 50-camera
registry.

## Scores

| Task | Score | Note |
| --- | --- | --- |
| 0 Checkout and baseline | 9 | |
| 1 Recorded contract in the API | 9 | |
| 2 CPU path | 9 | Real CPU inference proven |
| 3 Recorded file reader | 9 | Run on real VP9 4K and Theora files |
| 4 Recorded session and delivery | 9 | Run with real models; no real plate delivered |
| 5 Operator API for recordings | 9 | |
| 6 Both interfaces | 9 | was 7; now seen in real browser screenshots |
| 7 Launcher, image, documents | 9 | |
| 8 Restore and real CPU reading | 8 | Snapshot not imported; OCR comparison not done |
| 9 Evaluation tooling | 8 | Has scored real models on real footage |
| 10 Footage and labels | 2 | Permitted clips; no independent labels |
| 11 Baseline, proof, freeze, package | 1 | was 0; diagnosis started |
| **Total** | **91 of 120 (76%)** | |

The total measures the work around recognition. It does not mean recognition
is 76% of the way to working: on the evidence so far it accepts no plates.
