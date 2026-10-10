# Is everything done? Verification against the two handoff documents

Checked 10 October 2026, 23:15, against:

- `Sentinel_Teammate_Completion_Guide.pdf` (10 October, 13 pages)
- `Sentinel_Project_Status_and_Next_Steps.pdf` (8 October, 16 pages)

## Answer

**No. The project is not complete.** The software the two documents ask for is
built and tested. What they define as "finished" also needs evidence from real
models and real footage, and none of that exists yet.

| | Items | Share |
| --- | --- | --- |
| Done with real evidence | 15 | 26% |
| Built and tested, but only with a mocked recogniser or not yet seen running | 16 | 28% |
| Partly done | 8 | 14% |
| Not done | 19 | 33% |
| **Total checked** | **58** | |

The two documents overlap, so some requirements are counted in more than one
section; the shares are a guide, not a precise measure.

The 19 "not done" items share three blockers:

1. **Models are not on this machine yet.** The 5.7 GB model bundle is
   downloading (13 of 22 parts at the time of writing). Until it is restored,
   nothing real can run. This one I can finish.
2. **There is no permitted footage and no human labels.** Only you can supply
   these. Every accuracy item waits on them.
3. **Nobody has looked at the interfaces in a browser.**

How to read the status column:

- **Done**: shown with real services or real files, with the result recorded.
- **Tested (mock)**: code exists and automated tests pass, but the recogniser
  was a stub that always returns the same text, or the feature has not been
  run for real.
- **Partly**: some of the item is done; the rest is named.
- **Not done**: no work that counts toward it.

---

## A. Completion guide, page 1: "What counts as finished"

| # | Requirement | Status | Evidence or gap |
| --- | --- | --- | --- |
| A1 | A clean machine restores the code and required assets | Partly | Code restored and all suites run. Small bundles downloaded. Model bundle 13 of 22 parts. Restore script not yet run. |
| A2 | …runs both interfaces | Partly | Dashboard, API, database and media gateway are running on this machine (old API image). The engineering console has never started here. |
| A3 | …and processes a permitted recording through one shared session | Tested (mock) | Whole path run across real processes with a generated clip and a mocked recogniser. No permitted real recording exists. |
| A4 | Plate outputs checked against independently reviewed labels; correct, wrong, uncertain and unreadable all reported | Not done | The evaluation tool reports all four, but there are no labels and no real run. |
| A5 | A correctly recognised recorded passage reaches the central service, matches a synthetic watchlist entry and supports audited operator actions | Tested (mock) | Done end to end on PostgreSQL with acknowledge and resolve, but the "recognised" plate came from the stub. |
| A6 | Replay, EOF, pause, restart behave predictably | Done | Unit tests plus the cross-process audit (17 of 17). A pause defect was found there and fixed. |
| A7 | API outages behave predictably | Tested (mock) | Five-minute outage audit: 12 of 12 checks, zero missing, zero duplicated. Events were produced by the stub. |
| A8 | Setup, settings, model hashes and a labelled demonstration are reproducible | Partly | Setup and settings documented; hashes exposed in diagnostics and in evaluation reports. No demonstration recorded. |

## B. Completion guide, page 8: "Make the shared models run on CPU"

| # | Required change | Status | Evidence or gap |
| --- | --- | --- | --- |
| B1 | Explicit CPU/GPU option in the shared model loader; visible load failures | Done | `--device`, no fallback in either direction; a missing model is reported by code and not retried (tested). |
| B2 | A CPU launcher that does not request `--gpus all` | Done | `recorded-analysis` compose service and `start-recorded-analysis.ps1`; no GPU reservation. Validated; not yet started with models. |
| B3 | Use the isolated CPU Paddle dependencies, not just drop the GPU flag | Tested (mock) | Launcher builds a CPU-only import path. Whether the `deps` folder in the bundle loads cleanly is unknown until restore. |
| B4 | A verified CPU dependency/provider route instead of `onnxruntime-gpu` | Done | Image built: Torch `2.14.1+cpu`, ONNX Runtime CPU provider, no CUDA wheels. Full suite passes inside it. |
| B5 | Validate original versus adapted OCR on a bounded labelled sample on CPU | Not done | Needs the model bundle and the benchmark crops. |
| B6 | Expose model names, checkpoint hash, actual device and inference time | Done | `model_identity()` in the engineering snapshot; stage timings were already there; device also in operator status. |
| B7 | Sequential decoding with backpressure; no latest-frame dropping for files | Done | Blocking mode; 40 of 40 and 12 of 12 samples with a slow consumer, zero drops. |
| B8 | Sample by PTS, keep full-resolution frames, keep video time separate from processing time | Done | PTS preserved; `processed_at` stored separately from `observed_at`. |
| B9 | Record exact CPU, RAM, OS, package versions, free disk and timing | Partly | Machine recorded in `BUILD_INSTRUCTIONS.md`; evaluation reports capture environment. No inference timing exists yet. |

## C. Completion guide, pages 9 and 10: milestones

| # | Milestone | Status | Evidence or gap |
| --- | --- | --- | --- |
| M1a | Restore checkout and bundles, new keys, start central services and synthetic demo | Partly | Checkout, keys, services and demo simulator are up. Bundles not restored; `smoke-test.py` not run. |
| M1b | Confirm saved alerts/audits, queue files, report viewers, model hashes | Not done | Waits on the restore. |
| M1c | A real existing-checkpoint CPU reading | Not done | Waits on the model bundle. |
| M2 | Permitted footage, receipts, independent labels, splits by vehicle/video/scene | Not done | Needs you. Receipt tool, label format and split-leak check are ready. |
| M3a | `recorded` through API models, migration, snapshots, delivery, frontend types | Done | Migration ran on PostgreSQL from the old schema; 17 of 17 HTTP checks with the real edge client. |
| M3b | Keep recording ID, video PTS, processing UTC time, session ID, track ID | Done | All five stored and returned. `processed_at` was missing at first and was added after re-reading the guide. |
| M3c | Update every live-only assumption in the six named files | Done | `schemas.py`, `events.py`, `recognition_contract.py`, `shared_delivery.py`, `api.ts`, `recognition.ts` all changed. |
| M3d | Defined recorded states; never invent a recording time | Done | `processing`, `paused`, `completed`, error codes; time used only if the receipt states it, otherwise labelled processing time. |
| M3e | PyAV input with start/pause/resume/stop and documented seek/replay; resets on seek, EOF, cut | Done | Implemented, documented in `docs/RECORDED_VIDEO.md`, tested on a real file. |
| M3f | Idempotent replay IDs; new sessions traceable | Done | Shown on PostgreSQL: resend is a duplicate, a new session is a separate alert. |
| M3g | "Follow with actual CPU model output" | Not done | Waits on the model bundle. |
| M4a | File selection, RECORDED badges, video time, state, shared controls in both interfaces | Tested (mock) | Built; type-check, 22 frontend tests, scripted DOM runs. Never seen in a browser. |
| M4b | Controls from either view update the other; switching cannot mix evidence | Tested (mock) | One shared session by design; a new session clears the screen (tested each side). No single test spanning both pages. |
| M4c | Preview alone does not start analysis | Done | Unchanged behaviour; existing tests still pass. |
| M4d | Full candidates in the protected console only; masked metadata for the operator | Done | Allow-list tests on both sides; cross-process audit found no plate text or frame in operator output. |
| M5 | Baseline on labelled development footage; isolate the failing stage | Not done | Tool names the failing stage per passage. Needs models, footage, labels. |
| M6a | Genuine accepted recorded output through outbox, ingestion, synthetic watchlist | Tested (mock) | Path proven; output was not genuine. |
| M6b | Alert visibility, acknowledge, resolve, route order, reports | Tested (mock) | All exercised over HTTP on PostgreSQL. |
| M6c | Tests: API outage, queued retry, duplicate delivery, worker restart, replay, EOF, missing PTS, corrupt input, stale data, absent checkpoints | Tested (mock) | Every one has a passing test or audit. |
| M6d | Five-minute outage, zero missing or duplicate logical events | Tested (mock) | 300-second outage, 2 of 2 events delivered once, repeat delivery ignored. |
| M6e | Audit decisions survive restart | Partly | Already established in the handoff for the central service. Not re-run here. |
| M7 | Freeze, held-out run, report metrics, rehearse and record demos, publish | Not done | Needs everything above. |

## D. Completion guide, page 13: closing checklist

| # | Checklist line | Status |
| --- | --- | --- |
| D1 | Clean CPU restore, startup and real inference documented on the teammate machine | Not done (restore in progress, no inference) |
| D2 | Permitted recordings and independent labels exist | Not done |
| D3 | Recorded input, PTS, backpressure, controls, EOF/seek/reset and labels work through both interfaces | Tested (mock); not seen in a browser |
| D4 | Held-out outcomes with all denominators; targets measured | Not done |
| D5 | One correct recorded plate completes delivery, alert, operator action, incident/route report | Tested (mock) |
| D6 | Recorded replay/restart and five-minute outage pass without missing or duplicate events | Tested (mock) |
| D7 | Updated setup, limits, hashes, benchmark results and a labelled recorded demonstration are published; second clean restore verifies it | Not done |

## E. Status document: Priority 0 work (pages 11 and 12)

| # | Item | Status |
| --- | --- | --- |
| E1 | Restore and establish a reproducible baseline | Partly (suites reproduce the handoff counts; assets not restored) |
| E2 | Obtain permitted recorded footage and labels | Not done |
| E3 | Implement recorded input in the shared pipeline | Tested (mock): the code is complete, results come from a stub |
| E4 | Extend the contracts and both interfaces | Tested (mock): contracts proven on PostgreSQL, interfaces not seen in a browser |
| E5 | Prove the full operator workflow with a real labelled passage | Not done |
| E6 | Publish measurements and record the new demonstration | Not done |

Priority 1 and Priority 2 work in the status document (bottleneck
improvement, fair OCR comparison, tuning on broader data, calibration,
hardening, registry support, regional scaling) is **not started**. The
documents place it after a measured baseline, so that is expected.

## F. Status document, page 10: original acceptance conditions

| # | Condition | Status |
| --- | --- | --- |
| F1 | Two cameras concurrently for 15 minutes with measured freshness | Not done; out of the recorded-video direction unless you still claim it |
| F2 | A correctly read, labelled real passage through the whole workflow | Not done |
| F3 | Broad measurement on unseen, human-reviewed Indian footage | Not done |
| F4 | Repeated end-to-end latency and resource measurements | Not done |
| F5 | Full clean-machine recovery | Partly (this is that machine; restore under way) |
| F6 | Recorded-video workflow acceptance, playback controls and source labels | Tested (mock) |

---

## What I can still finish without you

1. Complete the model download and run the selective restore.
2. Start the recorded service with the real models and get the first real CPU
   reading and its timing (closes M1c, M3g, B3, raises Tasks 2, 4, 7).
3. Run `scripts/smoke-test.py` and check the saved snapshot counts (M1a, M1b).
4. Compare parent and adapted OCR on CPU, if the benchmark crops fit on disk
   (B5; that bundle is another 4.7 GB).

## What only you can do

1. **Provide permitted Indian CCTV recordings** with plates a person can read.
2. **Have a person label them** in the format in `docs/RECORDED_VIDEO.md`.
3. **Open both interfaces in a browser** and confirm they look and behave right.
4. Decide whether the two-camera 15-minute test (F1) is still a goal.

Once 1 and 2 exist, the remaining items (A4, A5 for real, M5, M7, D2, D4, D5,
D7, E5, E6, F2, F3) become runnable with tools that are already built.

## Honest limits of this verification

- Scores and statuses describe evidence on this machine today. They are not a
  forecast of how well the models will read real plates. The handoff documents
  record zero accepted readings on live Sentinel cameras; nothing here changes that.
- "Tested (mock)" items could still fail with real models, for example if the
  CPU Paddle runtime in the bundle does not load, or real video files behave
  differently from the generated test clip.
