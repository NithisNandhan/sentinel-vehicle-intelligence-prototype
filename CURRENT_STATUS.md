# Sentinel recorded-video prototype: current status

Written 10 October 2026, 23:40, on the teammate machine (AMD Ryzen 7 5700U,
13.9 GB RAM, no NVIDIA GPU). Nothing here is committed to Git yet.

## How much is complete

**About half, measured against what the handoff documents call "finished".**

| Way of measuring | Result | What it tells you |
| --- | --- | --- |
| Code the documents asked for | **about 90% written** | Almost all the software exists and passes its tests |
| Evidence against the "finished" criteria | **48%** | Half of the proof is still missing |
| Task scores (12 tasks, 10 points each) | **75 of 120 (63%)** | Nine tasks closed, one in progress, two not started |

The 48% comes from the 58 requirements checked in
`task-reports/VERIFICATION_AGAINST_HANDOFF.md`: 15 done with real evidence
(counted in full), 16 tested only with a mocked recogniser (counted at 0.6),
8 partly done (0.4) and 19 not done (0). The weights are my judgement; the raw
counts are the facts.

**The single biggest gap: no real model has read a real plate on this machine.**
Every "accepted plate" in every test so far came from a stub that always
returns the same text.

## Where we are

### Running right now

| Service | Address | State |
| --- | --- | --- |
| Operator dashboard | <http://127.0.0.1:5173/> | Up, rebuilt with the new code, opens already unlocked |
| Central API | <http://127.0.0.1:8000/docs> | Up, rebuilt; database upgraded to revision `0002_recorded_sightings` |
| PostgreSQL database | internal | Healthy; 2 cameras, 5 sightings, 4 alerts, about 1,044 health reports (demo data from this machine) |
| Demo simulator | internal | Up; sends labelled synthetic events |
| Media gateway | ports 8554, 8888 | Up |
| Engineering console / recognition service | <http://127.0.0.1:7780/> | **Not running.** Waiting for the model restore |

`scripts/smoke-test.py` passed all eight checks against the running API.

### In progress at the time of writing

- **Model restore.** All 22 parts of the model bundle (5.4 GB) are downloaded.
  The restore tool is verifying hashes and unpacking them into the
  `sentinel-prototype_model_cache` Docker volume. The first attempt stopped
  before changing anything (a PowerShell 5.1 quirk with redirected output);
  the second is running.

### Task by task

| Task | What it is | Score | State |
| --- | --- | --- | --- |
| 0 | Checkout, `.env` fix, test baseline | 9 | Closed |
| 1 | `recorded` source in the central API and database | 9 | Closed; migration now also applied to this machine's own database |
| 2 | CPU path for the shared session | 6 | Closed for wiring; no real CPU inference yet |
| 3 | Local video reader that never drops frames | 9 | Closed |
| 4 | Recorded mode, pause/resume/restart/seek, delivery | 8 | Closed for plumbing |
| 5 | Operator API for recorded sessions | 9 | Closed |
| 6 | Dashboard and engineering console | 7 | Closed for code; not confirmed on screen |
| 7 | Compose profile, launcher, CPU image, documentation | 8 | Closed; launcher not yet run with models |
| 8 | Restore models and prove a real CPU reading | 3 | **In progress**: download done, restore running |
| 9 | Evaluation tool and label format | 7 | Closed for tooling |
| 10 | Permitted footage and human labels | 0 | **Not started: needs you** |
| 11 | Baseline, end-to-end proof, freeze, package | 0 | Not started: needs 8 and 10 |

Scoring: 4 points built, 3 tested, 3 proven under real conditions.

### Test results

| Suite | Before I started | Now |
| --- | --- | --- |
| Backend | 53 passed | 92 passed |
| Edge, on Linux in the CPU image | 287 passed | 352 passed, 2 skipped, 0 failed |
| Edge, on this Windows host | 287 passed, 1 failed | 353 passed; 1 always fails and 1 fails intermittently, both Windows-only |
| Frontend | 15 passed | 22 passed; type-check clean; build passes |

Three audits on a throwaway stack with real services (mocked recogniser):

- Central API on PostgreSQL with the real edge delivery client: 17 of 17.
- Whole recorded path, file to alert, driven through the operator API: 17 of 17.
- Five-minute API outage: 12 of 12; both queued events arrived exactly once.

## What has to be updated

### On this machine

1. **Finish the model restore**, then start the recognition service
   (`scripts/start-recorded-analysis.ps1`). In progress.
2. **Put at least one permitted video in `data/recordings`** and register it
   with `tools/register_recording.py`. The folder is empty, so there is
   nothing to analyse even once the service is up.
3. **Disk space.** C: has 18.7 GB free (it had 40 GB before the CPU image was
   built). D: has 29.1 GB. The two skipped bundles (training 4.1 GB, benchmark
   4.7 GB) do not fit comfortably on C: once unpacked.
4. **The saved handoff snapshot is not imported.** The handoff's 6 cameras, 16
   sightings and 15 alerts are not in this database, because the restore tool
   refuses a database that already has rows and I did not delete yours.

### In the code and documents

5. **Nothing is committed.** 29 changed files and about 20 new ones are in the
   working tree only. They should go on a branch, as the guide asks.
6. **Dashboard auto-unlock is on.** I added `compose.autounlock.yaml` at your
   request. Anyone at this PC can open the dashboard without the key. Run
   `docker compose up -d dashboard` to turn it off.
7. **Task reports 2, 6, 7, 8** need re-scoring once the service has run with
   real models and someone has looked at the interfaces.
8. **Two tests fail on the Windows host only** (a hard-coded Linux path, and a
   timing assertion that hits Windows' 15.6 ms clock). They pass on Linux.
   Left unchanged.
9. **`docs/PROJECT_STATUS_AND_NEXT_STEPS.md` and the two PDFs are now out of
   date** where they say recorded input "is to be built". I did not edit them;
   they are the owner's record of 8 and 10 October.
10. **OpenCV 5.0.0** was pulled into the CPU image by a dependency, although
    the project's own pin is below 5. Tests pass; real model output on it has
    not been compared with the owner's recorded results.

### Known limits that are still open

- The recorded session sends sightings but no camera-health reports.
- The edge client retries a rejected event forever (unchanged behaviour).
- No calibration tooling; the acceptance threshold is unchanged and uncalibrated.
- The handoff records zero accepted readings on live Sentinel cameras. Nothing
  in this work changes or disproves that.

## Next steps

### I can do these without you

| # | Step | Closes |
| --- | --- | --- |
| 1 | Finish the restore and check the model hashes against the manifest | Task 8, M1b |
| 2 | Start the recognition service on CPU with the real models | Tasks 2, 7 |
| 3 | Get one real OCR reading on CPU and record how long it takes | Task 8, M1c, M3g |
| 4 | Run a generated clip through the real models to confirm the session works end to end (it will contain no real plates, so it proves the runtime, not accuracy) | Task 4 |
| 5 | Put the work on a Git branch, if you want it committed | Item 5 above |

### Only you can do these

| # | Step | Why it matters |
| --- | --- | --- |
| A | **Provide permitted Indian CCTV recordings** with plates a person can read | Every accuracy item waits on this |
| B | **Have a person label them** using the format in `docs/RECORDED_VIDEO.md` | Model output cannot be its own ground truth |
| C | **Look at both interfaces in a browser** and say whether they behave correctly | Task 6 stays at 7 until someone does |
| D | Decide whether the original two-camera, 15-minute live test is still a goal | It is outside the recorded-video direction |

### Once A and B exist

1. Run `tools/evaluate_recorded.py` on the development split to get the first
   real baseline and see which stage fails (vehicle, plate crop, OCR, fusion or gate).
2. Fix the smallest justified bottleneck. Do not lower the acceptance gate.
3. Push one correctly read real plate through delivery, the synthetic
   watchlist, and acknowledge/resolve.
4. Freeze the commit, models and settings; run the held-out test split once.
5. Record a labelled demonstration and publish a new release with hashes.

## Where to find things

| What | Where |
| --- | --- |
| The plan, rules and audit log | `BUILD_INSTRUCTIONS.md` |
| One report per task, with before/after and score | `task-reports/TASK_<n>_REPORT.md` |
| Line-by-line check against the two handoff PDFs | `task-reports/VERIFICATION_AGAINST_HANDOFF.md` |
| How to use recorded footage, receipts, labels, measurement | `docs/RECORDED_VIDEO.md` |
| Start recorded analysis | `scripts/start-recorded-analysis.ps1` |
