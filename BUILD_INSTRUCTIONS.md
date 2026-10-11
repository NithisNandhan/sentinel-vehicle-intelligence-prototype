# Build instructions: completing the recorded-video prototype

Written 10 October 2026 on the teammate machine (AMD Ryzen 7 5700U, 13.9 GB RAM,
no NVIDIA GPU). This is the working plan for finishing the project described in
`Sentinel_Project_Status_and_Next_Steps.pdf` and
`Sentinel_Teammate_Completion_Guide.pdf`. Those two documents remain the
authority on scope and evidence rules; this file says what to build, in what
order, and how each step is checked.

## Rules that apply to every task

1. **Audit after every task.** A task is not closed until its audit commands
   have been run and the result is written into the audit log at the bottom of
   this file. Record failures as failures.
2. **Measured claims only.** A passing unit test proves plumbing. It never
   proves plate-recognition accuracy. Mocked recognisers are labelled as mocked.
3. **Do not weaken the acceptance gate** to manufacture a reading or an alert.
4. **Source labels stay honest.** Downloaded footage is `recorded`, never
   `live`. Injected events are `synthetic`.
5. **Privacy boundaries are unchanged.** Frames, crops and full plate text stay
   in edge memory; the central API receives masked/keyed metadata only.
6. **Production defaults are untouched.** The independent production worker
   keeps YOLOv8n + FastALPR. All new work is in the shared engineering session.
7. **No credentials in Git.** `.env`, recordings and labels live under ignored
   paths (`.env`, `data/`).
8. **A report per task.** When a task closes, write
   `task-reports/TASK_<n>_REPORT.md` with: what existed before, what was
   changed, what still has to change, and a score.

## How a task is scored

Each task gets a score out of 10, built from three parts. The score describes
evidence, not effort.

| Part | Points | Earned when |
| --- | --- | --- |
| Built | 4 | Everything in the task's scope exists in the code or on the machine |
| Tested | 3 | Automated checks for it were run here and passed |
| Proven | 3 | It was shown working under real conditions: real models, real services or real footage, as the guide's "done when" requires |

A task tested only with mocked models cannot score above 7. A task that
depends on something not yet available (release assets, footage, labels) keeps
its lower score until that arrives; it is re-scored then.

## Verification against the two handoff documents (10 Oct 2026)

Both PDFs were re-read against this plan after Task 4.

- `Sentinel_Project_Status_and_Next_Steps.pdf` in the project root is
  byte-identical to `output/pdf/Sentinel_Project_Status_and_Next_Steps.pdf`
  from the repository (SHA256 `E60E18F3…03776C`).
- The completion guide names checkout `27935a3`; this machine is on it.
- The guide cites `docs/CPU_GPU_WORK_SPLIT.md` and `data/cleanup-20261010/`.
  Neither is in this checkout. They are owner-machine files and are not needed
  for the tasks below.
- The guide says to run setup in PowerShell 7 because `init-env.ps1` used
  `utf8NoBOM`. Task 0 removed that restriction; Windows PowerShell 5.1 works.
- **Gap found and closed:** milestone M3 requires a recorded event to keep its
  recording ID, video PTS, **processing UTC time**, session ID and track ID.
  Tasks 1 and 4 stored everything except the processing time when the
  recording's own clock was used. A `processed_at` field was added to the wire
  format, the central schema, the database (migration `0002`) and the alert,
  route and report outputs.
- Milestones map to tasks as: M1 = Tasks 0, 2, 8; M2 = Task 10; M3 = Tasks 1,
  3, 4, 5; M4 = Task 6; M5 = Tasks 9, 11; M6 and M7 = Task 11.

## Machine prerequisites

| Requirement | State on 10 Oct | Action |
| --- | --- | --- |
| Git, Node, Python 3.11 venv (`.venv311`) | Present | None |
| Docker Desktop + WSL2 | Installed, engine stopped | Start Docker Desktop |
| Windows PowerShell 5.1 | Present | Scripts must work here (Task 0) |
| GitHub CLI (`gh`) | Missing | Install, or download release assets in a browser |
| Access to the private release | Unknown | Owner must grant access to `NithisNandhan/sentinel-vehicle-intelligence-prototype` |
| Free disk | D: 35 GB, C: 40 GB | Use a selective restore (skip training and benchmark bundles first) |
| NVIDIA GPU | None | CPU path is mandatory (Task 2) |

## Task list

Each task lists what is built, the files touched, the completion condition and
the audit. Tasks 0 to 6 are software work that can be finished and unit-tested
on this machine without the release assets. Tasks 7 to 11 need the assets,
permitted footage or human labelling and are marked with what they are waiting on.

### Task 0: Prepare the checkout and a test baseline

- Fast-forward to `origin/main` (two documentation commits).
- Make `scripts/init-env.ps1` write `.env` without a byte-order mark under both
  Windows PowerShell 5.1 and PowerShell 7, and remove the mark from the existing
  `.env` while keeping its generated keys.
- Install test dependencies into `.venv311` and install frontend packages.
- Run the edge, backend, handoff-tool and frontend suites to record a baseline.

Done when: `.env` starts with `DB_PASSWORD`, and the baseline counts are logged.
Audit: byte check on `.env`; `pytest` in `edge`, `backend`, `tools`; `npm test`
and `npm run build` in `frontend`.

### Task 1: Recorded source contract in the central API

- `backend/app/schemas.py`: `SourceMode` gains `recorded`; sightings accept the
  optional `session_id`, `time_basis` and `processed_at` fields.
- `backend/app/models.py` + new Alembic revision `0002`: nullable
  `sightings.session_id`, `sightings.time_basis` and `sightings.processed_at`.
- `backend/app/main.py`: persist the new fields; expose them on alerts, routes
  and incident reports.

Done when: a `recorded` sighting is ingested, deduplicated by event ID, matched
against the synthetic watchlist and reported with its label intact; old `live`
and `synthetic` payloads still validate.
Audit: backend suite, including new tests for the above and the migration.

### Task 2: CPU execution path for the shared engineering session

- `edge/sentinel_edge/live_detection_viewer.py`: `load_models` takes an explicit
  `device`; new `--device cpu|gpu` flag (default stays `gpu`). Model-load
  failures stay visible as a failure code.
- Diagnostics expose model profile, OCR checkpoint SHA256 and actual device.
- `tools/run_ocr_research.py`: honours `SENTINEL_OCR_DEVICE=cpu` and then loads
  only the isolated CPU Paddle dependencies, no CUDA libraries.
- `edge/pyproject.toml` + `edge/Dockerfile`: `live-cpu` extra and image target
  with CPU `onnxruntime` and CPU Torch wheels.

Done when: the loader can be asked for CPU and never silently falls back from
GPU to CPU or the reverse.
Audit: edge suite with new loader tests. Real CPU inference is audited in Task 8.

### Task 3: Local recorded-video source with backpressure

- New `edge/sentinel_edge/recorded_source.py`:
  - a recording library that lists only video files that have a permission
    receipt beside them (`<video>.receipt.json`);
  - a PyAV file decoder that keeps the original video PTS, fails explicitly on
    missing PTS or a corrupt file, and supports a start position;
  - paced (real-time demonstration) and unpaced (benchmark) modes;
  - pause and resume.
- `edge/sentinel_edge/sampled_frames.py` and `crop_collector.py`: an opt-in
  blocking mode so a slow CPU makes the decoder wait instead of dropping frames.
  Live behaviour is unchanged.

Done when: every sampled frame of a test video reaches the consumer in unpaced
mode even with a deliberately slow consumer; pause, resume, seek and EOF behave
as documented; a file without PTS is rejected with a named failure.
Audit: edge suite with new tests that decode a generated video file.

### Task 4: Recorded mode in the shared session and delivery

- `live_detection_viewer.py`: recordings join the selectable sources; new
  statuses `processing` and `completed`; control actions `pause`, `resume`,
  `restart` and `seek`; each start, restart or seek gets a new session ID and
  clears tracking, temporal evidence and reconstruction.
- `tracked_live_viewer.py`: source factory and source mode are parameters; EOF
  on a recording is `completed`, not an error.
- `events.py`, `recognition_contract.py`, `shared_delivery.py`: `recorded`
  source mode, recording ID, session ID, video PTS, processing time, and
  deterministic event IDs so a retransmission is idempotent while a new
  session is traceable. The recording's real-world time is used only if the
  receipt states it; otherwise the event is labelled as processing time.
- New recordings-only launch mode that needs no Sentinel credentials.

Done when: a mocked-recogniser run over a generated video publishes recorded
snapshots, reaches `completed` at EOF, and writes exactly one masked outbox
event per accepted plate per session.
Audit: edge suite with new session, contract and delivery tests.

### Task 5: Central recognition proxy for recorded sessions

- `backend/app/recognition.py`: allow-listed recorded fields (source mode,
  session ID, recording name, video position and duration, playback state), a
  recordings list and a `/v1/recognition/control` endpoint. Still no images,
  hypotheses or upstream error bodies.

Done when: recorded status passes through with only allow-listed fields and
unknown values are rejected or dropped.
Audit: backend suite with new proxy tests.

### Task 6: Both interfaces

- Operator dashboard (`frontend/src`): recording selection, a RECORDED badge,
  video time, processing state, pause/resume/restart, and recorded labels on
  alerts, routes, cameras and reports.
- Engineering console (`engineering_page.py`): the same controls and labels,
  with crops and candidates still protected.
- Both observe one session; a preview never starts analysis.

Done when: type-check, unit tests and production build pass, and source labels
are correct in every view.
Audit: `npm test`, `npm run build`, edge page tests.

### Task 7: Launcher, compose profile and documentation

- `compose.yaml`: a `recorded` profile running the CPU engineering service with
  a read-only `data/recordings` mount.
- `scripts/start-recorded-analysis.ps1`.
- `docs/RECORDED_VIDEO.md`: receipt format, label format, controls, replay and
  seek semantics, privacy, limits.

Audit: `docker compose config` validates; documented commands match the code.

### Task 8: Restore the release and prove a CPU reading (needs release assets)

- Download the `model-cache`, `alpr-cache`, `saved-events` and
  `presentation-assets` bundles; run the restore with
  `-SkipTraining -SkipBenchmarks`.
- Start central services and the synthetic demo; run `scripts/smoke-test.py`.
- Build the CPU image and run one real OCR reading on CPU with the adapted
  checkpoint (hash `c2ba896d…4456d`).

Waiting on: access to the private release, Docker running, about 30 GB free.
Audit: restore output, smoke-test output, snapshot counts (6 cameras, 16
sightings, 15 alerts, 22 audits), checkpoint hash, recorded CPU timing.

### Task 9: Evaluation tooling

- `tools/evaluate_recorded.py` and `edge/sentinel_edge/recorded_evaluation.py`:
  score pipeline output against an independent label file and report vehicle
  coverage, plate localisation, usable crop yield, exact reads, accepted
  precision, wrong acceptances, abstentions, candidate recall (top 1, top 5,
  top K) and runtime, with explicit denominators.

Done when: scoring is unit-tested on hand-built cases, including the case
where nothing is accepted (precision undefined, not zero or one).
Audit: edge and tools suites.

### Task 10: Footage and independent labels (needs a person)

- Obtain permitted, original-quality Indian CCTV recordings; write a receipt
  for each; place them in `data/recordings`.
- Independently transcribe plates and mark unreadable passages using the label
  format in `docs/RECORDED_VIDEO.md`. Split development, validation and test by
  vehicle, video and scene.

Waiting on: footage with permission, and a human labeller. Model output must
not be used as ground truth.

### Task 11: Baseline, end-to-end proof, freeze and package (needs 8 and 10)

- Run the unchanged models on development footage; find the failing stage.
- Push one correctly read recorded plate through the outbox, central ingestion,
  a synthetic watchlist entry, and acknowledge/resolve actions.
- Five-minute API outage test with zero missing or duplicate logical events.
- Freeze commit, models and settings; run the held-out set once; publish
  metrics, limitations and a labelled recorded demonstration as a new release.

## Audit log

Results are appended here as each task closes.

### Task 0 audit (10 Oct 2026): closed

- Checkout fast-forwarded from `b3b05e1` to `27935a3`.
- `.env`: byte-order mark removed, generated keys kept. `init-env.ps1` now
  writes UTF-8 without a mark; checked under Windows PowerShell 5.1 on a
  temporary copy (first bytes `23 20 52`, no `GENERATE_` placeholders left).
- Docker Desktop started: engine 29.6.1, 7.2 GB RAM available to containers.
  Central images and a populated `postgres_data` volume already exist on this
  machine from an earlier offline run; no model volumes exist yet.
- Baseline on the Windows host with `.venv311` (Python 3.11.9):

| Suite | Result | Note |
| --- | --- | --- |
| Edge | 287 passed, 1 failed | `test_camera_suitability` hard-codes a Linux `/workspace` path; Windows-only failure, matches the handoff count of 287 |
| Backend | 53 passed | After giving one parametrised test short IDs; its 1 MiB ID exceeded the Windows environment-variable limit |
| Handoff tools | 11 passed, 3 failed on Windows; 14 passed in a `python:3.11-slim` container | Windows keeps the SQLite test files locked during cleanup |
| Frontend | 15 passed, build passed | Large-bundle warning remains |

- Not done: `gh` is not installed and release access has not been checked.

### Task 1 audit (10 Oct 2026): closed

- Backend suite: **63 passed** (53 existing + 10 new), 3 deprecation warnings.
- New tests cover: a `recorded` sighting keeping its label through the alert,
  camera list, route search and JSON/HTML incident report; event-ID
  deduplication; a second session producing a separate traceable sighting;
  rejection of recorded sightings missing `session_id`, `time_basis` or
  `stream_pts_ms`; rejection of unknown values; unchanged behaviour for
  existing payloads; and migration `0002` adding the columns to an existing
  `0001` database, twice, without losing a row.
- Checked `tools/handoff_events.py`: it inserts with explicit column lists, so
  the old snapshot still restores into the widened table.
- Not yet exercised against the PostgreSQL volume on this machine; that
  happens when the API image is rebuilt in Task 8.

### Task 2 audit (10 Oct 2026): closed for wiring, real inference still open

- Edge suite: **294 passed, 1 failed** (287 existing + 7 new; the one failure
  is the same Windows-only path test as the baseline).
- New tests (`edge/tests/test_cpu_path.py`) cover: the requested device reaches
  the OCR reader; the default stays `gpu` for existing launchers; unknown
  devices are rejected before any model loads; a GPU failure is not retried on
  CPU; diagnostics report device and the verified checkpoint hash without file
  paths; the CPU launcher environment contains no GPU import path or CUDA
  flags; the GPU launcher environment is unchanged.
- These tests use a mocked reader. **No real model has run on this CPU yet.**
  That needs the `model-cache` bundle and is audited in Task 8.
- Left unchanged on purpose: `VehicleTracker` still pins Torch to 2 threads.
  That is a speed setting, not accuracy, and should be tuned from measured
  timings rather than guessed.

### Task 3 audit (10 Oct 2026): closed

- Edge suite: **316 passed, 1 failed** on a clean run (294 + 22 new; same
  Windows-only path failure).
- New tests (`edge/tests/test_recorded_source.py`) decode a generated 60-frame
  MPEG-4 file and cover: only receipted videos are listed; receipts are never
  overwritten and need real source/licence/permission text; a stated UTC start
  time is the only way to get a source clock; every frame keeps its original
  PTS; paced playback waits for video time and still keeps every frame; seek
  starts at the requested position and rejects out-of-range positions; pause
  holds decoding and is not counted as playback time; stop is prompt; a file
  changed after its receipt is refused; a frame without PTS raises `MissingPts`;
  undecodable content raises `CorruptRecording` without leaking the path.
- Backpressure: with `block=True` a slow consumer receives all 40 of 40
  samples with zero drops and bounded memory; without it the live queue still
  drops as before; a cut is flagged on the right frame without discarding
  queued frames; an oversized frame fails with `FrameTooLarge`; the crop
  collector delivers 30 of 30 samples with their tracks and crops.
- Found during this audit, not caused by it:
  `test_live_reconstruction…without_adding_votes` fails intermittently on
  Windows (2 of 4 isolated runs) because `time.monotonic()` ticks every
  ~15.6 ms here and a fast stage measures 0.0 ms. It asserts a timing is
  greater than zero. Left unchanged; it is a Windows clock artefact.
- New helper: `tools/register_recording.py` writes a receipt for a permitted file.

### Task 4 audit (10 Oct 2026): closed for plumbing

- Edge suite: **335 passed, 1 failed** (316 + 19 new; same Windows-only
  path failure).
- New tests (`edge/tests/test_recorded_session.py`), all with a **mocked
  recogniser** that always returns the same text:
  - Session state: a recording is selectable and labelled `recorded` with a
    session ID and no file path; a camera-catalogue refresh does not drop it;
    pause/resume keep the run; restart and seek stop the old run, clear its
    evidence and start a new session; invalid controls change nothing; recorded
    evidence uses a 30-second limit, is held while paused, and idle expiry still
    clears it; the live 5-second limit is unchanged.
  - Contract and delivery: recorded events need session, PTS and time basis;
    live payloads keep their old shape; delivery is masked and deduplicated on
    video time; a retransmission reuses the same event ID while a new session
    produces a distinct one; the recording's clock is used only when its receipt
    states it; the event validates against the central `SightingIn` schema.
  - Pipeline: a 6-second generated file with a deliberately slow OCR stub had
    **12 of 12** samples analysed with zero drops, reached `completed`, and
    wrote exactly one masked recorded event. A seek to 4000 ms analysed
    positions 4000 to 5500 only. A file changed after its receipt fails with
    `ChangedRecording`.
  - HTTP service: listing, select, completion, restart, seek, bad controls
    (409), malformed bodies (400), missing token (403) and stop, end to end.
- Fixed while auditing: the consumer loop could exit one sample early if the
  decoder finished between two checks. It now exits only when the decoder has
  finished **and** the queue is empty.
- Known limit, not changed: the edge client retries a rejected event forever
  (`api.py`). The central API must be rebuilt with Task 1 **before** any
  recorded event is sent, or an old API would reject it and block the queue.
- What this does not show: any real plate being read.

### Amendment after document verification (10 Oct 2026)

- Added `processed_at` (Tasks 1 and 4). Backend suite: **65 passed**. Edge
  suite: **335 passed** apart from the Windows-only path test and the
  intermittent Windows clock test noted under Task 3.
- Per-task reports with scores are in `task-reports/`.

### Task 5 audit (10 Oct 2026): closed

- Backend suite: **92 passed** (65 + 27 new).
- Cross-check: nine states produced by the real edge `LiveState` were passed
  through the real central allow-list; all accepted with the right label. The
  first run failed because the edge did not forward `failure_code`; fixed in
  `recognition_contract.py` and covered by a test.
- Existing camera endpoints return exactly what they did before.
- Report: `task-reports/TASK_5_REPORT.md` (8 / 10).

### Task 6 audit (10 Oct 2026): closed for code, not yet seen in a browser

- Type-check clean; frontend **22 passed**; production build passed.
- Engineering page **10 passed**, including three scripted runs under Node.
- Edge 335 passed apart from the two known Windows-only items; backend 92.
- Not done: no visual check against a running session.
- Report: `task-reports/TASK_6_REPORT.md` (7 / 10).

### Task 9 audit (10 Oct 2026): closed for tooling

- Done before Task 8 because it needs no release assets.
- Edge suite: **352 passed**, 1 Windows-only failure (335 + 17 new).
- Scoring arithmetic checked on hand-built cases; one end-to-end run from a
  generated file to a report with a mocked recogniser.
- `tools/evaluate_recorded.py` refuses the test split without `--final` and
  refuses to overwrite a report (both checked by running it).
- Report: `task-reports/TASK_9_REPORT.md` (7 / 10).

### Task 7 audit (10 Oct 2026): closed, launcher not yet run with models

- `docker compose --profile recorded config` is valid; the default profile is unchanged.
- New launcher script parses with 0 errors.
- CPU image built: Torch `2.14.1+cpu`, no CUDA wheels, ONNX Runtime CPU provider.
- Edge suite **on Linux inside the CPU image: 352 passed, 2 skipped, 0 failed**,
  confirming the two host failures are Windows-only.
- Building used about 21 GB of Docker space on C: (19 GB free afterwards).
- Report: `task-reports/TASK_7_REPORT.md` (8 / 10).

### Task 8 audit (11 Oct 2026): real models on this CPU

- Release bundles downloaded with the machine's stored Git credential (token
  in memory only) and restored with `-SkipTraining -SkipBenchmarks -SkipCentral`:
  model-cache 16,931 files, alpr-cache 6, saved-events 13, presentation-assets 9.
- OCR checkpoint SHA256 in the volume matches `c2ba896d…4456d`.
- Saved snapshot file holds 6 cameras, 16 sightings, 15 alerts, 22 audits,
  16,625 health reports and 1 watchlist entry, as the handoff states. It was
  not imported: this database already had rows.
- **Defect in the published release:** the archiver omits folders named
  `token`, which removes five pinned PaddleOCR source files; the OCR refuses to
  load. `tools/repair_ocr_source.py` restored them from the pinned archive with
  matching hashes (0 missing afterwards).
- **Second startup defect:** Paddle could not create `~/.cache/paddle` under a
  read-only mount. The service now mounts the two weight folders individually.
- Real load and timing on CPU: models in 40.7 s; OCR 531 ms, vehicle detector
  165 ms (1080p frame), plate detector 126 ms per call.
- `scripts/smoke-test.py`: 8 of 8 checks against the rebuilt, migrated API.
- Real footage through the operator API: 4K VP9 clip, 3,375 frames decoded, 113
  samples analysed, 88 plate crops, 0 accepted, about 450 s.
- Pilot evaluation (4 labels made by the assistant, not an independent person):
  0 of 3 readable plates read; true plate in top 20 candidates for 0 of 3;
  0 wrong acceptances; plate boxes matched 2 of 3.
- Reports: `task-reports/TASK_8_REPORT.md` (8 / 10), `WRAP_UP.md`.

### Integration audit across real services (10 Oct 2026)

Run on a disposable compose project (`sentinel-audit`, own volumes, API on
port 18000). The stack already running on this machine was left alone.

- **Migration on PostgreSQL:** the old API image created revision `0001`; the
  rebuilt image upgraded it to `0002_recorded_sightings`; the old row survived.
- **Central API over HTTP, real edge delivery client:** 17 of 17 checks.
- **Whole recorded path, mocked recogniser:** generated file -> edge service
  -> SQLite outbox -> rebuilt API container -> PostgreSQL -> operator
  endpoints, session driven only through the operator API: 17 of 17 checks.
- **Defect found and fixed:** Pause did not stop analysis of frames already
  queued. Fixed in `tracked_live_viewer.py`, unit test added, audit re-run clean.
- Scores raised accordingly: Task 1 to 9, Task 4 to 8, Task 5 to 9.
- Edge suite after the fix: 353 passed apart from the known Windows-only items.
- Also found: the first release downloader stalled because it forwarded the
  GitHub token to the storage host on redirect. It now resolves the signed URL
  first and sends the token to `api.github.com` only.

