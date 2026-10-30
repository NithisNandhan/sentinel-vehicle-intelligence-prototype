# Recorded footage in the shared recognition session

The shared engineering session can analyse a permitted local video file
instead of a live camera. The vehicle tracking, plate cropping, OCR,
multi-frame fusion, reconstruction and acceptance gate are the same code. Only
the source of frames changes, and everything it produces is labelled
`recorded`.

This document describes what is built. It makes no accuracy claim. No
measurement on real recorded footage has been made yet.

## What you need

- The `model-cache` and `alpr-cache` bundles restored from the handoff release
  (`scripts/restore-teammate-handoff.ps1`). Models are never downloaded at run time.
- Docker Desktop running. No NVIDIA GPU and no Sentinel credentials.
- Video files you have explicit permission to download and use.

## 1. Register a recording

Put the video in `data/recordings/` (ignored by Git) and write its receipt:

```powershell
.\.venv311\Scripts\python.exe tools\register_recording.py data\recordings\junction.mp4 `
    --source "where the file came from" `
    --licence "its licence" `
    --permission "why you may download and use it" `
    --title "Junction A, evening"
```

This creates `junction.mp4.receipt.json` beside the video with the file's
SHA256. A video with no valid receipt is never listed. If the file changes
after the receipt was written, analysis is refused (`ChangedRecording`). A
receipt is never overwritten.

Add `--recorded-start-utc 2026-09-01T06:30:00Z` **only** if the real time of
the first frame is genuinely known. Without it, events carry the time of
analysis and are labelled `time_basis: processing_time`. A recording time is
never guessed.

Supported containers: `.mp4`, `.mkv`, `.mov`, `.avi`, `.m4v`, `.ts`, `.webm`.

## 2. Start the session

```powershell
.\scripts\start-recorded-analysis.ps1            # at most real time
.\scripts\start-recorded-analysis.ps1 -Unpaced   # as fast as this machine allows
```

This builds and starts the central API, the dashboard and the
`recorded-analysis` service (CPU). Then open the unified local dashboard,
choose a recording and start analysis. No manual operator unlock is required:

- Sentinel dashboard: <http://127.0.0.1:5173/>
- Model evidence: select a vehicle track and open its Inspector.

All sections share one session. Opening a page or a preview never starts
analysis. The API is rebuilt with the service on purpose: an API from before
this change would reject recorded events.

### Optional NVIDIA GPU mode

The dashboards remain browser applications. This option accelerates the shared
YOLO vehicle detector, ONNX plate detector and Paddle OCR service behind them.
It does not change model weights, frame sampling, plate thresholds or privacy.

```powershell
# Reuse the existing GPU-enabled live image and restored model volumes:
.\scripts\start-recorded-analysis.ps1 -Device gpu -SkipBuild
# On a GPU machine without the image, omit -SkipBuild to build it first.
# Return to the CPU-only workflow (also supported on the teammate's machine):
.\scripts\start-recorded-analysis.ps1 -Device cpu -SkipBuild
```

GPU mode layers `compose.recorded-gpu.yaml` over the default file and reserves
one NVIDIA GPU. Before replacing the running session, the launch preflight warms
the real OCR model, vehicle tracker and plate detector on in-memory blank inputs,
including actual CUDA inference and ONNX provider validation. Model startup also
rejects a missing GPU/provider rather than silently falling back. The console
model identity reports loaded OCR and detector devices. Restarting the service
ends the previous analysis; select the recording again to begin a new run.
Reported speedups must come from measurements on the same footage and settings;
GPU availability alone proves neither throughput nor recognition accuracy.

## Controls and what they mean

| Control | Effect |
| --- | --- |
| Start / Analyse recording | Begins a new run from the start of the file |
| Pause, Resume | Holds and continues the same run. The current frame stays visible while paused. |
| Restart | A new run from the start |
| Seek | A new run from the chosen video position |
| Stop | Ends the run and clears its evidence |

**A restart or seek is a new run.** It gets a new session ID, and vehicle
tracking, multi-frame evidence and reconstruction all start empty. Evidence is
never carried across a jump in video time. The same happens automatically at a
cut inside the file (a timestamp gap over 2.5 seconds or a backwards step).

**Replay and duplicates.** A recorded event's ID is derived from the session,
the recording, the keyed plate identifier and the video timestamp. Sending the
same event again is therefore ignored centrally. Running the file again is a
new session and produces new, separately traceable sightings. Within one run,
the same identifier is delivered at most once per 60 seconds of video time.

**Pacing.** Paced playback waits so the file is processed no faster than real
time. Unpaced processes as fast as possible. In both modes the decoder waits
for the analysis, so no sampled frame is dropped; on a slow CPU a paced run
simply takes longer than the video. Sampling is every 500 ms of video time,
as for live cameras.

**States.** `connecting`, `processing`, `paused`, `completed`, `unavailable`.
When unavailable, a failure code explains why: `ChangedRecording`,
`CorruptRecording`, `MissingPts` (a frame has no timestamp; the file is
refused rather than guessed), `NoVideoStream`, `FrameTooLarge`.

**Idle expiry.** As with cameras, the session ends and its evidence is cleared
60 seconds after the last interface heartbeat.

## Privacy

Unchanged from live mode. Frames, crops and full plate text stay in the edge
service's memory and the protected engineering console. The central API
receives masked plate text, a keyed identifier, the recording ID, video
timestamp, session ID and model version. File names appear only in the
engineering console's list of rejected files. File paths appear nowhere.

The recording itself stays where you put it, mounted read-only. Nothing is
copied or re-encoded to disk. **Retention of the files in `data/recordings` is
your responsibility** and should follow the terms under which you obtained them.

## 3. Label footage for measurement

Labels must be made by a person looking at the footage, never copied from the
pipeline's output. Write one JSON file (keep it under `data/`):

```json
{
  "schema_version": "1.0",
  "labeller": "name of the person who labelled",
  "labelled_on": "2026-10-12",
  "recordings": [
    {
      "recording_id": "rec-0123456789ab",
      "sha256": "0123456789ab… (the 64-character hash from the receipt)",
      "split": "development",
      "passages": [
        {
          "passage_id": "p001",
          "vehicle_id": "v001",
          "start_ms": 12000,
          "end_ms": 15500,
          "readable": true,
          "plate": "GJ01AB1234",
          "boxes": [
            {"position_ms": 13000, "vehicle": [410, 220, 780, 560], "plate": [540, 470, 660, 505]}
          ]
        },
        {
          "passage_id": "p002",
          "vehicle_id": "v002",
          "start_ms": 20000,
          "end_ms": 23000,
          "readable": false,
          "plate": null,
          "plate_visible": true
        }
      ]
    }
  ]
}
```

- A **passage** is one vehicle passing through view. `start_ms` and `end_ms`
  are positions in the video.
- `readable: true` means a person can transcribe the whole plate from the
  footage; give it in `plate`. `readable: false` needs `plate: null`.
- `plate_visible` says whether a plate can be seen at all. It defaults to true
  for readable passages.
- `boxes` are optional but needed for the vehicle-coverage and
  plate-localisation numbers. Coordinates are native pixels, `[x1, y1, x2, y2]`.
- `split` is `development`, `validation` or `test`, one per recording. Split
  by vehicle, video and scene. The loader rejects a `vehicle_id` that appears
  in more than one split.
- List every eligible vehicle in the labelled stretch, including the ones
  nobody can read. Leaving out failures inflates the result.

## 4. Measure

Run inside the CPU container so the same models and runtime are used:

```powershell
docker compose --profile recorded run --rm --no-deps `
    -v "${PWD}/data:/workspace/data" recorded-analysis `
    python /workspace/tools/run_ocr_research.py /workspace/tools/evaluate_recorded.py `
    --labels /workspace/data/labels/labels.json --split development `
    --recordings-dir /recordings --device cpu `
    --output /workspace/data/evaluation/development-001.json
```

The report gives each number with its denominator:

| Metric | Meaning |
| --- | --- |
| Vehicle coverage | Labelled vehicle boxes matched by a track (IoU ≥ 0.5 within 500 ms) |
| Plate localisation | Labelled plate boxes matched by a predicted plate box |
| OCR observation yield | Passages with a visible plate that produced at least one OCR observation |
| Exact read coverage | Readable passages accepted with exactly the labelled plate |
| Wrong reads, abstentions | Readable passages accepted with a different string, or not accepted |
| Accepted precision | Correct accepted readings out of all judged accepted readings |
| Candidate recall | Labelled plate at rank 1, within 5, within K |
| Runtime | Wall time, video time, analysed frames, peak memory |

Things to know when reading it:

- **Accepted precision is undefined, not zero and not 100%, when nothing was accepted.**
- Outcomes are per vehicle passage. Neighbouring frames and reconstructed
  images never count as extra successes.
- Readings on passages no person could read, or outside every labelled
  passage, are listed separately and kept out of precision, because they
  cannot be checked.
- Each failed passage names its first failing stage (vehicle, plate crop, OCR
  candidate missing, fusion, or withheld by the gate) so the right thing is
  fixed.
- The report contains masked plates only.
- Reports are never overwritten. The `test` split needs `--final` and should
  be run once, after the commit, models and settings are frozen. If you tune
  anything after seeing a test result, that test set is spent and a new one is needed.
- OCR ranking scores are not calibrated probabilities.

## Limits

- Recorded analysis uses the tracked detector profiles only (`general` by
  default). The independent production worker is not involved.
- The recorded session sends sightings but no camera-health reports.
- One recording at a time; this is the same single shared session as live mode.
- A file with any frame lacking a timestamp is refused. Remux it first if needed.
- Frames larger than the 48 MiB decode bound are refused (8K video).
- Small-sample CPU/GPU inference timings are recorded in
  [RECORDED_GPU_VALIDATION.md](RECORDED_GPU_VALIDATION.md); they do not establish
  full-video throughput. CPU processing can be slower than real time.
- The central API must be the rebuilt one. The edge retries a rejected event
  indefinitely, so an old API would block the delivery queue.
