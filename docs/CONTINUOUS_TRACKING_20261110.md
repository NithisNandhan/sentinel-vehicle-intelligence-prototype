# Tracking-only prototype and false-detection review

## What is implemented

The new local viewer at http://127.0.0.1:7798/ replays the full permitted Mumbai
recording with per-source-frame vehicle tracking. It provides dense tracking,
tentative tracks, raw detections and the sparse 500 ms baseline. Source-time
movement trails and the five previously problematic plate locations are available
for inspection. Native recorded video is used for playback; overlays select the
nearest recorded frame timestamp. Dense mode does not interpolate boxes into
unobserved frames. Sparse mode explicitly labels held rectangles.

This is a **tracking-only development viewer**, separate from the operator/OCR
workflow. OCR and plate models were not loaded for these runs. No watchlist events
were delivered. The recorded dashboard worker was restored after GPU capture;
its OCR tracking cadence is unchanged by this stage. Multi-frame OCR has not been
reconnected to this candidate tracker.

The detector is the existing cached YOLO26s checkpoint
`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.
No weights were downloaded or trained. Motorized classes are car, motorcycle,
bus and truck (COCO 2,3,5,7); bicycles are excluded from this plate-workload test.

## Tracking and false-detection safeguards

- ByteTrack receives all 928 source frames, rather than 62 samples.
- The screened candidate requires a detector score of at least 0.40 to create
  a new native track. Lower scores can still continue existing tracks.
- A new passage remains tentative until at least three source timestamps span
  150 ms, with two scores of at least 0.40 among its ten recent observations.
  This describes repeated model evidence, not verified physical identity.
- Passage IDs are never reused. A gap above 1,000 ms or a gross geometric
  discontinuity ends the binding. Returned boxes are current observations only.
- Car/truck and other class-label flicker no longer creates a new passage by
  itself. A ten-observation confidence-weighted vote stabilizes the label and
  exposes type ambiguity; geometry and native tracker identity remain separate.
- Stationary vehicles remain eligible. There is no zero-motion exclusion.
- Plate-location inspection returns all vehicle boxes covering at least 90%
  of the reference plate. Multiple candidates are labelled ambiguous; no smallest-
  box or highest-confidence shortcut assigns ownership.

Confidence and persistence cannot remove every false positive. A non-vehicle
consistently classified as a car may still qualify. The review queue therefore
shows first, largest and last available views of each screened passage. Reviewers
can label one vehicle, non-vehicle, mixed identities or uncertainty, add notes,
download the review and optionally hide human-flagged non-vehicles in playback.
Labels save atomically with revision checks and bind to the exact report hash.

The queue currently has **269 pending passage records**. These are fragments and
candidate histories, not 269 proven different vehicles. No human vehicle accuracy
or false-positive-removal percentage has been measured.

## Measured runs

Both inputs processed the same complete 30.933-second, 3840×2160 recording.

| YOLO input | Frames inferred on CUDA | Capture wall time | Median detector call |
|---|---:|---:|---:|
| 640 | 928 | 58.791 s | 23.960 ms |
| 960 | 928 | 58.234 s | 25.317 ms |

The 960 run has a detector-call p95 of 29.169 ms. Timing includes synchronisation
for detector calls; capture wall time additionally includes decoding, transfers,
Python overhead and serialisation. Playback at the source frame rate does not
demonstrate real-time inference: full capture took about 58 seconds for 31 seconds
of footage. The small wall-time difference between input sizes is not a speedup
claim or a controlled capacity study.

Each resolution's detections were frozen before replay. The sparse and dense
baselines use identical weights, classes, confidence and native ByteTrack settings.
The screened candidate changes only new-track threshold and adds the outer
qualification policy. The installed runtime is Ultralytics 8.4.162.

Results for 960 input:

| Mode | Updates | Native IDs | Passage records | Qualified passages | Tentative-only passages |
|---|---:|---:|---:|---:|---:|
| Sparse baseline | 62 | 54 | 122 | 57 | 65 |
| Dense baseline | 928 | 405 | 415 | 221 | 194 |
| Dense screened | 928 | 260 | 269 | 186 | 83 |

The 83 tentative-only screened histories are withheld from qualified-only view;
they have **not** been labelled as false vehicles. Fewer native IDs do not prove
better identity accuracy: they can also result from missed detections or mergers.
IDs remain fragmented and must be reviewed before feeding plate evidence to OCR.

The sparse baseline gives one candidate owner to all three different plates at
9.5, 10 and 11 seconds. In the 960 dense-screened replay these locations map to
separate candidate IDs (`V068`, `V072`, `V104`). The 28.0 and 30.5 second locations
map to `V183` and `V248`. This is geometric inspection of five known positions,
not complete physical-identity validation across the clip.

An extra fixed candidate disabled score fusion in motion association. Although
native-ID counts decreased, gross-discontinuity endings rose from 8 to 54 and
the 10-second plate became unassigned. That candidate is retained separately for
audit and was not selected for the viewer. Models were not changed.

ByteTrack's 30-update lost buffer represents different source durations at sparse
and dense cadences. That limitation is recorded; the outer passage timeout uses
PTS independently. No ground-truth vehicle inventory or complete track identities
were supplied, so IDF1, identity-switch accuracy, precision and recall are null.

## Verification

Eighteen pure policy tests and five review-server/browser-timing tests pass.
They cover multiple simultaneous tracks, generation after loss, stationary
vehicles, class-label stability, input validation, qualification timing, report-
bound persistent reviews, stale revisions, overlap ambiguity, video byte ranges
and origin/path restrictions. A Node regression checks nearest-frame selection
for all 928 fractional 30 FPS timestamps, preventing a one-frame overlay lag.

Browser checks confirmed video loading, 24 initial review cards, zero application
errors and actual overlay boxes at 10 seconds. No dummy labels were entered into
the real review. A screenshot attempt was cancelled after stalling; no screenshot
completion or human identity-accuracy result is claimed.

The MLE workflow and saved Graphify graph guided the work. A bounded subagent
owned the pure policy and tests; an independent QA agent audited source binding,
timestamps, IDs and review semantics. The graph launcher is broken, so incremental
refresh remains unavailable.

## Reproduction and handoff

Use the cached GPU image and model cache. First run
`tools/benchmark_vehicle_tracking.py --phase capture --image-size 960 --output <fresh-run>`
in the existing GPU Compose service, mounting `data` under `/workspace/data`.
Then run `--phase replay` and `--phase thumbnails` in the cached CPU/OpenCV image;
the thumbnails phase also requires permitted `data/recordings` at `/recordings`.
`--capture-dir` permits additional replays without another model-inference pass.
Use a fresh output folder; preserve frozen captures and reports.

Restart the local review without inference:

```powershell
.\scripts\start-tracking-review.ps1
```

The default run is `data/benchmarks/vehicle-tracking-960-20261110/`. Frozen
detections, timelines, report, vehicle thumbnails and any `vehicle-review.json`
stay local. The viewer exposes only the named source video, known thumbnails and
its APIs on localhost. GitHub receives code and aggregate evidence only.

Next, review non-vehicles and mixed histories, establish labelled detection and
identity measurements, and test a second clip. Then integrate dense vehicle
updates with explicit plate ownership, keeping OCR sampling asynchronous. Do not
promote this candidate merely because its playback looks smooth.
