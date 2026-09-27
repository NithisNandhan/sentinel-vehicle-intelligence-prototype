# Selected detectors and tracked OCR integration

## Decision and scope

The next stage connects the cached detector candidates to per-vehicle, multi-frame
OCR and makes the result visible in the opt-in localhost viewer. The viewer
launcher now selects **general YOLO26s-640 + parent YOLOv9-tiny-512 plate detector**.
The deeper-adapted Awiros OCR checkpoint is unchanged. Production `VisionPipeline`,
`.env`, CAM01/CAM02 workload, outbox and alert thresholds are unchanged.

There is **one correctly recognised Indian registration** in the automatic
licensed development-sequence test, but **no model-accepted registration in the
new CAM14/CAM17 live probes**. This stage is not live OCR acceptance or a 90%
accuracy claim. Fine-tuned detectors remain selectable experiments, not a global
replacement: earlier cross-source regressions and this sequence's missed vehicle
frames argue against promoting them indiscriminately.

## Pipeline implemented

1. Authorised catalogue discovery supplies the camera connection; no guessed IDs
   or stream URLs are exposed. The existing RTSP-over-TCP/HLS and protected gateway
   fallback is reused. Only selected cameras are opened.
2. A continuous decoding thread retains the latest frame rather than building a
   delayed queue. Dropped frames are counted; a dropped scene-discontinuity flag
   is carried forward. Decode failure and incomplete windows are reported.
3. Native video PTS drives the minimum 500 ms sampling interval. Missing PTS is
   skipped, not replaced by wall-clock arrival. Backward timestamps, a cut or a
   gap over 2.5 seconds reset trackers and OCR histories.
4. YOLO supplies vehicle proposals and ByteTrack IDs. Original-resolution vehicle
   regions are searched on a rotating ROI budget. Plate extraction is a native
   pixel slice, including a small margin; no display resizing enters OCR. The
   OCR network still has its normal fixed-size preprocessing, which is distinct
   from preserving source pixels during crop extraction.
5. Independent per-track histories combine character probabilities with the
   existing quality-weighted CTC reader. Duplicate crop hashes and too-close
   observations do not count as new evidence. Class/geometry jumps, displaced
   plate regions and old OCR observations clear the relevant history. Histories
   are bounded; track evidence is never pooled across unrelated vehicles.
6. At least three distinct observations are required. Existing .85 acceptance
   and character-support gates remain unchanged. Uncertain candidates do not
   become alerts. The live viewer shows masked hypotheses and labels acceptance
   as unverified. This diagnostic sends no sightings or central writes.

There is no reconstruction/deblurring ablation in this stage. Existing offline
hybrid experiments remain available separately. No registry lookup was used.
More frames, reconstruction or high internal scores do not guarantee a correct
reading when the native input lacks usable character information.

## Profiles and runtime compatibility

| Profile | Vehicle detector | Plate detector |
|---|---|---|
| `legacy` | YOLOv8n | Parent YOLOv9-tiny-512 |
| `general` | YOLO26s | Parent YOLOv9-tiny-512 |
| `plate_adapted` | YOLO26s | Indian-adapted YOLOv9-tiny-512 export |
| `indian_adapted` | Indian BMD-adapted YOLO26s | Adapted plate export |

`legacy` means the old detector weights in the **new tracked pipeline**, not the
exact old viewer algorithm. All profiles include bicycles consistently in this
experiment. The BMD checkpoint has 13 classes: all are requested, then explicitly
mapped to coarse tracking families. Applying the production COCO class filter
directly would silently remove relevant BMD vehicles.

Cached assets are SHA-256 pinned. Missing/changed weights fail rather than trigger
automatic downloads. The adapted plate export uses raw `[1,5,5376]` output and
external confidence/NMS decoding, not the parent's end-to-end ONNX interface.
Both choose the highest-confidence plate proposal within each searched vehicle.

Paddle OCR is warmed on GPU first in its isolated CUDA 12.6 runtime. Vehicle and
plate detectors run on CPU, with bounded thread settings, to avoid the observed
Paddle/Torch CUDA library conflict. This is compatible execution, not maximally
optimised GPU throughput. Loading four profiles is not a production capacity test.

## Measured evidence — 27 September 2026

### Paired live windows

Four profiles received the same sampled native frames per camera. Shared vehicle
tracking and identical-crop OCR caches avoid redundant computation; cache hits
are not additional temporal votes. Profile order alternates. ROI budget was two.
Both requested 90-second windows completed without decoder failures or missing
PTS. CAM14 ran 90.38 s (302 decoded / 35 analysed frames); CAM17 ran 90.45 s
(82 decoded / 11 analysed frames). These are sequential windows, not concurrent
two-feed endurance acceptance.

| Camera | Profile | Vehicle box instances | Plate crops | Max distinct observations on a track | Candidate windows | Model accepted |
|---|---|---:|---:|---:|---:|---:|
| CAM14 | legacy | 124 | 0 | 0 | 0 | 0 |
| CAM14 | general | 164 | 0 | 0 | 0 | 0 |
| CAM14 | plate_adapted | 164 | 2 | 1 | 0 | 0 |
| CAM14 | indian_adapted | 453 | 5 | 1 | 0 | 0 |
| CAM17 | legacy | 31 | 3 | 3 | 1 | 0 |
| CAM17 | general | 40 | 4 | 4 | 0 | 0 |
| CAM17 | plate_adapted | 40 | 4 | 4 | 0 | 0 |
| CAM17 | indian_adapted | 53 | 0 | 0 | 0 | 0 |

Box instances are **not unique vehicles or verified recall**. More proposals do
not imply more correctly localised plates. Median plate widths were approximately
33–45 native pixels where crops existed. No independent live plate transcription
was available; live accuracy is unknown. Saved evidence is aggregate metadata
only in `detector-ocr-live-20260927.json`, not live images or plate strings.

### General profile alone on CAM17

A separate 90.65-second window used ROI budget four: 255 decoded frames, 16
analysed frames, 78 vehicle box instances and nine plate crops. Eight distinct
OCR observations were collected across histories, with a maximum of six on one
track. Four fusion-eligible windows produced **zero complete candidate windows
and zero accepted identifiers**. Median native crop width was 40 pixels; median
quality score .399 is a heuristic, not accuracy. Median vehicle tracking was
106.77 ms and plate/OCR/fusion 188.61 ms, excluding stream waits and model startup.
These separate medians are not sustained FPS or end-to-end latency.

This rules out insufficient frame count as the only problem in that window:
repeat evidence existed but the reader still could not form a complete supported
candidate. Small/poorly localised crops, OCR domain fit, sampling and processing
gaps remain candidate bottlenecks; this test does not establish a single cause.
Recorded resets include large PTS gaps, not necessarily physical scene cuts.
Report: `detector-ocr-general-live-20260927.json`.

### Exact viewer profile and deployed image check

The deployed edge image loaded the general detectors and GPU OCR successfully.
Its first requested 30-second CAM17 check ended after 15.43 s on a clean stream
EOF: 24 analysed frames, five plate crops, two candidate windows, zero accepted
identifiers. That report is retained as **incomplete**, not a passed window.
The authorised preview reader previously returned after clean EOF; it now retries
the same preview with bounded backoff and marks the next frame discontinuous.
Decoder-local shutdown no longer cancels the outer operator session, and an
unexpected viewer source ending enters the existing delayed retry path.

A fresh recheck with the viewer's ROI budget three completed **30.87 s**, without
decoder failure or missing PTS: 104 decoded / seven analysed frames, 28 vehicle
box instances, six plate crops, up to five distinct observations on one track,
three candidate windows and **zero model-accepted identifiers**. This validates
the runnable profile, not recognition success or endurance; a completed repeat
does not prove all future reconnections. The browser also displayed the general
profile, correct tracking explanation, paused controls and 30 authorised live
catalogue entries. No genuine live screenshots were retained.

Reports: `detector-ocr-viewer-profile-smoke-20260927.json` (incomplete) and
`detector-ocr-viewer-profile-recheck-20260927.json` (completed).

### Automatic Indian development sequence

Three original video-derived images from the existing permitted
[NIVU source](https://universe.roboflow.com/nivu/indian-license-plate-knte7),
CC BY 4.0, were supplied as **full scenes**, not annotated plate crops. Vehicle
tracking, plate localisation and OCR ran automatically with ROI budget eight.
The visually reviewed label was used only after inference. Assigned 400 ms replay
PTS are not original video timestamps. This previously used one-vehicle sequence
is development evidence, not fresh held-out evaluation; original model training
overlap is unknown.

| Profile | Vehicle frames found / 3 | Plate crops | Median native plate width | Outcome |
|---|---:|---:|---:|---|
| legacy | 3 | 3 | 117 px | Correct: MH02EK0837 |
| general | 3 | 3 | 118 px | Correct: MH02EK0837 |
| plate_adapted | 3 | 3 | 112 px | Correct: MH02EK0837 |
| indian_adapted | 1 | 1 | 94 px | Abstained |

The final repeat reproduced these outcomes. This is **one registration recognised
by three profiles**, not three independently recognised vehicles or 100% general
accuracy. The general detector has not shown better plate-text accuracy than the
legacy detector on this one clear case. The contrast between this successful
sequence and small live crops is informative, but is not a universal pixel cutoff.

Local reports preserve input hashes and the explicit no-ground-truth-cropping
flag. Final artifacts: `detector-ocr-sequence-r2.json` and its HTML viewer.
The original run remains preserved. Public permitted crop images/text may appear
in these local benchmark files; genuine Sentinel media may not.

## View, reproduce and verify

- [Results summary](http://127.0.0.1:7777/detector-summary.html).
- [Automatic crop and OCR evidence](http://127.0.0.1:7777/detector-ocr-sequence-r2.html).
- [Opt-in live viewer](http://127.0.0.1:7780/). Select an authorised camera, such as
  CAM17. Loading the paused page is not itself live inference validation.

Start a new local live session with the existing Compose services and caches:

```powershell
.\scripts\start-live-detection-viewer.ps1
# Experimental alternative, without changing production:
.\scripts\start-live-detection-viewer.ps1 -DetectorProfile plate_adapted
```

It starts paused, expires an idle capture after 60 seconds without heartbeat and
shuts down after 30 minutes. No recording, external exposure or central alerts.
Host/Origin and per-process token checks remain in place. The training/model
volumes and scoped source mounts are read-only; credentials stay in local `.env`.

Inside the isolated research container, with existing model/training volumes,
read-only `/workspace/edge` and `/tools`, and ignored report directory writable:

```text
python /tools/run_ocr_research.py -m sentinel_edge.detector_ocr_shadow --cameras cam14,cam17 --seconds 90 --roi-budget 2 --report /workspace/data/benchmarks/reports/NEW-live.json
python /tools/run_ocr_research.py -m sentinel_edge.detector_ocr_shadow --cameras cam17 --profiles general --seconds 90 --roi-budget 4 --report /workspace/data/benchmarks/reports/NEW-general.json
python /tools/run_ocr_research.py /tools/validate_detector_ocr_sequence.py --name NEW-sequence
python /tools/render_detector_summary.py
```

Use new names: original experiment reports are not silently overwritten.
Live probes need the approved network and `.env`; the licensed-sequence probe
does not need network access. Models load from the previously acquired caches.

The complete edge suite passes **234 tests and five subtests (9.95 s)** in the
research runtime. New coverage includes native pixels, three distinct-frame
acceptance, masked viewer output, no source-image mutation by overlays, isolated
track histories, duplicate rejection, PTS reset/bounds, shared-cache evidence,
13-class mapping, model tamper rejection, dropped-cut propagation, decode-local
cancellation without cancelling the operator session, same-session authorised
preview reconnect on clean EOF with a discontinuity flag, and truthful escaped
benchmark/viewer labels. All 26 tool files and the PowerShell launcher also passed
syntax checks. This is not a new clean-dependency install or measured
accuracy test. Prior detector stage's clean-environment validation remains separate.

## Remaining bottlenecks and next evidence

The integration is functional on the clear licensed case, but genuine-live plate
reading remains unproven. Before production promotion, improve and measure
usable repeat crop yield, sample scheduling and per-vehicle localisation on
independently labelled permitted sequences. Separate detection misses from OCR
errors; compare unchanged thresholds and track association on the same inputs.
Then evaluate a broader unseen plate set and longer authorised live windows.
Do not lower confidence gates or present guesses as verified readings to hide
this boundary. No extra fine-tuning was performed in this integration stage.
