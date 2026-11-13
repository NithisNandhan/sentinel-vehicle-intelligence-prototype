# Recorded prototype validation — 13 November 2026

The current recorded-video application works through decoding, vehicle and
plate detection, OCR, masked central delivery, watchlist matching and operator
decisions. Recognition remains incomplete. The measurements below do not
establish reliable reading of every vehicle or arbitrary CCTV footage.

## Changes validated

- Added a bounded, opt-in recorded Inspector archive: at most 32 tracks or
  16 MiB of encoded diagnostic data in worker RAM. An accepted running-list row
  can retain its original crop, reading and capture timestamp after its rolling
  OCR history expires. The archive is labelled; it never combines an old crop
  with a newly changed reading. Scene/track identity and exact reading must
  still match. Stop, seek, source change and restart clear it. Live cameras
  never use this archive, and no diagnostic image is written to disk.
- Added actual CPU model execution checks alongside the compatible GPU checker.
  The Windows launcher validates the selected device/profile before replacing
  a working session; CPU launches also select CPU for the optional text stage.
- Added a model-only restore path. New installations need the model-cache and
  ALPR-cache archives, without historical events, presentations or training data.
- Added `tools/check_recorded_completion.py`, a bounded EOF check through the
  real shared inference path. It sends no central events and exports no images
  or OCR strings. Timestamp resets are reported rather than silently discarded.

The vehicle, plate and OCR weights and acceptance thresholds were not changed
in this validation. Exact hashes and current branch setup are in the
[current handoff guide](CURRENT_HANDOFF_20261113.md).

## Real recorded GPU runs

Two complete dashboard runs used the optional `general_quality` profile on the
31-second Mumbai recording, SHA256
`6b7064605b3d4d12c6c1da4566cbcec1cd2e3e3bd7b77814865fe9df22045f65`.
Each decoded **928 frames**, analysed **62 timestamp samples**, collected
**184 plate crops** and produced **three model-accepted identifiers**, with an
empty delivery outbox at completion. The last analysed sample was 30,500 ms;
the recording duration is 30,933 ms. Decoding to EOF does not mean every frame
was passed to the detector.

In the first run, two of the three central identifiers matched human reference
numbers **16 and 17**. The third was outside the available 18-reference set;
its correctness is unverified. This is not a 2/3 whole-video accuracy estimate:
the reference set is incomplete, and detection misses/abstentions remain.
Dashboard pause/resume held the sample position and preserved the same session.

The second run generated a clearly labelled demonstration watchlist alert from
reference 16. Alert **16** was acknowledged and resolved through the dashboard.
Both audit transitions and the resolved state survived API restart. Temporary
watchlist entry **2** was deactivated, preserving its history. It is a recorded
demonstration, not evidence of a genuinely suspicious vehicle.

An Inspector failure found during these runs was fixed after the second run.
The final post-fix replay completed on CPU, as recorded below.

## CPU validation

The CPU-only Docker image was built from source and the `general_quality`
model preflight passed on actual zero-valued images, including the optional CPU
text stage. The baseline `general` profile also completed a permitted 20.5-second
overpass recording through the real inference path:

| Measurement | Result |
|---|---:|
| Frames decoded | 615 |
| Samples analysed | 41 |
| Model loading | 11.425 s |
| Inference after loading | 10.608 s |
| Backward / duplicate source timestamps | 0 / 0 |
| Model-accepted identifiers | 0 |
| Central events sent by the check | 0 |

The recording hash is
`3ebc885f748271564703c658ba3e04b12b36ed56fa03d15dc47013c41e8e4f54`.
This test proves completion on that input, not OCR accuracy or Mumbai quality
profile throughput.

The final **CPU `general_quality` dashboard replay** then completed the entire
Mumbai file: **928 decoded frames, 62 samples, 181 plate crops, three accepted
identifiers**, last sample 30,500 ms and an empty delivery outbox. Central storage
contained exactly three sightings with three distinct keyed identifiers, all
labelled `recorded`. GPU crop count was 184; the counts are not claimed bitwise
identical across devices. The CPU session began at 05:17:36 UTC and was first
observed completed at 05:27:17 UTC, an upper bound of **9 min 41 s** including
model startup and polling delay, not pure inference time.

All three accepted rows returned HTTP 200 from the Inspector **after EOF**, with
nonempty readings and native images. Track 280 retained five frames; tracks 243
and 204 retained six each. The latter two used explicitly archived coherent
snapshots from 26,000 and 27,500 ms, rather than the previously empty panels.
The Inspector archive fix therefore passed both its regression tests and a
real complete recorded run. The actual dashboard Inspector for track 243 also
rendered its reading, archive notice and all seven displayed images, including
the unchanged native crop. Aggregate measurements are preserved in the
[machine-readable record](evidence/final-runtime-20261113.json).

Effective CPU runtime: Python 3.11.18, Torch 2.15.0+cpu, torchvision 0.29.2+cpu,
Ultralytics 8.4.211, PyAV 16.1.0, ONNX Runtime 1.31.1, NumPy 1.26.4,
OpenCV-headless 4.11.0.86, FastALPR 0.4.0 and PaddlePaddle 3.2.2. Some modules
come from the restored isolated OCR runtime, which takes precedence over image
packages. Image ID:
`sha256:5521a7d54d8b325bfca1e4c65821588b09bfcb4dde5cbd1a56cdb9bf5d97cd75`.
This local image identity is evidence, not a published downloadable image.

## Recovery and regressions

The existing isolated synthetic replay used eight scripted passages, a real
temporary FastAPI backend and SQLite outbox. All 11 queued events remained while
the backend was stopped, drained after recovery, and were deliberately resent.
There were two expected alerts, zero missed/extra/duplicate alerts, zero delivery
failures and an empty final queue. All eight recognition results persisted.
This was a short controlled outage, not a five-minute sustained-outage test,
and the scripted OCR inputs are not recognition accuracy evidence.

The affected test modules ran with **201 passes and one dependency-related
skip**: recorded diagnostic retention, diagnostic DTO/HTTP handling, model
preflight, live worker state, passed vehicles, result delivery, quality search,
handoff archives, recording sources, recorded sessions and 69 API proxy tests.
Recording tests ran inside the actual CPU image with PyAV. API tests ran from
`backend/`, their documented working directory, to avoid loading the real root
settings into synthetic fixtures. Windows launcher parsing and Git whitespace
checks passed. No frontend source was changed.

## Failures retained in the record

- The original parking recording emitted 236 backward timestamp steps in 513
  decoded frames. Its clock resets increased sampling and tracker resets. The
  CPU stress attempt on this source was stopped; it is not counted as a passed
  EOF or accuracy run. Two alternative sources had continuous timestamps.
- After earlier successful GPU runs and another passing GPU model preflight,
  Docker/WSL lost adapter access (`AcceleratorError`, then NVIDIA hook
  `WSL environment detected but no adapters were found`). Windows still saw the
  RTX 3050. A Docker Desktop restart did not restore container access. The
  final application was switched explicitly to CPU; no GPU success is claimed
  for a failed attempt. Verify the GPU preflight after recovering WSL/driver
  access before selecting GPU mode again.

## Remaining limits

Keep the optional quality search separate from its earlier selected-frame
[comparison](PLATE_QUALITY_ADAPTATION_20261112.md): that diagnostic improved
reference detection from 10 to 11 of 18 and exact readings from 7 to 8, but is
not a full-video accuracy result. Broader independent labels, complete vehicle
coverage scoring and difficult scene evaluation remain necessary. Fresh-OS
restore and execution on the teammate's AMD laptop have not been performed.
The current source is on `codex/enable-text-cropping`; the older default branch
and historical releases are not interchangeable with it.
