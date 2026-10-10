# Task 7 report: launcher, compose profile and documentation

Closed 10 October 2026. **Score: 8 / 10**

## What was there before

- The only way to start the shared engineering session was
  `scripts/start-live-detection-viewer.ps1`, which runs the GPU image with
  `--gpus all` and requires Sentinel credentials. It cannot start on this machine.
- `compose.yaml` had no service for it; the session was a hand-written
  `docker run`.
- No CPU image existed. An earlier attempt on this machine to build the GPU
  `live` image had ended in an error after 43 minutes (visible in the Docker
  build history).
- No document described recorded footage, receipts, labels or measurement.
- `README.md` said the recorded adapter "remains to be implemented".
- `.gitignore` did not ignore `.venv311/` or the `*.egg-info/` folder that an
  editable install creates.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `compose.yaml` | No engineering service | New `recorded-analysis` service under the `recorded` profile: CPU image, `./data/recordings` mounted read-only, models mounted read-only, no GPU reservation, no Sentinel variables, port 7780 on localhost only. New named volume entry for the existing shared-recognition queue. Existing services are unchanged. |
| `scripts/start-recorded-analysis.ps1` | Did not exist | Checks the model volume exists, refuses if the live service holds port 7780, creates `data/recordings`, then builds and starts the API, dashboard and recorded service. Options: `-DetectorProfile`, `-Unpaced`, `-CpuThreads`, `-SkipBuild`. |
| `edge/sentinel_edge/live_detection_viewer.py` | `--unpaced` flag only | Also reads `SENTINEL_RECORDED_UNPACED` so compose can set it |
| `docs/RECORDED_VIDEO.md` | Did not exist | Receipts, starting, controls, replay and seek rules, pacing, states, privacy, label format, measurement, limits |
| `README.md` | "adapter remains to be implemented" still stands as history | New section pointing to the above and stating that no accuracy has been measured |
| `.gitignore` | `.venv/` only | Also `.venv*/` and `*.egg-info/` |

## Audit result

| Check | Result |
| --- | --- |
| `docker compose --profile recorded config` | Valid. The default profile still lists exactly the four original services. |
| PowerShell parse of the new script | 0 errors |
| CPU image build (`live-cpu` target) | **Built**, 4.25 GB as reported by Docker |
| Image contents | Torch `2.14.1+cpu`, CUDA not available; ONNX Runtime 1.31.0 with CPU provider; Ultralytics 8.4.175; PyAV 16.1.0. No CUDA wheels (only `nvidia-ml-py`, a small monitoring binding pulled in by a dependency). Python packages total 1.8 GB. |
| Edge suite **on Linux inside that image** | **352 passed, 2 skipped, 0 failed.** The two tests that fail on the Windows host pass here, which confirms they are Windows-only. |

## What still has to change

- **The launcher has not been run end to end.** It needs the model volume,
  which is still downloading (Task 8).
- The image is 4.25 GB. That is far smaller than a CUDA image but not small.
  Building it took Docker's free space on C: from about 40 GB to 19 GB
  (image plus build cache). `docker buildx prune` would reclaim about 3 GB of
  cache; I have not run it because it also removes caches from your earlier builds.
- The image got OpenCV 5.0.0 through a dependency although the project's own
  pin is `<5`. Every test passes with it, but real model output on OpenCV 5
  has not been compared with the recorded results from the owner's machine.
- `-DetectorProfile indian_adapted` is not offered because it needs the
  12 GB-class training volume that the selective restore skips.
- Documentation describes behaviour; none of its commands have been run with
  real footage.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Service, launcher, image and documentation exist |
| Tested | 3 / 3 | Config validates, script parses, image builds, full suite passes inside it |
| Proven | 1 / 3 | The image runs real code on Linux; the launcher has not started a real session |
