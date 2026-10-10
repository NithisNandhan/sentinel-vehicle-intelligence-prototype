# Task 2 report: CPU execution path for the shared engineering session

Closed for wiring 10 October 2026. **Score: 6 / 10**

## What was there before

- `live_detection_viewer.py` loaded the OCR reader with `device='gpu'`
  hard-coded. On this machine (no NVIDIA GPU) the session could not start.
- `tools/run_ocr_research.py` always put the GPU Paddle build and CUDA
  libraries first on the import and library paths.
- The only edge runtime (`live` extra) depended on `onnxruntime-gpu`, and the
  image pulled the CUDA build of Torch.
- Diagnostics did not say which device or checkpoint had actually loaded.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `edge/sentinel_edge/live_detection_viewer.py` | `AwirosExperiment(cache, device='gpu', …)` | `load_models(…, device=…)` validates `cpu` or `gpu` and passes it on. New `--device` flag (default still `gpu`, or `SENTINEL_OCR_DEVICE`). New `model_identity()` reports OCR model, verified checkpoint SHA256, device, detector profile and model hashes, with no file paths. The engineering snapshot carries `models`; the operator status carries `inference_device`. |
| `tools/run_ocr_research.py` | One fixed GPU environment | `SENTINEL_OCR_DEVICE=cpu` loads only the CPU Paddle dependencies, sets no CUDA paths and hides GPUs. `SENTINEL_CPU_THREADS` sets thread count (2 on GPU as before, 4 on CPU). GPU behaviour is unchanged. |
| `edge/pyproject.toml` | `live` extra only | New `live-cpu` extra with CPU `onnxruntime` |
| `edge/Dockerfile` | `live` target only | New `live-cpu` target that installs CPU Torch first |
| `edge/tests/test_cpu_path.py` | Did not exist | 7 tests |

No silent fallback in either direction: a GPU request that fails raises, and
is not retried on CPU.

## Audit result

Edge suite at this point: **294 passed**, 1 Windows-only failure (unchanged
from baseline). The 7 new tests use a mocked OCR reader.

## What still has to change

- **Run a real model on this CPU.** Nothing real has loaded yet; that needs
  the `model-cache` release bundle (Task 8).
- Build the `live-cpu` image and confirm its size and that every dependency
  resolves (Task 7 and 8).
- Compare the parent and adapted OCR on a small labelled sample on CPU, as the
  guide asks, to confirm CPU output matches the recorded GPU results.
- Measure CPU timing, then decide whether `VehicleTracker`'s fixed 2 Torch
  threads should be raised. Left alone until there is a measurement.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 3 / 4 | Loader, launcher, extra and image target exist; the image has not been built |
| Tested | 3 / 3 | Wiring tests pass |
| Proven | 0 / 3 | No real CPU inference yet |
