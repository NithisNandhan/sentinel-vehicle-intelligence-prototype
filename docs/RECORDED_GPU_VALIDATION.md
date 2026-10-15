# Recorded GPU validation - 15 October 2026

## What changed

The browser dashboards share one inference service. NVIDIA mode now runs the
YOLO26s vehicle tracker, parent ONNX plate detector and adapted Awiros OCR on
the GPU. Previously the recording service was CPU-only, and both detector
adapters explicitly requested CPU even when OCR requested GPU.

CPU remains the default for machines without NVIDIA hardware. GPU requests
are explicit and reject missing CUDA or ONNX provider fallback. Model weights,
thresholds, native crops, reconstruction and sampling settings are unchanged.
The engineering console reports the loaded device for all three model stages.
Video decoding, tracking association, reconstruction alignment, candidate fusion,
JSON transport and browser rendering are not all moved to the GPU.

## Measured bounded comparison

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB. Same original permitted AGH
recording, SHA256 `34b977df9caade1f74f84ef6e409525ea4702c5989eec360acbbe23b3218568a`,
1280x720. The three native sample timestamps were 40, 640 and 1160 ms in both
runs. Each run used the same frozen model hashes, 9 vehicle-box instances,
5 plate crops and 5 OCR calls. Counts are not unique vehicles or correct reads.

| Measurement | CPU | GPU |
| --- | ---: | ---: |
| Three-sample inference total | 13.592 s | 4.573 s |
| Median sampled-frame inference | 4594.013 ms | 1117.312 ms |
| Individual sample inference | 4594.013, 2062.032, 6936.219 ms | 1117.312, 174.097, 3281.814 ms |
| Model load and blank-input warmup | 21.802 s | 25.202 s |

Aggregate inference speedup in this small comparison: **2.97x**. This is not
full-video FPS, an end-to-end dashboard latency guarantee, or an OCR accuracy
measurement. Decoding, queueing and rendering are excluded from the sample
times. Early real-input kernel/postprocessing initialization can still affect
these three samples despite blank-input warmup. The CPU viewing service was
stopped during the paired runs. Runtime images differ: Python 3.11.17 / Torch
2.14.1+cpu versus Python 3.11.16 / Torch 2.14.0+cu130.

Candidate fusion still took roughly 2.9 seconds on the third sample in both
runs; GPU acceleration does not eliminate that CPU bottleneck. A larger,
labelled full-video experiment is required before making throughput or
recognition-quality claims. The clip has Polish plates, not Indian plates.

Raw numeric-only local reports: `data/benchmarks/device-20261015/cpu.json` and
`gpu.json`. Neither contains media, OCR strings or per-character observations.

## Reproduce

```powershell
.\scripts\start-recorded-analysis.ps1 -Device gpu -SkipBuild
# CPU-only fallback:
.\scripts\start-recorded-analysis.ps1 -Device cpu -SkipBuild
```

The GPU override reuses the existing live GPU image when present; omit
`-SkipBuild` to build it on a new GPU machine. Before switching services the
launcher executes `tools/check_gpu_runtime.py` through the isolated OCR runtime.
This warms real Paddle OCR, Torch vehicle detection and ONNX plate detection
on in-memory blank images, not just advertised provider names. Model startup
order is Paddle warmup, Torch initialization, then ONNX session creation so
the cached CUDA 12 OCR and CUDA 13 Torch/ONNX libraries can be loaded correctly.
This follows ONNX Runtime's documented recommendation to preload Torch's CUDA
libraries before creating a CUDA session:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch

For the bounded timing tool, run `tools/benchmark_recorded_devices.py` through
`tools/run_ocr_research.py` in the corresponding recorded Compose service,
with `--device cpu` or `gpu`, the same `--recording-id`, and `--samples 3`.
It checks the recording's permission receipt and never writes imagery or OCR.
Output files are created exclusively and never overwritten.

## Verification

- Real CUDA computation worked in the cached NVIDIA container.
- Real Paddle OCR, Torch vehicle detector and CUDA ONNX plate detector loaded
  and executed together; actual model identity reported GPU for all three.
- CPU and GPU Compose configurations validated; launcher syntax passed.
- 9 GPU-adapter tests, 10 CPU/device-wiring tests, 18 viewer tests and 15 tracked
  OCR tests passed: **52 tests**. Device-routing tests are mocked; the real-model
  check and paired timing runs provide separate execution evidence.
- The GPU service was started using the launcher, both interfaces unlocked
  through their normal forms, and the recording started via the operator UI.
  A real analysed frame appeared in the engineering console. No browser
  screenshots, plate hypotheses or credentials were exported.
- The actual GPU replay subsequently reached `completed`, with 237 analysed
  frames, 447 cumulative plate crops, final analysed video position 20.44 s
  of a 20.52 s file, and no failure code. These are processing counts, not
  distinct vehicles or verified plates. Zero identifiers passed the model's
  acceptance gate on this foreign-plate clip; acceleration does not establish
  correct recognition.
- The browser test observed one HTTP 503 from the recognition-status endpoint
  during startup. The later completed-status request and central health check
  both succeeded. This is not a claim of a wholly error-free browser run.
