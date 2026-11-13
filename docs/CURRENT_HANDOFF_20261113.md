# Current recorded-video handoff — 13 November 2026

Use this guide for the single dashboard and a CPU-only teammate laptop. An AMD
processor is sufficient to run the CPU path; an NVIDIA GPU is optional. The
full Mumbai quality-profile replay completed on CPU on this laptop, with an
observed end-to-end upper bound of 9 min 41 s. This is not a timing prediction
for another machine. Slower than real-time processing is expected.

## Source and historical boundary

```powershell
git clone --branch codex/enable-text-cropping https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype.git
cd sentinel-vehicle-intelligence-prototype
git rev-parse HEAD
```

At the audit, this branch was **`a9bc5a0`**, the tested commit that introduced
`general_quality`; default `main` was `624ad007` and lacked that profile's core.
This guide and its model-only restore/preflight helpers are subsequent handoff
changes on the same branch. Use the current branch for setup; checking out
`a9bc5a0` alone reproduces the quality experiment but omits those later helpers.
Record the full checked-out commit with your run results.

The October 7 handoff release is a **model/runtime restore point**, not today's
application source. The November 6 snapshot contains historical reports,
annotations and metadata. Neither release establishes a fresh complete-machine
restore of the current code, and the November 12 local crop viewers are not
included in that earlier snapshot.

## Verified integration results today

- The CPU-only Docker image was freshly built. The actual `general_quality`
  zero-image preflight executed vehicle detection/tracking, plate detection,
  Awiros OCR and the optional text detector on **CPU**. The baseline `general`
  profile completed a separate permitted overpass recording: 615 decoded frames,
  41 samples, 11.425 seconds model loading and 10.608 seconds inference, with zero
  model-accepted identifiers. This is completion evidence, not OCR accuracy.
- The GPU dashboard replay reached EOF on the Mumbai clip: **928 decoded frames,
  62 sampled frames and 184 plate crops**, producing **three model-accepted
  identifiers**. Two matched human references **16 and 17**; the third has no
  human label. These counts are not an accuracy score or proof of full vehicle
  coverage.
- Dashboard pause/resume preserved the same analysis session. A temporary
  recorded-watchlist test using reference 16 produced alert **16**, which was
  acknowledged and resolved through the dashboard. Temporary watchlist entry
  **2** was then deactivated, preserving its history.
- The resolved alert and both operator audit transitions survived API restart.
  These checks used the current machine and restored caches, not a fresh
  operating-system installation or the teammate's laptop. See the
  [validation record](FINAL_VALIDATION_20261113.md), including the subsequent
  GPU/WSL fault and CPU fallback.
- The final CPU `general_quality` Mumbai dashboard run reached EOF: 928 decoded
  frames, 62 samples, 181 crops and three model-accepted identifiers. All three
  reached central storage, with no duplicates and an empty outbox. All three
  accepted Inspector rows retained readings and native crops after completion;
  two explicitly used the fixed RAM archive. These are not accuracy claims.

## Restore only what this application needs

Install Git, Python 3.11 or newer, Docker Desktop with Linux containers/WSL2 and
Windows PowerShell. Start Docker Desktop. No NVIDIA driver or Sentinel portal
credentials are required for CPU recorded mode.

From the public [October 7 release](https://github.com/NithisNandhan/sentinel-vehicle-intelligence-prototype/releases/tag/handoff-2026-10-07),
download these files into `data/handoff-assets/`:

- `model-cache.manifest.json` and **all 22** `model-cache.tar.gz.partNNN` files.
- `alpr-cache.manifest.json` and `alpr-cache.tar.gz.part001`.

The model-cache download is approximately **5.69 GB**, expanding to 8.94 GB of
archived files, before Docker images and temporary space. The ALPR bundle is
approximately 20 MB compressed. The larger archive includes historical variants
and Linux OCR dependencies, not just the three selected weight files. Do not
unpack these Linux runtime files directly into Windows Python.

GitHub CLI users can select only those bundles:

```powershell
gh release download handoff-2026-10-07 --repo NithisNandhan/sentinel-vehicle-intelligence-prototype --dir data/handoff-assets --pattern 'model-cache.*' --pattern 'alpr-cache.*'
.\scripts\check-prereqs.ps1
.\scripts\restore-teammate-handoff.ps1 -AssetsPath .\data\handoff-assets -ModelsOnly
.\scripts\init-env.ps1
```

Browser download of the same files is also supported. `-ModelsOnly` needs no
local settings file and restores only the two model volumes using a plain Linux
Python extraction helper. It does not build application services, import saved
events, or restore training data, reports or presentations. It may pull the
small `python:3.11-slim` helper image if missing. Parts are checksum verified;
existing nonempty destinations are refused. Do not delete existing volumes to
make a restore pass. The older full-restore path remains available for an empty
historical installation, but is unnecessary for a new recording/watchlist run.

The public model manifest includes all three exact weights used now:

| Component | Path within model volume | SHA256 |
|---|---|---|
| Vehicle detector | `experiments/detectors/yolo26s.pt` | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| Parent plate detector | `experiments/detectors/yolo-v9-t-512-license-plates-end2end.onnx` | `746fdd358ec110418775d7c9d8d07910d48b1a21471f92bf4421f6510d6daade` |
| Adapted OCR | `experiments/awiros/adaptations/indian-corpus-encoder.safetensors` | `c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d` |

`general_quality` uses these existing weights. The rejected November 12
fine-tuned detector is not needed. IWPOD, larger YOLO11 checkpoints and training
datasets are not prerequisites for this recorded workflow.

## Validate the CPU model runtime

The source code and restored weights are not by themselves proof that the CPU
runtime executes. Build the CPU service, then warm the actual OCR, vehicle and
plate stages on zero-valued memory images:

```powershell
docker compose --profile recorded build recorded-analysis
docker compose --profile recorded run --rm --no-deps -T -e SENTINEL_OCR_DEVICE=cpu -e SENTINEL_TEXT_DETECTOR_DEVICE=cpu -e RECORDED_DETECTOR_PROFILE=general_quality --entrypoint python recorded-analysis /workspace/tools/run_ocr_research.py /workspace/tools/check_model_runtime.py --device cpu --detector-profile general_quality
```

A passing result contains `model_preflight: passed`, `device: cpu`, the three
model identities and CPU device labels. The checker refuses conflicting runtime
flags, wrong checkpoint identity and GPU execution providers in CPU mode. It
does not read video, local settings files or credentials, download assets,
print plate text, or send central events. This is model execution verification,
not recognition accuracy or an end-to-end restore test. Today's real CPU
preflight passed as recorded above; the baseline CPU EOF check also passed.
Full Mumbai quality-profile completion is a separate dashboard test.

The existing `tools/check_gpu_runtime.py` remains a compatible GPU-only wrapper;
the GPU launcher still validates actual execution before replacing a working
CPU session. NVIDIA is useful for training and faster inference experiments.
Ask the GPU owner for help with those tasks, not routine dashboard, labels,
watchlist or CPU integration work.

## Supply permitted footage and run

Raw Mumbai footage and standalone crops were deliberately not mirrored in the
public releases. Obtain a permitted copy independently; the local source receipt
points to the [Pexels Mumbai clip](https://www.pexels.com/video/busy-urban-traffic-scene-in-mumbai-city-30400506/).
Read its current publisher terms. Exact replay of the existing reviewed
references requires SHA256
`6b7064605b3d4d12c6c1da4566cbcec1cd2e3e3bd7b77814865fe9df22045f65`.
A different cut, resolution or encoding requires its own receipt and labels.
Do not assume a newly downloaded copy has identical bytes.

Place the permitted file in `data/recordings/` and register it using real details:

```powershell
python tools/register_recording.py data/recordings/junction.mp4 --source 'ACTUAL SOURCE' --licence 'ACTUAL LICENCE' --permission 'ACTUAL PERMISSION BASIS' --title 'Junction recording'
$env:DASHBOARD_RECORDED_EVIDENCE = 'true'
.\scripts\start-recorded-analysis.ps1 -Device cpu -DetectorProfile general_quality -Unpaced
```

Open **http://127.0.0.1:5173/**, select the registered recording and press
**Start analysis**. No manual operator unlock is needed. `-Unpaced` removes
real-time pacing, not model work or source-frame safeguards. The optional quality
profile adds up to six native searches per vehicle region and can be expensive
on CPU. Use `-DetectorProfile general` for the original search. A failed or
uncertain reading is not a confirmed watchlist identity; acceptance rules are
unchanged.

Use a fresh permitted watchlist entry for your own run. Fresh local HMAC keys do
not reproduce identifiers in archived historical events. Recorded diagnostic
evidence is opt-in; central events use masked identifiers. Keep ports on localhost.

## Completion evidence still required

Before calling the current handoff complete, save the checked-out commit, model
preflight result, recording checksum and one full-recording integration result:

1. Analyse the complete recording to EOF without reconnect loops or a stuck
   session; record actual elapsed time and decoder/session status.
2. Inspect native crops and readings, preserving uncertain versus accepted
   states. Score only against independent human labels, with errors and misses.
3. Verify a labelled synthetic or genuinely accepted known-watchlist sighting
   reaches the alert queue. Do not force an uncertain reading into an alert.
4. Check acknowledgement/resolution, restart persistence and outbox recovery.
5. Start a second run and exercise navigation and pause/resume without losing
   the shared session. Preserve logs and failures rather than claiming a clean
   restore from this guide alone.

The results section records the checks already completed on this machine.
This document does **not** claim that the checklist has passed on a new CPU-only laptop or a
freshly reset OS.
