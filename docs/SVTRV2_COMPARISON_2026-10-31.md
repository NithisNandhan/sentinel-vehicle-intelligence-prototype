# SVTRv2-B vs Awiros: measured diagnostic comparison

Tested 31 October 2026 on the NVIDIA RTX 3050 6GB laptop GPU. Recommendation:
**retain adapted Awiros; do not replace it with the tested pretrained SVTRv2-B.**

This is an isolated recogniser comparison on existing licensed diagnostic crops,
not a measurement of full-video CCTV accuracy, automatic crop coverage or job
completion reliability. No production model, threshold or dashboard setting changed.

## Results

All three candidates were freshly run on the exact same 31 test crops and the same
three distinct frames from the NIVU Indian-plate development passage.

| Recogniser | Exact still plates | Character errors / reference characters | Median OCR ms | P95 OCR ms | Framework peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| Original Awiros | 22/31 (71.0%) | 23/295 | 47.306 | 64.216 | 155.30 |
| Our adapted Awiros | **28/31 (90.3%)** | **13/295** | 51.599 | 65.903 | 155.30 |
| SVTRv2-B, official CLI resize | 15/31 (48.4%) | 51/295 | 33.024 | 56.322 | 91.67 |

These are one timed pass after warm-up, batch one, FP32. OCR timings include input
preprocessing, transfers and posterior materialisation; they exclude detection,
temporal fusion, networking and UI rendering. Allocated-memory peaks are per
framework, not complete process/device memory or application co-residency capacity.
Paddle and PyTorch use different runtimes. No full-pipeline FPS claim follows.

The 31 references include AI-blind-review/publisher agreement and legacy visual
review, not a large independently human-verified CCTV holdout. Unknown pretraining
overlap and prior use of this diagnostic constrain generalisation. Stills were
scored as stills and never duplicated into fake temporal evidence. See
`DEEP_OCR_ADAPTATION.md` for the preserved label caveats.

## What the paired errors show

SVTRv2-B and adapted Awiros were both correct on 15 images. Adapted Awiros alone
was correct on 13; both were wrong on three. SVTRv2-B recovered none of the adapted
model's misses in this diagnostic, so it supplies no measured complementary gain
here.

Among eight crops with native width/height below 2.5, SVTRv2-B and original Awiros
each scored 0/8; adapted Awiros scored 6/8. Aspect ratio is a diagnostic slice, not
a manually verified two-row-layout annotation. The result points toward layout/
domain adaptation as an area to investigate, not proof of one exclusive cause.

## Three-frame sequence

Every SVTRv2-B native-frame top reading was correct. All three models ranked the
correct registration first after the existing sequence-likelihood fusion. Both
Awiros models passed the unchanged acceptance gate; SVTRv2-B abstained.

SVTRv2-B's top candidate had mean character-peak score 0.937, but minimum character
score 0.621, below the unchanged 0.75 minimum. Its mean score is not a calibrated
probability of correctness. No global threshold was relaxed to manufacture an
accepted event. One development passage cannot establish acceptance precision or
video-wide recognition recall.

The three source images have assigned 400ms replay spacing; original video PTS
are unknown. Manually supplied plate boxes isolate OCR. This is not an end-to-end
automatic-detection test. Source: [NIVU](https://universe.roboflow.com/nivu/indian-license-plate-knte7), CC BY 4.0.

## Preprocessing and checkpoint checks

Official English SVTRv2-B Union14M-L-Filter weights and their matching configuration
were acquired from the Google Drive folder linked by the publisher's model table.
The model's 19,755,551 inference parameters loaded strictly; only 29 keys belonging
to the auxiliary training/SMTR branch were omitted as required by the published
CTC inference configuration. `torch.load(weights_only=True)` was used.

- OpenOCR revision: `1ccfc6ee6161f7133b192e16af3a9265273997ba`.
- SVTRv2 weights SHA256: `b65dd4675ef061a116390ba1590f538b69ab18795a55fd30a14bd98af1338f49`.
- Config SHA256: `9b3918ec96a1b7f3a6686906826207fbe37d4380d130f608779e2aa29245a873`.
- Original/adapted Awiros hashes and input manifest hashes are recorded in
  `docs/evidence/svtrv2-20261031.json`.
- GPU PyTorch runtime: `2.14.0+cu130`; the existing isolated Paddle GPU runtime
  was used for Awiros. Candidates ran sequentially in a separate test container.
- Our RGB/bicubic/normalisation transform matched the pinned publisher's actual
  CLI transform exactly across four input shapes: maximum tensor difference 0.

The pinned publisher CLI assigns a tuple to `self.ceil`, making it truthy even
when the intended setting is false. We reproduced that actual CLI behavior and
separately checked the publisher dataset transform's rounded-ratio policy on the
37-image validation split: CLI policy 20/37, dataset-round policy 18/37. Thus the
alternate documented resize policy did not provide a validation improvement.
No new parameters were fitted or selected from test labels.

Locally computed acquisition hashes pin these downloaded assets. The publisher did
not provide an independently verified weight checksum. Source and licence files
were retained with an integrity manifest. The OpenOCR code repository declares
Apache-2.0; no separate weight-specific licence was found in the model folder.
Dataset terms are recorded separately from source-code licensing.

Sources: [SVTRv2 model table](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2),
[OpenOCR](https://github.com/Topdu/OpenOCR), [Awiros](https://huggingface.co/Awiros/anpr-ocr).

## Reproduce and inspect

Code: `tools/prepare_svtrv2.py`, `tools/compare_svtrv2.py`,
`tools/render_svtrv2_comparison.py` and `edge/sentinel_edge/svtrv2_experiment.py`.

Local assets and full diagnostic readings remain ignored under
`data/benchmarks/svtrv2/`. The side-by-side viewer, including licensed source
attributions, is `data/benchmarks/reports/svtrv2-comparison.html`, served locally
at `http://127.0.0.1:7777/svtrv2-comparison.html`. Plate-free aggregate results are
tracked in `docs/evidence/svtrv2-20261031.json`.

Prepare/download once on Windows:

```powershell
& backend/.venv/Scripts/python.exe tools/prepare_svtrv2.py --root data/benchmarks/svtrv2/assets
```

In the isolated GPU container, bind `edge` and `tools` read-only under `/workspace`,
the existing benchmarks read-only at `/benchmarks`, downloaded candidate assets at
`/research`, and a writable results directory at `/results`. Existing model volume
is read-only at `/models`. Use the cached NVIDIA image, not the CPU dashboard image.

```text
PYTHONPATH=/workspace/edge python /workspace/tools/compare_svtrv2.py --backend svtrv2 --manifest /benchmarks/indian-corpus/adaptation_manifest.json --sequence /benchmarks/nivu/manifest.json --output /results/svtrv2-test.json
python /workspace/tools/run_ocr_research.py /workspace/tools/compare_svtrv2.py --backend awiros-parent --manifest /benchmarks/indian-corpus/adaptation_manifest.json --sequence /benchmarks/nivu/manifest.json --output /results/awiros-parent-test.json
python /workspace/tools/run_ocr_research.py /workspace/tools/compare_svtrv2.py --backend awiros-adapted --manifest /benchmarks/indian-corpus/adaptation_manifest.json --sequence /benchmarks/nivu/manifest.json --output /results/awiros-adapted-test.json
```

17 focused tests passed across input transforms, benchmark-integrity boundaries,
CTC fusion and the existing OCR selection harness; Python compilation passed. Real
GPU checkpoint loading and transform parity were also checked. Docker suffered an
engine interruption during setup and was restored before the measured runs; the
interruption is not an OCR accuracy outcome or an endurance test.

## Decision and next work

Keep adapted Awiros as the current recogniser. SVTRv2-B was faster in this small
diagnostic but sufficiently less accurate that a replacement is not supported.
An Indian fine-tuned SVTRv2 remains an untested option; no claim is made about its
potential after adaptation. The next useful step is to compare checked versus
automatic crops on representative human-labelled CCTV passages, then evaluate
another challenger or targeted adaptation only if those failures justify it.
