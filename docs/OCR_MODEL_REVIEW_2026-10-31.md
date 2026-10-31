# OCR model review and selection recommendation

Reviewed 31 October 2026 using publisher repositories, model cards and research
papers. This is a catalogue review, not a measured comparison on Sentinel footage.
No weights, dependencies, acceptance settings or running services were changed.

Subsequent measured update: SVTRv2-B was tested against both Awiros checkpoints.
It scored 15/31 exact stills versus 28/31 for adapted Awiros. The current decision
is to retain adapted Awiros; see `SVTRV2_COMPARISON_2026-10-31.md`. The shortlist
below records the research rationale before that experiment, not a measured win.

## Recommendation

Retain the untouched Awiros parent and our adapted Awiros checkpoint as baselines.
The preferred new replacement candidate is OpenOCR SVTRv2-B, with SVTRv2-S as
the smaller option. Also benchmark PP-OCRv6_medium_rec, PARSeq and the existing
CCT-S v2 global recogniser. Promote a replacement only after a controlled,
independently labelled Indian CCTV comparison demonstrates an improvement.

This ordering is an engineering judgement about task fit, integration and available
assets. It is not a claim that SVTRv2 already beats Awiros on our recordings.

## Current implementation

The recorded-analysis launcher loads Awiros through
`edge/sentinel_edge/awiros_experiment.py`, using the adapted checkpoint
`indian-corpus-encoder.safetensors`. The separate production worker and historical
FastALPR/CCT experiments must not be confused with that recorded-analysis path.

The parent model is an Indian-plate specialist based on PP-OCRv5, with about 37M
parameters and a 3x48x320 input. The publisher reports 98.42% on its own held-out
validation set; that does not measure our CCTV accuracy or RTX 3050 throughput.
The model card covers single-row and dual-row plates and declares Apache-2.0.

Our bounded adaptation trained 133,864 parameters and improved a small still-image
diagnostic from 22/31 to 28/31 exact strings. Labels were partly publisher-silver
and AI-reviewed rather than independently human verified. See
`docs/DEEP_OCR_ADAPTATION.md`. Neither the parent nor adaptation is proven to be
the best option for our target video distribution.

Publisher: https://huggingface.co/Awiros/anpr-ocr

## Catalogue and disposition

| Family | Available assets and relevance | Disposition |
|---|---|---|
| Awiros ANPR OCR | Indian-specific PP-OCRv5 specialist, released weights; Apache-2.0 model card. Already integrated and adapted locally. | Keep both parent and adapted checkpoint in the comparison. |
| OpenOCR SVTRv2-B / SVTRv2-S | Public pretrained models, training/evaluation code and CTC recognition. Repository declares Apache-2.0. General scene text, not a demonstrated Indian CCTV specialist. | First new challenger; test B and use S if resource limits justify it. |
| PP-OCRv6_medium_rec | Official inference and training weights. The recognition model card declares Apache-2.0, 19M parameters and a CTC+NRTR decoder. Improvements reported on general OCR benchmarks do not establish Indian-plate superiority. | Second new challenger. Isolate its newer Paddle runtime from the working environment. |
| PARSeq / PARSeq-tiny | Public pretrained scene-text models and fine-tuning code; primary implementation Apache-2.0, with separately licensed reproduced methods. Exposes token logits but uses autoregressive decoding. | Useful architecture-diverse challenger. Its outputs need sequence-aware adaptation; do not feed them directly into CTC fusion as though identical. |
| fast-plate-ocr CCT-S v2 global | Publisher's recommended current plate recogniser; toolkit MIT, ONNX inference and training support. An existing project experiment already wraps its probabilities. | Low-cost control and potential CPU option; not an assumed accuracy upgrade. |
| LPRNet | Lightweight CTC architecture. The reviewed Apache-2.0 PyTorch implementation supplies Chinese-plate-oriented weights and expects adaptation for other plate distributions. | Lower priority: no verified ready-made Indian CCTV advantage over our specialist. |
| TrOCR base printed | Public encoder-decoder checkpoint fine-tuned on SROIE, intended for single text lines. | Lower priority for this deadline: not an Indian-plate specialist, and requires an additional sequence adapter and likely domain adaptation. |
| EasyOCR | Public general OCR implementation; repository Apache-2.0 and CPU/GPU inference support. | Optional inexpensive sanity baseline, not the preferred new training investment. |
| ICPR 2026 multi-frame recognition methods | Research includes temporal attention and joint visual-feature fusion over five low-resolution frames. The reviewed public team repositories describe training/inference, but reusable weight availability and explicit reuse licences were not established. | Research leads; do not describe these as fully cleared, downloadable production replacements. |

Sources:

- SVTRv2 models/configuration: https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2
- OpenOCR repository and licence: https://github.com/Topdu/OpenOCR
- PP-OCRv6 recognition model: https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec
- Official recognition catalogue: https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html
- PARSeq: https://github.com/baudm/parseq
- CCT model zoo: https://github.com/ankandrew/fast-plate-ocr/blob/master/docs/inference/model_zoo.md
- fast-plate-ocr repository/licence: https://github.com/ankandrew/fast-plate-ocr
- LPRNet implementation reviewed: https://github.com/sirius-ai/LPRNet_Pytorch
- TrOCR printed checkpoint: https://huggingface.co/microsoft/trocr-base-printed
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Official low-resolution competition report: https://arxiv.org/abs/2604.22506
- Public multi-frame research code reviewed: https://github.com/Fluuvys/ICPR_2026_LRPR_Competition
- Additional multi-frame research code reviewed: https://github.com/ducquan19/MultiFrame-LPR

## Multi-frame finding

The ICPR 2026 competition reports a winning whole-plate recognition rate of 82.13%
on its blind test. Its five-frame low-resolution tracks and Brazilian/Mercosur
plate distribution differ from our Indian CCTV data. This is evidence of research
progress, not a transferable accuracy promise. Dataset access is restricted and
non-commercial; research-code visibility does not establish weight/data rights.

Our OCR-output fusion and checked reconstruction should be compared with a learned
visual-feature fusion method only after the stronger single-frame encoders and
input handling are evaluated. A temporal neural model requires labelled sequences
and its own training/integration work; it is not a flag on a single-frame reader.

## Controlled selection procedure

1. Human-label 50-100 development vehicle passages from permitted target recordings,
   covering single/dual-row, size, angle, lighting and blur; mark unreadable cases.
   This is diagnostic data, not a sufficient general training or final test set.
2. Keep a separate set of vehicles/recordings untouched for final evaluation.
   Adjacent views of one registration must not cross training/test boundaries.
3. Freeze detector outputs and native crops. Score recognisers first on checked
   crops, then automatic crops to distinguish OCR from crop/layout failures.
4. Use each publisher's own preprocessing, charset and decoder. Export a shared
   candidate/evidence interface while preserving differences between CTC, fixed-slot
   and autoregressive output alignment.
5. Compare parent Awiros, adapted Awiros, SVTRv2, PP-OCRv6, PARSeq and CCT. Include
   a best-frame diagnostic, current multi-frame fusion and optional reconstruction
   ablations. Do not count reconstructed images as independent observations.
6. Measure top-1 whole-plate accuracy, top-5 candidate recall, character error rate,
   correct/wrong accepted outputs, abstentions, per-passage latency and peak memory.
   No acceptances means accepted precision is undefined, not 100%.
7. Fine-tune only the leading one or two candidates using verified representative
   Indian plate data; select settings on validation data and run final tests once.
8. Replace the application default only for a repeatable improvement with acceptable
   wrong-read, latency and memory behaviour. Keep the old model/config recoverable.

On the 6GB NVIDIA laptop, load one candidate at a time and begin with small batches.
Actual training/inference memory and co-residency with the detector must be measured;
publisher RTX 3090 timings and model-file size do not establish local capacity.
The CPU teammate can prepare labels, crops and scoring and review failures. GPU
comparisons, training and final full-video resource measurements belong on the
owner's laptop. No additional model download is required until this comparison is
implemented and inputs are ready.
