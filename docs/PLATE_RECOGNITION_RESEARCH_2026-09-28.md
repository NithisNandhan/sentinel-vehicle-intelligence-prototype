# Sentinel plate recognition: research review and recommended direction

Research date: 28 September 2026. Status: literature review and proposal, not an implemented upgrade or a new local benchmark.

## Bottom line

The strongest direction is a **plate-first, native-resolution, multi-frame recognition pipeline with a checked reconstruction branch**. Improve the supply of usable, correctly associated crops before replacing OCR again. Train restoration for character correctness, not attractive images. Compare original-crop temporal recognition against reconstruction-assisted recognition on exactly the same independently labelled vehicle passages.

This is a scoped review of 44 related papers, not a claim to have found every paper or fully read every publication. Core method/evaluation sections were inspected for MF-LPR2, the ICPR 2026 competition, and UFPR-SR-Plates. The remaining entries were screened through primary paper abstracts, publisher records, author manuscripts, or official repositories. A screened paper is a research lead, not a replicated result.

No camera configuration, model, running service, or application code was changed for this review. No footage, model, or dataset was downloaded. Public availability of a paper does not establish permission to use its code, weights, or dataset.

## 1. Findings that actually change our decision

### Multi-frame reconstruction is a real research direction

MF-LPR2 uses learned optical flow, temporal rejection of bad alignments, and planar refinement appropriate to a rigid plate. It then combines aligned frames and recognizes the result. Its RLPR evaluation uses 200 sequences of 31 frames and reports **86.44% character recognition accuracy**. This is not 86.44% exact Indian whole-plate accuracy. Its alignment ablation is particularly relevant: flow alone reaches 74.70%, while adding filtering and spatial refinement reaches 86.44%. We should study those safeguards, not assume that any optical-flow model automatically provides correct alignment. [Paper and method](https://arxiv.org/html/2508.14797v1).

### The best recent competition supports more than one approach

The ICPR 2026 competition's winning exact whole-plate rate was **82.13%** on 3,000 Brazilian/Mercosur tracks, with five LR images each. Leading approaches included jointly trained restoration/OCR, feature fusion, and direct OCR without dedicated super-resolution. This supports comparing approaches rather than assuming reconstruction always wins. These are not Indian or Sentinel results. [Competition paper](https://arxiv.org/html/2604.22506v1).

### Real low-quality data matters

UFPR-SR-Plates contains 10,000 tracks and 100,000 paired LR/HR images. Its benchmark reports 1.7% recognition on LR inputs, 31.1% after LCDNet super-resolution, and 44.7% after character-position voting across five enhanced images. The improvement is substantial, but the remaining errors show why an enhancement example cannot justify a universal accuracy promise. [Dataset and benchmark paper](https://arxiv.org/abs/2505.06393).

### Restoration must be measured by OCR correctness

LCDNet uses layout/character-oriented training, with OCR guiding restoration. TextZoom and TPGSR similarly treat recognition as the goal rather than photographic quality. These motivate a restoration loss containing character supervision, not merely pixel similarity. We still need an untouched temporal OCR baseline because restoration can alter characters. [LCDNet](https://arxiv.org/abs/2408.15103), [TextZoom](https://arxiv.org/abs/2005.03341), [TPGSR](https://arxiv.org/abs/2106.15368).

### A high score is not a demonstrated probability

Calibration and forensic LPR research distinguish a confident prediction from a correct prediction. Our output needs separately measured whole-plate accuracy, accepted-result precision, coverage and candidate recall. A larger candidate list may improve search usefulness without proving that its first entry is correct. [Calibration](https://proceedings.mlr.press/v70/guo17a.html), [Probabilistic LPR](https://arxiv.org/abs/2302.01427).

## 2. Why this applies to the current code

Read-only inspection found these concrete limits in the diagnostic pipeline:

- `tracked_live_viewer.py` samples at a minimum 500 ms source-PTS interval, so it analyses at most two frames per second, before processing overhead. This is not dense burst collection.
- `experimental_detectors.py` runs vehicle detection at 640 input resolution on CPU. The adapted plate ONNX session explicitly uses CPU execution.
- Vehicle results without a tracker ID are excluded from the downstream vehicle list. A visible vehicle can therefore be absent from plate processing even if a detection exists.
- `tracked_ocr.py` defaults to three vehicle ROIs per analysed frame. This bounds compute but can leave crowded scenes poorly sampled.
- `reconstruction.py` preserves exact decoded native crops. Its bank retains up to eight crops per track with FIFO removal: it is not a best-quality-and-diversity selector.
- The live reconstruction aligns working copies with ECC affine registration and fuses them at the reference crop dimensions. It is **not** a trained multi-frame super-resolution network. The live diagnostic explicitly reports `learned_restoration='not_run'`.
- OCR and reconstruction run in the analysis path. Slow recognition limits how much fresh evidence that path can collect.

These facts suggest collection and localisation are plausible bottlenecks. They do not prove the percentage of failures caused by each one; that needs a labelled audit. We must distinguish a missed plate, a bad crop, a tracking mix-up, failed alignment, wrong characters and a correct candidate rejected by an acceptance gate.

## 3. Proposed solution, in priority order

The following is my engineering recommendation inferred from the literature and the inspected code. It is not a published single architecture or a promised accuracy result.

### A. Build a reliable crop collector

1. Keep the vehicle detector for context, but add periodic direct plate detection in configured road/parking regions so missed vehicle tracks do not prevent localisation.
2. Run plate detection on native-resolution vehicle regions. If it fails, use bounded overlapping tiles, merge duplicate boxes, and return coordinates to the original decoded frame. SAHI supports this small-object strategy; its published gains are on aerial datasets, not guaranteed plate gains.
3. Prefer four-corner/keypoint output, or add a corner-refinement stage. Rectification should correct perspective without cutting off leading/trailing characters or mishandling two-row plates.
4. Detect a plate, then track it between detector refreshes. Reject ambiguous associations and reset across scene cuts. Never combine crops merely because two vehicles look similar.
5. Preserve original crops unchanged. Resize only working copies used by a model. This preserves available pixels; it cannot recover information already lost in camera optics, encoding, or transmission.
6. Separate bounded in-memory crop collection from slower OCR/restoration. Use camera-specific burst policies, but select the final rates from measured throughput and queue age, not a guessed FPS target.

CAM17: stationary plates allow longer observation windows, but repeated identical images add no new spatial detail. Noise variation can help averaging; measurable subpixel motion is needed for some super-resolution gains. CAM01: prioritise short dense windows while vehicles occupy the most readable part of the scene. These are policies to test, not assumptions that every vehicle will become readable.

### B. Retain informative sequences, not just the latest frames

Use a bounded bank that balances focus, native plate dimensions, exposure, occlusion, valid corners and temporal/viewpoint diversity. Keep the clearest reference and several complementary observations. Initially compare banks of 5, 8 and 16 distinct observations; expand only if accuracy/latency results justify it.

Track-specific crop fingerprints prevent exact repeats from becoming extra evidence. Near-duplicates must also be considered when weighting confidence. Retain source PTS and crop-to-frame transforms. Wrong-track fusion is more harmful than having fewer frames.

### C. Compare two recognition branches

**Original branch:** run the current fine-tuned OCR and a challenger such as an Indian-adapted SVTRv2/OpenOCR model on the retained originals. Aggregate character likelihoods with alignment-aware temporal fusion, not a highest-score-frame rule. Existing PARSeq remains a useful comparator. No challenger is selected as the winner until it beats the current reader on the same held-out passages.

**Reconstruction branch:** keep ECC as the inexpensive baseline. Prototype geometry-constrained learned alignment inspired by MF-LPR2, including rejection masks, rigid/planar consistency and protection against flow failures. Compare its output against a recognition-trained restoration model such as LCDNet. Research-grade LP-Diff belongs in the comparison only after rights, weights, runtime and memory checks; it is not the default because it is newer or generative.

Treat originals and all reconstructions derived from them as one correlated evidence bundle. Different reconstructions are not extra independent witnesses. A restored crop can add a useful hypothesis, but a visually sharper result must not override clear contradictory original evidence without validated justification.

### D. Train restoration and recognition together where data allows

Preferred training direction: clear-image teacher supervision, recognition and reconstruction losses, and character-distribution consistency. Add temporal attention only with labelled sequences. This follows the competition's joint-training direction rather than stacking unrelated restorers.

Generate degraded training copies from permitted Indian HR examples: realistic downscaling, directional blur, defocus, noise, clipping, perspective, partial occlusion and codec artefacts. Match the observed feed degradation where measurable. Synthetic degradation is augmentation, not proof of real-camera performance. Static images alone cannot validate a learned temporal model.

Indian-format guidance should be a soft/layout-aware constraint, not a universal hard regex. Support one-row and two-row arrangements and alternate layouts; keep an unconstrained fallback so format assumptions do not remove the true answer. Do not use a registry match as a replacement for image evidence.

### E. Give useful outputs without inventing certainty

For every sufficiently localised track, return the top hypothesis, bounded alternatives, character uncertainties, observation count and failure reason. Where no text hypothesis is supported, return an explicit unreadable result rather than fabricated alternatives. Distinguish `confirmed`, `candidate/review`, and `unreadable` in the engineering view and operator workflow.

Calibrate confidence on held-out labelled passages. A target such as 90% precision among accepted results is different from recognising 90% of all vehicles. Always report coverage alongside precision. Choose operational alert thresholds separately from the diagnostic candidate display.

## 4. Alternatives and why I would not lead with them

| Approach | Strength | Main issue | Decision |
|---|---|---|---|
| Better crop collection + original/reconstructed temporal recognition | Addresses missing evidence and tests reconstruction fairly | More pipeline work and labelled sequence evaluation | Recommended |
| Fine-tune only a stronger OCR | Easier controlled comparison; direct LR recognition can be competitive | Cannot read a plate never localised/captured | Run as the original branch, not the whole solution |
| Large diffusion restorer or visual LLM as the main reader | Can produce plausible detail or additional hypotheses | Character invention, domain mismatch and compute; sharper is not necessarily correct | Experimental comparator only |

## 5. Data and permission gates

- **Indian-labelled crops:** each needs verified full transcription, permitted use, plate bounds/corners and layout where possible. Record provenance per source; do not silently merge uncertain licences.
- **Indian vehicle passages:** ordered original crops of the same plate with verified text and source timing. These are the missing training/evaluation asset for a truly learned temporal reader.
- **Clear-reference pairs:** useful for restoration supervision, but HR/LR frames from different moments must be aligned and must not be treated as perfectly corresponding pixels.
- **Scene annotations:** visible vehicles/plates, corners, occlusion and readability. Include missed detections, not just successful crops.
- **Independent splits:** group by vehicle identity/passage and acquisition source. Do not put adjacent frames or duplicate images on both sides of a train/test split.
- **Sentinel footage:** retain the agreed live RAM-only policy. Do not build a saved Sentinel training set without explicit authorised permission. If independent ground truth is unavailable, report model candidates and pipeline operation, not measured reading accuracy.

Verified resource cautions:

1. The [Indian_LPR authors' repository](https://github.com/sanchit2843/Indian_LPR) explicitly says its 16,192-image dataset cannot be made public because of legalities. The paper is not evidence that we can download or redistribute that dataset.
2. [LRLPR-26 official repository](https://github.com/raysonlaroca/lrlpr-26-dataset) requires an institutional-email request and agreement; use is academic and non-commercial. It is not Indian data or an unrestricted production asset.
3. [RLPR official data page](https://data.mendeley.com/datasets/4rs5wpvckz/2) labels Korean Hangul as blurred and provides only numerical recognition labels. Its terms explicitly exclude surveillance/law-enforcement/commercial applications. Do not use this dataset in Sentinel; studying the paper's method is separate from using its data.
4. [OpenOCR](https://github.com/Topdu/OpenOCR) offers SVTRv2 code, training/evaluation and model links under an Apache-2.0 repository licence. Individual weights/data provenance still needs checking; these are not verified Indian checkpoints or a published competition-winning checkpoint.
5. [LCDNet author code](https://github.com/valfride/lpsr-lacd) and [LP-Diff author code](https://github.com/haoyGONG/LP-Diff) exist. A reusable licence, available checkpoint and compatibility with our setup were not established by this review. LP-Diff's README and published abstract differ in dataset counts; use the published version when reporting paper statistics and verify the downloaded release separately.
6. A complete official MF-LPR2 implementation/checkpoint was not located in this search. Its paper identifies reusable flow and recognizer components, but a faithful reproduction is still engineering work.

## 6. Experiments before any success claim

### First isolate where failure occurs

For a permitted, labelled set of sequences, compare:

- automatic plate crops versus manually supplied correct native crops;
- current sampling versus denser crop collection;
- FIFO crop retention versus quality/diversity retention;
- original temporal OCR versus ECC-assisted recognition;
- ECC versus checked learned alignment;
- current fine-tuned OCR versus a fine-tuned challenger;
- separate training versus jointly recognition-supervised restoration.

Every comparison uses the same passages and fixed test split. A single-frame result is an ablation/control only; the product remains multi-frame. Use correct transcription only for scoring, never as an input to inference or crop selection at test time.

### Required metrics

| Stage | Measure |
|---|---|
| Localisation | Recall of visible/readable plates, corner/crop accuracy, missed vehicle versus missed plate counts |
| Collection | Usable distinct crops per track, native plate width/height, queue age, wrong-track associations |
| Alignment | Rejected frames, valid support, corner/reprojection consistency, visible character ghosting |
| Recognition | Exact whole-plate accuracy per passage, character error rate, top-1 and candidate recall@5/@20/@200 |
| Decision | Precision among confirmed outputs, coverage, false confirmations, abstentions, confidence calibration |
| Runtime | Per-stage latency, end-to-end delay, sustained throughput, dropped frames, actual VRAM and RAM |

Failure curves should vary native plate width, blur, compression, angle and sequence length. Report sample counts and uncertainty intervals. A synthetic failure curve and a real-camera curve must remain separate. Do not publish a universal minimum pixel width from one vehicle passage.

### Acceptance sequence

1. Demonstrate accurate multi-frame reading on permitted labelled Indian passages with manual correct crops.
2. Repeat with automatic crops; quantify the lost performance attributable to detection/cropping.
3. Compare reconstruction against untouched temporal OCR; keep it only where it improves exact reading or candidate recall without unacceptable false confirmations.
4. Validate sustained laptop throughput without unbounded backlog.
5. Run CAM17 and CAM01, show native crops, derived crops, per-frame hypotheses and final candidate/confirmation state live. Confirm accuracy only on independently verifiable examples.

No literature result supplies our acceptance rate. Any inability to obtain lawful sequences or independent labels is an explicit validation limit, not something a larger model fixes.

## 7. Annotated primary-source bibliography: 44 papers

Years below refer to the linked paper/preprint or stated conference year; preprint appearance and final publication can differ. `Core` means relevant method/evaluation sections were inspected. `Screened` means abstract, primary bibliographic record, or author description was inspected, not a complete reproduction or exhaustive full-text critique.

### Plate-specific restoration and multi-frame methods

1. **Core — ICPR 2026 Competition on Low-Resolution License Plate Recognition (2026).** Exact-match multi-frame benchmark; teacher-student restoration, feature fusion and direct OCR all appear among leading approaches. [Paper](https://arxiv.org/abs/2604.22506).
2. **Core — MF-LPR2: Multi-Frame License Plate Image Restoration and Recognition using Optical Flow (2025).** Filter/refine flow before combining frames; distinguish its character metric from exact whole-plate accuracy. [Paper](https://arxiv.org/abs/2508.14797).
3. **Core — Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark (2025).** Real LR/HR tracks and OCR-output fusion; useful evaluation model with restricted data access. [Paper](https://arxiv.org/abs/2505.06393).
4. **Screened — Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach (2024).** LCDNet/LCOFL directs restoration toward plate layout and character recognition. [Paper](https://arxiv.org/abs/2408.15103).
5. **Screened — LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate (CVPR 2025).** Multi-frame cross-attention and diffusion-based restoration on real degraded plates; experimental comparator, not a guaranteed reader. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.pdf).
6. **Screened — A Dataset and Model for Realistic License Plate Deblurring (IJCAI 2024).** LP-Blur uses a dual-camera collection process; plate-specific motion blur is distinct from generic deblurring. [Paper](https://www.ijcai.org/proceedings/2024/0086.pdf).
7. **Screened — Forensic License Plate Recognition with Compression-Informed Transformers (ICIP 2022).** Recognition informed by compression severity; especially relevant to encoded CCTV streams. [Paper](https://arxiv.org/abs/2207.14686).
8. **Screened — Embedding Similarity Guided License Plate Super Resolution (2025).** Embedding consistency guides restoration; not evidence of Indian-camera generalisation. [Paper](https://arxiv.org/abs/2501.01483).
9. **Screened — Superresolution of License Plates in Real Traffic Videos (2007).** Classical fusion of shifted observations; demonstrates that this direction predates deep learning. [Author manuscript](https://www.ee.iitm.ac.in/~raju/journals/j27.pdf).
10. **Screened — License Plate Recognition based on Temporal Redundancy (ITSC 2016).** Aggregates evidence across a vehicle track instead of trusting one frame. [Author manuscript](https://homepages.dcc.ufmg.br/~william/papers/paper_2016_ITSC.pdf).
11. **Screened — Character Time-series Matching For Robust License Plate Recognition (2023).** Alignment and character association across frames; published dataset/GPU results cannot be transferred directly to us. [Paper](https://arxiv.org/abs/2307.11336).
12. **Screened — Geometry-Constrained Multi-Frame Character Association for License Plate Recognition on Moving Cameras (Sensors 2026).** Highlights how incorrect motion/association can make temporal fusion worse. Publisher abstract/text was discoverable; full page access was intermittent. [Paper](https://www.mdpi.com/1424-8220/26/18/5704).

### Localisation, plate datasets and recoverability

13. **Screened — Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (2022).** Targeted overlapping crops can preserve small-object scale at detector input; costs more inference. [Paper](https://arxiv.org/abs/2202.06934).
14. **Screened — License Plate Detection and Recognition in Unconstrained Scenarios (ECCV 2018).** WPOD-Net detects and rectifies distorted plates; useful corner/geometry design precedent. [Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.html).
15. **Screened — Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline (ECCV 2018).** Introduces CCPD; useful localisation benchmark, but Chinese plate layouts differ. [Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.html).
16. **Screened — A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector (2018).** UFPR-ALPR and a staged ALPR baseline; latency and reading must be evaluated end to end. [Paper](https://arxiv.org/abs/1802.09567).
17. **Screened — Indian Licence Plate Dataset in the wild (2021).** Indian corner/character annotations and localisation/OCR pipeline; associated dataset is not publicly released by authors. [Paper](https://arxiv.org/abs/2111.06054).
18. **Screened — LSV-LP: Large-Scale Video-Based License Plate Detection and Recognition (TPAMI, accepted 2022).** Chinese video-based plate benchmark covering varied capture conditions. [Author manuscript](https://crabwq.github.io/pdf/2022%20LSV-LP_final.pdf).
19. **Screened — FANVID: A Benchmark for Face and License Plate Recognition in Low-Resolution Videos (2025, revised 2026).** Temporal LR benchmark; only its plate task is relevant, not face recognition. [Paper](https://arxiv.org/abs/2506.07304).
20. **Screened — Mapping License Plate Recoverability Under Extreme Viewing Angles for Opportunistic Urban Sensing (2026 preprint).** Sweeps degradation/angle space to map failure boundaries; not a universal real-camera accuracy guarantee. [Paper](https://arxiv.org/abs/2604.23814).

### Recognizers and text-directed enhancement

21. **Screened — LPRNet: License Plate Recognition via Deep Neural Networks (2018).** Lightweight segmentation-free plate recognizer; useful speed baseline, not the similarly named point-cloud network. [Paper](https://arxiv.org/abs/1806.10447).
22. **Screened — Scene Text Recognition with Permuted Autoregressive Sequence Models (ECCV 2022).** PARSeq supports context-aware decoding; domain-specific plate training remains necessary. [Paper](https://arxiv.org/abs/2207.06966).
23. **Screened — SVTR: Scene Text Recognition with a Single Visual Model (2022).** Visual text representation with efficient variants; challenger family rather than automatic replacement. [Paper](https://arxiv.org/abs/2205.00159).
24. **Screened — SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition (ICCV 2025).** Improved CTC recognizer; competition also uses an AR variant, which should not be conflated with every SVTRv2 checkpoint. [Paper](https://arxiv.org/abs/2411.15858).
25. **Screened — Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition (CVPR 2021).** ABINet separates visual/language reasoning; natural-word priors require caution on registration strings. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.html).
26. **Screened — Scene Text Image Super-Resolution in the Wild (ECCV 2020).** TextZoom/TSRN prioritises recognition on real text pairs over synthetic-only photographic SR. [Paper](https://arxiv.org/abs/2005.03341).
27. **Screened — Text Prior Guided Scene Text Image Super-resolution (2021 preprint; TIP 2023).** Character probabilities guide reconstruction; useful model coupling, with feedback-error risk to test. [Paper](https://arxiv.org/abs/2106.15368).

### Alignment and general restoration foundations

28. **Screened — RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (ECCV 2020).** Learned dense motion; low-quality plate flow still requires independent consistency checks. [Paper](https://arxiv.org/abs/2003.12039).
29. **Screened — FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation (CVPR 2023).** Flow estimator used by MF-LPR2; motion benchmarks do not establish plate character accuracy. [Paper](https://arxiv.org/abs/2303.01237).
30. **Screened — Deep Burst Super-Resolution (CVPR 2021).** Learned alignment and attention fusion; RAW Bayer bursts are not our compressed decoded RGB stream. [Paper](https://arxiv.org/abs/2101.10997).
31. **Screened — Towards Real-World Burst Image Super-Resolution: Benchmark and Method (ICCV 2023).** RealBSR includes RGB/RAW and geometry-aware fusion; potential research basis, not plate-specific validation. [Paper](https://arxiv.org/abs/2309.04803).
32. **Screened — EDVR: Video Restoration with Enhanced Deformable Convolutional Networks (2019).** Feature alignment and temporal attention instead of naive pixel averaging. [Paper](https://arxiv.org/abs/1905.02716).
33. **Screened — BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment (2021 preprint; CVPR 2022).** Recurrent temporal restoration; broad image metrics are not exact plate-reading metrics. [Paper](https://arxiv.org/abs/2104.13371).
34. **Screened — Investigating Tradeoffs in Real-World Video Super-Resolution (CVPR 2022).** RealBasicVSR warns that propagation can amplify severe degradation; supports explicit cleaning/rejection comparisons. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.html).
35. **Screened — Simple Baselines for Image Restoration (ECCV 2022).** NAFNet provides efficient generic deblurring; no plate correctness guarantee. [Paper](https://arxiv.org/abs/2204.04676).
36. **Screened — Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (2021).** Useful degradation-modelling baseline; photographic plausibility cannot establish true characters. [Paper](https://arxiv.org/abs/2107.10833).
37. **Screened — Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022).** General image restoration; retain only if measured plate recognition improves. [Paper](https://arxiv.org/abs/2111.09881).

### Reliability, tracking, data quality and fair evaluation

38. **Screened — ByteTrack: Multi-Object Tracking by Associating Every Detection Box (2021 preprint; ECCV 2022).** Lower-score detections can support tracking rather than being discarded prematurely. [Paper](https://arxiv.org/abs/2110.06864).
39. **Screened — On Calibration of Modern Neural Networks (ICML 2017).** Model scores need empirical calibration; temperature scaling is a baseline, not a universal cure. [Paper](https://proceedings.mlr.press/v70/guo17a.html).
40. **Screened — Reliability Scoring for the Recognition of Degraded License Plates (AVSS 2022).** Explicit reliability for forensic plate reading; supports uncertain candidate states. [Author manuscript](https://faui1-files.cs.fau.de/public/publications/mmsec/2022-Maier-AVSS.pdf).
41. **Screened — Benchmarking Probabilistic Deep Learning Methods for License Plate Recognition (2023).** Uncertainty under noise, blur and domain shift; scores should expose failure rather than hide it. [Paper](https://arxiv.org/abs/2302.01427).
42. **Screened — Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition (IJCNN 2023).** Avoid duplicate/frame leakage before believing fine-tuning gains. [Paper](https://arxiv.org/abs/2304.04653).
43. **Screened — On the Cross-dataset Generalization in License Plate Recognition (VISAPP 2022).** Same-dataset performance can conceal substantial cross-domain failure. [Paper](https://arxiv.org/abs/2201.00267).
44. **Screened — The Perception-Distortion Tradeoff (CVPR 2018).** Perceptual realism and reconstruction fidelity are distinct objectives; do not equate a better visual with correct identity. [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html).

Additional relevant lead, outside the 44-paper count: [Advancing Multinational License Plate Recognition Through Synthetic and Real Data Fusion: A Comprehensive Evaluation](https://arxiv.org/abs/2601.07671). Its evaluation of real/synthetic training combinations supports investigating controlled Indian augmentation, not assuming that merging every available dataset helps.

## 8. Final recommendation

Do not discard the existing system or start another unmeasured model-upgrade cycle. Preserve it as a baseline. First improve crop recall and sequence acquisition, then compare original temporal OCR with geometry-checked reconstruction and an Indian-adapted recognition challenger. Pursue joint restoration/recognition training only with adequate permitted labels and actual held-out gains.

The goal is the largest **measured number of correct whole-plate readings**, with useful uncertain candidates and transparent failure explanations. The research establishes that this direction is credible; it does not establish our local accuracy or guarantee recovery from every camera.
