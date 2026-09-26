# Multi-frame plate recognition: implementation and evidence boundaries

Implemented on 2026-09-26. This extends the existing edge-to-central pipeline; it does not change the two monitored cameras, central contracts, operator workflow or privacy policy.

## What is deep learning here?

YOLOv8n detects vehicles; a pretrained YOLOv9-t 512-pixel model detects plate regions inside vehicle crops; FastALPR's pretrained `cct-s-v2-global-model` reads plates. Those are neural models. ByteTrack associates vehicle observations. Our new temporal component aggregates neural OCR probability distributions across frames. It is **not** a newly trained temporal neural network, optical-flow reconstruction, video super-resolution or registry-assisted identification.

The original baseline used a 384-pixel plate detector, CCT-XS OCR and two identical high-scoring whole-string readings. The new path can combine evidence even when complete frame-level strings disagree.

## Processing sequence

1. Decode continuously; sample detection frames using presentation timestamps, not reported camera FPS. Each camera owns its tracker and evidence buffers.
2. For each tracked vehicle, attempt plate recognition at most every 300 ms of video time, with at most 40 attempts. Expire inactive vehicle state after 30 seconds; cap active tracks at 128.
3. Detect plate regions in the vehicle crop. Select the highest detector-scoring region, add small bounded padding, and reject crops smaller than 20 by 6 pixels. These thresholds are engineering guards, not proof that a crop is readable. Wider candidate processing is deliberately allowed than in the old baseline.
4. Run the pretrained OCR network. The public OCR API exposes only winning characters and their scores; our narrow adapter retrieves all alternatives for each character slot using the same preprocessing and ONNX output. A runtime check compares that adapter with the public API on an in-memory synthetic text image, including probability row sums.
5. Keep the latest twelve probability matrices in RAM for that vehicle. Weight observations by plate-crop area, bounded between 0.1 and 1. Average each character's probabilities across observations. Do not multiply scores as if adjacent frames were independent evidence.
6. Decode a bounded beam of at most 200 alternatives, considering the four highest-probability tokens plus padding per slot. Padding must trail the text. Filter for broad common state-registration and Bharat-series layouts. Grammar is a hint, not an exhaustive legal validator; no assumed state or registration database is used. Filtering does not renormalize the scores.
7. Accept only after at least three temporally separated observations. Require mean character score at least the configured threshold (default 0.85), weakest character score at least 0.75, a log-score margin of at least 0.5 over the next retained valid alternative, and support of at least 0.60 for each character in two observations. Otherwise abstain.
8. Emit once per track; suppress repeat plate sightings for five minutes. Existing event construction immediately masks the plate and computes its keyed HMAC. Only protected metadata enters the SQLite outbox and central API. Live events identify the model as `yolov8n+fastalpr-cct-s-v2+temporal-v1`.

Timestamp discontinuities reset the tracking evidence. A large shift of a plate region within the vehicle crop clears that plate's fusion buffer. These guards reduce accidental mixing but cannot guarantee correct vehicle association. Scene cuts with continuous timestamps and small spatial changes can still escape these guards; no appearance-matching system was added.

## Scores and limitations

- The `confidence` wire field remains compatible with the existing API. For the temporal path it means **uncalibrated mean character score**, not the probability that the whole plate is correct. A score of 0.90 is not a measured 90% accuracy claim.
- The deployed CCT-S model has **ten character slots**. It cannot represent longer registrations. The grammar helper supports some longer layouts for future models, but that does not remove the deployed model's limit.
- Slot-level fusion assumes the network consistently aligns character positions. There is no geometric frame alignment, two-row segmentation or learned temporal alignment. Persistent blur, lost pixels, inconsistent character alignment and wrong plate crops remain limitations.
- The 200-entry beam is a computational limit, not a guarantee that the correct plate appears among 200 possibilities. Even high-scoring, syntactically valid alternatives may be wrong.
- Repeated detections are not unique correctly identified vehicles or plates. Non-empty OCR output is not a successful recognition. A larger model is not automatically better for these scenes.
- Genuine frames, crops, candidates and full plate text stay in memory. The benchmark emits aggregate counts/timings only. No live images or candidate list is saved or sent to the browser.
- Authorised registry access is not implemented. It could constrain or rank candidates if the true plate is retained, but any improvement must be evaluated against labelled data. No automatic confidence boost is claimed.

## Reproducible live comparison

With the online relays already running and local credentials configured:

```powershell
docker compose --profile online stop edge-worker
docker compose --profile online build edge-worker
try {
    docker compose --profile online run --rm --no-deps edge-worker python -m sentinel_edge benchmark-live --probe-seconds 180
} finally {
    docker compose --profile online up -d --no-deps edge-worker
}
```

Each camera applies three methods to the same sampled frames: original models/consensus, upgraded models/original consensus, and upgraded models/temporal processing. Initialization and synthetic warm-up occur before the probes. Each camera measures the requested observation interval from its first decoded frame, with bounded startup. Report startup and actual observed time rather than assuming perfect continuity. Methods have separate tracker state and run sequentially on each sampled image; the two cameras run concurrently. This comparison increases workload over the normal worker and is not a production-throughput test.

The temporal method also changes crop gating and attempt budget, so this is a comparison of complete pipeline variants, **not a clean fusion-only ablation**. Latency covers the vision call, not camera-to-dashboard alert delivery. Models use CUDA execution providers with CPU fallback. Counts of accepted identifiers are measured; accuracy remains unknown without ground truth.

## Validation required before accuracy claims

Dependency-light tests cover complementary character errors, consistent evidence, weak inputs, ties, duplicate timestamps, invalid probabilities, bounded storage, reset behavior, deduplication and the real pipeline's acceptance flow with mocked model outputs. They validate software behavior, not real-camera recognition accuracy. The neural-model parity check validates the adapter, not recognition of the synthetic label.

For accuracy, obtain permitted labelled Indian-plate examples and labelled vehicle passages spanning resolution, viewing angle, daylight/night and blur. Split by vehicle/passage and camera rather than placing adjacent frames in both development and test sets. Compare exact whole-plate accuracy, character accuracy, accepted-reading precision and coverage/abstention, false watchlist alerts, and candidate recall at K. If calibration is needed, fit it only on held-out development data and report performance on untouched test data. The live results and remaining acceptance gates are in [VALIDATION.md](../VALIDATION.md).
