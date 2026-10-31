# SVTRv2-B comparison contract

Goal: determine whether official pretrained SVTRv2-B is a promising replacement
for the Awiros recogniser in recorded Indian plate recognition. The owner chooses
promotion; this experiment changes no dashboard default or acceptance threshold.

Compare SVTRv2-B, untouched Awiros and our adapted Awiros on identical inputs.
Use the existing 31-image test diagnostic first, preserving its explicit AI-review,
publisher-label and unknown-pretraining-overlap caveats. Then compare the licensed
three-frame NIVU development passage with the existing CTC sequence scorer. Stills
must never be represented as independent video frames or accepted temporal events.

Report exact whole-plate readings, character errors, per-image inference timings,
GPU peak memory, raw/normalised text and temporal candidates/acceptances. Inputs,
weights, source revision and settings are hashed. Use publisher preprocessing and
safe tensor-only checkpoint loading. Genuine live imagery is outside this task.

Promotion requires a representative human-labelled CCTV holdout plus repeatable
whole-plate improvement and acceptable wrong-acceptance, memory and latency. This
small diagnostic alone cannot promote a replacement or demonstrate full-video
reliability. No training, dataset search or broad environment upgrades are included.

Run candidates sequentially in an isolated container with the cached NVIDIA image,
separate from the CPU dashboard worker. Reuse the saved Graphify code index and
existing OCR benchmark/CTC utilities. Stop and report a concrete blocker only after
reasonable authorised alternatives; never silently substitute another checkpoint.
