# Evaluation protocol

## Unit of evaluation

The **vehicle passage after plate-level association and deduplication** — never a frame, a crop, or a tracker ID.
Frames of one passage and repeated recordings of one vehicle are not independent. Every rate carries its denominator.

## Data splits

| Split | Purpose | Rules |
| --- | --- | --- |
| `development` | debugging, tuning, calibration | ≥2 recordings kept. Model-assisted pre-filled labels allowed if a person verifies them (`label_source: human_verified_prefill`); review-derived labels allowed (`review`). |
| `sealed_a` | first independent evaluation | ≥2 recordings **not** used in development, ≥50 readable passages, **blind human** labels (annotators never see model output). |
| `sealed_b`, `sealed_c` | only after a previous sealed failure | prepared on demand. |
| `test`, `validation` | legacy names kept for old label files | `test` is treated as sealed. |

Enforced by `recorded_evaluation.load_labels` and `annotations.validate`: sealed splits reject any label source other
than `blind_human`; review labels are development-only; the same registration or recording hash in two splits is an
error; duplicate overlapping passages are an error; sealed_a with <50 readable passages or <2 recordings is an error.

**Sealed-set lifecycle.** `tools/evaluate_recorded.py --split sealed_a --final` appends to an append-only ledger
(`data/evaluation/sealed-ledger.json`). A second opening is refused unless `--allow-reopen`, and the report then states
`previously opened: development data, not unseen`; the release gates cannot pass on a reopened set. Once a sealed set has
influenced development it is development data forever; open the next set or collect new recordings. Never tune on it.

## Metrics (all computed by `recorded_evaluation.score_states` unless noted)

| Metric | Definition |
| --- | --- |
| Readable-passage coverage | readable passages with an overlapping CONFIRMED outcome equal to the label / readable passages; Wilson 95 % |
| Accepted precision | correct CONFIRMED / all CONFIRMED on readable passages; undefined (not 0, not 1) if nothing confirmed |
| Unsupported confirmations | CONFIRMED on a passage no person could read; counts as a gate failure |
| Wrong confirmed | CONFIRMED text ≠ label; zero required; error bound reported (rule of three ≈ 3/n and exact one-sided bound) |
| Partial passages / useful rate | PARTIAL whose length equals the label and whose every displayed known character equals the label |
| Wrong known characters | displayed known characters ≠ label (a wrong length counts every known character wrong); Wilson/rule-of-three over known characters |
| Unreadable rate | readable passages ending UNREADABLE or with no outcome |
| False partial watchlist leads | leads whose candidate fingerprints do not include the true plate (`tools/seeded_watchlist_replay.py`) |
| Localisation recall, OCR yield, candidate recall, failing stage | existing `score()` rules, unchanged |
| Exact-plate accuracy / CER | single-crop OCR: `tools/eval_plates.py` (existing); passage-level exact read is the coverage metric |
| Slices | `layout:one_row`, `layout:two_row`, `condition:{night,low_light,blur,small,perspective,glare,occluded,rain,motion}` from annotations; a condition with no annotated passages is **absent (untested)**, never zero |
| Runtime | wall seconds, speed vs real time, OCR calls, peak memory (existing runtime block) |

## Release gates (initial release; `release_gates()`)

1. first opening of a sealed split; 2. ≥50 readable passages; 3. observed coverage ≥60 %; 4. Wilson lower bound ≥45 %;
5. zero wrong confirmed and zero unsupported confirmations; 6. zero wrong known characters in partial results.
Also required but outside that function: seeded-watchlist replay clean, delivery/duplicate tests passed, rollback tested.

At n=50, 30 correct gives 60 % observed and Wilson lower 46.2 % (`wilson(30,50)` = 0.4618): the observed-60 % rule decides.

## Required statements in every report

- accepted sample count and upper error bound (≈3/n when zero errors); **a clean sealed set of ~50 does not show 99.9 %**:
  that needs on the order of 3,000 independent accepted passages with no error;
- per-recording results and the number of recordings;
- Wilson intervals assume independent passages; passages in one recording are correlated, so true uncertainty is wider;
- denominators for partial results, however small.

## Shadow mode

`tools/shadow_compare.py` compares a baseline report and a candidate report produced on identical labels (hash-checked):
confirmed/partial/unreadable counts, wrong-confirmed, wrong known characters, runtime, gates. Promotion is blocked when
the candidate confirms a wrong plate, shows a wrong known character, lacks state scoring, or fails the gates.

## Calibration (development only)

`calibration.py`: risk-versus-coverage over accepted readings, ECE, leave-one-recording-out (needs ≥3 recordings),
versioned `ThresholdConfig` that cannot be looser than the existing gate. Provisional; never described as a guarantee.

## Seeded-watchlist replay

`tools/seeded_watchlist_replay.py` drives the real edge pipeline, outbox, HTTP client and a live backend, including a
central outage and double delivery. Reports missed/false/duplicate confirmed alerts, delivery failures, alerts from pending
reviews; with partial matching on also review recall/precision/false leads, else those are "not applicable". The shipped
scripted cases are **synthetic**; to use real data run it on human-labelled passages (not yet possible; no labels exist).
