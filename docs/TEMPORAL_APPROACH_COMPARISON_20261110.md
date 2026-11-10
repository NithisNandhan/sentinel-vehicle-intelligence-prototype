# Comparing one frame, temporal OCR and reconstruction

This experiment answers whether temporal combination improves the deployed OCR
on already human-reviewed Mumbai targets. It changes no model weights, detector
settings, sampling, deployed gates or dashboard behaviour.

## Measured outcome

The complete GPU capture decoded 928 frames, analysed 62 frames and recorded all
172 admitted original OCR observations in 98 uninterrupted OCR-reader segments.
Capture took 260.36 seconds including diagnostic evidence collection. Each
method reused the same deployed model outputs; no training was performed.

| Approach | Exact on the six safely matched references | After excluding one mixed-vehicle history | On retained targets with at least three observations |
|---|---:|---:|---:|
| One image-quality-selected frame | 3/6 | 2/5 | 0/2 |
| Multi-frame, without reconstruction | 2/6 | 2/5 | 0/2 |
| Current multi-frame with reconstruction | 2/6 | 2/5 | 0/2 |

Across all 18 reference targets the conservative raw counts are 3/18, 2/18 and
2/18. Twelve targets lack a safe geometry association at their reference instant;
this does **not** establish that the detector missed their entire passage. One
additional full-plate crop is visible but falls below the prespecified polygon-
bounds IoU threshold because it includes substantial surrounding vehicle context.

The raw single-frame advantage occurs entirely on reference 19. Independent main
and QA visual inspection found that OCR segment `ocr-0090`, vehicle tracker 117,
contains two different plates: a yellow state-series plate at 28.0 seconds and a
white Bharat-series plate at 30.5 seconds. Their 2,500 ms gap does not exceed the
current evidence-expiry threshold. Combining them changes the first target's
correct single-frame output into an incorrect reading with a missing character.

This is an **evidence-association failure**, not a clean test showing that temporal
recognition is intrinsically worse. The 2/5 comparison is an explicitly post-review
integrity sensitivity analysis; it does not replace the frozen raw report.
Reconstruction neither recovered nor lost an exact result on this matched subset.
Neither temporal approach emitted a model-accepted reading on these six references;
the three unique acceptances elsewhere in the clip are not verified by these labels.

**Decision:** the quality-selected single frame performs best in the current raw
diagnostic, but the correctly grouped subset does not select a winner. Keep model
weights and runtime settings unchanged. Prioritise preventing evidence from crossing
vehicle passages, then repeat the comparison with more labelled multi-frame passages.
Only two matched references supplied three or more observations, so this is weak
evidence about the value of correctly grouped temporal recognition.

## Fixed protocol

1. Process the complete permitted 30.933-second recording on GPU with the deployed
   Awiros adaptation, current detector configuration and normal 500 ms sampling.
2. Capture the original OCR probabilities and independently generated per-frame
   candidate beams, native crops, timestamps and plate coordinates. Capture does
   not read human labels. Each uninterrupted OCR-reader instance is a separate
   segment, including resets that occur within the same vehicle tracker ID.
3. Compare three strategies on those same original probabilities:
   - **One frame:** choose the highest image-quality observation in the segment,
     with earlier timestamp as tie-breaker, then use the existing candidate decoder.
     Neither truth nor OCR confidence chooses the image.
   - **Multi-frame originals:** apply the unchanged quality-weighted CTC fusion,
     rolling eight-observation window and acceptance rules, without reconstruction.
   - **Multi-frame with reconstruction:** use the operational run's actual final
     candidate ranking and acceptance decisions, including ECC reconstruction
     proposals and its disagreement veto. Derived images add no independent votes.
4. Only after capture, load the earlier human labels and associate references by
   timestamp (within 20 ms) and geometry. Use the bounding rectangle of existing
   manual plate/text polygons where available, otherwise the original manual box.
   Require IoU of at least 0.30. Reject overlapping alternatives whose IoU is within
   0.05 of the best. Transcription never determines association or frame choice.
   A segment linked to conflicting existing human transcriptions is quarantined
   rather than selecting whichever truth produces a favourable result.
5. Exclude human-unreadable IDs 08 and 14 and the previously disputed ID 21,
   leaving 18 reference targets. Preserve unmatched targets as missing outputs.
   Report the matched subset and the subset with at least three observations
   separately, so single-frame cases cannot masquerade as a temporal experiment.

All methods receive the same geometric OCR working views already used in capture,
including text rectification when enabled. "Originals" means independent source
frame evidence, not an additional preprocessing change. The one-frame policy may
retain an earlier good view, while the existing temporal policy keeps its rolling
window; this difference is intentional and documented.

## Evidence boundary

This is a paired diagnostic on previously inspected reference targets, not a new
test set, an exhaustive vehicle inventory or the full-video accuracy report.
Multiple reference targets may refer to the same passage; they are not assumed
to be independent vehicles. Source/model/decoder/trace/label identities are hashed.

Exact top guesses and actual model acceptances are separate. A one-frame result
cannot meet the existing requirement for repeated independent support, so its
confirmation count is not compared as though it used the temporal gate. Every
distinct wrong temporal acceptance remains an error even if corrected later.

The comparison reuses model outputs to isolate aggregation. Its replay timings
are not separate end-to-end throughput measurements for the three approaches.
No model or runtime option is promoted automatically from these small results.

## Reproduction and files

With the installed cached GPU runtime, pause the recorded dashboard worker and
run `tools/compare_temporal_approaches.py --phase capture --output <fresh-folder>`
through `tools/run_ocr_research.py` in the existing GPU Compose service, mounting
the ignored `data` folder under `/workspace/data`. This needs no downloads.
Restore the recorded dashboard worker once capture completes.

Then run locally:

```powershell
python tools/compare_temporal_approaches.py --phase score `
  --output data/benchmarks/temporal-approaches-20261110
python -m unittest discover -s tools -p test_temporal_approaches.py -q
```

The ignored run folder contains `trace.json.gz`, native `assets/`, detailed
`report.json`, a side-by-side `index.html` and `progress.json`. Human truths,
full predictions and crops remain local. Only aggregate evidence is published.
The explicitly reviewed `integrity-audit.json` and `integrity.html` preserve the
mixed-vehicle finding and sensitivity results without modifying `report.json`.
`capture-runner.py` and `scoring-runner.py` preserve the exact scripts identified
by the trace and raw report; the scoring identity is recorded separately.

To reproduce the explicit visual integrity audit after inspecting the images:

```powershell
python tools/compare_temporal_approaches.py --phase audit `
  --output <new-scored-run-folder> --mixed-segment ocr-0090 `
  --integrity-reviewer "Name of actual reviewer" `
  --integrity-note "Explain the visually established identity mismatch"
```

This requires an existing scored run and a fresh audit file; it never edits the
original report. The 2026-11-10 audit records actual assistant and QA inspection.

Six controlled tests cover exact decoder replay, image-only selection, reader
segmentation, timestamp/geometry association and the possibility that conflicting
frames overturn a correct clear-frame reading, plus conflicting-label quarantine. These establish evaluator
behaviour; real recognition measurements come from the captured footage.

QA found two evaluator edge cases before final scoring: whole-segment hash
deduplication (the capture's 172 observations equal the pipeline's 172, proving
this run was unaffected) and shared mutable per-reference result dictionaries.
Both are corrected in the reusable tool. The original capture and raw report
remain preserved. Graphify's existing graph was reused; its installed launcher
remains broken, so incremental refresh is unavailable.
