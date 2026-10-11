# Task 9 report: evaluation tooling

Closed 10 October 2026. **Score: 7 / 10**

## What was there before

- The repository had benchmark tools for still images and for one development
  passage (`tools/validate_crop_collection.py` and others). None could take a
  recorded video and a label file and report vehicle-level results.
- There was no label format for recorded footage.
- The handoff states the metrics that must be reported (completion guide,
  page 11) but nothing computed them.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `edge/sentinel_edge/recorded_evaluation.py` | Did not exist | Label loader with validation; scoring of one recording; totals across recordings; a collector that observes the real pipeline; `evaluate()` that produces the report |
| `tools/evaluate_recorded.py` | Did not exist | Command-line runner. Loads the unchanged models, writes a JSON report with settings, model hashes, environment and runtime. Refuses to overwrite a report. Needs `--final` for the test split. |
| `edge/sentinel_edge/tracked_live_viewer.py` | No way to observe a sample | Optional `on_sample` hook, called after each analysed sample. It does not change what is analysed or accepted, so the evaluator measures the same path the interfaces use. |
| `docs/RECORDED_VIDEO.md` | Did not exist | Label format and how to read the report (written with Task 7) |
| `edge/tests/test_recorded_evaluation.py` | Did not exist | 17 tests |

What the report contains, each with numerator and denominator: time-overlap
coverage, vehicle coverage, plate localisation, OCR observation yield, exact
read coverage, wrong reads, abstentions, accepted precision, candidate recall
(top 1, top 5, top K), and runtime. Each failed passage names its first
failing stage.

Rules built in, following the guide:

- Accepted precision is **undefined** when nothing was accepted.
- Outcomes are per vehicle passage; repeated frames do not multiply a success.
- Readings that cannot be checked (on passages nobody could read, or outside
  all labelled passages) are listed separately and kept out of precision.
- Totals are summed across recordings, never averaged.
- Labels must name a human labeller; a vehicle in two splits is rejected;
  labels for a different file are rejected.
- The report holds masked plates only.

## Audit result

Edge suite: **352 passed**, 1 Windows-only failure (335 + 17 new).

The 17 tests check the arithmetic on hand-built cases (correct, nothing
accepted, wrong read, each failing stage, box overlap rule, unverifiable
readings, totals) and one full run from a generated file to a report with a
**mocked** recogniser: 12 of 12 samples analysed, one track, scored correct
against a matching label and as a wrong read against a non-matching one.

## What still has to change

- **It has never scored a real model on real footage.** That needs Task 8
  (models) and Task 10 (footage and labels).
- Matching is by video-time overlap. With several vehicles in view at once and
  no labelled boxes, a reading can be credited to the wrong passage. Labelling
  boxes removes most of this; the limit is stated in the report's rules.
- "Usable crop yield" in the guide means crops a person would call readable.
  That cannot be judged automatically, so the report gives "OCR observation
  yield" instead and says so.
- Peak memory is reported for the evaluation process only, and only on Linux.
  GPU memory is not measured.
- No calibration tooling yet (choosing a threshold on validation data). That
  is Priority 1 in the status document, after a baseline exists.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Loader, scorer, runner, hook and documentation exist |
| Tested | 3 / 3 | 17 tests pass, including an end-to-end run |
| Proven | 0 / 3 | Never run with real models or real labels |
