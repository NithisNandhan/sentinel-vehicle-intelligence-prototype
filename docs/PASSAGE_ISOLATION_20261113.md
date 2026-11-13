# Recognition passage isolation — 13 November 2026

## Observed failure

The former `track-s0-3` human-review card combined several different vehicles.
Two faults contributed: the exporter grouped by scene and raw tracker ID even
after OCR history resets, and the vehicle tracker sometimes switched between
nearby motorcycles while retaining its ID. Similar bounding boxes and vehicle
classes did not establish a shared physical identity.

## Implemented changes

- Review records and running dashboard rows use recognition passage identity.
  Reusing a raw tracker ID starts a clean row when the history changes. A
  supported reassociation carrying the same passage retains one row.
- Crop observations retain their source timestamp, vehicle box, tracker ID,
  passage ID and pixel hash. The selected crop belongs to the actual admitted
  OCR observation, including when a later duplicate has no new OCR vote.
- Before OCR, a plate must have one plausible current vehicle owner. A second
  vehicle box containing the plate centre and at least 85% of its area causes
  the proposal to be withheld. Partial overlap alone does not prove ambiguity.
- A plate aspect-ratio change over twofold breaks continuity, in addition to
  existing time, scale, displacement and vehicle-consistency checks.
- Independent OCR disagreement can split evidence: three differing characters
  scored at least 0.9 in both equal-length readings, or two differing terminal
  digits when both observations end in four digits. This is a veto, not an
  identity match, calibrated confidence estimate or acceptance shortcut.
- Inspector requests are pinned to the selected row/passage. Reused IDs cannot
  silently replace the displayed vehicle. Missing retained evidence is explicit.

The neural weights and automatic acceptance thresholds were not changed.

## Recorded development validation

The permission-receipted Mumbai recording is 3840×2160 and about 31 seconds.
Both corrected GPU captures decoded all 928 frames and analysed 62 samples.
The final frozen capture is local at
`data/benchmarks/full-video-passage-review-v2-20261113/`.

| Measurement | Previous raw-ID capture | Passage-aware v2 capture |
| --- | ---: | ---: |
| Decoded frames | 928 | 928 |
| Analysed samples | 62 | 62 |
| Retained plate crop observations | 183 | 144 |
| Ownership-ambiguous proposals withheld | Not measured | 39 |
| Unique model-accepted readings, unverified | 3 | 3 |
| Review records | 51 raw tracker records | 163 recognition fragments |

The record counts are not unique physical vehicle counts. Splitting is
deliberately conservative, and human duplicate review is still required. The
9.5-second scooter and following motorcycles at 10 and 11 seconds from tracker
3 now occupy separate records with one crop each. All frozen record IDs,
prediction IDs, evidence hashes and selected crop timestamps reconcile.

The updated running dashboard also completed a GPU run: 928 decoded frames,
62 analysis samples and 164 records. A UI rehearsal verified normal-speed
sampled replay, pause, scrubbing, historical rows and no new inference requests
during replay. Counts vary slightly between runs; no deterministic-output or
latency claim follows from these checks.

The new capture starts with 163 pending human decisions. Whole-video accuracy
remains unavailable. Existing manual labels and earlier frozen captures were
preserved, and no OCR output was turned into ground truth. The observer sends
no central events.

## Remaining limits

The changes fix the demonstrated tracker-3 contamination; they do not establish
perfect physical identity. Visual review of all 14 saved multi-crop sheets found
a residual tracker-148 switch at 13.5–15 seconds involving a tiny non-plate trim
proposal, and truck-280 windscreen false proposals in another passage. Similar
geometry with weak/non-plate OCR can evade the current identity vetoes. These
records require correction before treating them as single-vehicle accuracy
units. A better physical association check and plate/non-plate rejection remain
follow-up work; confidently wrong OCR can also cause excessive fragmentation.

The model-accepted count is not an accuracy score. Source video, derived crops
and local review labels are not automatically published with the code.

## Open the corrected reviewer

```powershell
python tools/serve_video_review.py --run data/benchmarks/full-video-passage-review-v2-20261113 --video data/recordings/footage.mp4 --port 7797
```

Open http://127.0.0.1:7797/. The operator dashboard is at
http://127.0.0.1:5173/. Keep the older mixed-record review as historical evidence;
do not overwrite its labels or transfer them by matching OCR text.
