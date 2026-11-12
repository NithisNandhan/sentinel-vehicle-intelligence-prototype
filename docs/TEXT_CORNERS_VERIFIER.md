# Text corners and a learned check before OCR

This experimental path implements:

`automatic vehicle region -> text outlines -> oriented text enclosure -> visual plate verifier -> OCR -> existing reconstruction`

It is opt-in. It does not replace the deployed OCR weights, change watchlist
matching or relax operational acceptance thresholds.

## Geometry and verification

`text_plate_geometry.text_plate_candidates` preserves text-detector quadrilaterals,
joins compatible fragments into up to two text rows and expands along neighbouring
strokes in an orientation-aligned working copy. Weak text seeds remain candidates
even when a complete plate or confident OCR reading is unavailable. Known distinct
plate regions veto an enclosure spanning both. Native source images are unchanged.

The returned four corners are an **estimated enclosure of the writing**, not
proof of a visible physical plate border. This supports painted registrations
and plates whose background blends into the vehicle. It does not establish that
every painted registration will be found.

`PlateRegionVerifier` is a trained 14.5 KB HOG/logistic classifier. It looks at
image gradients and texture, without invoking OCR. Its output is `likely_plate`,
`uncertain` or `likely_non_plate`; the numeric score is not a calibrated real-world
probability. Vehicle-context metadata is recorded, but the current learned
classifier itself is image-only. Spatial ownership remains the existing router's
responsibility; a new temporal plate/non-plate classifier is not claimed.

The original strict pre-OCR gate passes only `likely_plate` regions. The newer
`--provisional-ocr` option also reads uncertain regions as described below. It preserves uncertain and
rejected native/rectified crops and reasons in the engineering audit. Their OCR
readings are deliberately absent. A low score does not prove a region is not a
plate; one labelled test plate was rejected, so these decisions must be inspectable.

## Training evidence

The existing frozen Gujarat/Make ML detection manifest supplies 360 training,
130 validation and 77 test images. Group/image identities do not cross splits.
Positive regions use publisher annotations; negatives are sampled away from all
known plate boxes and are **assumed background**, not independently verified
negative labels. Padding/contrast augmentation is confined to training.

Mumbai footage, transcriptions and crops are excluded from training and threshold
selection. The verifier's model, exact feature contract, hash and provenance are
in `edge/assets/plate-region-verifier/`. The test result is 56/83 plate regions
accepted, 26 uncertain and one rejected; among 462 assumed backgrounds, zero
accepted, 102 uncertain and 360 rejected. These are crop-classifier results on
proxy labels, not plate-reading or whole-scene detection accuracy. Keeping the
uncertain regions in review retains 82/83 plates; only 56/83 currently reach OCR.

## Integration and local review

The existing research runner gains `--text-corners --plate-verifier <model.npz>`
alongside `--plate-first --character-guided`. Standard tracking/reconstruction
options continue to apply. Text-corner mode requires the verifier and allows
tentative vehicle regions to seed discovery without granting confirmed ownership;
reviewed non-vehicles remain excluded. The verifier acts before identity OCR, and same-frame
deduplication retains one selected view. The earlier `--refine-characters` path
depends on OCR and must not be combined with pre-OCR verification; this path
uses the independent text-corner expansion instead.

For a bounded check before another long full-video run:

```text
python tools/run_ocr_research.py tools/probe_text_corners.py --schedule data/benchmarks/text-corner-schedule-20261111/schedule.json --output data/benchmarks/<fresh-audit> --device cpu --retry-contrast
python tools/render_text_corner_audit.py --capture data/benchmarks/<fresh-audit> --baseline data/benchmarks/crop-extent-replay-20261111 --review data/benchmarks/recorded-footage-20261104/manual-review --output data/benchmarks/<fresh-viewer>
```

These commands use the existing cached OCR/text assets and usual read-only
model/recording mounts. CPU uses `SENTINEL_OCR_DEVICE=cpu`. Native rendering
requires Python/Pillow/Git; no inference model is loaded by the renderer.

The schedule contains only reference IDs and timestamps. Capture searches all
automatic vehicle boxes at those timestamps, including tentative boxes, and
excludes already reviewed non-vehicles. It never reads plate answers or manual
plate rectangles. Existing human timestamps select this diagnostic subset;
this is not an independent whole-video benchmark.

Only after capture does the renderer use human references to suggest an
illustrative geometry correspondence. It shows the original vehicle region,
all raw text outlines, expanded corners, classifier status/reasons and native
versus rectified crops. Unretained text outlines stay visible. Manual vehicle/
candidate browsing clears any implied reference identity. New readings are
single-frame diagnostic outputs, not a new reconstructed coverage score.

Raw crops, plate strings, audit HTML and dataset membership remain local in
ignored benchmark folders. Code, the small verifier and aggregate evidence can
be shared without redistributing the footage or raw dataset.

## Completed targeted check

On the 14 existing reference timestamps, capture decoded 901 frames up to the
last requested time and searched 149 automatic vehicle regions. It found 373 raw
text outlines and produced 331 corner candidates: 128 likely-plate regions
reached OCR, 175 uncertain and 28 likely non-plate regions were retained without
OCR. These counts include multiple views and overlapping vehicle regions; they
are not unique vehicles or verified plates. CPU capture took 574.19 seconds.

After capture, illustrative geometric checks found corner candidates overlapping
14 of the 18 references. Three references have multiple alternatives rather than
a unique region association; the viewer shows the highest-overlap crop explicitly
for inspection without asserting identity. The remaining four lack a qualifying
corner candidate: one lacks covering automatic vehicle regions, one lacks an
overlapping raw text outline, and two have overlapping text but unsuccessful
expansion/retention under this diagnostic's geometric thresholds. This is a
stage trace, not a new plate-reading accuracy score.

The local viewer is `data/benchmarks/text-corner-review-final-20261111`; aggregate
evidence is `docs/evidence/text-corners-verifier-20261111.json`. All original
native and rectified PNG hashes were checked. A tolerance of one millionth of a
pixel permits perspective roundoff at a boundary; genuinely out-of-frame corners
are still rejected. The inputs were not rewritten to make rendering pass.

**114 focused tests pass**, covering geometry, independent verification, source/
split/artifact integrity, skipping OCR, same-frame handling and safe rendering.
Browser checks verified eighteen reference choices, fourteen timestamp choices,
corner overlays, loaded images and the explicit uncertain/skip explanation.
The opt-in full multi-frame runner is wired to the new gate and source-reviewed;
this targeted run does not claim a completed new full-video reconstruction test.

## Provisional OCR and registration resemblance

With `--provisional-ocr`, both likely-plate and uncertain regions may be read;
likely-non-plate regions remain blocked. The original visual verdict, classifier
score and model thresholds do not change. OCR eligibility and visual certainty
are separate decisions.

The selected OCR top is checked by `provisional_plate_reading.check_reading`.
A recognised project format or common registration-shaped letter/digit layout
produces a **provisional plate**. Unknown state/district details can still be
shown under `registration_shape_only`; this is not a registry check. Case and
formatting separators may be normalised, but characters are not substituted,
completed, or selected from lower-ranked hypotheses to force a match. Raw text
is always retained. Other layouts remain review-needed, not proof of a non-plate.

These readings cannot confirm a vehicle or independently create a watchlist
alert. Histories/parents containing uncertain-region evidence have their
automatic acceptance cleared. A format-looking string can still be wrong.

The saved-crop replay avoids another detector/video run:

```text
python tools/run_ocr_research.py tools/replay_provisional_readings.py --capture data/benchmarks/text-corner-audit-20261111 --output data/benchmarks/<fresh-provisional-audit> --device cpu
python tools/render_text_corner_audit.py --capture data/benchmarks/<fresh-provisional-audit> --baseline data/benchmarks/crop-extent-replay-20261111 --review data/benchmarks/recorded-footage-20261104/manual-review --output data/benchmarks/<fresh-provisional-viewer>
```

Original images, corners and classifier decisions are frozen. Old OCR outputs
were already format-filtered, so eligible crops are reread using the unfiltered
CTC prefix top, with identical-image outputs cached. This keeps resemblance
checking independent of hypothesis selection. The old ranked text remains in
the audit for comparison. No plate answers, weight tuning, new detection or central events
are involved. A versioned audit policy authorises uncertain OCR; the viewer
recomputes every reading check and rejects tampered displays or operational
eligibility. Old strict captures retain their original validation rules.

Multi-frame provisional mode likewise admits unfiltered text hypotheses before
the final resemblance check. This mode cannot return an automatically accepted
CTC result, and provisional child evidence cannot confirm a parent vehicle.
The original format-aware reader remains the default outside provisional mode.

### Completed frozen-crop replay, 12 November 2026

The replay read 303 eligible saved crops, including all 175 uncertain regions;
28 likely-non-plate regions remained unread. Of those OCR outputs, 156 resembled
a registration, including 68 from uncertain regions; 147 raw readings remain
review-needed. These are repeated candidate-region counts, not verified plates,
unique vehicles or a new recognition-accuracy score. Unknown state/district
layouts account for 60 shape-only results; 96 fit the stricter project patterns.

The first reference's provisional output matches its existing human label. Its
original uncertain visual decision and classifier score are unchanged. Source
corners, classifier decisions, OCR weights and every crop's bytes were verified
unchanged; no plate answers were used by the replay. CPU OCR/decoding took
371.97 seconds after setup. Full-video reconstructed coverage was not rerun.

The local review is `data/benchmarks/provisional-reading-review-20261112` and
aggregate evidence is `docs/evidence/provisional-readings-20261112.json`.
**209 focused tests pass**, including CTC/fusion/reconstruction regression,
parent-confirmation blocking, policy integrity and viewer tests. A flaky legacy
timing assertion now permits zero rounded milliseconds while retaining its
successful-reconstruction, native-pixel and observation-count assertions.
