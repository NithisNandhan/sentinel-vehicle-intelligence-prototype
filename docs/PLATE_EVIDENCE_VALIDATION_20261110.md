# Crop-level plate validation

For the subsequent crop-boundary experiments and side-by-side images, see
[Native crop recovery](CROP_RECOVERY_20261111.md).

The engineering review now separates suspected non-plate evidence from normal
vehicle plate cards. It preserves all original crops, IDs and raw hypotheses.
The correction is opt-in and does not change the operator worker or send events.

## Problem and approach

The reported VT0117 vehicle contained two window histories and a real plate
history. The window crops had high image-quality scores and could displace real
plate samples when the parent reader chose its six inputs. Image sharpness and
consistent vehicle ownership do not establish that a crop is a number plate.

An initial check that relied on missing OCR alone also flagged blurry real
plates. It was rejected. The implemented rule requires all of these cues:

- No confident ASCII alphanumeric CTC character in any retained original, using
  the existing 0.6 character-peak threshold. Any such character protects the
  history from this rejection rule; that does not verify the text.
- Fewer than three character-sized image components in most retained originals,
  checked in both image polarities. This is an image heuristic, not a recogniser.
- At least two independent, confirmed-owner observations, with the crop fully
  inside the upper half of an actually observed vehicle in most originals.
  Predicted boxes, unlocked ownership and missing context cannot supply this cue.
- Independent originals must have distinct source hashes and at least 150 ms
  separation. Resizing, reconstruction and duplicated frames add no votes.

The resulting label is **suspected non-plate**, never a definitive declaration.
Suspected histories are excluded from parent OCR fusion and placed in an
expandable inspection queue. Their image evidence, ownership metadata and raw
single/multi-frame/reconstruction candidates remain available. The acceptance
gates are not relaxed. Other unreadable crops remain explicitly uncertain.

## Verification

The comparison uses the immutable `parent-anchored-20261110` capture: 928 source
frames, 124 search timestamps, 304 histories and 493 retained native crops.
It reuses saved CTC posteriors and native PNGs; it does not decode the video or
rerun detection, OCR or reconstruction. Before applying the filter, the tool
requires its parent-fusion replay to equal the frozen baseline exactly.
Every loaded native crop is checked against its source-pixel hash. Source-code
hashes must remain unchanged during the run. Ground-truth labels are loaded only
by the separate scorer after validation.

The source IDs PT0173 and PT0182 are genuine plate crops and their existing
method-specific readings are preserved. PT0236 is a genuine two-line plate with
an incorrect OCR hypothesis, not a false plate localisation. It remains visible
and unconfirmed. This filter does not repair its line layout or missing digits.

The final capture is `data/benchmarks/plate-validated-final-20261110/`.
Aggregate measurements are published separately in
`docs/evidence/plate-evidence-validation-20261110.json`; crops, per-frame OCR
posteriors, actual plate strings and human review annotations stay local.

## Measured result

The final rule quarantines **PT0192 and PT0258**, the two reported window
histories, while preserving PT0288 as VT0117's real plate evidence. That vehicle
now combines only its real plate inputs and passes the unchanged text gate.
This is an additional model-gated reading, not an independently measured new
accuracy target. No other parent reading changed.

PT0173, PT0182, PT0236 and PT0288 retain exactly their prior single-frame,
multi-frame and reconstruction outputs. The same eighteen human-labelled
development targets remain 7/18 for single-frame and 10/18 for each multi-frame
method; recognition linked to confirmed ownership remains 6/18. No correct
reference reading was lost. Quarantined real targets would count as misses,
even when their preserved raw hypothesis happens to be correct.

All 304 evidence histories and 493 native images remain available. There are
302 active histories: 153 have at least one confident character and 149 remain
uncertain. These are not unique-vehicle counts or verified plate detections.
The CPU saved-evidence check, including two parent-fusion replays, took 87.305
seconds. No model weights, detection thresholds or OCR thresholds changed.

**122 focused tests passed.** Coverage includes predicted/unlocked-parent
exclusion, independent-source counting, raw-hypothesis preservation, parent
fusion isolation, scoring rejected true plates as misses, and viewer evidence
preservation. Browser checks verified all 493 images load, the 4K source plays,
the VT0117 card contains PT0288 only, and the two window histories appear in the
closed inspection queue. The local viewer is **http://127.0.0.1:7805/#VT0117**;
the unfiltered control remains on port 7804.

## Simple numbered results page

The landing page on port 7805 now uses a dropdown numbered 1 through 18 for the
eighteen evaluated human labels, in their original review order. Each selection
shows the labelled reference crop, an example automatic crop when available,
the reconstructed OCR reading and the actual human reading. Original reference
IDs remain in the local data; the displayed numbering has no gaps.

At the user's request, this view counts a label when its complete text appears
as an active reconstructed **top output anywhere in the clip**, irrespective of
timestamp, crop geometry or the confirmation threshold. Count each labelled
example once; extra incorrect guesses do not reduce this found-at-least-once
score. The measured result is **12/18 (66.7%)**. This is recognition coverage on
these selected labels, not prediction precision. Single-frame outputs and lower
ranked alternatives cannot count towards it.

When no exact result exists, the page shows the previously associated incorrect
reconstructed reading if one is available. Otherwise it says no exact reading
was found and displays the human reference crop; it does not invent a vehicle
association to fill the empty automatic-crop panel. The original detailed page
is preserved as `engineering.html`. The original `accuracy.json` is unchanged.

Regenerate the local landing page with:

```text
python tools/render_plate_results.py --capture data/benchmarks/plate-validated-final-20261110 --review data/benchmarks/recorded-footage-20261104/manual-review
```

The renderer verifies prediction/label bindings and reference-image hashes.
Generated HTML, crops and `simple-results.json` contain local review data and
remain outside Git. Four focused tests cover the new counting rule and safe
HTML rendering. Browser verification exercised all eighteen choices, loaded
their available images and checked desktop and narrow layouts for overflow.

## Reproduce

In the existing CPU-capable research image, with edge/tools on `PYTHONPATH` and
the project data mounted at `/workspace/data`, run:

```text
python tools/validate_plate_evidence.py --capture data/benchmarks/parent-anchored-20261110 --output data/benchmarks/<fresh-output>
python tools/run_owned_multiframe.py --phase score --output data/benchmarks/<fresh-output> --review data/benchmarks/recorded-footage-20261104/manual-review
python tools/summarize_plate_evidence.py --baseline data/benchmarks/parent-anchored-20261110 --current data/benchmarks/<fresh-output> --output docs/evidence/<fresh-report>.json
```

New model runs can request the same rule using `--validate-plates` together with
`--plate-first --plate-flow --vehicle-lifecycle --visual-identity`. The existing
unfiltered profile remains available for comparison. No model download is needed.

## Limits and next work

This is a conservative fix for the observed upper-body/window failure, evaluated
on the development clip. It cannot remove every non-plate. Reflections with
character-like texture, branding, lower-body objects and confidently hallucinated
OCR may survive. An unreadable real plate unusually high on a vehicle could be
quarantined; original evidence remains accessible for that reason.

PT0173/PT0182 method selection and PT0236's two-line recognition require separate
recognition work. The next detection improvement should collect human-reviewed
plate/non-plate examples across other clips, especially windows, lights, grilles
and branding, then evaluate detector hard-negative training or a dedicated
verifier on vehicles not used for training. No whole-video accuracy or detector
precision claim follows from the selected development reference scores.
