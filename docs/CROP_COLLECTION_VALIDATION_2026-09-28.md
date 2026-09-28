# Crop collection and recognition validation (28 September 2026)

This is a **development diagnostic**, not a measured deployment accuracy claim. The
NIVU CC-BY-4.0 fixture contains three frames from **one previously used Indian
vehicle sequence**. Its timestamps are assigned for replay; the plate/vehicle
boxes are approximate visually reviewed annotations, not an independent new
human-labelled test set. Real Sentinel pixels and plaintext plates were never
written to these reports.

## Changes tested

- The engineering viewer now separates continuous decode, bounded
  timestamp-sampled vehicle/plate crop collection, and OCR/fusion. Queue budgets
  are eight frames, 48 MiB and 2.5 seconds. Old packets are dropped; a scene
  cut clears old evidence. The displayed frame age includes queue waiting.
- Plate detection first uses the normal full vehicle ROI. Only when that yields
  no usable crop, it searches four overlapping **native-pixel** tiles (five
  detector calls maximum including the original). It never enlarges source
  pixels before storing the crop. The tile choice is a geometric heuristic, not
  model confidence.
- The RAM-only reconstruction bank retains a sharp older crop while keeping
  recent crops, under an age and memory limit. Reconstructed views suggest
  candidates but cannot vote as extra original frames.
- The protected engineering console now reports collected crops, dropped
  packets, memory peak, reconstruction, per-frame OCR and ranked alternatives.
  This diagnostic path does not replace the production edge worker.

## Paired result and boundary

The [local report](http://127.0.0.1:7777/crop-collection-20260928.html)
compares four methods on the **same 21 degraded versions of that one sequence**.
Each method accepted the labelled identifier in 13 conditions, withheld a
decision in 8, and accepted no wrong identifier. This is **13 conditions**, not
13 different vehicles. The new fallback plus checked reconstruction did not
increase the accepted-condition count on this sequence. The fallback was not
invoked when the primary detector offered a geometrically usable but incorrect
plate region, so an apparent crop is still not proof of correct localisation.

At 50% resolution, the correct identifier was accepted with an automatic plate
crop about 59 native pixels wide. At 37.5%, it ranked first but failed the
existing evidence gate even with a manually annotated crop about 41 pixels
wide. At 25%, the identifier left the candidate beam even with the manual crop.
Moderate synthetic motion blur showed the same pattern: at kernel 5 the target
ranked first but was withheld; at kernel 9 it was absent. With strong noise,
automatic localisation missed two of three frames while manual crops still
ranked the target first but failed acceptance. These are **one-sequence
boundaries**, not camera-independent thresholds.

The earlier [four-profile baseline](http://127.0.0.1:7777/crop-first-baseline-20260928.html)
accepted the same undegraded target with legacy, general, and plate-adapted
profiles. The India-adapted vehicle profile found only one of three frames and
abstained. This is evidence not to promote that vehicle profile on the current
fixture, not proof that it is generally inferior.

## Model decision

No additional weight tuning was run from this one reused sequence. The paired
manual-crop result isolates a genuine low-information OCR boundary, while the
strong-noise case isolates some localisation loss. Fitting either detector or
reader on this same sequence would invalidate it as a test. Next model training
requires a larger, independently labelled Indian multi-frame set with separate
vehicle/scene groups, clear footage permission, and blind post-training scoring.
False accepted identifiers must be counted separately from abstentions.

Automated checks: 275 edge tests passed (one optional browser test skipped in
the Docker image); nine standalone engineering-page tests passed with Node.
The current live camera check is reported separately after completion. A
browser rendering or local-server HTTP response is not proof of live plate
recognition.
