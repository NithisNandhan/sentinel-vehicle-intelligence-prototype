# Persistent vehicles and their plate views

## Verified outcome

The final corrected run is available locally at **http://127.0.0.1:7801/**.

| Same eighteen development targets | Previous plate-motion path | Final vehicle-lifecycle path |
| --- | --- | --- |
| Highest-quality single native crop | 6/18 | 7/18 |
| Multi-frame OCR | 8/18 | **10/18** |
| Multi-frame plus checked reconstruction | 8/18 | **10/18** |

The final multi-frame path gained references 02 and 11 and lost no previously
correct target. Eleven targets had a unique matched crop; ten were exact. Five
matched references passed the unchanged child-text gate, all five correct.
This selected development evaluation does not establish whole-video accuracy.

Stricter recognition tied to a confirmed vehicle was **6/18**: seven targets had
a vehicle-associated hypothesis, six were exact, and four passed the unchanged
acceptance gate, all four correct. Correct OCR and correct vehicle attribution
are separate requirements. Unassigned vehicles are not given guessed owners.

The run decoded all 928 frames, sampled 124 plate-search timestamps, made 583
original-input OCR calls and took 434.46 seconds after model loading. The prior
path took 317.17 seconds. Neither is real-time end-to-end processing. There were
185 vehicle histories and 336 plate histories; these are not unique physical
vehicle counts. The optional proposal time budget means exact optional calls
can vary between executions.

An initial integration run exposed car/truck label flicker causing **120**
parent-ID transitions for existing native IDs. The final class-vote fix reduced
those transitions to **zero** on the same frozen source tracks. The initial
`vehicle-lifecycle-checked-20261110/` output is explicitly superseded; its higher
11/18 text score does not justify promoting a fragmented vehicle-identity path.

The final lifecycle recorded 39 geometrically supported boundary exits and 119
unresolved timeouts. Those are inferred closing reasons, not manually verified
exit counts. Remaining active histories are retained at end of the recording.
There were 327 plate-history retention checks supported by an existing vehicle
parent; these are repeated decisions, not 327 distinct occlusions. No central
sightings or alerts were emitted by either benchmark.

Of the reviewed cases, previous PT0036/PT0041 now map to one plate history under
one confirmed parent. Previous PT0006/PT0007 also map to one history, while the
earlier PT0004 remains unassigned. PT0049/PT0050 collapse to one side view, but
its relationship to the front view PT0047 remains unconfirmed. PT0015/PT0016 and
PT0068/PT0069 remain partly unresolved. PT0023 is not forced into the silver
car's history. The viewer includes this mapping explicitly; the change does
not claim all reported duplicates are fixed.

**83 focused tests pass**, including occlusion/recovery, stationary tracks,
boundary versus unresolved loss, scene reset, bounded state, class flicker,
cross-parent evidence isolation, reviewed-region vetoes and same-frame vote
suppression. The normal GPU worker was restored after benchmarking. Source
fingerprints and final aggregate evidence are published in
`docs/evidence/vehicle-lifecycle-20261110.json`; raw crops, labels and plate texts
remain local.

This change implements the reviewed vehicle-lifecycle approach in the local
engineering pipeline. Model weights, OCR acceptance thresholds and the existing
928-frame source are unchanged. The operator event-delivery worker is not
switched to an unvalidated experiment.

## Behaviour

Each persistent VT vehicle can have several PT plate histories. Temporary
detector misses preserve the VT identity. Up to one second of velocity-predicted
position is shown as a dashed box; after that, an unresolved identity is retained
without a drawn position for up to five seconds. Source timestamps drive both
limits. Neither prediction nor an unresolved record supplies OCR evidence or
owner confirmation. Stationary vehicles can remain active while observed.

An unresolved record can reconnect to a new detector ID with compatible geometry
and strong appearance, or checked optical motion with appearance support.
Competing associations abstain. The same raw ID must still pass geometry checks.
Car/truck label flicker keeps its identity and a rolling ten-observation class
vote. New-ID recovery retains a compatible-class gate. New-parent recovery
uses no plate text. A compact tracking
descriptor is separate from the original-resolution plate crop.

An exit requires the last observed box to be near an image boundary, outward
velocity and the entire predicted box to cross that boundary. Other missing
vehicles expire with `unresolved_timeout`, never a fabricated boundary exit.
Camera cuts explicitly detected by the caller close identities with
`scene_reset`; tracking state and descriptors reset. Automatic scene-cut
detection is conservative and is not guaranteed to detect every cut.

State is bounded: 128 active vehicle parents, two recent observations and
signatures, ten class votes, 32 raw aliases per parent, bounded closed summaries,
and at most 128 optical-flow states. Gross geometry changes still fork a native
ID instead of incorrectly combining vehicles. Vehicle identity remains a heuristic.

Plate searching includes visible parents and short-lived predicted regions.
Only a currently observed, eligible parent can confirm ownership. Consistent
plate-versus-parent motion can resolve overlapping expanded vehicle boxes;
reviewed mixed/unsafe regions remain hard ownership vetoes. Once a plate history
has a confirmed parent, a different visible parent forces a separate history
before old OCR evidence can be reused.

Nested overlapping plate fragments are suppressed before OCR. Separate front
and side views survive as separate PT histories under one confirmed vehicle.
Combined vehicle recognition uses only original observations with confirmed
ownership. Several surfaces from one timestamp contribute at most one view,
and at most six distinct timestamps are retained for combination. Strong
conflicting plate readings remain explicitly flagged and block acceptance.
The existing CTC/reconstruction rules and original-image evidence limits remain.

## Evaluation and known risks

The previous motion-supported result is frozen in
`data/benchmarks/plate-flow-checked-20261110/` (8/18 selected reference targets).
The final output is `data/benchmarks/vehicle-lifecycle-stable-20261110/`.
The eighteen development labels are read only after inference. Child-plate
recognition and stricter confirmed-parent recognition are scored separately;
missing parents and contradictory target identities remain misses. These
numbers are not exhaustive vehicle counts or unseen-camera accuracy.

Human-reported groups are diagnostic checks after inference. They do not become
hard-coded associations. In particular, PT0004/PT0006/PT0007 in the previous
viewer belong to a silver car; PT0023 belongs to a different autorickshaw.
The front/side markings in prior PT0047/PT0049/PT0050 were verified as one
autorickshaw in the native 7-second frame. New run IDs cannot be compared by
number alone; source time and image geometry must bind the comparison.

Remaining risks include overlapping vehicle boxes, false detector classes,
long occlusion, similar adjacent vehicles, tiny plates with insufficient motion
features, scene cuts that evade the conservative detector and inaccurate OCR
despite a complete crop. Persistent tracking cannot recover missing image detail.
The original run and operator delivery remain available for rollback.

## Reproduction and review

Use the cached recorded GPU Compose service with the isolated
`tools/run_ocr_research.py` launcher, working directory `/workspace`, and the
project data mounted at `/workspace/data`. Run:

```text
tools/run_owned_multiframe.py --phase run --plate-first --plate-flow \
  --vehicle-lifecycle --output /workspace/data/benchmarks/<fresh-run>
```

Then run `--phase score --output <that-folder>` with the local human-review
directory. Run folders are immutable captures; source files, input review and
their hashes are saved with each inference run. No new models are downloaded.

The viewer adds a complete timeline replay with observed/predicted state labels
and a vehicle table linking each parent to its child plate views. Its source
video comes from the existing localhost tracking server on port 7798. Raw
footage, native crops, plate texts and human labels stay local. Code, tests,
methodology and aggregate evidence may be published.
