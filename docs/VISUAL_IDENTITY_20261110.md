# Visual identity, tracker comparison and vehicle-first review

For the subsequent opt-in crop validation that quarantines the reported window
histories without deleting evidence, see
[Plate evidence validation](PLATE_EVIDENCE_VALIDATION_20261110.md).

The change keeps a visually established plate identity through conflicting OCR
readings, while flagging the recognition as uncertain. It also adds bounded
plate-search bursts and groups confirmed plate evidence under vehicle cards.
The existing `vehicle-lifecycle-stable-20261110/` run is preserved as the control.
This remains an engineering experiment; operator event delivery is not switched
to unvalidated associations.

## Recommended engineering profile: parent-anchored association

Retain ByteTrack, the existing model weights and the original 250 ms sampling
cadence. Enable `--visual-identity` with the parent-position correction; leave
`--adaptive-plates` off. The new review is on **http://127.0.0.1:7804/**; the prior
stable control remains on port 7801 and the unsuccessful burst trial on port 7803.

The parent-position correction allows a changing plate crop to continue when
it has the same unique observed vehicle candidate and stays within 0.20 of its
previous normalized position inside that vehicle. Standard displacement/scale,
competing-assignment, parent-conflict and reviewed-region vetoes remain active.
The first matching parent candidate alone does not create confirmed ownership;
the subsequent independent observation must confirm it. Spatially distant
front/side surfaces remain separate views under their parent rather than being
concatenated into a fictitious plate image.

The narrow saved-input truck probe used five previously captured crops and
their original OCR posteriors. The control split them into two histories; the
parent-position correction kept one. No new OCR inference, plate transcription
or training was used by that diagnostic. The full original source then verified
the correction independently of the selected-crop replay.

| Same eighteen development targets / same cadence | Stable control | Parent-anchored run |
| --- | --- | --- |
| Single native frame | 7/18 | 7/18 |
| Multi-frame OCR | 10/18 | 10/18 |
| Checked reconstruction plus multi-frame | 10/18 | 10/18 |
| Recognition tied to a confirmed vehicle | 6/18 | 6/18 |
| Fully resolved reported example clusters | 1/5 | 2/5 |
| Plate evidence histories, not unique vehicles | 336 | 304 |

No exact reference reading was gained or lost. The original 68/69 truck views now
map to one history under VT0013, and the previously resolved 36/41 case remains
resolved. Original 6/7 remain collapsed, but 4 remains unassigned; 15/16 and the
front-versus-side 47/49/50 relationship remain partly unresolved. No new known
conflicting-reference history was detected. The car-versus-autorickshaw negative
check was not forcibly merged, although ownership remains incomplete there.

The run decoded all 928 frames and sampled the original 124 timestamps, using 582
original-input OCR calls versus 583 for the control. Runtime after model loading
was 455.68 seconds versus 434.46. Vehicle histories remained 185, with zero parent
switches for continuous native identities. Five matched child-text acceptances
and four matched parent acceptances agreed with human labels; unlabelled
acceptances remain unverified. These are selected development measurements,
not whole-video accuracy, unique-vehicle counts or real-time performance.

**104 focused regression tests pass.** The viewer groups assigned views into
expandable vehicle cards, preserves all original evidence cards and makes
unassigned histories explicit. Aggregate evidence is in
`docs/evidence/parent-anchored-20261110.json` and the isolated probe in
`docs/evidence/parent-anchor-probe-20261110.json`. The normal GPU worker is
restored. No experimental sightings or alerts were delivered centrally.

## Initial comparison and sampling trial

**The first combined trial was not promoted. The vehicle-first viewer remains useful;
new tracker and adaptive-search settings remain opt-in experiments.** None of the
tracker candidates improved the selected identity checks, and the combined
visual-identity/adaptive-search trial did not resolve an additional reported
vehicle group.

| Same eighteen human-labelled development targets | Stable control | Experimental flags |
| --- | --- | --- |
| Best-quality single native crop | 7/18 | 8/18 |
| Multi-frame OCR | 10/18 | 10/18 |
| Checked reconstruction plus multi-frame OCR | 10/18 | 10/18 |
| Recognition tied to a confirmed vehicle | 6/18 | 5/18 |

No multi-frame exact target was gained or lost. The extra searches collected
863 original-input OCR calls instead of 583, and processing took 571.99 seconds
after model loading instead of 434.46. They did not improve the multi-frame
score. More observations are not automatically better evidence. No retained
history in this source exercised the new OCR-contradiction override; its
behaviour is verified by synthetic regression tests, not field improvement.

The experimental run decoded all 928 frames, scheduled 713 plate-search frames
(589 above the 124 base samples), and produced 506 plate histories under 185
vehicle histories. These are evidence/track counts, not unique vehicles.
The five reported groups are tied to the original plate-flow prediction hash;
IDs from a newer baseline must never silently substitute for the user's IDs.
Previous 36/41 remain resolved, 6/7 remain collapsed, and 49/50 remain collapsed,
but the other ownership gaps remain. Zero native-ID parent switches alone is
not proof that every physical vehicle has one identity.

The observed bottleneck is still reliable crop-to-vehicle ownership in an
overlapping scene, including short-lived front/side views. A cleaner UI does not
resolve those associations. The next useful test would evaluate instance-level
ownership with labelled vehicle trajectories; this change does not claim that
test or a duplicate-free end product is complete.

The stable control viewer is on port 7801. The rejected adaptive-search run
is separately reviewable on port 7803. Aggregate evidence is in
`docs/evidence/vehicle-association-20261110.json` and
`docs/evidence/visual-identity-20261110.json`. The raw video, crops, per-frame
posteriors and assistant review rectangles remain local. No experimental event
was delivered centrally.

## Tracker comparison

All strategies replay the same frozen YOLO26s detections and all 928 aligned
native frames. The detector is not retrained or rerun. ByteTrack and BoT-SORT use
the same new-track threshold 0.4; both BoT-SORT variants disable global camera
motion compensation for this fixed camera. Other library defaults are recorded.

| Strategy | Native IDs | Passage histories | Qualified passages | Median tracker work |
| --- | --- | --- | --- | --- |
| Screened ByteTrack | 260 | 269 | 186 | 1.132 ms |
| BoT-SORT motion | 239 | 250 | 172 | 1.117 ms |
| BoT-SORT, exploratory person encoder | 304 | 312 | 183 | 93.822 ms |
| BoT-SORT, vehicle-trained OmniScaleNet | 309 | 316 | 176 | 213.203 ms |

These counts are not unique vehicles or tracking accuracy. Eight approximate
source-frame rectangles across four selected physical-vehicle cases were
reviewed after inference by the assistant. They are not human-verified MOT
ground truth. All four methods matched all eight anchors, preserved two of the
three repeatedly observed cases and split the silver-car case. The fourth
vehicle has only one anchor and is a negative check, not a continuity test.
No selected cross-vehicle ID merge was observed.
No replacement demonstrated better identity behaviour on these checks, so the
existing ByteTrack path remains the integration control. Timings describe this
laptop run, including crop/feature work, and do not establish deployment capacity.

The first off-the-shelf Ultralytics encoder's metadata identified MSMT17 person
training; its installed dynamic-input default 224 was recorded. It is retained
as an exploratory diagnostic, not evidence for vehicle-specific model quality.
The subsequent 8.84 MB OmniScaleNet vehicle model was verified against the
publisher's size/SHA384 manifest and a pinned SHA256. We derive its original
RGB float32 208x208 input range 0..255 from the publisher's unscaled demo input
and channel-reversal-only conversion arguments. The adapter follows that
contract. CUDA execution and 512-dimensional finite embeddings were
verified. Each learned comparison encoded 13,938 detection crops.

Sources:

- [Ultralytics tracker interfaces and ReID support](https://docs.ultralytics.com/modes/track/)
- [Vehicle encoder description, VeRi-776 evaluation and original RGB input](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/README.md)
- [Publisher download manifest and conversion arguments](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/model.yml)
- [Publisher demo input preparation](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/multi_camera_multi_target_tracking_demo/python/utils/ie_tools.py)

## Identity and recognition separation

`--visual-identity` changes plate association only after independent visual
support: strong checked optical flow plus compatible crop appearance, or a
previously confirmed matching vehicle parent with visual continuity. The parent-position
correction additionally supports changing crop appearance under the same unique
observed parent when relative position is stable; it cannot override the strong
OCR-conflict veto without the independent support above. Geometry,
one-to-one association, competing-parent checks and reviewed-region vetoes
remain active. OCR agreement alone is disabled as a positive association cue.
Insufficiently supported contradictions can still split an uncertain history;
this is not an unrestricted merge of nearby yellow plates.

A retained history records conflicting OCR explicitly. Original crops and all
candidate readings remain available, but conflicting text cannot pass the child
acceptance gate. Parent-level conflicting-view checks remain unchanged. The
YOLO, plate-detector and OCR weights and CTC thresholds stay fixed.

## Extra observations and review UI

`--adaptive-plates` retains the 250 ms base cadence and schedules at most three
extra ROI searches for a new or reappearing visible parent. Extra searches are
100 ms apart, expire after 500 ms, and process at most two parents per extra
frame. A one-second per-parent cooldown prevents rapid detector flicker from
continually restarting the burst. Existing source-image deduplication and
150 ms original-evidence spacing still limit OCR confirmations.

The viewer defaults to one expandable card per confirmed vehicle parent.
Its plate histories and native crops remain within that card. Unassigned plate
histories appear in a counted, separate review queue. No ownership is invented
to make the UI look cleaner, and no original evidence card is dropped. Fragment
links automatically expand their containing cards. The complete observed,
predicted and unresolved timeline remains available.

## Reproduction

Restore exact public encoders with `tools/download_tracking_models.py --model
vehicle` or `--model ultralytics`. Both write receipts and verify pinned sizes
and SHA256 hashes. Run `benchmark_vehicle_association.py` with the existing
frozen capture, explicit encoder and a fresh output. The vehicle-specific pass
uses `--encoder-kind vehicle --modes botsort_reid`; no silent CPU or native-feature
fallback is accepted. Generated timelines remain available for review.

Run the recommended unchanged-cadence profile in the cached GPU OCR environment:
`tools/run_owned_multiframe.py --phase run --plate-first --plate-flow
--vehicle-lifecycle --visual-identity --output <fresh-folder>`, followed by its
separate scoring phase. The full verified capture is
`data/benchmarks/parent-anchored-20261110/`. To recreate its vehicle-first HTML
without model inference, use `--phase render --output <capture>
--review-summary <matching-aggregate.json>`; hash checks bind both the score and
review table to the capture.

The unsuccessful extra-search trial used
`tools/run_owned_multiframe.py --phase run --plate-first --plate-flow
--vehicle-lifecycle --visual-identity --adaptive-plates --output <fresh-folder>`,
then run its separate scoring phase with the existing human labels. The new
capture is `data/benchmarks/visual-identity-20261110/`. Labels and review identity
checks are not loaded to choose or tune detections. Published evidence contains
aggregate counts and hashes; footage, crops, plate texts and manual labels stay
local. No central alerts are sent by these experiments.

When comparing later captures, pass the original
`plate-flow-checked-20261110/` as `--review-baseline` to
`summarize_vehicle_lifecycle.py`. The transcript's reported PT IDs belong only
to that capture; recognition baseline selection is independent of this ID binding.
