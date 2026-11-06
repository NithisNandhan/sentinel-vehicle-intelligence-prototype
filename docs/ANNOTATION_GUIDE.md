# Annotation guide

## Readability (definition)

A passage is **readable** when a person can identify **every character** from the best available original frame using up
to **4× zoom**, **without** relying on plate-format assumptions or on any model prediction. If the complete text cannot be
identified, the passage is **unreadable for exact-plate coverage**. Do not call a passage unreadable merely because OCR
fails on it, and do not guess characters to make it readable.

## What to label per vehicle passage

| Field | Notes |
| --- | --- |
| `passage_id`, `vehicle_id` | one passage = one physical vehicle's visible span; the same vehicle never appears in two splits |
| `start_ms`, `end_ms` | video positions (ms) covering when a plate could be read |
| `readable`, `plate` | `plate` null when unreadable; uppercase letters and digits only |
| `layout` | `one_row` or `two_row` (square plates are `two_row`) |
| `conditions` | any of `night, low_light, blur, small, perspective, glare, occluded, rain, motion` — drives the quality slices |
| `boxes[]` (optional) | `{position_ms, vehicle, plate}` boxes in native pixels, for localisation metrics |
| `corners` (optional) | four `[x, y]` plate corners in native pixels for selected samples only |
| `second_check` | `{readable, plate, adjudicated_by?}` from an independent second annotator |
| per recording | `recording_id` (`rec-` + 12 hex of the file hash), `sha256`, `split`, `label_source`, optional `camera_id` |

`label_source`: `blind_human` (annotator saw no model output; **required for sealed sets**), `human_verified_prefill`
(model pre-filled, a person verified every label; development only), `review` (derived from review-page decisions;
development only, never sealed).

## Process

1. Register the recording with its permission receipt (`tools/register_recording.py`); the file hash fixes identity.
2. Development set: model-assisted pre-fill is allowed; the person must verify each label against the frame.
3. Sealed set A: annotators work from the video only. Do not show predictions, crops chosen by the pipeline, or earlier
   review decisions. List unreadable passages too — the denominator matters.
4. A second annotator independently re-judges about **20 %** of passages (readability and text). Record each as
   `second_check`; any disagreement stays open until an adjudicator is recorded in `adjudicated_by`.
5. Run `python tools/validate_annotations.py --labels <file> --manifest <manifest.json>`. Fix every error. The manifest
   pins label and recording hashes for the evaluation record.
6. Do not edit sealed labels after the model has been run on them.

## Expected effort

About 150–250 passages across development, sealed A, unreadable examples and second checks; roughly 15–40 hours depending
on footage. **No human labels beyond the existing AI-drafted pilot exist in the repository** (`data/labels/hyderabad-pilot.json`
and the local Mumbai crops were drafted by an AI assistant and are not ground truth). This work fabricated none and
created no sealed set.

## Common mistakes

- Reading the plate "as it should be" from the state format. Format knowledge is not evidence.
- Treating look-alikes (O/0, I/1, B/8, S/5, Z/2) as obvious. If you cannot tell, it is unreadable.
- Labelling the same vehicle twice because the tracker lost it; that is one passage.
- Putting two recordings of one scene into different splits.
