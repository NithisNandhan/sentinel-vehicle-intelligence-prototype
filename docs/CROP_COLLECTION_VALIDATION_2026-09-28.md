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

Automated checks: 278 edge tests passed (one optional browser test skipped in
the Docker image); nine standalone engineering-page tests passed with Node.

## First live check (before relay-fallback repair)

Both authorised feeds were polled for three minutes. CAM17 yielded 2,969
decoded frames, 187 analysed frames, 13 plate crops, and a maximum of seven
distinct observations on a track; four reconstruction windows were counted.
CAM01 yielded 2,007 decoded frames, 77 analysed frames, nine plate crops, and
at most two distinct observations. **Neither produced a model-accepted plate.**
No live ground-truth transcript exists, so neither OCR accuracy nor correctly
read registrations can be claimed from these counters.

Only 47/88 CAM17 polls and 32/90 CAM01 polls had evidence no more than five
seconds old. CAM01's last view was 111 seconds old. A separate probe found its
local `rtsp://mediamtx:8554/cam01` relay returned **404 Not Found**. The initial
diagnostic reader kept retrying that relay. Follow-up comparison found the
long-running relay container had older credential values than the current
local `.env` and diagnostic viewer; no value was printed. Recreating the relay
and edge-worker containers from `.env` started two FFmpeg publishers, and
FFprobe confirmed both local camera paths now expose 1920×1080 H.264. This
identifies an environment/configuration failure, not an OCR-model failure.
The viewer also marks stale evidence unavailable and bounds relay retries
before trying authorised direct RTSP/HLS. A third live run checks the repaired
relays separately; the connectivity repair alone does not prove plate accuracy.

The live reports contain only aggregate counters and UTC timings. No camera
pixels, source URLs, plaintext registrations or credentials are retained.

## Final live acceptance status

A second 90-second check, before refreshing the relay containers, still yielded
zero accepted identifiers. After credential refresh, a 120-second check per
camera **failed** the two-feed acceptance gate: CAM17 analysed seven frames
with no plate crop; CAM01 analysed zero frames. Both ended unavailable. A
separate authorised transport probe had decoded a single 1920×1080 H.264
CAM01 frame through RTSP and HLS, and both local relay paths briefly exposed
the same codec/resolution. Those probes establish momentary connectivity,
**not sustained feed delivery**. Subsequently the portal catalogue reported a
sign-in failure and the catalogue gateway returned HTTP 503. The viewer
recorded a safe `TimeoutError` category without exposing the authenticated
address. Continued live testing therefore needs restored Sentinel access.

The benchmark development result and the offline interface remain available;
they are not substitutes for a verified live plate read. No live registration
was model-accepted or independently verified in these runs. Accuracy and
end-to-end live readiness remain **unproven**. Do not use a candidate or an
uncalibrated model score as a confirmed suspicious-vehicle match.

## Follow-up diagnostic run

The [per-frame evidence viewer](http://127.0.0.1:7777/crop-collection-frame-evidence-20260928.html)
was regenerated from the same frozen three-frame development sequence. It
adds each sampled native crop and its OCR reading to the visual comparison;
it is **not** a new dataset or an independent accuracy test. The four methods
again accepted the labelled identifier in 13/21 degradation conditions each,
with zero wrong accepted identifiers on this one sequence. Reconstruction did
not improve that count.

The diagnostic transport probe was extended to use only the explicitly
approved configured camera pair when the catalogue is unavailable, and it
labels that source separately. For both CAM01 and CAM17, the catalogue was
unavailable, direct RTSP returned an HTTP authentication failure, and HLS
portal sign-in failed. The probe prints failure classes, not credentials or
authenticated URLs. These results explain why another sustained live OCR run
would be invalid now. They do not establish that the media server is down or
that the recognition model cannot read a clear live plate; authorised access
must be restored before either claim can be tested.
