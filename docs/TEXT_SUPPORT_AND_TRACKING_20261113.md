# Withheld proposals and tracking comparison — 13 November 2026

## Selected behaviour

Keep sampled vehicle tracking and OCR, with the new text-support safeguard.
Continuous vehicle tracking is implemented as a recorded-only trial and remains
disabled by default because the controlled comparison lost accepted readings.
No model weights or automatic recognition thresholds were changed.

A plate proposal with no greedy CTC character reaching a 0.6 run-peak score is
now withheld before it can move the plate clock, reset its location or add a
fusion/reconstruction vote. The peak is taken over the entire character run,
including later CTC timesteps. This is **insufficient text evidence**, not proof
that the region is not a real plate.

The Inspector shows a separate **Withheld proposals** section with native crop,
raw OCR, character scores and reason. These images never appear as admitted
observations. Up to three previews per passage are retained across at most 32
passages in worker RAM; an individual preview over 80,000 base64 characters is
omitted explicitly. The protected API validates the metadata and always marks
`used_for_fusion=false`.

The local blind reviewer preserves native withheld crops in a separate section;
they are not additional vehicle/accuracy units. Raw OCR remains in the frozen
predictions file, hidden during labelling. Capture retention is bounded to
512 proposals / 32 MiB, with explicit omission/truncation counts.

## Recorded GPU comparison

Both runs used the same permitted 4K Mumbai source, identical model hashes,
identical source code and recognition safeguards. Only the continuous-tracking
setting differed. Each decoded 928 frames and analysed 62 plate/OCR samples.

| Measurement | Selected sampled tracking | Continuous tracking trial |
| --- | ---: | ---: |
| Vehicle tracker updates | 62 | 928 |
| Admitted plate observations | 103 | 127 |
| Preserved withheld proposals | 41 | 67 |
| Review fragments | 144 | 291 |
| Unique model-accepted readings, unverified | 3 | 1 |
| Previously accepted controls lost | 0 | 2 |
| Observed analysis-stage wall time | 127.18 s | 220.14 s |

These are measurements from individual development runs, not throughput or
general accuracy claims. The timed section starts after loading the primary
vehicle/plate/OCR models and can include optional text-detector initialization.
More native tracker updates produced more fragments, but that alone does not
explain the lost acceptances: both controls remained the leading OCR guesses
with several observations. Their fused scores weakened, and one history included
an early clipped/occluded plate view. This points to evidence selection as a
remaining problem; lowering acceptance thresholds would not fix it. The trial
added no new accepted registration. All twelve retained multi-crop sheets in the
trial showed no obvious vehicle mixing on visual inspection; this is not a
labelled identity-accuracy measurement.

The selected capture is
`data/benchmarks/full-video-text-supported-sparse-v2-20261113/`. The rejected
continuous candidate is
`data/benchmarks/full-video-text-supported-continuous-20261113/`. Both remain
local and immutable. All withheld crops were retained without truncation.

All eight previously inspected windscreen/trim negatives matched exact pixel
hashes and timestamps in the selected run's withheld section, with none admitted
to fused track evidence. The known tracker-148 weak/trim observations no longer
contribute to a combined plate reading. This does not prove its physical vehicle
identity is correct. The earlier tracker-3 passage safeguards remain in force.

## Positive controls and limitations

The diagnostic used 173 saved images, including 21 unique human reference crops
and 17 saved crops from the three accepted control passages. The text-support
rule withheld all eight inspected negatives and none of the 17 accepted control
crops. It also withheld two human reference crops: readable reference 09 and
source-unreadable reference 08. Their pixels/raw OCR remain inspectable.

The older optional shape/confidence filter would reject three unique reference
crops, including a valid crop just below its aspect-ratio floor. The existing
small visual verifier missed three inspected negatives. Neither was enabled as
a blanket plate/non-plate decision.

The three model-accepted readings are not three independently verified truths.
Full-video accuracy remains unavailable until the human inventory, duplicate
decisions and transcriptions are completed. Similar-looking vehicles can still
exchange tracker IDs, and confident OCR mistakes can still split a real passage.
Preserving weak proposals avoids hiding failures; it does not make those proposals
readable. No genuine live-camera footage was recorded for these tests.

## Run and review

```powershell
# Selected GPU configuration; source models must already be restored.
$env:SENTINEL_CONTINUOUS_TRACKING='false'
$env:DASHBOARD_RECORDED_EVIDENCE='true'
./scripts/start-recorded-analysis.ps1 -Device gpu -DetectorProfile general_quality -Unpaced -SkipBuild

# Corrected local blind review; preserves earlier runs and labels.
python tools/serve_video_review.py --run data/benchmarks/full-video-text-supported-sparse-v2-20261113 --video data/recordings/footage.mp4 --port 7798
```

The dashboard is at http://127.0.0.1:5173/ and the new reviewer at
http://127.0.0.1:7798/. The previous reviewers can remain open for comparison.
For deliberate research only, `-ContinuousTracking` enables the rejected trial;
`-ContinuousTracking:$false` disables it again. It is never enabled for live feeds
by this implementation. CPU mode remains supported with the default sampled path.
