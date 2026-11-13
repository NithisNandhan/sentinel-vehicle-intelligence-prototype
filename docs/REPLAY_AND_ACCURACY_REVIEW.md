# Normal-speed replay and full-clip human review

Use the current `codex/enable-text-cropping` source. Models and acceptance
thresholds are unchanged. The operator dashboard remains at
http://127.0.0.1:5173/.

## Replay for a dashboard recording

1. Analyse a permission-receipted recording to completion.
2. Choose **Replay at 1×**. Loading retrieves the completed session's immutable
   sampled frames and the readings available at each original video timestamp.
3. Use pause/play and the position slider. **Exit replay** returns to the final
   analysis and enables the Inspector again.

Playback follows source video time, independent of inference speed. It is a
replay of sampled analysis frames, not smooth full-frame video or fresh AI
inference. Later readings are not displayed early. Replay does not restart
models, alter the watchlist or create alerts. Original/overlay mode is selected
before entering replay; inference controls and the Inspector are disabled while
showing historical samples.

Replay is recorded-only and requires the existing recorded-evidence opt-in.
It is held in worker RAM: at most 300 samples and 64 MiB, with no live-camera
frames recorded. Completed results remain available when changing browser tabs;
Stop, a new analysis, seek, worker restart or worker lifetime expiry clears them.
An oversized recording or invalid/backward source clock explicitly disables
replay rather than presenting an incomplete clip as complete. API pages are
operator-protected, no-store and bounded to one sample.

The local capture helper discovers controls and rehearses pause, source-time
playback, scrubbing, historical rows and the absence of inference-control calls
before recording a captioned real dashboard WebM:

```powershell
node tools/record_analysis_replay.cjs --rehearse
node tools/record_analysis_replay.cjs
```

It uses installed Chrome and Playwright. Output defaults to
`output/demos/analysis-replay-20261113/`. Set `REPLAY_DEMO_OUTPUT` to a fresh
directory for another recording. No credentials are embedded in the script.

## Finish the actual accuracy report

Use the passage-aware Mumbai capture at
`data/benchmarks/full-video-passage-review-v2-20261113/`. The older
`full-video-human-review-20261113` capture groups by raw tracker ID and contains
mixed-vehicle records (including `track-s0-3`); do not score those as one vehicle.
Its saved human labels remain untouched. Launch the blind human
reviewer if it is not already running:

```powershell
python tools/serve_video_review.py --run data/benchmarks/full-video-passage-review-v2-20261113 --video data/recordings/footage.mp4 --port 7797
```

Open http://127.0.0.1:7797/. The capture decodes all 928 frames and analyses 62
samples. Review cards represent separate recognition passages, not a verified
inventory of unique vehicles. All decisions initially remain pending and the
full inventory is not attested. The 18 older human reference crops are only partial ground truth;
timestamp/geometry links are review suggestions, and no old transcription was
automatically imported.

The dashboard and reviewer now bind readings/crops to the same passage identity.
Tracker ID reuse, geometry breaks and independent confident OCR disagreements
start a clean history. A proposal covered substantially by two current vehicle
regions is withheld before OCR fusion. A plate aspect-ratio change over twofold
also breaks continuity. OCR disagreement is a veto only: three confident
conflicting characters in equal-length readings, or two conflicting terminal
digits when both readings end in four digits. Agreement never establishes
identity or relaxes acceptance. These heuristics can split one real vehicle into
several records, especially when OCR is confidently wrong. Review duplicates
against the video rather than automatically joining similar registrations.

The Inspector is pinned to the selected passage; if its evidence is unavailable,
it reports that instead of displaying another vehicle with the reused tracker ID.

For each record, independently inspect the vehicle and native plate crops:

- **Readable:** enter the actual plate you can read from the source evidence.
- **Unreadable:** the source itself does not support a reliable transcription.
- **No plate:** a real vehicle without a visible plate in the reviewed evidence.
- **Not a vehicle:** a detector false positive.
- **Duplicate:** link another fragment of the same physical vehicle.

Review the complete source video and add any vehicles the automatic pipeline
missed. Then attest the full-video inventory and finalize. A missing OCR output
is not automatically an unreadable source plate. Predictions must not be
copied into the truth field simply because their model score is high.

Finalization requires complete human decisions. The report separates exact
plate coverage, wrong accepted readings, unknown/unreadable passages, crop
coverage, missed vehicles and false/duplicate tracks. Its selected hypothesis
is frozen before human truth: the strongest-window policy used by the current
dashboard; equal-strength duplicate fragments use earliest timestamp then stable
track ID. Older captures retain their original latest-window scoring policy.

Until this human work is complete, whole-video accuracy is **unavailable**, not
0% and not the model confidence. Results describe this development clip and do
not establish general accuracy on unseen CCTV footage. Source footage/crops and
local recordings are not automatically uploaded to GitHub.
