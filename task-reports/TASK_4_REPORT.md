# Task 4 report: recorded mode in the shared session and delivery

Closed for plumbing 10 October 2026, amended the same day. **Score: 8 / 10** (raised from 7 after the cross-process audit below)

## What was there before

- The shared session (`LiveState` and `serve` in `live_detection_viewer.py`)
  could select only catalogue cameras. Its statuses were `paused`,
  `connecting`, `live` and `unavailable`.
- `observe_tracked` treated the end of a stream as a failure.
- `events.py` rejected any source other than `live` or `synthetic`.
- `shared_delivery.py` hard-coded `source_mode='live'` and deduplicated on the
  wall clock, which is wrong for a file processed faster or slower than real time.
- `recognition_contract.py` hard-coded `source_mode='live'`.
- The service needed Sentinel credentials to start at all.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `edge/sentinel_edge/live_detection_viewer.py` (`LiveState`) | Cameras only | Recordings are selectable. New `control()` with `pause`, `resume`, `restart`, `seek`. Each start, restart or seek is a new session ID and a new capture generation, which clears tracking, temporal evidence and reconstruction. Snapshots carry `source_mode`, session ID, recording name, duration and time basis. Recorded evidence uses a 30 s stale limit and is held while paused or completed; idle expiry (60 s without a heartbeat) still clears it. The live 5 s limit is unchanged. |
| same file (`serve`) | Catalogue discovery blocked everything when it failed | Recordings folder is rescanned every 15 s; a failed catalogue no longer blocks a recording. New `/control` and `/internal/control` endpoints. New flags `--recordings-dir`, `--recordings-only` (no Sentinel credentials needed), `--unpaced`. A bad file is reported once by failure code and not re-read in a loop. |
| `edge/sentinel_edge/tracked_live_viewer.py` | Live source only; EOF raised an error | Takes a `recorded` run. Uses blocking decode, publishes status `processing`, adds video position and pacing, returns normally at end of file. Also fixed a pre-existing race where the loop could exit one sample early. |
| `edge/sentinel_edge/recorded_source.py` | (Task 3) | Adds `RecordedRun` |
| `edge/sentinel_edge/events.py` | `live` / `synthetic` | Accepts `recorded` with required session, PTS, time basis; adds `processed_at`. Live payloads keep their exact old shape. |
| `edge/sentinel_edge/shared_delivery.py` | Wall-clock dedup, always `live` | Recorded path: dedup per identifier per 60 s of video time within a session; deterministic event ID from session, source, identifier and PTS; uses the recording's clock only when its receipt states it |
| `edge/sentinel_edge/recognition_contract.py` | Always `live` | Passes the recorded label and fields; new `completed` state |
| `edge/tests/test_recorded_session.py` | Did not exist | 19 tests |

## Audit result

Edge suite: **335 passed**, apart from the Windows-only path test and the
intermittent Windows clock test. Backend still 65 passed.

With a deliberately slow **mocked** OCR, a 6-second generated file had 12 of
12 samples analysed, zero drops, reached `completed` and wrote exactly one
masked recorded event that validates against the central schema. Restart,
seek, pause, bad controls and a tampered file were exercised over HTTP.

### Cross-process audit (added later on 10 October)

The real edge service ran on port 7780 and was driven **only through the
rebuilt central API container** (disposable stack, PostgreSQL), still with a
mocked recogniser. **17 of 17 checks passed**: listing shows only the
receipted file; start, completion with 12 of 12 samples; the accepted reading
travelled outbox -> API -> PostgreSQL and appeared as one recorded watchlist
alert with video time and a masked plate; a seek started a new session,
analysed only the remaining 6 samples and produced a second, separately
traceable alert; an out-of-range seek was refused; pause held the position;
resume finished; stop cleared the session.

**A defect was found and fixed by this audit.** The first run failed one
check: after Pause, the position kept advancing, because frames already
decoded and queued were still being analysed (up to 16 samples, which is many
seconds on a slow CPU). `observe_tracked` now holds analysis itself while
paused, so it stops at the sample in hand. A unit test covers it
(`test_pause_stops_analysis_at_the_sample_in_hand…`).

## What still has to change

- **Run it with the real models.** Every accepted plate in these tests comes
  from a stub that always returns the same text.
- See the alert in the dashboard in a browser (Task 6 and Task 11).
- The recorded session sends no health reports to the central API.
- The edge client still retries a rejected event forever; unchanged.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Session, controls, delivery and contract are complete on the edge |
| Tested | 3 / 3 | 19 new tests pass, including the HTTP service end to end |
| Proven | 1 / 3 | Real services, real HTTP, real PostgreSQL end to end; the recogniser is still mocked |
