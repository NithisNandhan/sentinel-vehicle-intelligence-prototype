# Task 6 report: both interfaces

Closed 10 October 2026. **Score: 7 / 10**

## What was there before

**Operator dashboard (`frontend/src`)**

- `SourceTag` showed `SIMULATED` for synthetic events and `LIVE SOURCE` for
  everything else, so a recorded event would have been shown as a live source.
- The recognition panel listed cameras only, treated only `live` as current
  with a fixed 5-second limit, had one "Pause" button (which stopped the
  session) and no video time.
- The incident report banner had two cases: simulated or live.

**Engineering console (`edge/sentinel_edge/engineering_page.py`)**

- Camera list only. Evidence shown only when status was `live` and under 5
  seconds old. No playback controls, no video position, no session ID, no
  indication of which device or checkpoint was loaded.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `frontend/src/api.ts` | `source_mode: 'live'`; 4 statuses | `'live' \| 'recorded'`; adds `processing`, `completed` and the recorded fields; `getRecognitionRecordings()` and `controlRecognition()`; alerts and sightings carry video PTS, session, time basis, processing time |
| `frontend/src/recognition.ts` | 4 small helpers | Adds `sourceLabel`, `sourceDescription`, `staleLimit`, `isRecorded`, `recognitionIsRunning`, `formatVideoTime`, `sessionLabel`, `failureMessage`. A camera can never use the recorded limit or states. |
| `frontend/src/RecognitionPanel.tsx` | Cameras only | Cameras and recordings in one selector (grouped); RECORDED tag; video position and duration with a seek slider; Pause/Resume, Restart and Stop for a recording; recordings stay selectable when the camera catalogue is down; plain-language failure messages; heartbeats keep a paused or finished recording alive |
| `frontend/src/App.tsx` | Two source labels | Three: alerts, cameras, routes and the incident report all show RECORDED. Alert detail adds video time, analysis run and a note saying whether the time shown is the recording's clock or the analysis time. |
| `frontend/src/styles.css` | No recorded style | `.source-tag.recorded` and the playback bar |
| `edge/sentinel_edge/engineering_page.py` | Cameras, live only | Recordings in the selector with reasons for files that were not listed; Pause/Resume, Restart, Seek; facts for source, video position, analysis run and OCR device with checkpoint hash prefix; evidence limit taken from the session for recordings only; a new run from either interface clears the screen first. The old "Pause shared session" button is now labelled "Stop shared session", which is what it always did. |
| Tests | 15 frontend tests; 1 browser-behaviour run | 22 frontend tests; 3 browser-behaviour runs |

Unchanged: crops, full candidates and original frames are still only in the
protected engineering console; the dashboard still shows masked plates only.

## Audit result

| Check | Result |
| --- | --- |
| TypeScript type-check (`tsc -b`) | Clean |
| Frontend tests | **22 passed** (15 before) |
| Frontend production build | Passed; large-bundle warning remains (1.95 MB) |
| Engineering page tests | **10 passed**, including the script run under Node 24 with a minimal DOM |
| Edge suite | 335 passed apart from the two known Windows-only items |
| Backend suite | 92 passed |

The Node run checks, among other things: a camera cannot borrow the recorded
limit; 12-second-old recorded evidence stays visible; a paused recording keeps
its frame; a new session clears the old one before anything new is shown; an
invalid seek is never sent; a recording name or failure code containing markup
is rendered as text.

## What still has to change

- **Nobody has looked at it in a browser with a running session.** Layout,
  the seek slider and the grouped selector are untested visually. This needs
  the services running with models (Task 8).
- Two-way control (start in one interface, see it in the other) is covered by
  design and by separate tests on each side, not by one test across both.
- The dashboard's map and camera-health counters ignore recordings, which have
  no location or health. That is intended but should be checked on screen.
- Bundle size warning is unchanged (listed as later hardening in the guide).

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Selection, labels, controls and video time exist in both interfaces |
| Tested | 3 / 3 | Type-check, unit tests, build and scripted DOM runs pass |
| Proven | 0 / 3 | Not yet seen working in a real browser against a real session |
