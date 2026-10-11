# Task 5 report: central recognition proxy for recorded sessions

Closed 10 October 2026. **Score: 9 / 10** (raised from 8 after the cross-process audit)

## What was there before

`backend/app/recognition.py` is the operator-only gateway between the
dashboard and the edge session. It:

- accepted only the statuses `paused`, `connecting`, `live`, `unavailable`, so
  a recorded session's `processing` or `completed` would have returned 503;
- always answered `"source_mode": "live"`, whatever the edge said;
- had no way to list recordings or send pause, resume, restart or seek;
- dropped the reason a session was unavailable.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `backend/app/recognition.py` (status) | 4 statuses, always `live` | Accepts `processing` and `completed` for a recording only. Rejects impossible mixes (a camera that is `processing`, a recording that is `live`). Reports `recorded` when the edge does. Adds allow-listed `session_id`, `recording_name`, `video_position_ms`, `video_duration_ms`, `time_basis`, `pacing`, `stale_after_seconds`, `inference_device`, and `failure_code` (a bare class name, only when unavailable). Bad values are dropped, never forwarded. |
| same file (new endpoints) | None | `GET /v1/recognition/recordings` returns ID, name, duration and time basis only. `POST /v1/recognition/control` validates the action before it reaches the edge. Both need the operator key. |
| same file (errors) | One 409 message | A separate, still redacted, 409 message for a rejected playback control |
| `edge/sentinel_edge/recognition_contract.py` | Did not forward `failure_code` | Forwards it (found by the cross-check below) |
| `backend/tests/test_recognition_recorded.py` | Did not exist | 24 tests |
| `backend/tests/test_recognition.py` | 3 invalid-status cases | 3 more cases for impossible status and source mixes |

The existing camera endpoints return exactly what they did before; their
tests are unchanged and pass.

## Audit result

- Backend suite: **92 passed**, 0 failed.
- Edge suite: **335 passed** apart from the two known Windows-only items.
- Cross-check script: nine real edge session states (idle, recorded
  connecting, processing, paused, completed, failed, camera connecting, camera
  live, stopped) were each produced by the actual edge `LiveState` and passed
  through the actual central allow-list. All nine were accepted with the
  correct label and no media. The first run of this check failed on the
  missing failure code, which is how that gap was found and fixed.

- Cross-process audit (later the same day): the rebuilt API **container**
  talked to a running edge service over HTTP. Listing, session start, status
  polling, heartbeat, seek, restart, pause, resume and stop all went through
  `/v1/recognition/*`; 17 of 17 checks passed (details in the Task 4 report).
  With no edge running, the recordings endpoint fails safe with 503 and an
  invalid control is rejected with 422 before the edge is contacted.

## What still has to change

- Repeat the same run in this machine's own `sentinel-prototype` stack once
  its API is rebuilt, and with real models.
- The API has no automated test that a recording's file name or path can never
  appear; the allow-list design prevents it and the tests check several
  attempts, but it is not an exhaustive proof.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Status, listing and control are complete |
| Tested | 3 / 3 | 27 new backend tests pass; existing ones unchanged |
| Proven | 2 / 3 | Running container to running edge service over HTTP; recogniser mocked and not yet in the machine's own stack |
