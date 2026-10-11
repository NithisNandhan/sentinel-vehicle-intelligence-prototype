# Task 1 report: recorded source contract in the central API

Closed 10 October 2026, amended the same day. **Score: 9 / 10** (raised from 8 after the PostgreSQL audit below)

## What was there before

- `backend/app/schemas.py` allowed only two sources: `live` and `synthetic`.
  Downloaded footage could only have been sent mislabelled as `live`.
- A sighting had no way to say which replay run produced it, whether its
  timestamp was the recording's clock or the processing time, or when it was
  processed.
- Incident reports had a notice for synthetic events only.
- One database migration (`0001_initial`).

## What I changed

| File | Before | After |
| --- | --- | --- |
| `backend/app/schemas.py` | `SourceMode = Literal["live", "synthetic"]` | Adds `"recorded"`. New optional `session_id`, `time_basis` (`source_clock` or `processing_time`) and `processed_at` on sightings. A `recorded` sighting is rejected unless it carries all three plus `stream_pts_ms`. Alerts return the same fields. |
| `backend/app/models.py` | No such columns | Nullable `session_id`, `time_basis`, `processed_at` on `sightings` |
| `backend/migrations/versions/0002_recorded_sightings.py` | Did not exist | Adds the three columns to an existing database; safe to run twice; a fresh database already has them |
| `backend/app/main.py` | Stored and returned none of this | Stores the new fields; returns them on alerts, route search and incident reports; adds `recorded_notice` ("RECORDED FOOTAGE · NOT A LIVE CAMERA", plus "TIME SHOWN IS PROCESSING TIME" when that is the basis) |
| `backend/tests/test_api.py` | 53 tests in the suite | 12 new tests |

Amendment: the first version of this task omitted `processed_at`. Re-reading
the completion guide (milestone M3, "processing UTC time") showed the gap and
it was added.

## Audit result

Backend suite: **65 passed**, 0 failed, 3 deprecation warnings.

New tests show: a recorded sighting keeps its label through the alert, camera
list, route search and JSON/HTML report; duplicate event IDs are ignored; a
second session is stored as a separate traceable sighting; missing or invalid
recorded fields return 422; old `live` and `synthetic` payloads still work;
the migration upgrades an old database without losing a row.

### Integration audit on PostgreSQL (added later on 10 October)

Run on a disposable stack (`sentinel-audit`, its own database volume and port
18000) so the services and database already running on this machine were not
touched.

1. The **old** API image created the old schema on PostgreSQL (revision
   `0001_initial`, 14 columns on `sightings`) and one old row was inserted.
2. The **rebuilt** API image started and logged
   `Running upgrade 0001_initial -> 0002_recorded_sightings`. The three columns
   exist (`session_id`, `time_basis`, `processed_at timestamp with time zone`)
   and the old row is intact.
3. An audit script used the **real edge delivery client** to send a recorded
   sighting over HTTP: **17 of 17 checks passed**. They cover the watchlist
   alert, the recorded label with run/PTS/basis/processing time, masked plate,
   duplicate handling, a second session as a separate traceable sighting,
   rejection without a session (422) or edge key (401), acknowledge and
   resolve with audit trail, the incident report notice, route search and the
   camera list.

## What still has to change

- Rebuild the API in the **existing** `sentinel-prototype` stack so its
  database gets migration `0002`. The currently running API container on this
  machine is still the old image. `scripts/start-recorded-analysis.ps1` does this.
- Health reports have no recorded-specific fields. None are needed yet because
  the recorded session does not send health events.
- The API must be rebuilt **before** the edge sends any recorded event; an old
  API would reject it and the edge would retry forever.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Contract, storage, migration and outputs are complete |
| Tested | 3 / 3 | 65 of 65 pass, including the migration |
| Proven | 2 / 3 | Shown on PostgreSQL through the rebuilt container with the real edge client; not yet applied to this machine's own running stack |
