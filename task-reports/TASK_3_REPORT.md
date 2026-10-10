# Task 3 report: local recorded-video source with backpressure

Closed 10 October 2026. **Score: 9 / 10**

## What was there before

- The only video reader (`stream.py`) opened RTSP or HLS camera streams. There
  was no way to read a local file.
- The decode queues (`sampled_frames.py`, `crop_collector.py`) were built for
  live feeds: when analysis fell behind they dropped the oldest frames. On a
  slow CPU that would discard most of a recording before OCR saw it.
- Nothing checked that a file was permitted to be used.

## What I changed

| File | Before | After |
| --- | --- | --- |
| `edge/sentinel_edge/recorded_source.py` | Did not exist | `RecordingLibrary` lists only videos with a valid `<video>.receipt.json` (source, licence, permission, SHA256). `read_recording` decodes with PyAV, keeps original PTS, supports a start position, paced or unpaced playback, pause and resume. Named failures: `UnpermittedRecording`, `ChangedRecording`, `CorruptRecording`, `MissingPts`, `NoVideoStream`, `InvalidSeek`. Error text never contains the file path. |
| `edge/sentinel_edge/sampled_frames.py` | Always dropped when full | New `block=True` mode: the decoder waits for room instead. A cut is flagged on the correct frame without discarding queued frames. An oversized frame raises `FrameTooLarge`. Default behaviour is unchanged. |
| `edge/sentinel_edge/crop_collector.py` | Always dropped when full | Honours the same `block` mode in both of its stages |
| `tools/register_recording.py` | Did not exist | Writes a receipt for a permitted file; never overwrites one |
| `edge/tests/test_recorded_source.py`, `recorded_fixtures.py` | Did not exist | 22 tests on a generated MPEG-4 file |

Recording IDs are `rec-` plus the first 12 hex digits of the file's SHA256, so
the same file has the same ID on any machine.

## Audit result

Edge suite at this point: **316 passed**, 1 Windows-only failure.

Measured in the tests: all 60 frames decoded with PTS 0 to 5900 ms; a slow
consumer received 40 of 40 samples in blocking mode with zero drops, while the
unchanged live mode still dropped; paced playback waited about 5.9 s of video
time for a 6 s clip; seek to 3000 ms started at exactly 3000 ms.

## What still has to change

- Try real CCTV files. The tests use a small generated MPEG-4 clip. Real
  footage may be H.264 or HEVC, have variable frame rate, B-frames or a
  non-zero start time. The code is written for these but they are untested.
- A recording with any frame lacking a PTS is rejected outright. That is the
  rule the guide asks for; if real files commonly hit it, they will need
  remuxing first.
- A side finding: `test_live_reconstruction` fails intermittently on Windows
  because the system clock ticks every ~15.6 ms. Not caused by this task and
  left unchanged.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Library, decoder, controls, backpressure and receipt tool exist |
| Tested | 3 / 3 | 22 new tests pass with a real decoder on a real file |
| Proven | 2 / 3 | Real decoding is exercised, but not yet on genuine CCTV footage |
