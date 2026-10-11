# Task 8 report: restore the release and prove a real CPU reading

Closed 11 October 2026. **Score: 8 / 10**

## What was there before

- No `data/` folder: none of the 14.6 GB of release assets were on this machine.
- No model volumes in Docker. Nothing real could load; every earlier test used
  a stub recogniser.
- GitHub CLI was not installed.
- The API container was still the pre-change image and the database was at
  revision `0001`.

## What I changed

| Item | Before | After |
| --- | --- | --- |
| Release assets | None | 32 files (5.8 GB) downloaded to `data/handoff-assets`: model-cache, alpr-cache, saved-events, presentation-assets and three receipts. The training (4.1 GB) and benchmark (4.7 GB) bundles were skipped for disk space. |
| How they were fetched | No `gh` | A small script used the Git credential already stored on this machine. The token is sent to `api.github.com` only and is never printed or saved. |
| Model volumes | None | `sentinel-prototype_model_cache` (16,931 files, 8.4 GB) and `sentinel-prototype_alpr_cache` (6 files), both hash-verified by the restore tool |
| Supplemental files | None | `data/handoff-restored/saved-events` (13 files) and `presentation-assets` (9 files) |
| `tools/repair_ocr_source.py` | Did not exist | Restores five PaddleOCR source files the handoff archive omitted |
| `compose.yaml` | Weights cache mounted read-only at `/root/.cache` | The two weight folders are mounted individually so Paddle can create its own cache folder |
| Running API and database | Old image, revision `0001` | Rebuilt image, revision `0002`; existing rows intact |
| `edge/sentinel_edge/recorded_source.py` | 7 container types | Also `.ogv` and `.ogg` |
| `data/recordings` | Empty | Two CC BY-SA 4.0 clips from Wikimedia Commons with receipts |
| `data/labels/hyderabad-pilot.json` | Did not exist | Four pilot labels, marked as made by the assistant |

## Defects found in the handoff itself

1. **The published release cannot load the OCR.** `tools/handoff_archives.py`
   omits any path component named `token` as a possible secret. PaddleOCR has
   a source folder `ppocr/data/imaug/vqa/token/`, so five files listed in the
   pinned manifest are missing after a restore and the reader raises
   `FileNotFoundError`. The repair tool fixes a restored volume; **the archiver
   is unchanged and the owner should be told.**
2. **Paddle needs to create `~/.cache/paddle` on import**, which a read-only
   mount of `/root/.cache` prevents. The owner's GPU launcher uses the same
   read-only mount and would hit this on a fresh machine too.

Neither showed up in any mocked test. Both appeared within seconds of the
first real start, reported by failure code as designed.

## Audit result

| Check | Result |
| --- | --- |
| Restore tool | Exit 0; four bundles restored with hash verification |
| OCR checkpoint SHA256 | Matches `c2ba896d994b174071c95ace0f5ffc8a61cec3fc6eecce8f0bc5feb7dec4456d` |
| Missing pinned source files after repair | 0 of 337 |
| Saved snapshot counts | 6 cameras, 16 sightings, 15 alerts, 22 audits, 16,625 health reports, 1 watchlist entry: exactly as the handoff states |
| `scripts/smoke-test.py` on the rebuilt API | 8 of 8 checks |
| Real model load on CPU | 40.7 s |
| Real inference timing | OCR 531 ms; vehicle detector 165 ms on a 1080p frame; plate detector 126 ms on a vehicle crop |
| Real footage through the operator API | 4K VP9, 56 s: 3,375 frames decoded, 113 samples analysed, 88 plate crops, 0 accepted |
| Pilot evaluation | 0 of 3 readable plates read; true plate in the top 20 for 0 of 3; 0 wrong acceptances; plate boxes matched 2 of 3; 420 s wall time |

The pilot uses one handheld clip and labels made by the assistant. It is a
first diagnostic, not an accuracy measurement.

## What still has to change

- **The saved snapshot is not imported.** This database already had rows, the
  restore tool refuses to overwrite, and I did not delete your data.
- **Parent versus adapted OCR on a labelled sample on CPU** (guide, page 8) is
  not done. It needs the benchmark bundle (4.7 GB) and C: has about 13 GB free.
- **This was not a clean-machine restore in the strict sense.** Central images
  already existed here from an earlier run.
- **Fix the archiver** so the next release includes the `token` folder.
- **Recognition on this footage does not work yet.** That is the subject of
  Task 11, not a restore problem.
- Speed is 0.13 of real time on 4K. 1080p has not been timed.

## Score

| Part | Points | Reason |
| --- | --- | --- |
| Built | 4 / 4 | Assets restored, service running with real models, repair tool added |
| Tested | 3 / 3 | Hashes, counts, smoke test and timings all checked |
| Proven | 1 / 3 | Real inference is proven; a correct real reading, the snapshot import and the OCR comparison are not |
