# Opt-in plate-recovery validation (2026-09-28)

`-PlateRecovery` adds a bounded direct full-image plate search after unusable
vehicle-ROI proposals and changes PTS sampling from 500 ms to 250 ms while a
sufficiently large tracked vehicle is recently visible. It does not relax the
multi-frame acceptance rules or store live media.

## Automated checks

- Edge suite in the existing container: 283 tests passed, one skipped.
- New cases cover native-pixel fallback attribution, rejecting a plate outside
  the vehicle box, one-second fallback cooldown, and adaptive sample selection.
- `git diff --check` found no whitespace errors.

## Licensed-data diagnostic that motivated the switch

On ten Bapatla Indian-plate test stills, the parent full-image plate detector
localized all ten and the adapted OCR read nine exactly. The earlier
vehicle-first path read seven exactly on the same split. This small still-image
comparison does not estimate live-camera accuracy. On one NIVU three-frame
development sequence with *assigned replay PTS*, 500 ms sampling retained two
distinct observations and could not meet the three-observation acceptance
rule; 250 ms retained all three and accepted the correct plate. These are not
real capture timestamps or independent live holdout data.

## Short live runtime check

The authorised catalogue and two streams were exercised for 30 seconds each
through the opt-in engineering viewer, using metadata-only polling and no
saved images or plaintext registrations:

| Camera | Decoded | Analysed | Vehicle-box instances | Native plate crops | Full-image calls | Full-image recovered crops | Accepted identifiers |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM17 | 183 | 15 | 6 | 0 | 1 | 0 | 0 |
| CAM01 | 403 | 77 | 188 | 8 | 18 | 0 | 0 |

Both cameras produced fresh live frames during this run. The fallback executed
but recovered no crop in this window; no plate was accepted. There are no
verified per-vehicle live labels, so this run cannot measure recognition
accuracy or establish improvement over the default path. A longer paired
default-versus-opt-in run with labelled vehicles and latency/drop metrics is
required before making the experiment the default.
