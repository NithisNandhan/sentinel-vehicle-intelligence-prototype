# Approved dual-interface prototype

The user approved a clean operator dashboard and a protected local engineering
console, developed together around the same recognition session. This records
the approved chat design, not a new recognition-accuracy claim.

## Boundaries

- Extend the existing React command centre and local Python diagnostic viewer.
- One selected diagnostic camera, one decoder/tracker/OCR run, two consumers.
  The existing two-camera production worker remains separate and is not silently
  replaced. Its model versions must not be confused with this selected pipeline.
- An operator-protected central proxy exposes only allowlisted recognition
  metadata. Genuine images and full hypotheses never pass through that proxy.
- Explicit Start/Pause changes the shared session. Heartbeats cannot switch it.
  Opening either page alone does not start inference. Idle expiry clears media.
- The engineering console requires the operator key for images and full evidence,
  as well as the existing local Host/Origin and process-token protections.
- Native crops and OCR evidence are bounded in RAM, cleared on resets, switching,
  expiry and shutdown. No recording, full-text logs or download endpoints.
- Existing accepted-reading thresholds remain. Uncertain candidates do not enter
  watchlist alerts. Scores are uncalibrated, not probabilities of correctness.

## Operator view

Keep camera search/preview, alerts, map, routes, reports and operator actions.
Add a clean shared-recognition panel with camera selection, explicit controls,
active camera, freshness, reading/uncertain/model-accepted/unreadable states and
masked per-track outcomes. Clearly distinguish source and scope. Failures clear
stale recognition data rather than implying current processing. Existing camera
preview remains independent playback, not evidence of inference.

## Engineering view

Show aligned vehicle/plate overlays, selectable tracks, distinct native crops,
per-frame OCR and character evidence, bounded ranked candidates, acceptance gate
reasons, native dimensions/PTS, model profile and stage timings. The page explains
that these are observable diagnostics, not a model's internal thoughts.
Reconstruction is shown as inactive unless it actually runs; it is not invented
for presentation. Full local evidence is a privileged, short-lived view.

## Integration and verification

Shared contracts are fixed before parallel frontend/backend/edge implementation.
The central proxy authenticates the operator and sends the existing edge key to
fixed internal paths; it does not forward upstream arbitrary fields or errors.
Test authentication, pixel preservation, isolated track evidence, repeat evidence,
session expiry/switching, two-consumer consistency, no duplicate inference from
polling, metadata-only proxy responses, honest acceptance states and UI build.
Rebuild only changed services and verify with the browser. No new model family,
dataset collection, scale claim, or live accuracy claim is part of this change.
