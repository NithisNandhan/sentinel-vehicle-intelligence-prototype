# Reconstruction in the shared live pipeline

Implemented 27 September 2026 following the user's request to connect the
previously approved reconstruction flow to the live diagnostic pipeline.

## Runtime

The operator and engineering interfaces continue to share one selected analysis
session. `observe_tracked` now enables reconstruction in `TrackedOCR`:

1. Detect a plate inside the tracked vehicle and copy its original native crop.
2. Run the fine-tuned Awiros CTC OCR on distinct crops, spaced by video PTS.
3. Once three crops are retained, choose the highest-quality reference, align
   neighbours using the existing checked ECC affine registration, and fuse
   valid pixels with quality weights. At least three aligned sources and 70%
   supported overlap are required. Unsupported borders retain reference pixels.
4. Run the same GPU OCR model on this reconstructed image. Keep at most 32
   derived suggestions, alongside original frame hypotheses, in a total bounded
   200-candidate search. Rescore each suggestion using **only original-frame
   CTC likelihoods and character support**. No generated text is rendered back
   into recognition evidence.
5. Keep the existing three-distinct-original, character-score and ranking-margin
   acceptance gates. A strong contradictory derived reading withholds an
   otherwise accepted reading. Processed-only suggestions remain diagnostic.

This can recover a candidate missed by the original-frame beam search when its
characters are supported by the originals; it cannot manufacture missing visual
evidence or turn repeated views into more independent observations. Scores
remain uncalibrated. Model acceptance is not independent registration verification.

ECC is geometric alignment, not a newly trained deep reconstruction model. The
OCR is the existing fine-tuned deep model. Learned deblurring/RAFT and the separate
offline hybrid experiments are **not newly promoted into this live session**.
Native-grid reconstruction must not be advertised as recovered resolution.

## Operator, engineering and privacy

The engineering console places a `RECONSTRUCTED · DERIVED` image card beside
original crops, with its OCR, top suggestions, source count and measured timing.
Its `0 extra observations` label distinguishes correlated evidence. Original
images remain unchanged. The `reconstruction` timing stage includes alignment,
fusion and derived OCR. The existing `ocr_calls` counter counts original OCR
calls; reconstruction attempts are reported separately.

The central/operator projection still receives metadata and masked plate text,
never reconstruction pixels or full hypotheses. Delivery uses the existing
accepted-only masked/HMAC outbox and now stamps the ECC-proposal processing
identity. No central API/schema changes or production-worker model changes.

Native retention stays at eight crops/32 tracks/32 MiB. Working crops exceeding
1.5 MiB are rejected for reconstruction. There are at most six reconstruction
attempts per evidence segment. A changed crop window after budget exhaustion
clears the old reconstruction/suggestions and continues original OCR. Duplicate
or too-close frames never trigger an additional attempt. Gap, track/class/plate
inconsistency, camera/scene change, expiry and eviction release reconstruction
state. Existing stale HTTP/browser cleanup applies to the derived image too.
Errors are redacted and fall back to original OCR, rather than killing capture.

## Measured validation

- Final edge regression: 265 tests run, 264 passed, one optional Node-in-container test
  skipped. All nine engineering renderer tests ran successfully on the Windows
  host, including reconstructed-card rendering, safe text and evidence clearing.
- Frontend: 15 tests passed and the production build passed after the camera
  scope label update. The PowerShell launcher parsed without syntax errors.
- New tests cover real ECC reconstruction with native-pixel preservation,
  no extra votes, suggestions pruned from frame beams, original-only rescoring,
  conflict withholding, failure fallback/redaction, duplicates, expiry, eviction
  and the bounded reconstruction budget.
- A cached GPU OCR plumbing check on explicitly rendered synthetic pixels
  reconstructed three distinct inputs and decoded ten characters. It kept three
  original observations, zero extra temporal votes and one model-accepted
  synthetic hypothesis. Reconstruction plus derived OCR took 331.68 ms in this
  cold diagnostic. This is not road-footage accuracy or a live reading.
- Sequential 90-second authorised live checks: CAM17 decoded 251 frames,
  analysed 20, produced seven plate crops and three reconstructed windows, with
  zero reconstruction errors and zero model-accepted readings. The maximum
  reconstruction-stage time observed by the two-second poll was 81.68 ms.
  CAM14 decoded 109 frames, analysed 13 and produced zero plate crops/windows.
- These feeds had intermittent delivery: CAM17's last snapshot was seven
  seconds old and its sensitive debug evidence was consequently omitted.
  These short windows do not establish stable throughput, accuracy, or endurance.
  No genuine images, reconstructed images or plaintext readings were saved.
  No labelled live ground truth was available, so live accuracy remains unknown.
- After changing the selected pair, sequential 90-second live checks connected
  both CAM19 and CAM26 from the authorised 30-camera catalogue. CAM19 decoded
  1,310 frames/analysed 81; CAM26 decoded 1,684/analysed 107. Both had zero plate
  crops, reconstruction attempts and accepted readings. Their final frame ages
  were 0.9 s and 0.1 s respectively. Reconstruction cannot run on either camera
  until plate localisation supplies at least three usable same-track crops;
  this is not evidence of reconstruction execution on the new pair.

The selected camera pair is now CAM17/CAM01 following the user's subsequent
request. `.env`, its example and future smoke checks use this pair. The launcher
no longer overrides `.env` with CAM01/CAM02. Full catalogue search remains
available; the configured pair is ordered first. Production pair selection also
uses the updated setting, without deleting historical camera records.
The earlier CAM19/CAM26 measurements above are retained as historical evidence,
not current camera selection. Model settings and reconstruction are unchanged.
The AI runtime was updated source-only using `edge/Dockerfile.source-update`
on the previously validated installed image, avoiding an unrelated dependency
upgrade/download. Future dependency changes should use the normal Dockerfile.

The shared viewer was restarted with the updated mounted source and returned to
paused after validation. Reload http://127.0.0.1:7780/, unlock with the local
operator key and start a camera to see reconstruction when enough crops align.
The main pipeline implementation remains a separate dependency from reliable
plate localisation and actual OCR accuracy.

## Repeating the checks

`tools/check_reconstruction_gpu.py` runs through the cached GPU runtime launcher
inside the viewer container. Its inputs are deliberately synthetic and no images
are saved. `tools/check_live_reconstruction.py --seconds 90` runs inside the
viewer, requires an initially paused session, consumes keys from its existing
environment, prints metadata only, and returns the session to paused. It tests
both `.env` camera IDs only when present in the authorised catalogue or approved
configured-relay scope; it never scrapes portal cards or fabricates camera access.
