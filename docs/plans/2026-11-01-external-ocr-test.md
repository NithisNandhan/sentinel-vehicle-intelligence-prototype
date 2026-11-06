# Separate-source OCR evaluation

Use a dataset not used in this project's model adaptation. Search publisher sources
for actual plate transcriptions and permission, checking files rather than dataset
card counts. DataCluster currently exposes 47 images with text in only 20 annotation
files; other candidates lack downloadable images or require registration. The
academic Saisirishan Indian vehicle dataset provides XML plate-text and bounding-box
annotations and includes separate web and video-image acquisition groups.

Acquire the academic release for private noncommercial evaluation only, respecting
its CC BY-NC-ND terms: no training, no redistributing images/crops. Keep source,
permission, original bytes and annotation hashes separate from the app's licence.

Freeze a new test manifest before model predictions. Exclude previously seen
registration strings, exact crop pixels and original-image hashes, check perceptual
similarity to prior training crops, and keep at most one observation per registration
in the scored comparison. Record every exclusion. No OCR model creates labels or
chooses the easiest images. Reference labels are publisher transcriptions unless a
separate human review is recorded; original-model pretraining overlap remains unknown.

Run original Awiros, adapted Awiros and SVTRv2-B with the already-frozen settings.
Measure whole-plate exact reads, character errors, latency and source/size slices.
These images do not establish full-video completion, temporal acceptance or actual
fixed-camera CCTV recall. Keep new test images out of any subsequent training;
use a separate development subset if further adaptation is authorised later.

No application model replacement from one small or poorly labelled result. Report
the measured comparison and label limitations; preserve the previous recogniser.
