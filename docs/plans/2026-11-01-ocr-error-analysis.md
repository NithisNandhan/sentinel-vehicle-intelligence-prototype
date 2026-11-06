# OCR error analysis and bounded preprocessing experiment

Goal: improve recognition without changing the dashboard, detectors, production
thresholds or model weights while the teammate fixes uninterrupted video jobs.

Baseline: adapted Awiros, 137/200 whole-plate exact against publisher XML labels.
Review all 63 differences visually; this is AI-assisted diagnosis, not independent
human ground truth. Preserve the original labels and predictions.

Development: the existing 37 validation crops from indian-corpus. They have been
used during earlier adaptation, so gains are development evidence only. No image
or prediction from the 178 reserved external registrations will be opened.

Bounded ablations: unchanged native crop, replicate-edge padding, modest luminance
CLAHE, guarded two-row recognition, and limited deskew if a reliable image-only
orientation can be established. No generated characters, grammar correction or
per-example selection using the expected answer. Record correct-to-wrong regressions
as well as recoveries, character error and OCR cost on the entire development set.

Choose at most one policy before replaying the already-inspected 200-image set.
That replay is diagnostic regression evidence, not a newly blind test. No further
policy search on those 200 outcomes; no automatic production promotion. Freeze all
configuration and input/checkpoint hashes, preserve licensing and keep private
image-containing reports local. Publish only code and plate-free aggregate findings.

Second-reader and real multi-frame work follows the crop experiments if supported
by evidence; frame fusion needs actual distinct-frame labelled passages and cannot
be demonstrated by transformed copies of still images.
