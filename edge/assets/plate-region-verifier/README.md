# Experimental visual plate-region verifier

`model.npz` contains 3,786 linear logistic-regression weights and one bias.
`model.json` binds the exact model bytes, feature contract, validation-selected
thresholds, frozen training manifest and source declarations. Loading uses
`allow_pickle=False` and bounded array dimensions. No OCR model is invoked.

The feature extractor uses original grayscale appearance, padded 128 x 64
working-copy HOG gradients and six texture/shape values. The input image is
not modified. Inference requires NumPy and OpenCV; scikit-learn is required
only to rerun `tools/train_plate_region_verifier.py`.

The classifier's `allow_ocr` describes its original strict recommendation. The
separate opt-in provisional reading policy may read an `uncertain` region and
display registration-shaped text without changing this visual decision or
permitting automatic confirmation. See `docs/TEXT_CORNERS_VERIFIER.md`.

## Data provenance and evaluation

The frozen source manifest has 360 training, 130 validation and 77 test images.
Groups and decoded-image identities may not cross these splits. Source images
are not included in this folder:

- [Gujarat vehicle plates](https://www.kaggle.com/datasets/paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready),
  publisher declaration Apache 2.0; Ghanshyam and Aabid Ansari.
- [Make ML / Larxel plate detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection),
  publisher declaration CC0/public domain.

Positive labels come from the publishers' plate boxes. Negative regions are
sampled outside all known plate annotations, including adjacent vehicle context.
They are **assumed background**, not independently human-verified non-plates;
missing publisher annotations remain a potential source of label noise.
Augmentation is confined to training positives. The Mumbai footage and its
human transcriptions were not opened by the trainer or used for model fitting,
threshold selection or the reported test evaluation.

Validation selected thresholds before test evaluation. On the untouched 77-image
test partition, 83 plate crops and 462 assumed background regions yielded:

| Outcome | Publisher plate | Assumed background |
|---|---:|---:|
| Likely plate | 56 | 0 |
| Uncertain | 26 | 102 |
| Likely non-plate | 1 | 360 |

Keeping uncertain regions retains 82/83 labelled plates (98.8%); passing only
likely-plate regions accepts 56/83 (67.5%). These are **crop-classifier metrics
on proxy labels**, not OCR accuracy, full-scene plate detection recall, performance
on painted registrations, or Mumbai CCTV results. Scores are not calibrated
real-world probabilities. The single rejected labelled plate is a reason to keep
rejected native crops visible in engineering review and avoid automatic promotion.

Training/evaluation report and per-image membership stay in the ignored local
`data/benchmarks/plate-region-verifier-20261111/run` folder. Source bytes and
decoded pixels were checked against the existing manifest. Original dataset
images retain their own publisher declarations; this folder does not redistribute
or relicense them. Keep the verifier opt-in until reviewed on actual candidate
regions, including vehicle branding and other text-like hard negatives.
