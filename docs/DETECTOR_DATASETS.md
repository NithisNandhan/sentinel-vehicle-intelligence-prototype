# Detection dataset audit

This is a bounded, permission-conscious internet search, not a claim to acquire
every dataset in existence. Detection training requires full scenes and actual
bounding-box labels. The existing 30,110-crop OCR corpus cannot simply substitute
for that supervision; most of its generated crops are not CCTV scenes.

## Acquired for this experiment

| Source | Publisher declaration / labels | Treatment |
|---|---|---|
| [Gujarat vehicle plates](https://www.kaggle.com/datasets/paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready) | Apache 2.0; Ghanshyam and Aabid Ansari; 355 YOLO-labelled images, version 1 | Acquired 45,874,064-byte ZIP; real regional plate-location examples. Group nearby timestamped photos, duplicates and similar images before split. Publisher labels, not new independent verification. |
| [Make ML / Larxel plate detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) | CC0/public-domain declaration; 433 PASCAL VOC images, version 1 | Acquired 212,910,636-byte ZIP. Broader non-Indian plate-location supplement, not Indian OCR text supervision. Retain source notice. |
| [Madrid self-learning traffic dataset](https://zenodo.org/records/10912886) | CC BY 4.0; Guerrero-Contreras, Balderas-Diaz, Garcia-Pascual, Munoz | Acquired publisher manual train/validation archives with verified MD5, ZIP CRC and SHA256. Excluded auto-labelled archive from supervised training/scoring. Actual export 117 train / 11 val / 6 test pictures; training includes augmentation variants from fewer source frames. |
| [IISc BMD-45](https://huggingface.co/datasets/iisc-aim/BMD-45) | CC BY 4.0; Indian traffic CCTV, consensus COCO annotations | Acquired 1,300 originals (4,332,531,693 bytes), revision `3e3f6a0b4834d282890c34986216c487914c5dff`. One image is excluded: publisher dimensions 1920 x 1080, actual 1620 x 1080; boxes were not rescaled speculatively. Before near-image guard: 1,000 train / 149 val / 150 test. Full repository API reports >160 GB; only a bounded subset was acquired. |

The BMD-45 README describes 14 classes, but the retrieved annotation files have
**13**, without `Other`. Conversion uses the actual publisher taxonomy after
checking train/val agreement; it does not invent an extra class. The complete
dataset, camera-diversity coverage and camera-identity-separated holdout are not
claimed by our small subset.

BMD originals with local integrity receipts remain preserved, including the
excluded image. Training uses explicit file lists, never directory discovery
that could re-admit an excluded file. Publisher train and validation pools are
retained; the validation pool is deterministically divided before training.
Exact cross-split pixel duplicates are quarantined. A further transitive pHash
cross-split guard (Hamming distance <= 6) excluded two more images from a
connected group. Final learning split: **999 train / 148 val / 150 test**.
Excluded originals are preserved but absent from all explicit training lists.
Local SHA256 receipts are not publisher-verified image checksums.

The admitted BMD subset has 10,402 training, 1,541 validation and 1,592 test
vehicle annotations. It is imbalanced: two-wheelers have 4,814 training boxes,
three-wheelers 1,801, but mini-buses only 38 (two validation, five test).
Overall pooled box results must not be presented as reliable accuracy for every
rare subclass. Frozen guarded manifest SHA256:
`0dfe0ee13dad6a3bfcf7862287c5070f2c1080b34c61de9434abfe936ce0fa94`.

Plate preparation admitted 567 unique non-conflicting photographs: 360 train /
130 val / 77 test. 135 exact duplicate observations and 86 conflicting pixel
groups were identified; all observations from conflicting groups were excluded.
Split/source counts: Gujarat 234/85/34; Make ML 126/45/43. No annotations rejected
for malformed coordinates in this export. Native images remain unmodified.

An independent audit corrected an initial summary counter: 788 observations
contain 653 unique pixel images (135 duplicates), then 86 conflicting pixel
groups are excluded, leaving 567 admitted images. The original frozen training
manifest retains its incorrect summary counter for hash reproducibility;
sample membership, splits, checkpoints and evaluation were not changed.

Archives stay under ignored `data/benchmarks/detector-training/archives`, with
download receipts and original publisher metadata. Fast training exports and
checkpoints use the separate Docker volume `sentinel_detector_training`.
Original provenance declarations do not independently establish every photo's
acquisition history. These data are not relicensed under the app's licence.

## Other relevant sources reviewed

The existing [OCR corpus audit](INDIAN_OCR_CORPUS.md) records 24 Kaggle and ten
Hugging Face candidates; its licences, exclusions and synthetic-origin warnings
still apply. This detection-specific search adds the following distinctions:

| Candidate | Decision / reason |
|---|---|
| [IISc UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26) | Relevant CC BY Indian CCTV source; BMD-45 chosen first. Do not indiscriminately merge related releases: overlap must be checked. |
| [IndiaScene365](https://data.mendeley.com/datasets/pwffhg6nhz/2) | Relevant CC BY Indian road scenes; public API access attempt returned 403. Not acquired, and no authentication barrier bypassed. |
| [NIVU Indian plates](https://universe.roboflow.com/nivu/indian-license-plate-knte7) | Existing three permitted development frames retained only for diagnostics. Full export API returned 401; no account key available. Do not train on the regression vehicle. |
| [Indian LPR in the wild](https://github.com/sanchit2843/Indian_LPR) | Creator states full dataset cannot be made public for legal reasons. No unauthorised mirror acquisition or contact made. |
| [Indian commercial-truck plates](https://github.com/siddagra/Indian-Commercial-Truck-License-Plates-Dataset) | Relevant restricted noncommercial source, access request form required. No form submitted on user's behalf. |
| [UA-DETRAC research](https://arxiv.org/abs/1511.04136) | Relevant fixed-camera vehicle data; full scale and original permission need source review. A Hugging Face re-export exists and contains images, but its relabelling/licence declaration alone is not treated as permission from the original creators. |
| [thundarstrom Indian plate repository](https://huggingface.co/datasets/thundarstrom/indian-license-plate-detection) | Advertises thousands of images, but current API reports zero stored image bytes; only metadata/config files. Not an acquired training corpus. |
| [Bapatla](https://zenodo.org/records/13954136) | Complete permitted OCR archive already local; lacks independent full-scene plate boxes. Current-model pseudo-crops are not detector ground truth. Initial diagnostic targets remain excluded from new supervised training. |
| abtexp / Umar / small TR OCR corpus | Already local for OCR; generated plate-only artwork and origin-unverified crops stay separate. Not counted as real detection scenes. |
| Saisirishan Indian vehicle plate collection | Noncommercial NoDerivatives declaration: excluded from adaptation pool. |
| DataTurks / unknown-licence mirrors | Missing or conflicting photo permission: excluded. |
| CCPD / AOLP / UFPR / SSIG | Potential broader plate-detection research sources, not automatically Indian OCR data; access conditions, size and source-specific rights must be reviewed before acquisition. Not acquired in this run. |

## Quality gates and limitations

Publisher full-scene bounding boxes only; no OCR-generated boxes used as truth.
Archive hashes/CRC, image decode/dimensions, annotation ranges, exact-pixel
duplicates and split guards are checked. Gujarat timestamp groups and perceptual
similarity groups prevent obvious repeated shots crossing partitions. This is
not exhaustive vehicle identity verification or an independently labelled test
set. Camera identity, near-duplicate coverage and pretrained overlap remain
explicit limitations; reserved test labels never enter training or checkpoint
selection. Stress degradation affects test inputs only, not annotations' meaning.

No genuine Sentinel footage is downloaded or retained, no plates queried against
a registration registry, no credentials exposed, no paid datasets purchased and
no access-request messages sent during acquisition.
