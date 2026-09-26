# Expanded Indian OCR corpus — 27 September 2026

This is dataset preparation, not a deeper training run or an accuracy claim.
The existing experimental OCR checkpoint and live worker are unchanged.
The combined images, labels, archives, receipts and review pages stay local
under ignored `data/benchmarks`; GitHub receives tools, tests and this audit only.

Follow-up: a separate bounded GPU encoder adaptation and diagnostic evaluation
are recorded in [DEEP_OCR_ADAPTATION.md](DEEP_OCR_ADAPTATION.md). They do not change
the corpus's original review statuses or establish independent live accuracy.

## Selected sources

| Source | Publisher permission / labels | Treatment |
|---|---|---|
| [Bapatla Number Plate Number Identification](https://zenodo.org/records/13954136), DOI 10.5281/zenodo.13954136 | CC BY 4.0; 1,700 photographs and original Excel transcriptions | Map filenames, filter supported Indian-style labels, automatically localise native plate crops, reuse explicit prior reviews, quarantine ambiguous mappings and known bad samples. Publisher labels are not new human verification. |
| [abtexp synthetic Indian plates](https://www.kaggle.com/datasets/abtexp/synthetic-indian-license-plates), version 1 | CC0; 18,000 generated plate images; filenames contain generated text | Separate synthetic supplement. These are not 18,000 real vehicles, temporal frames or CCTV examples. |
| [umar1103 Final Licence](https://www.kaggle.com/datasets/umar1103/final-licence/versions/1) | CC BY 4.0; mixed origins; filenames contain text | Actual archive: 30,450 image files. Normalising spaces/hyphens admits 17,012 supported Indian-style observations. Exactly 6,050 decoded-image duplicates match abtexp and inherit its synthetic origin. The other 10,962 remain origin-unverified; inspected 512×256 samples are obvious two-row generated artwork, not real CCTV. |
| [AnirudhLanka2002 TR OCR Training License Plate](https://huggingface.co/datasets/AnirudhLanka2002/TR_OCR_Training_License_Plate/tree/b3cda9572099a7023cea30a2ee0537d0647fd62e) | Publisher declares MIT; 100 cropped JPGs and `lpr.csv` | Pinned revision; 82 supported label/crop pairs admitted, 18 unsupported labels excluded. Original photo provenance and manual versus automated label creation are unspecified. Keep origin-unverified, not verified real ground truth. |

Bapatla attribution: Bhuma Chandra Mohan; CH V M S N Pavan Kumar; Katakam;
Bethala Venkata Varma; Vatada Sai; Kagitha Sambaiah; Bapatla Engineering
College, 2024. The original workbook is unchanged. Publisher licence declarations
do not independently prove ownership of every contributed image; acquisition
history and pretrained-model overlap remain uncertain. In particular, similar
state-wise image naming in other repositories is a provenance-review flag,
not proof that their photographs can be relicensed. Known conflicting-licence
sources are excluded below.

All combined observations retain individual source URLs, attribution, archive
hashes, member names, original-image hashes, label kind and licence. The corpus
is **not relicensed under this application's MIT licence**.

## Actual local outputs

The authoritative measured counts and integrity digest are in
`data/benchmarks/indian-corpus/summary.json`; do not sum advertised dataset sizes.

Final measured merge, after all four selected sources and the orientation check:

| Tier | Unique native crops | Training-eligible crops |
|---|---:|---:|
| Publisher-declared photographs | 1,066 | 951 |
| Known generated plates | 18,000 | 16,153 |
| Mixed origin, not independently established | 11,044 | 9,956, separate opt-in pool |
| Total | **30,110** | **27,060** |

There are 29,972 distinct registration *strings*, not proven physical vehicles.
The photographic tier has 946 strings, including 840 training registrations.
Exactly 48 crops retain the prior explicit visual reviews; 12,062 publisher
labels remain unverified and 18,000 are generator-labelled. There is no newly
human-verified test set. The 89 photographic proposed holdout crops represent
80 registrations requiring review, alongside the preserved old test/validation
registrations. Across all tiers, 3,024 crops are proposed holdouts, six are
quarantined and ten each remain legacy test and validation.

The merge removed 6,051 exact decoded-image duplicates. Bapatla's remaining
non-admitted records include 193 localisation failures, one tiny crop, four
unmapped/conflicting-name cases, 182 unsupported orientation records and 253
unsupported/non-Indian-style labels. Unsupported labels are not necessarily
legally invalid plates. Of admitted photographic crops, 101 are 32–63 pixels
wide, 479 are 64–127 pixels wide and 486 are at least 128 pixels wide; width alone
does not establish readable text. None of these counts measures OCR accuracy.

Final corpus manifest SHA-256:
`8b593cfa6424fcb5a1282bc593b68e33bdf975c2018c7abe4886c9a4f2137195`.

Completed verification checked **all 30,110** saved crops: file hashes, decoded
pixel hashes, dimensions, source licence/attribution references, registration
split separation and exact training-index membership all passed. This does not
certify publisher label correctness. **174 edge tests passed**, including 13
new corpus tests for native pixels, lossless orientation, duplicate/conflict
handling, provenance, extension integrity and split protection. The Windows-only
full-suite attempt lacked OpenCV; the complete suite passed in the existing
Docker runtime. The small native corpus-only suite also passed.

- `archives/`: untouched downloaded ZIPs or explicitly locally assembled pinned
  snapshots, with SHA-256/CRC receipts. Bapatla also requires its publisher MD5.
- `source_audit.json`: machine-readable publisher metadata, licence declarations,
  sample file listings and access failures. An audit entry is not an acquired dataset.
- `crops/`: lossless PNGs named by decoded RGB pixel hash; integer crops only.
- `corpus.jsonl`: deduplicated index with every source observation preserved.
- `train_photographs.jsonl`: training-eligible publisher-declared photographs;
  most labels require review. Only explicit prior reviews are marked verified.
- `train_synthetic.jsonl`: generated training supplement, distinct from photos.
- `train_origin_unverified.jsonl`: optional mixed-source data, **not** the default
  real-camera pool; inspect provenance, generation and label quality first.
- `holdout_needs_review.jsonl`: proposed fresh held-out registrations; not a
  scored test dataset until crops and transcriptions are independently reviewed.
- `review_queue.jsonl`, `quarantine.jsonl`, `rejected.jsonl`: inspectable quality
  gates and reasons, not silently discarded observations.
- `data/benchmarks/reports/indian-corpus.html`: local source-balanced review
  sheets and aggregate counts. Labels displayed are publisher proposals, **not
  OCR predictions or new recognition results**.

## Cleaning and leakage controls

The acquisition tool accepts only specifically reviewed downloadable sources;
a permissive metadata tag alone cannot authorise every search result. ZIP
paths, expansion, individual file size and available disk headroom are bounded.
No third-party training scripts, credentials, portal scraping, personal-data
forms or live Sentinel footage are used.

Labels are uppercased with whitespace/hyphens removed, without inventing O/0,
B/8 or missing characters. Supported common Indian/BH formats are an eligibility
filter, not a complete legal-plate validator. Unsupported, partial or foreign
labels are retained in the rejection audit, not declared universally invalid.
Workbook names are mapped explicitly; missing/conflicting mappings are rejected.
Automatic localisation picks the highest-confidence plate detection without
consulting transcription text. A detector crop can select the wrong plate in a
multi-vehicle image: those crops remain review proposals.

Standard EXIF orientation is normalised with lossless 90-degree rotations or
pixel flips before localisation, without resampling or changing pixel values.
Original files remain untouched; the source orientation and normalised box
coordinate space are recorded for newly oriented inputs. The targeted recovery
checked all 182 excluded orientation-tagged photographs: all carried unsupported
orientation value **0**, not standard orientations 1–8. It therefore added no
crops. We did not silently assume a correct rotation; these remain review cases.

Exact decoded RGB duplicates are collapsed even when PNG compression differs.
Every source claim remains attached. Different labels on the same pixels
quarantine both labels, including their other images. Native crops preserve
decoded source pixels; later network-input resizing is a separate operation.
Legacy cropped-image hashes are rechecked before review status is transferred.

Registration strings, rather than images, determine deterministic splits before
augmentation. The previous 10 test and 10 validation registrations cannot enter
training. NIVU's existing multi-frame regression registration is also reserved.
Repeated photographs of a registration cannot straddle splits. Exact-dedup
coverage is measured; comprehensive cross-registration perceptual dedup and
vehicle-identity verification are **not** claimed. Publisher-pretraining overlap
is unknown. All legacy benchmarks are already development-visible, not new
untouched evaluation evidence.

## Recommended use for the next tuning run

1. Review held-out photographic crop/text pairs first; reject uncertain labels
   rather than use OCR agreement as ground truth. Establish a genuinely fresh
   validation/test set with one-row, two-row, small, angled and difficult plates.
2. Use the photographic training pool as the main adaptation data. Audit and
   explicitly opt into suitable mixed-origin examples; don't let rendered
   templates masquerade as real-camera diversity.
3. Use generated examples for character/layout coverage and optional pretraining.
   For the real-image adaptation stage, begin with at most 25% generated examples
   per batch, a proposed starting policy to validate—not a measured optimum.
4. Preserve originals; add mild blur, downsampling, noise and exposure changes
   **only to training images**. Generated degradations are not independent frames.
   Two-row examples require the existing row-aware recognition path or a suitable
   explicit training representation, not blind one-line squeezing.
5. Stage the selected training export on a Linux data volume before larger
   container training: thousands of individual files over a Windows/WSL mount
   are slow. Check free space first; no relocation/deletion is implicit in this
   preparation. Then unfreeze selected encoder blocks with a lower learning rate, compare to
   the untouched parent, early-stop on reviewed validation data and score the
   fresh held-out set plus the existing multi-frame degradation regression.
   No broader trainer was enabled by weakening the old reviewed-pilot guards.

More downloaded images alone do not solve missing image detail. The main new
resource is a larger *candidate* adaptation corpus; label quality, real-camera
coverage and independent held-out evaluation remain consequential gates.

## Additional sources investigated, but not merged

### Useful candidates with an access or annotation gap

- [IURS ANPRINR](https://data.mendeley.com/datasets/znghtgfnnf/1), DOI
  10.17632/znghtgfnnf.1: CC BY 4.0, publisher reports 8,168 character-annotated
  COCO images. Augmented flips/rotations mean these are not 8,168 independent
  vehicles. Public download requests returned HTTP 403; browser acquisition did
  not complete. No files or character-derived transcriptions from it are merged.
- [Indian commercial truck plates](https://github.com/siddagra/Indian-Commercial-Truck-License-Plates-Dataset):
  images require author/form access; repository specifies noncommercial
  share-alike terms. No form submitted or personal details sent.
- [Indian LPR in the wild](https://github.com/sanchit2843/Indian_LPR): paper reports
  16,192 images and 21,683 plates with character annotations. The maintainer says
  the dataset cannot be public because of legalities; not an available download.
- [OCR-On-the-go](https://github.com/rohitsaluja22/OCR-On-the-go): potentially
  relevant vehicle-video research, but code permission is not an image-dataset
  licence and a usable permitted OCR corpus was not established.
- [Indian plate restoration](https://github.com/yashchavan11/indian-license-plate-restoration):
  describes 643 labelled crops, but the inspected repository tree did not ship
  `data/raw` images or `labels.csv`; MIT code is not evidence of image permission.
- [NIVU](https://universe.roboflow.com/nivu/indian-license-plate-knte7): CC BY 4.0,
  detection-oriented. Our three explicitly labelled existing frames stay a
  regression passage, never newly merged training data.

### Detection/classification labels are not plate transcriptions

- [Gujarat YOLO-ready](https://www.kaggle.com/datasets/paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready):
  publisher declares Apache 2.0, 355 resized images and plate boxes; useful for
  later explicit transcription work, not immediately OCR labelled.
- [Kedarsai](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels):
  CC0 tag; YOLO boxes, including machine-proposed annotations, not text labels.
- [Ankitbarai](https://www.kaggle.com/datasets/ankitbarai507/indian-vehicle-license-plates):
  CC0 tag; image/bounding-box text files, not OCR text.
- [TKM22092](https://www.kaggle.com/datasets/tkm22092/indian-number-plate-images):
  CC0 tag; detection images, boxes and weights. Not a transcription corpus.
- [Deepakat002](https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation):
  CC0 tag; number-plate boxes, not transcriptions.
- [Santosh collection](https://www.kaggle.com/datasets/santoshvishwakarma99/indian-license-plates)
  and [same-owner mirror](https://www.kaggle.com/datasets/santoshvishwakarma99/indian-license-plate-dataset):
  CC0 tags, matching metadata/sample files; HSRP/non-HSRP scene collections.
  A complete image-to-text table was not established. Don't sum mirrors as new data.
- [Roboflow yolomdata](https://universe.roboflow.com/yolomdata/indian-license),
  [Indian license plate](https://universe.roboflow.com/license-plate-3w6fl/indian-license-plate-h6fll)
  and [Indian plate detection](https://universe.roboflow.com/license-plate-detection-khhkb/indian-license-plate-detection-6tmbr):
  detection projects; text-label verification/reannotation is still needed.
- [NPDS](https://data.mendeley.com/datasets/sxrnr7hwtk/1): CC BY 4.0, plate
  detection with augmented images, not a verified full-plate OCR table.
- [DATS_2022](https://data.mendeley.com/datasets/nfc34n8svj/2): CC BY 4.0 Indian
  road scenes/video-derived frames, mainly vehicle detection/classification.
  Useful future temporal annotation material, not currently plate-text-labelled.
- [ThirdEye Indian road data](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset):
  CC BY 4.0 card, gated road/vehicle annotations, not a public OCR-labelled corpus.

### Licence, provenance or availability prevents inclusion

- [DataCluster Kaggle sample](https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset)
  says CC0, whereas the [publisher's HF mirror](https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset)
  says CC BY-NC-ND 4.0 and its [official repository](https://github.com/datacluster-labs/Indian-Licence-Plate-Image-Dataset)
  describes limited samples/full-data requests. Conflicting terms are not resolved
  by picking the most permissive mirror. The same-owner
  [EV collection](https://www.kaggle.com/datasets/dataclusterlabs/electric-vehicle-number-plate-image-dataset)
  also needs explicit permission/provenance clarification.
- [SaiSirisha](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset):
  CC BY-NC-ND 4.0; not automatically admitted for cropped derivative training.
  [Annotated derivative](https://www.kaggle.com/datasets/vedantgupta007/indian-vehicle-license-plate-dataset-result)
  has an Apache tag but references this restrictive upstream source; excluded.
- [Thamizhsterio](https://www.kaggle.com/datasets/thamizhsterio/indian-license-plates),
  [Gauravsanwal](https://www.kaggle.com/datasets/gauravsanwal/indian-licence-plate),
  [character segmentation](https://www.kaggle.com/datasets/thekeh/indian-license-plate-letter-segmentation-dataset),
  [Siddhesh synthetic](https://www.kaggle.com/datasets/siddheshmm/synthetic-license-plates-indian),
  [Vishanktomar](https://www.kaggle.com/datasets/vishanktomar/indian-car-number-plate-detection),
  [Shaikhshahid](https://www.kaggle.com/datasets/shaikhshahid/indian-car-number-plate)
  and [Dataturks](https://www.kaggle.com/datasets/dataturks/vehicle-number-plate-detection):
  unknown licence in the inspected official metadata, or insufficient text-label
  and reuse information. [Sunrajbishnolia](https://www.kaggle.com/datasets/sunrajbishnolia/license-plate-detection)
  has custom/other terms; [Ckay16](https://www.kaggle.com/datasets/ckay16/indian-number-plate-detection)
  and [Bellair](https://www.kaggle.com/datasets/bellair/indian-licence-plates)
  retain original-author image rights rather than a clear permissive image licence.
- [Chiraggodaw resized](https://www.kaggle.com/datasets/chiraggodaw/license-plate-dataset-resized):
  share-alike tag, OCR label mapping/provenance not established; not silently mixed.
- [Overfitowl synthetic mirror](https://huggingface.co/datasets/overfitowl/synthetic-indian-licenseplates-dataset):
  no declared licence in inspected card metadata; possible duplicated synthetic
  coverage does not justify another unreviewed download.
- [Thundarstrom OCR corpus](https://huggingface.co/datasets/thundarstrom/indian-anpr-ocr-corpus):
  advertises 18,537 crops but inspected API reports zero stored image data and
  only README/attributes files. Advertised size is not usable local data.
- [Military/civilian plates](https://huggingface.co/datasets/adityaver/Indian-Military_and_Civilian-Plates)
  and [paired HR/LR plates](https://huggingface.co/datasets/adityamangal98/HR-LR-images-of-the-Indian-License-plates):
  no clear image reuse licence; former also gated. Paired images would be useful
  for restoration, but cannot be merged merely because the files are visible.
- [UniqueData](https://huggingface.co/datasets/UniqueData/license_plates) and
  [UniDataPro](https://huggingface.co/datasets/UniDataPro/license-plate-detection):
  noncommercial/no-derivatives metadata and non-Indian focus; excluded.
- [AY000554 OCR](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset):
  Russian-oriented collection, not an Indian-domain addition for this run.
- [Fayazhussain ALPR](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition):
  reports a hand-labelled Indian/German test sample from permitted Roboflow data,
  but inspected tree did not expose the paired labelled crop corpus. Code and
  evaluation summaries are not downloadable training images.
- [INDO-ALPR](https://data.mendeley.com/datasets/j3256db3fz/1): the inspected record
  reports 6,174 tropical traffic images but does not establish Indian plates.
  The similar name alone is not evidence of Indian-domain suitability; not selected.

This is a broad targeted survey, not a guarantee of every dataset on the internet.

## Reproduction

Acquisition uses Python's standard library. Corpus export needs Pillow;
Bapatla workbook parsing also needs openpyxl. Automatic scene localisation uses
the existing cached FastALPR detector and its installed runtime.

```text
python tools/acquire_plate_corpus.py --root data/benchmarks/indian-corpus --audit
python tools/acquire_plate_corpus.py --root data/benchmarks/indian-corpus --kaggle abtexp/synthetic-indian-license-plates
python tools/acquire_plate_corpus.py --root data/benchmarks/indian-corpus --kaggle umar1103/final-licence
python tools/acquire_plate_corpus.py --root data/benchmarks/indian-corpus --hf-trocr
python tools/acquire_plate_corpus.py --root data/benchmarks/indian-corpus --bapatla
python -m sentinel_edge.plate_corpus --root data/benchmarks/indian-corpus --sources abtexp umar trocr bapatla --pilot data/benchmarks/bapatla/training_manifest.json --regression data/benchmarks/nivu/manifest.json --viewer data/benchmarks/reports --detect
```

Use `PYTHONPATH=edge` from the repo root, or the established isolated Docker
research environment. `--extend --sources <new-source>` incorporates only a new
source into an integrity-checked completed manifest; already-merged sources are
rejected so observation counts are not accidentally inflated. Downloads can be
resumed. Bapatla's precise bounded range mode recovered stalled long transfers.
Keep at least 4 GiB free; preparation never deletes other files to make space.

The research container needs **both** existing caches: models at `/models`
and `sentinel-prototype_alpr_cache` read-only at `/root/.cache`. The latter
contains FastALPR detector/OCR assets; the model volume alone does not. The small
openpyxl/et-xmlfile dependencies were installed only in
`/models/experiments/corpus-deps`, added to this preparation process's import path,
not to the running production worker. The detector export runs with networking
disabled and no live `.env`.

After the merge, verify every exported file and native pixel hash, licence
reference, registration split and training-index membership:

```text
python -m sentinel_edge.plate_corpus --root data/benchmarks/indian-corpus --verify-only
```

`integrity_check.json` records this result. It verifies file/data consistency,
not the truth of publisher transcriptions. The source-balanced gallery is at
[the existing loopback viewer](http://127.0.0.1:7777/indian-corpus.html).
