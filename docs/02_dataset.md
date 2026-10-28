# 2. Dataset

## Primary: PlantVillage

**Correction (confirmed by actually running the download — see
`docs/RUN_LOG.md`):** the Kaggle mirror this project pulls
(`emmarex/plantdisease`) is a **15-class subset covering only Pepper,
Potato, and Tomato** — not the full 38-class/54,000-image academic release
some PlantVillage papers describe. Actual numbers from this mirror:

- **20,638 RGB images, 15 classes** (Pepper × 2, Potato × 3, Tomato × 10 —
  healthy + disease states per crop). E.g. `Tomato_Late_blight`,
  `Potato___healthy`. Note the inconsistent `_`/`___` separator across
  classes in this particular mirror — cosmetic, doesn't affect
  `ImageFolder` loading.
- Class imbalance ratio ≈ **21x** (3,208 images for
  `Tomato__Tomato_YellowLeaf__Curl_Virus` vs. 152 for
  `Potato___healthy`) — worse than typically assumed, see
  `docs/03_preprocessing_eda.md` and the fairness analysis in
  `docs/07_trust_robustness.md`.
- Lab-condition images (uniform background), which makes training fast and
  accuracy high — good for getting a working baseline quickly.
- If your project actually needs the full 38-class/14-species set (with
  Apple, Corn, Grape, etc.), use the GitHub mirror
  `spMohanty/PlantVillage-Dataset` instead and adjust
  `src/config.py:kaggle_dataset` / `download.py` accordingly — this repo
  defaults to the smaller, faster-to-download 15-class mirror.

## Optional stretch / robustness dataset: PlantDoc

- ~2,600 images, 13 species, 17 classes, captured in **real field
  conditions** (cluttered background, varying lighting). Much harder.
- Useful in `docs/07_trust_robustness.md` as an out-of-distribution test set:
  train on PlantVillage, evaluate Grad-CAM/accuracy on PlantDoc images to see
  how much the "lab-condition" model degrades on real photos. This is a
  strong, easy-to-explain robustness result for the paper if you have time.
- Kaggle mirror: `nirmalsankalana/plantdoc-dataset`.

## Getting the data

Requires a free Kaggle account + API token (`~/.kaggle/kaggle.json`, or set
`KAGGLE_USERNAME` / `KAGGLE_KEY` env vars). Then:

```bash
python -m src.data.download
```

This runs `src/data/download.py`, which uses `kagglehub` to pull
`emmarex/plantdesease` into `data/raw/plantvillage/` (path configurable in
`src/config.py`). If you'd rather download manually, unzip into that same
path — the folder structure expected is:

```
data/raw/plantvillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___healthy/
├── ...
└── Tomato___healthy/
```
(i.e. one subfolder per class, standard `ImageFolder` layout — this is
exactly how PlantVillage ships.)

## Splits

`src/data/dataset.py` builds train/val/test splits (70/15/15 by default,
configurable in `src/config.py`). Split is stratified by class and done at
the **file level with a fixed random seed** so results are reproducible.

⚠️ **Known caveat to mention in the paper** (this is a real, citable issue
with PlantVillage): many images are near-duplicates from the same source
photo session, which can leak between train/test splits and inflate reported
accuracy in a lot of published work. We do plain stratified random splitting
here (documented, not hidden) — call this out explicitly as a limitation in
the paper's Discussion/Limitations section rather than claiming
state-of-the-art accuracy. If time allows, a perceptual-hash-based
dedup pass before splitting is a good "innovation" line for the rubric.
