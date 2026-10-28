# 3. Preprocessing & EDA

## Preprocessing (`src/data/dataset.py`)

- Resize to 224×224 (standard input size for both MobileNetV2 and ResNet50
  ImageNet backbones).
- Normalize with ImageNet mean/std (`[0.485,0.456,0.406]` /
  `[0.229,0.224,0.225]`) since both models start from ImageNet-pretrained
  weights (transfer learning — see `docs/04_baseline_models.md`).
- Train-time augmentation: random horizontal flip, small rotation (±15°),
  mild color jitter (brightness/contrast only — **not** hue, since hue shift
  can mimic or erase disease discoloration and would corrupt the label
  semantics).
- Val/test: resize + normalize only, no augmentation.

## EDA (`src/data/eda.py`)

Run with:
```bash
python -m src.data.eda
```

Produces, into `outputs/eda/`:
- `class_counts.png` — bar chart of images per class (PlantVillage is
  imbalanced — some classes have 150 images, others 5,000+; this matters
  later for the per-class fairness analysis in step 7)
- `class_counts.csv` — raw numbers for the report
- `sample_grid.png` — a few sample images per crop species, for the paper's
  dataset figure
- `image_stats.json` — mean/std image dimensions, channel stats, total image
  count, class count

## What to report in the paper

- Total images, number of classes, class imbalance ratio (max/min class
  count)
- The near-duplicate/leakage caveat from `02_dataset.md`
- 1 figure: sample grid; 1 figure: class distribution bar chart
