# 6. Comparison of Explanation Methods Across 2 Models

This is the rubric's explicit requirement: "Comparison of explanation
methods with 2 different learning models." Implemented in
`src/compare_explanations.py`.

## What it does

1. Samples N test images per class (default N=5, configurable) that both
   MobileNetV2 and ResNet50 classify **correctly**, plus a smaller set they
   both get **wrong**, so the comparison covers both cases.
2. For each image, runs Grad-CAM (and optionally SHAP) on both models.
3. Computes **IoU (Intersection over Union)** between the two models'
   thresholded heatmaps (top 25% activation region) — a single number per
   image quantifying how much the two architectures "look at" the same
   region.
4. Aggregates IoU by class and produces:
   - `outputs/comparison/iou_by_class.csv`
   - `outputs/comparison/iou_distribution.png` (histogram)
   - `outputs/comparison/side_by_side/<class>_<idx>.png` — grid of
     [original | MobileNetV2 Grad-CAM | ResNet50 Grad-CAM] for qualitative
     inspection

```bash
python -m src.compare_explanations --n_per_class 5 --top_pct 0.25
```

## Interpreting the results

- **High IoU** (models agree on region) + both correct → the disease's
  visual signal is unambiguous, both architectures converge on it. Good sign
  for trust.
- **Low IoU** + both correct → models reach the right answer via different
  evidence — worth flagging as a case where accuracy alone would have hidden
  a meaningful difference, which is the whole point of doing XAI rather than
  just reporting accuracy.
- **Low IoU** + one or both wrong → useful failure-mode examples for the
  paper's Discussion section (pick 1–2 for the "qualitative failure
  analysis" figure).
- Deeper model (ResNet50) generally expected to produce tighter, more
  lesion-localized heatmaps than MobileNetV2 due to greater representational
  capacity — check whether that holds; if it doesn't, that's an interesting
  (reportable) negative result too.

## What to report in the paper

- Mean/median IoU across all sampled images (one number, headline stat)
- IoU-by-class table or bar chart — which classes show most/least agreement
- 2–3 side-by-side qualitative examples (one high-IoU, one low-IoU, one
  failure case)
