# 7. Trust, Fairness & Robustness Evaluation

## Trust: sanity checks (`src/robustness/sanity_checks.py`)

Based on Adebayo et al. (2018), "Sanity Checks for Saliency Maps" — the
canonical test for whether an explanation actually depends on what the model
learned, or is just an artifact of the architecture/input edges.

Method: take the trained model, then progressively **randomize its weights**
layer by layer from the output back toward the input (a "cascading
randomization" test), regenerating Grad-CAM at each step. A trustworthy
explanation method should degrade toward visual noise as more layers are
randomized — if the heatmap looks identical on a fully random network, the
explanation isn't actually explaining the *trained* model's reasoning.

```bash
python -m src.robustness.sanity_checks --model mobilenet_v2 \
    --image path/to/example.jpg
```

Outputs `outputs/robustness/sanity_<model>_<image>.png` — a strip of
heatmaps from "fully trained" to "fully randomized," plus a similarity score
(SSIM or rank correlation) between consecutive steps.

## Robustness: perturbation stability (`src/robustness/perturbations.py`)

Applies realistic perturbations to the same input and checks whether both
the **prediction** and the **explanation** stay stable:

- Gaussian blur (simulates poor camera focus)
- Brightness shift (± simulates outdoor lighting variation)
- Additive Gaussian noise (sensor noise)
- Random occlusion (simulates a leaf partially blocked by another leaf —
  the single most agriculturally realistic corruption in this list)

For each perturbation level, records:
- Does the predicted class change? (robustness of the model itself)
- IoU between original-image heatmap and perturbed-image heatmap (robustness
  of the *explanation*, independent of whether the prediction changed)

```bash
python -m src.robustness.run_robustness --model mobilenet_v2
python -m src.robustness.run_robustness --model resnet50
```

Outputs `outputs/robustness/perturbation_curves.png` — accuracy and mean
heatmap-IoU vs. perturbation severity, one line per perturbation type, for
each model. Two models on the same plot makes for a good comparison figure.

## Fairness (reframed as per-class fairness — no protected demographic
attribute exists in this dataset)

Uses the per-class precision/recall from `src/evaluate.py`
(`docs/04_baseline_models.md`) cross-referenced against the class imbalance
data from EDA (`docs/03_preprocessing_eda.md`):

- Does per-class accuracy correlate with per-class training set size?
  (expected: yes, minority classes underperform — quantify with a simple
  correlation coefficient)
- Does explanation quality (heatmap IoU-with-itself under mild perturbation,
  as a proxy for explanation stability) also degrade for minority classes,
  or only raw accuracy?

This analysis is just aggregation over outputs already produced by
`evaluate.py` + `run_robustness.py` — no new model code needed, write it up
directly from `outputs/eval/*.csv` and `outputs/robustness/*.csv`.

## What to report in the paper

- Sanity check figure (heatmap degrading under weight randomization) — this
  is a strong, visually convincing "we didn't just assume Grad-CAM works"
  result
- Perturbation-robustness curves, both models overlaid
- One paragraph + 1 small table on the per-class fairness correlation
- Honest limitations: PlantVillage's lab-condition images mean robustness
  results here are optimistic relative to real field deployment (tie back to
  the PlantDoc note in `02_dataset.md` if you ran that cross-dataset check)
