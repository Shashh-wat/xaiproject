# 1. Problem Statement & Literature Survey

## Problem statement

> Build an explainable convolutional neural network (CNN) pipeline for crop
> leaf disease classification that not only predicts the disease class but
> also justifies its prediction visually (Grad-CAM, SHAP) and textually
> (LIME), and quantify how trustworthy and robust those explanations are
> under two different CNN architectures and realistic input perturbations.

## Objective

1. Classify crop leaf images into healthy / diseased categories (15 classes
   in the Kaggle PlantVillage copy used here: Pepper, Potato and Tomato; see
   `02_dataset.md`).
2. Explain individual predictions with Grad-CAM, SHAP, and LIME.
3. Compare explanation quality/agreement between a lightweight CNN
   (MobileNetV2) and a deeper CNN (ResNet50).
4. Evaluate whether explanations are trustworthy (sanity checks against a
   randomly-initialized network) and robust (stable under blur, brightness
   shift, occlusion, noise).
5. Ship a dashboard for interactive inspection and an IEEE-format writeup.

## Why this fits CNNs + XAI well

- Disease symptoms are localized (lesions, spots, discoloration) — Grad-CAM's
  spatial heatmap directly answers "did the model look at the lesion?"
- PlantVillage is a large, clean, well-cited public dataset — low friction to
  get a working baseline in week 1–2, leaving more time for the XAI/
  robustness analysis the rubric actually weights heavily (9 of the listed
  22 marks + "innovation").
- Two architecturally different backbones (shallow/mobile vs. deep/residual)
  give a genuine axis for the required "comparison of explanation methods
  with 2 different learning models" — not just two random model names.

## Literature survey (target: 10+ papers)

Search terms to use on IEEE Xplore / Google Scholar / arXiv:
- "explainable plant disease detection CNN"
- "Grad-CAM crop disease classification"
- "PlantVillage deep learning interpretability"
- "SHAP image classification agriculture"
- "trustworthy AI plant pathology"
- "sanity checks for saliency maps" (Adebayo et al., 2018 — canonical paper
  for the trust experiment in step 7)
- "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization" (Selvaraju et al., 2017 — the base method itself)
- "LIME: Why Should I Trust You?" (Ribeiro et al., 2016)
- "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee,
  2017 — SHAP)

Track the survey in a simple table (paper, year, dataset used, model,
XAI method, key finding, gap it leaves) — this table drops directly into the
IEEE paper's Related Work section. Keep it as `docs/literature_survey.md` or
a spreadsheet once you start reading; not scaffolded here since it's manual
research work, not code.

## Deliverable checklist for this step

- [ ] 10+ papers logged with the table above
- [ ] 2–3 sentence summary of the gap this project addresses (most
  PlantVillage papers report accuracy only, few evaluate explanation
  robustness/trustworthiness — that's the angle)
