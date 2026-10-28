# 9. IEEE Paper & Presentation

## Final submission checklist (from the assignment brief)

- [ ] Report in IEEE conference format, 6–8 pages
- [ ] Source code (this repo)
- [ ] GitHub repository (public or shared with instructor)
- [ ] 5–7 minute demonstration video
- [ ] Final presentation covering: Problem statement, Objective, Literature
  review, XAI framework, Dataset, Methodology (learning model + XAI
  methods), Explainability analysis, Results and comparison

## Suggested IEEE paper structure

1. **Abstract** — problem, method, headline result (e.g. accuracy of both
   models + the IoU agreement finding + the sanity-check result), 150–200
   words
2. **Introduction** — why crop disease detection matters, why
   explainability matters for agri-tech adoption (a farmer/agronomist needs
   to trust *why*, not just *what*), contributions bullet list
3. **Related Work** — the 10+ paper survey from `01_problem_statement.md`,
   organized into: (a) CNN-based plant disease detection, (b) XAI methods
   for CV (Grad-CAM/SHAP/LIME), (c) trust/robustness evaluation of
   explanations. End with the gap you're filling.
4. **Dataset** — PlantVillage description, EDA figures, the near-duplicate/
   leakage limitation, split methodology (`docs/02_dataset.md`,
   `docs/03_preprocessing_eda.md`)
5. **Methodology**
   - 5a. Model architectures (MobileNetV2, ResNet50), training setup
     (`docs/04_baseline_models.md`)
   - 5b. Explainability methods (Grad-CAM, SHAP, LIME) and how each is
     computed (`docs/05_explainability_methods.md`)
   - 5c. Evaluation protocol: IoU-based model comparison, sanity checks,
     perturbation robustness (`docs/06_model_comparison.md`,
     `docs/07_trust_robustness.md`)
6. **Results**
   - Classification performance table (both models)
   - Explanation comparison (IoU stats, side-by-side figures)
   - Sanity check figure
   - Robustness curves
   - Fairness/per-class correlation
7. **Discussion** — what the results mean, where the two models
   agree/disagree and why that matters beyond accuracy, limitations (lab
   images, split methodology, dataset imbalance)
8. **Conclusion & Future Work** — summary, and a pointer to the PlantDoc
   cross-dataset extension if you didn't have time to run it fully
9. **References** — IEEE numbered format, from the literature survey table

## Presentation deck outline (maps 1:1 to the paper sections)

Problem statement → Objective → Literature review (condensed to 2–3 key
citations) → XAI framework diagram (input → CNN → prediction, with Grad-CAM/
SHAP/LIME branching off as parallel explanation paths) → Dataset →
Methodology (both models) → Explainability analysis (example figures) →
Results and comparison (the IoU + robustness headline numbers) → Dashboard
demo screenshot → Conclusion.

## Demo video (5–7 min) script

1. (30s) Problem + why explainability matters here
2. (1 min) Dataset + a couple of EDA figures
3. (1 min) Both models' accuracy, quick confusion matrix glance
4. (2 min) Live dashboard: upload image → prediction → Grad-CAM → toggle
   SHAP/LIME → model-compare mode with IoU
5. (1 min) Robustness/sanity-check highlight (show the heatmap degrading
   under randomization — it's visually convincing)
6. (30s) Wrap-up / key takeaway
