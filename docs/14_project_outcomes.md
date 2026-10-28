# 14. Project outcomes

What the assignment asked for, what was done, and what the results say.
Details and sources for every number: [`RUN_LOG.md`](RUN_LOG.md).
Source for the list of steps: `XAI_Domain_Wise_Project.docx`, section "Common
Workflow" and "Final Submission".

## In one paragraph

Two CNNs classify crop leaves in the Kaggle PlantVillage copy (15 classes,
20,637 images). Grad-CAM, SHAP and LIME explain them. A web frontend with a
FastAPI backend runs the models and shows the three explanations for an
uploaded leaf. After the instructor's feedback, the project also tried to link
the three kinds of explanation with a Siamese encoder trained with SupCon. The
link between the methods exists but is weak (Grad-CAM and LIME the strongest),
and the trained encoder did not improve on plain correlation of the maps. The
IEEE paper, the demo video and the literature survey are not done.

## The nine workflow steps

| # | Step in the brief | Status | Evidence |
|---|---|---|---|
| 1 | Literature survey (10+ papers) | **Not done.** Doc 01 names four well-known papers from memory; none was read for this project and there is no table of 10+ papers | `01_problem_statement.md` |
| 2 | Dataset | Done. 15 classes, 20,637 images (not the 38 classes first assumed) | `02_dataset.md` |
| 3 | Preprocessing and EDA | Done. 224×224, ImageNet normalisation, light augmentation; class counts, imbalance (21 times), sample grid | `03_preprocessing_eda.md`, `outputs/eda/` |
| 4 | Baseline model | Done. MobileNetV2 and ResNet50, linear head only, 3 epochs. Test accuracy 0.918 and 0.889 | `04_baseline_models.md`, RUN_LOG |
| 5 | Explainability | Done. Grad-CAM, SHAP, LIME. A SHAP bug (wrong maps with shap 0.49) was found and fixed. The "any new method" part is the linking encoder, below | `05_explainability_methods.md` |
| 6 | Compare explanation methods on 2 models | **Partly.** Grad-CAM was compared across both models (75 images, mean IoU 0.435). SHAP and LIME maps exist for MobileNetV2 only | `06_model_comparison.md`, RUN_LOG |
| 7 | Trust, fairness, robustness | Done at small scale. Sanity check on 1 image and 1 model. Four image changes on 20 images per model. Per-class check: no clear link between class size and recall | `07_trust_robustness.md`, RUN_LOG |
| 8 | Dashboard | Done. Web frontend at `/`, FastAPI backend, older Streamlit page | `08_dashboard.md` |
| 9 | IEEE paper and presentation | **Not done.** An outline exists. A slide deck drafted earlier has placeholder numbers and is not part of the repo | `09_paper_presentation.md` |

Final submission items: source code and GitHub repository are done
(<https://github.com/Shashh-wat/xaiproject>). The 6–8 page IEEE paper, the 5–7
minute demo video and the final presentation are not done.

## The linking work

The ask, in the team's words: the explanations from SHAP, LIME and Grad-CAM
need to be correlated, "something like sup-con or siamese learning". Doc 12
explains how that was read and what was found. In short:

| Question | Answer | Where |
|---|---|---|
| Can the three kinds of explanation be put in a form that can be compared? | Yes. Each becomes a 64×64 map of the evidence for the predicted class | spec 13, section 1 |
| Do the methods agree on where the evidence is? | A little. Grad-CAM and LIME: correlation 0.23 for the same image, 0.09 for different images. Grad-CAM and SHAP: 0.14 against 0.11. SHAP and LIME: 0.04 against 0.03 | doc 12, result 1 |
| Does a learned shared space (Siamese, SupCon) link them better than plain correlation? | **No.** Six settings, 5-fold cross-validation: top-1 same image 0.075 to 0.117, against 0.092 untrained, 0.192 plain correlation, 0.083 chance | doc 12, result 2 |
| Why? | Not proven. The embeddings group by method, not by class. Likely causes: 48 training images per fold, and noisy SHAP maps | doc 12 |

What this gives the project: a tested way to measure how much two
explanation methods agree on an image (it is in the Explain tab and the API),
a negative result for the learned encoder with the evidence behind it, and a
list of what to change to test it fairly (doc 12, last section).

## How the numbers were kept honest

- Every number in the docs is copied from a file under `outputs/`, named next
  to it.
- Encoders were tested on images they were not trained on, in 5 folds, against
  an untrained encoder, plain correlation and chance. All settings that were
  run are in the table; none was left out.
- A first run showed the encoder at chance. The maps were looked at, which
  led to the SHAP bug. After the fix, the same grid was run again and its
  results are the ones reported.
- The 12-image single split first used gave different numbers from the
  cross-validation (for example top-1 same image 0.15 for the trained encoder in
  one fold, 0.117 at best over five). With 12 images, a single split is not
  reliable.

## Known limits

- Few images: 60 for the linking work, 20 for the robustness runs, 75 for the
  model comparison. Differences of a few points are noise.
- One classifier (MobileNetV2) for the linking work. One image for the sanity
  check.
- The classifiers are trained for 3 epochs with a frozen backbone. They are
  baselines, not tuned models. ResNet50 was still improving.
- PlantVillage images are lab photos on plain backgrounds. Results will not
  carry to field photos.
- SHAP changes slightly from run to run.

## What is left

1. Read and tabulate 10+ papers (step 1).
2. Write the IEEE paper and the final presentation, using the real numbers in
   RUN_LOG (step 9). Use the negative result for the encoder as it is.
3. Record the demo video from the frontend.
4. If a stronger result for the learned link is wanted: more images and a
   less noisy SHAP (doc 12), then ResNet50.
