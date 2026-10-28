# 10. Evaluation Rubric & Semester Timeline (from the assignment brief)

## Rubric mapping

| Component | Marks | Where it's earned in this repo |
|---|---|---|
| Objective, Literature Review, Baseline model implementation | 10 | `docs/01_problem_statement.md` (survey) + `docs/04_baseline_models.md` + `src/train.py`/`src/evaluate.py` |
| Appropriate XAI methods, Explainability — Evaluation & Comparison | 9 | `docs/05_explainability_methods.md`, `docs/06_model_comparison.md`, `docs/11_explanation_correlation.md`, `src/explain/*`, `src/compare_explanations.py`, `src/correlate/*` |
| Presentation and Report without Plagiarism | 8 | `docs/09_paper_presentation.md` — write-up work, keep citations honest |
| Any Innovation or productive outcome | 3 | The linking work (`docs/12_linking_xai_methods.md`): a way to measure how much Grad-CAM, SHAP and LIME agree, and a Siamese / SupCon encoder tested on held-out images. Result: the methods agree a little; the encoder did not beat plain correlation. Report it as a tested negative result |
| **Total** | **30** | |

## Semester timeline mapping

| Weeks | Brief's activity | This repo |
|---|---|---|
| 1–2 | Domain selection and proposal | Done — see `docs/00_overview.md`, `docs/01_problem_statement.md` |
| 3–4 | Dataset and preprocessing | `src/data/download.py`, `src/data/dataset.py`, `src/data/eda.py` |
| 5–6 | Model development | `src/models/factory.py`, `src/train.py`, `src/evaluate.py` |
| 7–8 | Explainability | `src/explain/gradcam.py`, `shap_explain.py`, `lime_explain.py` |
| 9–10 | Evaluation | `src/compare_explanations.py`, `src/robustness/*` |
| — | Explanation correlation (added post-feedback) | `src/correlate/*.py`, `docs/11_explanation_correlation.md` |
| 11–12 | Dashboard & GitHub | `src/dashboard/app.py`, `src/api/main.py` (`/correlate` endpoint), push repo, tag a release |
| 13–14 | Paper drafting | Use `docs/09_paper_presentation.md` outline, pull figures from `outputs/` |
| 15 | Presentation & final submission | Deck from `docs/09_paper_presentation.md`, record demo video off the dashboard |

Use this table as the actual sprint plan — each week's target output already
has a named script/doc to point at, so "what do we do this week" should
never be an open question.
