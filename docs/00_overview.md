# Overview


## Chosen domain & topic

**Agriculture → Crop Disease Detection.** Also listed under "Computer Vision"
as "Plant Disease Detection" — it's the same task and the natural fit for
CNNs + Grad-CAM, since Grad-CAM was built specifically for convolutional
architectures and disease detection is inherently about *where* on the leaf
the model is looking.

## How the workflow maps to this repo

| # | Workflow step | Doc | Code |
|---|---|---|---|
| 1 | Literature survey | [`01_problem_statement.md`](01_problem_statement.md) | — |
| 2 | Dataset selection | [`02_dataset.md`](02_dataset.md) | `src/data/download.py` |
| 3 | Preprocessing & EDA | [`03_preprocessing_eda.md`](03_preprocessing_eda.md) | `src/data/dataset.py`, `src/data/eda.py` |
| 4 | Baseline model | [`04_baseline_models.md`](04_baseline_models.md) | `src/models/factory.py`, `src/train.py`, `src/evaluate.py` |
| 5 | Explainability | [`05_explainability_methods.md`](05_explainability_methods.md) | `src/explain/*.py` |
| 6 | Comparison across 2 models | [`06_model_comparison.md`](06_model_comparison.md) | `src/compare_explanations.py` |
| 7 | Trust, fairness, robustness | [`07_trust_robustness.md`](07_trust_robustness.md) | `src/robustness/*.py` |
| 8 | Frontend and backend | [`08_dashboard.md`](08_dashboard.md) | `src/api/main.py`, `src/api/static/index.html` |
| 9 | Paper & presentation | [`09_paper_presentation.md`](09_paper_presentation.md) | — |
| — | Rubric / timeline mapping | [`10_rubric_timeline.md`](10_rubric_timeline.md) | — |
| — | Linking the three methods (Siamese, SupCon): how to run | [`11_explanation_correlation.md`](11_explanation_correlation.md) | `src/correlate/*.py` |
| — | Linking the three methods: idea and results | [`12_linking_xai_methods.md`](12_linking_xai_methods.md) | — |
| — | Linking the three methods: technical specification | [`13_correlation_model_spec.md`](13_correlation_model_spec.md) | — |
| — | What was set out, what was done | [`14_project_outcomes.md`](14_project_outcomes.md) | — |
| — | Log of the run, bugs found | [`RUN_LOG.md`](RUN_LOG.md) | — |

## Read order

Start with `14` for the summary. Then `01` to `08` in order, `12` and `13` for
the linking work (added after the instructor's feedback), `09` for the paper and
slides, `10` for the rubric and timeline.
