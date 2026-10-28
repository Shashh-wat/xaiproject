# Explainable Crop Disease Detection

BCSE418L (Explainable AI) digital assignment. Domain: agriculture. Topic: crop
disease detection with CNNs.

Two CNNs (MobileNetV2, ResNet50) classify leaf images. Grad-CAM, SHAP and LIME
explain the predictions. A small Siamese encoder trained with supervised
contrastive loss (SupCon) links the three kinds of explanation to each other.
A web frontend and a FastAPI backend let you try it on an image.

Start with [`docs/14_project_outcomes.md`](docs/14_project_outcomes.md) for what
was done and the results.

## Docs

| File | Content |
|---|---|
| [`docs/14_project_outcomes.md`](docs/14_project_outcomes.md) | What was set out, what was done, results, limits |
| [`docs/12_linking_xai_methods.md`](docs/12_linking_xai_methods.md) | How learning from Grad-CAM, SHAP and LIME is linked |
| [`docs/13_correlation_model_spec.md`](docs/13_correlation_model_spec.md) | Technical specification of that model |
| [`docs/11_explanation_correlation.md`](docs/11_explanation_correlation.md) | Commands to run it |
| [`docs/RUN_LOG.md`](docs/RUN_LOG.md) | Log of the actual run: numbers, bugs found |
| `docs/00` to `docs/10` | One file per step of the assignment workflow (see [`00_overview.md`](docs/00_overview.md)) |

## Code

```
src/
  config.py             paths and settings
  data/                 download, dataset and split, EDA
  models/factory.py     MobileNetV2 and ResNet50 builders
  train.py, evaluate.py
  explain/              gradcam.py, shap_explain.py, lime_explain.py
  compare_explanations.py   Grad-CAM overlap between the two models
  robustness/           perturbations, sanity check, robustness run
  correlate/            the correlation model (encoder, loss, training, analysis)
  api/main.py           FastAPI backend, also serves the frontend
  api/static/index.html the web frontend
  dashboard/app.py      older Streamlit dashboard
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.download                       # needs a Kaggle token, see docs/02_dataset.md
python -m src.train --model mobilenet_v2 --epochs 3
python -m src.train --model resnet50 --epochs 3
python -m src.evaluate --model mobilenet_v2
python -m src.evaluate --model resnet50
python -m src.compare_explanations
python -m src.robustness.run_robustness --model mobilenet_v2
python -m src.robustness.run_robustness --model resnet50

# correlation model, see docs/11_explanation_correlation.md
FORCE_CPU=1 python -m src.correlate.build_explanation_dataset --model mobilenet_v2 --n_images 60 --lime_samples 150
python -m src.correlate.train_contrastive --model mobilenet_v2
python -m src.correlate.analyze --model mobilenet_v2

# frontend + backend
uvicorn src.api.main:app --port 8000              # then open http://localhost:8000
```

`data/` and `outputs/` (dataset, model files, cached maps, figures) are not in
git. Train the models and build the maps locally.

## Notes

- The Kaggle copy of PlantVillage used here has 15 classes and 20,637 images
  (Pepper, Potato, Tomato). See `docs/02_dataset.md`.
- On macOS, `num_workers` is 0 in `src/config.py` because PyTorch data loader
  workers hung on the test machine.
