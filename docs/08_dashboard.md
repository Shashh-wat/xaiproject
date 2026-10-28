# 8. Frontend and backend

## Run

```bash
uvicorn src.api.main:app --port 8000
```

Open <http://localhost:8000>. API docs are at <http://localhost:8000/docs>.
Set `FORCE_CPU=1` before the command if the GPU path misbehaves.

The page is one file, `src/api/static/index.html`, with no build step. The
FastAPI app serves it at `/` and answers the page's requests.

## Tabs

| Tab | What it does | Calls |
|---|---|---|
| Explain | Upload a leaf image, pick a model. Shows top-3 predictions and Grad-CAM at once. Then runs SHAP and LIME (about a minute on CPU) and shows the three maps, the similarity between each pair, and the closest cached maps from the other methods | `/explain/gradcam`, `/correlate` |
| Compare models | Grad-CAM from both models on one image, and their overlap (IoU) | `/compare` |
| Results | Accuracy table, Grad-CAM overlap between models, how well the three methods link on held-out images, robustness charts | `/results` |
| How it works | Short description and links to the docs | none |

The Explain tab needs a trained encoder for the chosen model
(`docs/11_explanation_correlation.md`). Without one it shows Grad-CAM only.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /models` | which classifiers and encoders are on disk |
| `POST /predict?model=` | top-3 classes |
| `POST /explain/gradcam?model=` | top-3 classes and Grad-CAM overlay |
| `POST /explain/shap?model=`, `POST /explain/lime?model=` | single-method views |
| `POST /compare` | Grad-CAM from both models and the IoU |
| `POST /correlate?model=` | all three maps, similarity between them, nearest cached maps. Fields are listed in [`13_correlation_model_spec.md`](13_correlation_model_spec.md) section 7 |
| `GET /results` | numbers read from `outputs/` |

Uploads must be JPEG or PNG. Anything else returns 400.

Requests that use the models run one at a time (a lock in `main.py`).
The server is meant for one user on one machine. CORS is open (`*`); tighten it
before putting the server anywhere else.

## Streamlit

`src/dashboard/app.py` is the first dashboard. It still runs
(`streamlit run src/dashboard/app.py`) but the web frontend above replaces it.
