"""FastAPI backend for the Explainable Crop Disease Detection project.

Serves the web frontend (src/api/static) at "/" and the JSON API used by it.

Run:
    uvicorn src.api.main:app --port 8000
    open http://localhost:8000          (frontend)
    open http://localhost:8000/docs     (API docs)

All heavy endpoints are plain `def` functions, so FastAPI runs them in a
thread pool and the server stays responsive. A lock lets only one of them
use the models at a time (Grad-CAM, SHAP and LIME are not safe to run on one
model from two threads).
"""
from __future__ import annotations

import base64
import io
import itertools
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import CFG, SUPPORTED_MODELS
from src.compare_explanations import heatmap_iou
from src.correlate.analyze import load_encoder
from src.correlate.link import ReferenceIndex, embed, raw_embed
from src.correlate.maps import METHODS, compute_all_maps
from src.data.dataset import build_transforms, denormalize
from src.explain.common import load_model, predict
from src.explain.gradcam import compute_gradcam

STATIC_DIR = Path(__file__).resolve().parent / "static"
CORR_DIR = CFG.outputs_dir / "correlate"

app = FastAPI(
    title="Explainable Crop Disease Detection API",
    description="CNN crop disease classification with Grad-CAM, SHAP and LIME, "
                "and a learned link between the three explanations.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, tuple] = {}
_ENCODER_CACHE: dict[str, object] = {}
_INDEX_CACHE: dict[str, ReferenceIndex] = {}


def get_cached_model(name: str):
    if name not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unknown model '{name}'. "
                             f"Choose one of {list(SUPPORTED_MODELS)}.")
    if name not in _MODEL_CACHE:
        ckpt_path = CFG.checkpoints_dir / f"{name}.pt"
        if not ckpt_path.exists():
            raise HTTPException(
                503, f"No checkpoint for '{name}' at {ckpt_path}. "
                     f"Train it first: `python -m src.train --model {name}`.")
        _MODEL_CACHE[name] = load_model(name)
    return _MODEL_CACHE[name]


def get_cached_encoder(name: str):
    if name not in _ENCODER_CACHE:
        try:
            _ENCODER_CACHE[name] = load_encoder(name)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _ENCODER_CACHE[name]


def get_optional_encoder(name: str):
    try:
        return get_cached_encoder(name)
    except HTTPException:
        return None


def get_reference_index(name: str):
    """Cached maps of other images, for nearest-map lookups. None if there
    are none for this model."""
    if name not in _INDEX_CACHE:
        try:
            _INDEX_CACHE[name] = ReferenceIndex(get_optional_encoder(name), name)
        except (FileNotFoundError, ValueError):
            return None
    return _INDEX_CACHE[name]


def image_bytes_to_tensor(raw: bytes):
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image.")
    tensor = build_transforms(train=False)(pil_img).unsqueeze(0)
    rgb_float = np.array(denormalize(tensor[0])).transpose(1, 2, 0)
    return tensor, rgb_float


def array_to_base64_png(arr: np.ndarray) -> str:
    """arr: HxW, HxWx3 uint8, or float in [0,1]."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Upload must be a JPEG or PNG image.")
    return file.file.read()


def overlay_map(rgb_float: np.ndarray, small_map: np.ndarray) -> np.ndarray:
    """Draws a 64x64 explanation map over the 224x224 image."""
    size = rgb_float.shape[0]
    big = cv2.resize(small_map.astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC)
    # Display only: stretch so the brightest 1% of pixels reach the top of the
    # colour scale. Sparse maps (SHAP) are otherwise nearly invisible. The
    # numbers returned by /correlate use the unstretched maps.
    hi = float(np.percentile(big, 99))
    big = np.clip(big / hi if hi > 1e-8 else big, 0, 1)
    return show_cam_on_image(rgb_float.astype(np.float32), big, use_rgb=True)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- schemas --
class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


class GradCamResponse(BaseModel):
    model: str
    predictions: list[Prediction]
    heatmap_overlay_png_base64: str


class CompareResponse(BaseModel):
    mobilenet_v2_prediction: str
    resnet50_prediction: str
    iou: float
    mobilenet_v2_overlay_png_base64: str
    resnet50_overlay_png_base64: str


# ------------------------------------------------------------------ routes --
@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_checkpoints": [
            n for n in SUPPORTED_MODELS
            if (CFG.checkpoints_dir / f"{n}.pt").exists()
        ],
        "available_encoders": [
            n for n in SUPPORTED_MODELS
            if (CORR_DIR / "checkpoints" / f"encoder_{n}.pt").exists()
        ],
    }


@app.get("/models")
def list_models():
    return {
        name: {
            "label": label,
            "checkpoint_ready": (CFG.checkpoints_dir / f"{name}.pt").exists(),
            "encoder_ready": (CORR_DIR / "checkpoints" / f"encoder_{name}.pt").exists(),
        }
        for name, label in SUPPORTED_MODELS.items()
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(model: str, file: UploadFile = File(...)):
    raw = read_upload(file)
    net, class_names = get_cached_model(model)
    tensor, _ = image_bytes_to_tensor(raw)
    with _LOCK:
        topk = predict(net, tensor, class_names, topk=3)
    return PredictResponse(
        model=model,
        predictions=[Prediction(label=l, confidence=c) for l, c in topk])


@app.post("/explain/gradcam", response_model=GradCamResponse)
def gradcam_endpoint(model: str, file: UploadFile = File(...)):
    raw = read_upload(file)
    net, class_names = get_cached_model(model)
    tensor, rgb_float = image_bytes_to_tensor(raw)
    with _LOCK:
        topk = predict(net, tensor, class_names, topk=3)
        heatmap, _ = compute_gradcam(net, tensor)
    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
    return GradCamResponse(
        model=model,
        predictions=[Prediction(label=l, confidence=c) for l, c in topk],
        heatmap_overlay_png_base64=array_to_base64_png(overlay))


@app.post("/explain/shap")
def shap_endpoint(model: str, file: UploadFile = File(...),
                  n_background: int = 30):
    from src.explain.shap_explain import compute_shap, sample_background

    raw = read_upload(file)
    net, class_names = get_cached_model(model)
    tensor, rgb_float = image_bytes_to_tensor(raw)
    with _LOCK:
        background = sample_background(n=n_background)
        attribution, pred_idx = compute_shap(net, tensor, background)

    vmax = float(np.percentile(np.abs(attribution), 99) + 1e-8)
    norm = np.clip((attribution + vmax) / (2 * vmax), 0, 1)
    heat_rgb = np.stack([1 - norm, np.zeros_like(norm), norm], axis=-1)
    return {
        "model": model,
        "prediction": class_names[pred_idx],
        "attribution_png_base64": array_to_base64_png(heat_rgb),
        "note": "Blue supports the predicted class, red opposes it. Simple "
                "colors, no colorbar. Use src/explain/shap_explain.py for the "
                "full figure.",
    }


@app.post("/explain/lime")
def lime_endpoint(model: str, file: UploadFile = File(...),
                  num_samples: int = 300):
    from skimage.segmentation import mark_boundaries

    from src.explain.lime_explain import compute_lime

    raw = read_upload(file)
    net, class_names = get_cached_model(model)
    tensor, rgb_float = image_bytes_to_tensor(raw)
    with _LOCK:
        temp, mask, pred_idx = compute_lime(net, rgb_float, class_names,
                                            num_samples=num_samples)
    return {
        "model": model,
        "prediction": class_names[pred_idx],
        "superpixels_png_base64": array_to_base64_png(
            mark_boundaries(temp / 255.0, mask)),
    }


@app.post("/compare", response_model=CompareResponse)
def compare_endpoint(file: UploadFile = File(...), top_pct: float = 0.25):
    raw = read_upload(file)
    model_a, class_names = get_cached_model("mobilenet_v2")
    model_b, _ = get_cached_model("resnet50")
    tensor, rgb_float = image_bytes_to_tensor(raw)
    with _LOCK:
        heat_a, pred_a = compute_gradcam(model_a, tensor)
        heat_b, pred_b = compute_gradcam(model_b, tensor)
    return CompareResponse(
        mobilenet_v2_prediction=class_names[pred_a],
        resnet50_prediction=class_names[pred_b],
        iou=heatmap_iou(heat_a, heat_b, top_pct),
        mobilenet_v2_overlay_png_base64=array_to_base64_png(
            show_cam_on_image(rgb_float, heat_a, use_rgb=True)),
        resnet50_overlay_png_base64=array_to_base64_png(
            show_cam_on_image(rgb_float, heat_b, use_rgb=True)))


@app.post("/correlate")
def correlate_endpoint(model: str, file: UploadFile = File(...),
                       map_size: int = 64, lime_samples: int = 150):
    """Grad-CAM, SHAP and LIME maps for one image, and how much they match.

    Returns:
      pearson      Pearson correlation between the maps of each pair of
                   methods, for this image (no training involved)
      similarity   cosine similarity of the trained encoder's embeddings for
                   each pair; null if no encoder is trained for this model
      reference    mean value for the same image and for different images over
                   the cached images, so the numbers above can be read
                   against them (from outputs/correlate/analysis/summary)
      nearest      for this image's Grad-CAM map, the closest cached SHAP and
                   LIME maps of other images by Pearson correlation, with the
                   class predicted there
      overlays     the three maps drawn over the image

    Needs cached maps for the reference lists (not for the main numbers):
        python -m src.correlate.build_explanation_dataset --model <model>
    The encoder is optional:
        python -m src.correlate.train_contrastive --model <model>
    """
    raw = read_upload(file)
    net, class_names = get_cached_model(model)
    encoder = get_optional_encoder(model)
    tensor, rgb_float = image_bytes_to_tensor(raw)

    with _LOCK:
        top = predict(net, tensor, class_names, topk=1)
        maps, pred_idx = compute_all_maps(
            net, tensor, rgb_float, class_names,
            map_size=map_size, lime_samples=lime_samples)
        raw_emb = {m: raw_embed(maps[m][None])[0] for m in METHODS}
        enc_emb = ({m: embed(encoder, maps[m][None])[0] for m in METHODS}
                   if encoder is not None else None)

    pairs = list(itertools.combinations(METHODS, 2))
    pearson = {f"{a}-{b}": float(np.dot(raw_emb[a], raw_emb[b])) for a, b in pairs}
    similarity = ({f"{a}-{b}": float(np.dot(enc_emb[a], enc_emb[b])) for a, b in pairs}
                  if enc_emb is not None else None)

    nearest = {}
    index = get_reference_index(model)
    if index is not None:
        for target in ("shap", "lime"):
            nearest[f"gradcam->{target}"] = index.nearest(
                raw_emb["gradcam"], target, space="raw", k=3)

    reference = None
    summary = read_json(CORR_DIR / "analysis" / f"{model}_summary.json")
    if summary and "raw_all_images" in summary:
        ra = summary["raw_all_images"]
        reference = {
            "n_images": ra["n_images"],
            "pearson": {
                f"{a}-{b}": {
                    "same_image": ra["pairs"][f"{a}->{b}"]["mean_sim_same_image"],
                    "other_image": ra["pairs"][f"{a}->{b}"]["mean_sim_other_image"],
                } for a, b in pairs},
        }

    return {
        "model": model,
        "prediction": class_names[pred_idx],
        "confidence": top[0][1],
        "pearson": pearson,
        "similarity": similarity,
        "reference": reference,
        "nearest": nearest,
        "image_png_base64": array_to_base64_png(rgb_float),
        "overlays_png_base64": {
            m: array_to_base64_png(overlay_map(rgb_float, maps[m]))
            for m in METHODS
        },
    }


@app.get("/results")
def results():
    """Numbers from the saved run outputs, for the frontend's Results tab.
    Anything missing on disk comes back as null."""
    out = {"classification": {}, "comparison": None,
           "robustness": {}, "correlation": {}, "cross_validation": {}}
    for name in SUPPORTED_MODELS:
        rep = read_json(CFG.eval_dir / f"{name}_classification_report.json")
        hist = read_json(CFG.checkpoints_dir / f"{name}_history.json")
        if rep:
            out["classification"][name] = {
                "test_accuracy": rep["accuracy"],
                "test_macro_f1": rep["macro_f1"],
                "best_val_accuracy": max(hist["val_acc"]) if hist else None,
                "epochs": len(hist["val_acc"]) if hist else None,
            }
        rob = CFG.robustness_dir / f"{name}_perturbation_summary.csv"
        if rob.exists():
            out["robustness"][name] = pd.read_csv(rob).to_dict("records")
        corr = read_json(CORR_DIR / "analysis" / f"{name}_summary.json")
        if corr:
            out["correlation"][name] = corr
        cv = read_json(CORR_DIR / "analysis" / f"{name}_cross_validation.json")
        if cv:
            out["cross_validation"][name] = cv
    iou = CFG.comparison_dir / "iou_raw.csv"
    if iou.exists():
        d = pd.read_csv(iou)
        out["comparison"] = {"n_images": int(len(d)),
                             "mean_iou": float(d["iou"].mean()),
                             "median_iou": float(d["iou"].median())}
    return out


# Frontend. Mounted last so it does not shadow the routes above.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
