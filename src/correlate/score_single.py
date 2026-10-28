"""Single-image inference: compute Grad-CAM/SHAP/LIME maps for one image,
embed them with the trained correlation encoder, and return the pairwise
cosine-similarity ("agreement") scores. This is what the FastAPI /correlate
endpoint and the dashboard's "Explanation Correlation" tab both call.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from src.config import DEVICE
from src.correlate.analyze import load_encoder
from src.correlate.maps import METHODS, compute_all_maps


def score_image(model, tensor, rgb_float, class_names, encoder_model_name: str,
                 map_size: int = 64, lime_samples: int = 300):
    """Returns (maps: dict[str, np.ndarray], pred_idx: int,
    similarities: dict[str, float]) for one image.

    `encoder_model_name` selects which trained encoder checkpoint to use
    (encoders are trained per-backbone, since MobileNetV2 and ResNet50
    Grad-CAM maps have different statistics)."""
    maps, pred_idx = compute_all_maps(model, tensor, rgb_float, class_names,
                                       map_size=map_size,
                                       lime_samples=lime_samples)

    encoder = load_encoder(encoder_model_name)
    with torch.no_grad():
        embeddings = {}
        for method in METHODS:
            m = torch.from_numpy(maps[method]).unsqueeze(0).unsqueeze(0).float()
            embeddings[method] = encoder(m.to(DEVICE))[0].cpu().numpy()

    similarities = {}
    for m1, m2 in itertools.combinations(METHODS, 2):
        similarities[f"{m1}-{m2}"] = float(np.dot(embeddings[m1], embeddings[m2]))

    return maps, pred_idx, similarities
