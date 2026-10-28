"""Shared helpers for computing normalized, fixed-size explanation maps
from Grad-CAM, SHAP, and LIME on a single image -- the common representation
the contrastive correlation model (encoder.py) is trained on.

Each map is reduced to a single-channel [map_size, map_size] array in
[0, 1] that shows evidence for the predicted class, so all three methods -- despite very different native outputs
(a gradient-based heatmap, a signed pixel attribution, a superpixel
segmentation) -- become directly comparable inputs to one shared encoder.
"""
from __future__ import annotations

import numpy as np
from skimage.transform import resize

from src.config import CFG
from src.explain.gradcam import compute_gradcam
from src.explain.lime_explain import compute_lime_dense
from src.explain.shap_explain import compute_shap, sample_background

METHODS = ("gradcam", "shap", "lime")


def normalize01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    m = m - m.min()
    denom = m.max()
    if denom > 1e-8:
        m = m / denom
    return m


def resize_map(m: np.ndarray, size: int) -> np.ndarray:
    return resize(m, (size, size), anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def compute_all_maps(model, tensor, rgb_float, class_names,
                      map_size: int = 64,
                      lime_samples: int = 300,
                      shap_background=None):
    """Returns ({"gradcam": [S,S], "shap": [S,S], "lime": [S,S]}, pred_idx).

    All three maps are computed against the *model's own predicted class*
    for this image (not the ground truth), since we're asking "what does
    each method say justified THIS prediction", not grading correctness.
    """
    heat, pred_idx = compute_gradcam(model, tensor)
    gradcam_map = resize_map(normalize01(heat), map_size)

    background = (shap_background if shap_background is not None
                  else sample_background(n=30))
    # All three maps show evidence FOR the predicted class: Grad-CAM is
    # already non-negative (ReLU), so SHAP and LIME keep only their positive
    # values. Evidence against the class is dropped.
    shap_attr, _ = compute_shap(model, tensor, background)
    shap_map = resize_map(normalize01(np.maximum(shap_attr, 0)), map_size)

    lime_attr, _ = compute_lime_dense(model, rgb_float, class_names,
                                       num_samples=lime_samples)
    lime_map = resize_map(normalize01(np.maximum(lime_attr, 0)), map_size)

    return {"gradcam": gradcam_map, "shap": shap_map, "lime": lime_map}, pred_idx
