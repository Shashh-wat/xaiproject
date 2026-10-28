"""Linking explanation maps across methods.

Given a map from one method (for example Grad-CAM), find the closest maps
from another method (for example SHAP) in the encoder's embedding space.

Two uses:
  - `retrieval_metrics`: held-out test used by analyze.py. For every image,
    does the Grad-CAM map's nearest SHAP map belong to the same image? The
    same predicted class?
  - `ReferenceIndex`: nearest cached maps for a new image, used by the API.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import DEVICE
from src.correlate.build_explanation_dataset import MAPS_DIR, META_PATH
from src.correlate.maps import METHODS


@torch.no_grad()
def embed(encoder, maps: np.ndarray) -> np.ndarray:
    """maps: [N, S, S] float -> [N, D] L2-normalized embeddings."""
    x = torch.from_numpy(maps).unsqueeze(1).float().to(DEVICE)
    return encoder(x).cpu().numpy()


def raw_embed(maps: np.ndarray) -> np.ndarray:
    """No learning: flatten, subtract the mean, L2-normalize. The dot product
    of two such rows is the Pearson correlation of the two maps."""
    flat = maps.reshape(len(maps), -1).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    return flat / np.maximum(norm, 1e-12)


def load_split(model_name: str, split: str | None, fold: int = 4):
    """Returns (sample_ids, maps{method: [N,S,S]}, classes[N], image_names[N])
    for images that have all three methods cached."""
    from src.correlate.dataset_loader import ExplanationMapDataset

    df = ExplanationMapDataset(model_name=model_name, split=split, fold=fold).df
    ids = sorted(df["sample_id"].unique())
    maps = {m: [] for m in METHODS}
    classes, names = [], []
    kept = []
    for sid in ids:
        rows = df[df["sample_id"] == sid]
        if set(rows["method"]) != set(METHODS):
            continue
        kept.append(sid)
        for m in METHODS:
            maps[m].append(np.load(MAPS_DIR / rows[rows["method"] == m]["file"].iloc[0]))
        classes.append(rows["pred_class"].iloc[0])
        names.append(Path(rows["path"].iloc[0]).name)
    return kept, {m: np.stack(v) for m, v in maps.items()}, classes, names


def retrieval_metrics(emb: dict, classes: list) -> dict:
    """emb: {method: [N, D]} with unit-length rows. For every ordered pair
    (query method a -> target method b) and every image i, rank the N target
    maps by similarity to the query map of image i.

    Reports top-1 / top-3 hit rate for the SAME image, top-1 hit rate for the
    same predicted class, and the mean similarity of same-image pairs versus
    different-image pairs. Chance levels are computed from the same data.
    """
    n = len(classes)
    cls = np.array(classes)
    idx = np.arange(n)
    result = {}
    for a in METHODS:
        for b in METHODS:
            if a == b:
                continue
            sim = emb[a] @ emb[b].T
            order = np.argsort(-sim, axis=1)
            same_image = np.diag(sim)
            off_diag = sim[~np.eye(n, dtype=bool)]
            result[f"{a}->{b}"] = {
                "top1_same_image": float((order[:, 0] == idx).mean()),
                "top3_same_image": float(np.mean([i in order[i, :3] for i in idx])),
                "top1_same_class": float((cls[order[:, 0]] == cls).mean()),
                "mean_sim_same_image": float(same_image.mean()),
                "mean_sim_other_image": float(off_diag.mean()),
                "chance_top1_same_image": 1.0 / n,
                "chance_top3_same_image": min(3, n) / n,
                "chance_top1_same_class": float(
                    np.mean([(cls == cls[i]).mean() for i in idx])),
                "n_images": n,
            }
    return result


class ReferenceIndex:
    """Embeds every cached map once, then answers nearest-map queries.

    Two spaces: "raw" (Pearson correlation of the 64x64 maps, needs no
    training) and "encoder" (only if a trained encoder is given)."""

    def __init__(self, encoder, model_name: str):
        ids, maps, classes, names = load_split(model_name, None)
        self.ids, self.classes, self.names = ids, classes, names
        self.space = {"raw": {m: raw_embed(maps[m]) for m in METHODS}}
        if encoder is not None:
            self.space["encoder"] = {m: embed(encoder, maps[m]) for m in METHODS}

    def nearest(self, query_vec: np.ndarray, target_method: str, space: str = "raw",
                k: int = 3):
        sim = self.space[space][target_method] @ query_vec
        top = np.argsort(-sim)[:k]
        return [{"image": self.names[i], "predicted_class": self.classes[i],
                 "similarity": float(sim[i])} for i in top]
