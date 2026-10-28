"""Caches Grad-CAM / SHAP / LIME maps for a sample of test images, so the
contrastive encoder can be trained without recomputing SHAP/LIME (the slow
methods) on every epoch.

Usage:
    python -m src.correlate.build_explanation_dataset --model mobilenet_v2 --n_images 150
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG
from src.data.dataset import build_datasets
from src.explain.common import load_image_tensor, load_model
from src.explain.shap_explain import sample_background
from src.correlate.maps import METHODS, compute_all_maps

MAPS_DIR = CFG.outputs_dir / "correlate" / "maps"
META_PATH = CFG.outputs_dir / "correlate" / "metadata.csv"


def build(model_name: str, n_images: int, map_size: int, lime_samples: int,
          seed: int = 42):
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    model, class_names = load_model(model_name)

    _, _, test_ds, _ = build_datasets()
    random.seed(seed)
    idxs = random.sample(range(len(test_ds.paths)),
                          min(n_images, len(test_ds.paths)))

    background = sample_background(n=30)  # shared SHAP reference distribution
    rows = []
    for i, idx in enumerate(tqdm(idxs, desc="caching explanation maps")):
        path, true_label = test_ds.paths[idx], test_ds.labels[idx]
        tensor, rgb_float = load_image_tensor(path)
        maps, pred_idx = compute_all_maps(
            model, tensor, rgb_float, class_names,
            map_size=map_size, lime_samples=lime_samples,
            shap_background=background,
        )
        for method in METHODS:
            fname = f"{model_name}_{i:04d}_{method}.npy"
            np.save(MAPS_DIR / fname, maps[method])
            rows.append({
                "sample_id": i,
                "method": method,
                "file": fname,
                "model": model_name,
                "path": str(path),
                "true_class": class_names[true_label],
                "pred_class": class_names[pred_idx],
            })

    df = pd.DataFrame(rows)
    if META_PATH.exists():
        existing = pd.read_csv(META_PATH)
        existing = existing[existing["model"] != model_name]  # replace this model's rows
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(META_PATH, index=False)
    print(f"[correlate] cached {len(idxs)} images x {len(METHODS)} methods "
          f"-> {MAPS_DIR}")
    print(f"[correlate] metadata -> {META_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--n_images", type=int, default=150)
    parser.add_argument("--map_size", type=int, default=64)
    parser.add_argument("--lime_samples", type=int, default=300)
    args = parser.parse_args()
    build(args.model, args.n_images, args.map_size, args.lime_samples)


if __name__ == "__main__":
    main()
