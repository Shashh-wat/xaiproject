"""Compare Grad-CAM explanations between MobileNetV2 and ResNet50 across
sampled test images, computing heatmap IoU per image and aggregating by
class. Implements step 6 of the workflow (docs/06_model_comparison.md).

Usage:
    python -m src.compare_explanations --n_per_class 5 --top_pct 0.25
"""
from __future__ import annotations

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import CFG, DEVICE
from src.data.dataset import build_datasets
from src.explain.common import load_model
from src.explain.gradcam import compute_gradcam


def heatmap_iou(h1: np.ndarray, h2: np.ndarray, top_pct: float) -> float:
    """IoU between the top `top_pct` activation regions of two heatmaps."""
    t1 = np.quantile(h1, 1 - top_pct)
    t2 = np.quantile(h2, 1 - top_pct)
    m1, m2 = h1 >= t1, h2 >= t2
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(m1, m2).sum()
    return float(inter / union)


def sample_test_images(n_per_class: int, seed: int = 42):
    _, _, test_ds, class_names = build_datasets()
    random.seed(seed)
    by_class: dict[int, list] = {}
    for path, label in zip(test_ds.paths, test_ds.labels):
        by_class.setdefault(label, []).append(path)
    samples = []
    for label, paths in by_class.items():
        chosen = random.sample(paths, min(n_per_class, len(paths)))
        samples += [(p, label) for p in chosen]
    return samples, class_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_per_class", type=int, default=5)
    parser.add_argument("--top_pct", type=float, default=0.25)
    parser.add_argument("--n_qualitative", type=int, default=6,
                         help="How many side-by-side figures to save.")
    args = parser.parse_args()

    print("[compare] loading both models ...")
    model_a, class_names = load_model("mobilenet_v2")
    model_b, _ = load_model("resnet50")

    print(f"[compare] sampling {args.n_per_class} test images per class ...")
    samples, class_names = sample_test_images(args.n_per_class)
    print(f"[compare] {len(samples)} total images sampled")

    from src.explain.common import load_image_tensor, predict
    from pytorch_grad_cam.utils.image import show_cam_on_image

    rows = []
    saved_qualitative = 0
    for path, label in tqdm(samples):
        input_tensor, rgb_float = load_image_tensor(path)

        heat_a, pred_a = compute_gradcam(model_a, input_tensor)
        heat_b, pred_b = compute_gradcam(model_b, input_tensor)
        iou = heatmap_iou(heat_a, heat_b, args.top_pct)

        rows.append({
            "path": str(path),
            "true_class": class_names[label],
            "pred_mobilenet_v2": class_names[pred_a],
            "pred_resnet50": class_names[pred_b],
            "both_correct": (pred_a == label) and (pred_b == label),
            "iou": iou,
        })

        if saved_qualitative < args.n_qualitative and random.random() < 0.3:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(rgb_float); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(show_cam_on_image(rgb_float, heat_a, use_rgb=True))
            axes[1].set_title(f"MobileNetV2 -> {class_names[pred_a]}")
            axes[1].axis("off")
            axes[2].imshow(show_cam_on_image(rgb_float, heat_b, use_rgb=True))
            axes[2].set_title(f"ResNet50 -> {class_names[pred_b]}")
            axes[2].axis("off")
            fig.suptitle(f"true={class_names[label]}  IoU={iou:.2f}")
            fig.tight_layout()
            out_dir = CFG.comparison_dir / "side_by_side"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{class_names[label]}_{saved_qualitative}.png",
                        dpi=150)
            plt.close(fig)
            saved_qualitative += 1

    df = pd.DataFrame(rows)
    df.to_csv(CFG.comparison_dir / "iou_raw.csv", index=False)

    by_class = df.groupby("true_class")["iou"].agg(["mean", "std", "count"])
    by_class.to_csv(CFG.comparison_dir / "iou_by_class.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["iou"], bins=20)
    ax.set_xlabel(f"Grad-CAM heatmap IoU (top {args.top_pct:.0%} region)")
    ax.set_ylabel("Count")
    ax.set_title("MobileNetV2 vs ResNet50: Grad-CAM agreement")
    fig.tight_layout()
    fig.savefig(CFG.comparison_dir / "iou_distribution.png", dpi=150)
    plt.close(fig)

    print(f"[compare] mean IoU={df['iou'].mean():.3f} "
          f"median IoU={df['iou'].median():.3f}")
    print(f"[compare] wrote iou_raw.csv, iou_by_class.csv, "
          f"iou_distribution.png, {saved_qualitative} side-by-side figures "
          f"to {CFG.comparison_dir}")


if __name__ == "__main__":
    main()
