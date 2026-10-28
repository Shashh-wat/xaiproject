"""Robustness experiment: for a sample of test images, apply each
perturbation type at increasing severity and record (a) whether the
prediction flips and (b) Grad-CAM heatmap IoU between original and
perturbed image. Produces per-model curves; run for both models and
compare the two output CSVs/plots.

Usage:
    python -m src.robustness.run_robustness --model mobilenet_v2 --n_images 30
"""
from __future__ import annotations

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from src.config import CFG, DEVICE
from src.compare_explanations import heatmap_iou
from src.data.dataset import build_datasets
from src.explain.common import load_model
from src.explain.gradcam import compute_gradcam
from src.robustness.perturbations import PERTURBATIONS


def load_sample_test_tensors(n_images: int, seed: int = 42):
    from src.explain.common import load_image_tensor
    _, _, test_ds, class_names = build_datasets()
    random.seed(seed)
    idxs = random.sample(range(len(test_ds.paths)),
                          min(n_images, len(test_ds.paths)))
    items = []
    for i in idxs:
        path, label = test_ds.paths[i], test_ds.labels[i]
        input_tensor, rgb_float = load_image_tensor(path)
        items.append((input_tensor, label, class_names))
    return items, class_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--n_images", type=int, default=30)
    parser.add_argument("--top_pct", type=float, default=0.25)
    args = parser.parse_args()

    model, class_names = load_model(args.model)
    print(f"[robustness] sampling {args.n_images} test images ...")
    items, class_names = load_sample_test_tensors(args.n_images)

    rows = []
    for input_tensor, label, _ in tqdm(items):
        with torch.no_grad():
            orig_pred = int(model(input_tensor.to(DEVICE)).argmax(1).item())
        orig_heat, _ = compute_gradcam(model, input_tensor)

        for pert_name, (fn, severities) in PERTURBATIONS.items():
            for severity in severities:
                perturbed = fn(input_tensor, severity)
                with torch.no_grad():
                    pert_pred = int(
                        model(perturbed.to(DEVICE)).argmax(1).item())
                pert_heat, _ = compute_gradcam(model, perturbed)
                iou = heatmap_iou(orig_heat, pert_heat, args.top_pct)

                rows.append({
                    "perturbation": pert_name,
                    "severity": severity,
                    "true_class": class_names[label],
                    "orig_correct": orig_pred == label,
                    "pred_flipped": pert_pred != orig_pred,
                    "heatmap_iou": iou,
                })

    df = pd.DataFrame(rows)
    csv_path = CFG.robustness_dir / f"{args.model}_perturbation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[robustness] wrote {csv_path}")

    summary = df.groupby(["perturbation", "severity"]).agg(
        flip_rate=("pred_flipped", "mean"),
        mean_iou=("heatmap_iou", "mean"),
    ).reset_index()
    summary_path = CFG.robustness_dir / f"{args.model}_perturbation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[robustness] wrote {summary_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for pert_name in PERTURBATIONS:
        sub = summary[summary["perturbation"] == pert_name]
        axes[0].plot(sub["severity"], sub["flip_rate"], marker="o",
                     label=pert_name)
        axes[1].plot(sub["severity"], sub["mean_iou"], marker="o",
                     label=pert_name)
    axes[0].set_xlabel("severity"); axes[0].set_ylabel("prediction flip rate")
    axes[0].set_title(f"{args.model}: prediction robustness")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("severity"); axes[1].set_ylabel("Grad-CAM IoU vs. original")
    axes[1].set_title(f"{args.model}: explanation robustness")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig_path = CFG.robustness_dir / f"{args.model}_perturbation_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[robustness] wrote {fig_path}")


if __name__ == "__main__":
    main()
