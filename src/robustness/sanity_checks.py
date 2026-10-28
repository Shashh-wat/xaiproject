"""Sanity checks for Grad-CAM (Adebayo et al., 2018 style): cascading
weight randomization from output back toward input. A trustworthy
explanation should visibly degrade as more layers are randomized -- if the
heatmap looks the same on a fully-random network, Grad-CAM isn't actually
reflecting what the trained network learned.

Usage:
    python -m src.robustness.sanity_checks --model mobilenet_v2 --image path.jpg
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from src.config import CFG, DEVICE
from src.explain.common import load_image_tensor, load_model
from src.explain.gradcam import compute_gradcam


def _randomize_module(module: nn.Module):
    """Re-initializes a module's parameters in place, if it knows how to."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def cascading_randomization(model, input_tensor, n_steps: int = 5):
    """Yields (step_label, heatmap) pairs, randomizing progressively more
    of the network from the output layer backward.

    Implementation note: rather than requiring architecture-specific layer
    ordering, we randomize the top-N% of *parameterized* modules ordered
    from output to input, increasing N each step -- this generalizes across
    MobileNetV2 and ResNet50 without hardcoding layer names.
    """
    param_modules = [m for m in model.modules()
                     if hasattr(m, "reset_parameters")]
    # modules() yields roughly input->output order; reverse for output->input
    param_modules = list(reversed(param_modules))
    n_total = len(param_modules)

    results = []
    working_model = copy.deepcopy(model).to(DEVICE)
    heatmap, _ = compute_gradcam(working_model, input_tensor)
    results.append(("trained (0% randomized)", heatmap))

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        n_randomize = int(round(frac * n_total))
        working_model = copy.deepcopy(model).to(DEVICE)
        modules_now = [m for m in working_model.modules()
                       if hasattr(m, "reset_parameters")]
        modules_now = list(reversed(modules_now))
        for m in modules_now[:n_randomize]:
            _randomize_module(m)
        heatmap, _ = compute_gradcam(working_model, input_tensor)
        results.append((f"{frac:.0%} randomized", heatmap))

    return results


def save_sanity_figure(rgb_float, results, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    correlations = []
    prev_heatmap = None
    for ax, (label, heatmap) in zip(axes, results):
        ax.imshow(heatmap, cmap="jet")
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        if prev_heatmap is not None:
            corr, _ = spearmanr(prev_heatmap.flatten(), heatmap.flatten())
            correlations.append(corr)
        prev_heatmap = heatmap
    fig.suptitle("Grad-CAM under cascading weight randomization "
                  "(should degrade toward noise)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--n_steps", type=int, default=5)
    args = parser.parse_args()

    model, class_names = load_model(args.model)
    input_tensor, rgb_float = load_image_tensor(args.image)

    print("[sanity] running cascading randomization "
          f"({args.n_steps} steps) ...")
    results = cascading_randomization(model, input_tensor, args.n_steps)

    out_path = (CFG.robustness_dir /
                f"sanity_{args.model}_{Path(args.image).stem}.png")
    correlations = save_sanity_figure(rgb_float, results, out_path)

    print(f"[sanity] step-to-step rank correlations: "
          f"{[round(c, 3) for c in correlations]}")
    print("[sanity] (expect correlation to drop toward 0 as randomization "
          "increases -- a flat, high correlation across all steps would "
          "mean Grad-CAM isn't actually sensitive to the trained weights)")
    print(f"[sanity] wrote {out_path}")


if __name__ == "__main__":
    main()
