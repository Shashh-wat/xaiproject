"""SHAP explanations for a single image, using GradientExplainer.

Usage:
    python -m src.explain.shap_explain --model mobilenet_v2 --image path.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.config import CFG, DEVICE
from src.data.dataset import build_transforms, get_class_names
from src.explain.common import load_image_tensor, load_model, predict


def sample_background(n: int = CFG.shap_background_samples,
                       root=CFG.raw_data_dir, seed: int = 42):
    """Randomly samples n training images (across classes) to use as the
    SHAP background/reference distribution."""
    random.seed(seed)
    class_names = get_class_names(root)
    transform = build_transforms(train=False)
    all_paths = []
    for cname in class_names:
        cdir = root / cname
        all_paths += [p for p in cdir.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    sampled = random.sample(all_paths, min(n, len(all_paths)))
    tensors = []
    from PIL import Image
    for p in sampled:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors)


def compute_shap(model, input_tensor, background: torch.Tensor):
    explainer = shap.GradientExplainer(model, background.to(DEVICE))
    shap_values, indexes = explainer.shap_values(
        input_tensor.to(DEVICE), ranked_outputs=1, nsamples=50)
    # The layout depends on the shap version:
    #   older: a list with one array [1, 3, H, W] per ranked output
    #   newer (0.49 seen here): one array [1, 3, H, W, 1]
    # Either way we want the [3, H, W] values of the one ranked output.
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.asarray(sv)
    sv = sv[0, ..., 0] if sv.ndim == 5 else sv[0]  # [3, H, W]
    if sv.ndim != 3 or sv.shape[0] != 3:
        raise ValueError(f"unexpected SHAP value shape {sv.shape}")
    pred_idx = int(np.asarray(indexes).reshape(-1)[0])
    # sum over the colour channels -> signed per-pixel attribution [H, W]
    attribution = sv.sum(axis=0)
    return attribution, pred_idx


def save_shap_figure(rgb_float, attribution, class_names, pred_idx,
                      confidence, out_path):
    vmax = np.percentile(np.abs(attribution), 99) + 1e-8
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(rgb_float); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(rgb_float)
    im = axes[1].imshow(attribution, cmap="bwr", alpha=0.6, vmin=-vmax,
                         vmax=vmax)
    axes[1].set_title(f"SHAP: {class_names[pred_idx]} ({confidence:.1%})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--n_background", type=int,
                         default=CFG.shap_background_samples)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, class_names = load_model(args.model)
    input_tensor, rgb_float = load_image_tensor(args.image)
    topk = predict(model, input_tensor, class_names)

    print(f"[shap] sampling {args.n_background} background images ...")
    background = sample_background(n=args.n_background)

    print("[shap] computing SHAP values (this can take a minute) ...")
    attribution, pred_idx = compute_shap(model, input_tensor, background)
    confidence = dict(topk)[class_names[pred_idx]]

    out_path = args.out or (
        CFG.explanations_dir / f"shap_{args.model}_"
        f"{Path(args.image).stem}.png"
    )
    out_path = Path(out_path)
    save_shap_figure(rgb_float, attribution, class_names, pred_idx,
                      confidence, out_path)

    print(f"[shap] prediction: {class_names[pred_idx]} ({confidence:.1%})")
    print(f"[shap] wrote {out_path}")


if __name__ == "__main__":
    main()
