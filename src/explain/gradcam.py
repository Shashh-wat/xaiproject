"""Grad-CAM explanations for a single image.

Usage:
    python -m src.explain.gradcam --model mobilenet_v2 --image path/to.jpg
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import CFG, DEVICE
from src.explain.common import load_image_tensor, load_model, predict


def compute_gradcam(model, input_tensor, target_class: int | None = None):
    """Returns (heatmap [H,W] in [0,1], predicted_class_idx).

    If target_class is None, Grad-CAM explains the model's own top
    prediction (the common default).
    """
    cam = GradCAM(model=model, target_layers=[model.get_target_layer()])
    targets = None
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import \
            ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]

    # Our baselines train with a frozen backbone (build_model's default),
    # so every backbone parameter has requires_grad=False. With a plain
    # image tensor (requires_grad=False too), autograd never builds a
    # graph through those layers at all, and grad-cam's backward hook
    # captures None for the activation gradients -- crashes downstream.
    # Setting requires_grad on the *input* forces the graph to exist
    # regardless of which weights are frozen (output.requires_grad is an
    # OR over its inputs'), without changing any prediction or training.
    input_tensor = input_tensor.to(DEVICE).clone().requires_grad_(True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    heatmap = grayscale_cam[0]  # [H, W]

    with torch.no_grad():
        pred_idx = int(model(input_tensor.to(DEVICE)).argmax(1).item())
    return heatmap, pred_idx


def save_gradcam_figure(rgb_float, heatmap, class_names, pred_idx,
                         confidence, out_path):
    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_float); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(
        f"{class_names[pred_idx]} ({confidence:.1%})"); axes[2].axis("off")
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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, class_names = load_model(args.model)
    input_tensor, rgb_float = load_image_tensor(args.image)

    heatmap, pred_idx = compute_gradcam(model, input_tensor)
    topk = predict(model, input_tensor, class_names)
    confidence = dict(topk)[class_names[pred_idx]]

    out_path = args.out or (
        CFG.explanations_dir / f"gradcam_{args.model}_"
        f"{'_'.join(str(args.image).split('/')[-1].split('.')[:-1])}.png"
    )
    from pathlib import Path
    out_path = Path(out_path)
    save_gradcam_figure(rgb_float, heatmap, class_names, pred_idx,
                         confidence, out_path)

    print(f"[gradcam] prediction: {class_names[pred_idx]} ({confidence:.1%})")
    print(f"[gradcam] top-{len(topk)}: {topk}")
    print(f"[gradcam] wrote {out_path}")


if __name__ == "__main__":
    main()
