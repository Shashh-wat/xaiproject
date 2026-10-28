"""LIME explanations for a single image.

Usage:
    python -m src.explain.lime_explain --model mobilenet_v2 --image path.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift

from src.config import CFG, DEVICE
from src.explain.common import load_image_tensor, load_model, predict


def make_predict_fn(model, mean, std):
    """LIME needs a function: np.ndarray [N,H,W,3] uint8 in [0,255] ->
    np.ndarray [N,num_classes] of class probabilities."""
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict_fn(images: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - mean_t) / std_t
        logits = model(batch.to(DEVICE))
        return torch.softmax(logits, dim=1).cpu().numpy()

    return predict_fn


def explain_instance(model, rgb_float, num_samples: int = CFG.lime_num_samples):
    """Runs LIME's superpixel perturbation and returns the raw `explanation`
    object plus the predicted class index. Shared by compute_lime (CLI
    figure) and compute_lime_dense (continuous map for the correlation
    pipeline in src/correlate/)."""
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)
    predict_fn = make_predict_fn(model, CFG.imagenet_mean, CFG.imagenet_std)

    explainer = lime_image.LimeImageExplainer()
    segmentation_fn = lambda img: quickshift(img, kernel_size=4,
                                              max_dist=200, ratio=0.2)
    explanation = explainer.explain_instance(
        rgb_uint8, predict_fn, top_labels=1, hide_color=0,
        num_samples=num_samples, segmentation_fn=segmentation_fn,
    )
    pred_idx = explanation.top_labels[0]
    return explanation, pred_idx


def compute_lime(model, rgb_float, class_names,
                  num_samples: int = CFG.lime_num_samples):
    explanation, pred_idx = explain_instance(model, rgb_float, num_samples)
    temp, mask = explanation.get_image_and_mask(
        pred_idx, positive_only=True, num_features=8, hide_rest=False)
    return temp, mask, pred_idx


def compute_lime_dense(model, rgb_float, class_names=None,
                        num_samples: int = CFG.lime_num_samples):
    """Returns a continuous, per-pixel LIME importance map (signed weight
    of each pixel's superpixel segment for the predicted class) instead of
    the CLI's binary top-k mask. This is what src/correlate/ needs, since a
    binary mask can't be compared/embedded the way a Grad-CAM or SHAP map
    can."""
    explanation, pred_idx = explain_instance(model, rgb_float, num_samples)
    segments = explanation.segments
    weights = dict(explanation.local_exp[pred_idx])
    dense = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, w in weights.items():
        dense[segments == seg_id] = w
    return dense, pred_idx


def save_lime_figure(rgb_float, temp, mask, class_names, pred_idx,
                      confidence, out_path):
    boundary_img = mark_boundaries(temp / 255.0, mask)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(rgb_float); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(boundary_img)
    axes[1].set_title(f"LIME: {class_names[pred_idx]} ({confidence:.1%})")
    axes[1].axis("off")
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
    parser.add_argument("--num_samples", type=int,
                         default=CFG.lime_num_samples)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model, class_names = load_model(args.model)
    input_tensor, rgb_float = load_image_tensor(args.image)
    topk = predict(model, input_tensor, class_names)

    print(f"[lime] running LIME with {args.num_samples} perturbation "
          "samples (this can take a minute) ...")
    temp, mask, pred_idx = compute_lime(model, rgb_float, class_names,
                                         num_samples=args.num_samples)
    confidence = dict(topk)[class_names[pred_idx]]

    out_path = args.out or (
        CFG.explanations_dir / f"lime_{args.model}_{Path(args.image).stem}.png"
    )
    out_path = Path(out_path)
    save_lime_figure(rgb_float, temp, mask, class_names, pred_idx,
                      confidence, out_path)

    print(f"[lime] prediction: {class_names[pred_idx]} ({confidence:.1%})")
    print(f"[lime] wrote {out_path}")


if __name__ == "__main__":
    main()
