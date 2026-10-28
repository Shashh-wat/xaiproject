"""Shared helpers for the explainability scripts: loading a checkpoint,
loading/preprocessing a single image, and running a prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import CFG, DEVICE
from src.data.dataset import build_transforms, denormalize
from src.models.factory import build_model


def load_model(model_name: str):
    ckpt_path = CFG.checkpoints_dir / f"{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run "
            f"`python -m src.train --model {model_name}` first."
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(model_name, len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["class_names"]


def load_image_tensor(image_path: str | Path):
    """Returns (input_tensor [1,3,H,W] normalized, rgb_float_image [H,W,3]
    in [0,1] for overlay visualizations)."""
    img = Image.open(image_path).convert("RGB")
    transform = build_transforms(train=False)
    tensor = transform(img).unsqueeze(0)
    rgb_float = np.array(
        denormalize(tensor[0])
    ).transpose(1, 2, 0)
    return tensor, rgb_float


@torch.no_grad()
def predict(model, input_tensor, class_names, topk: int = 3):
    logits = model(input_tensor.to(DEVICE))
    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(min(topk, len(class_names)))
    return [
        (class_names[i], float(p))
        for p, i in zip(top_probs.tolist(), top_idx.tolist())
    ]
