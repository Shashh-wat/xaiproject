"""Input perturbations for robustness testing: blur, brightness shift,
Gaussian noise, and random occlusion. Each returns a function
tensor[1,3,H,W] (normalized) -> perturbed tensor of the same shape.

Perturbations are applied in de-normalized [0,1] pixel space, then
re-normalized, so severity parameters are interpretable independent of the
ImageNet normalization constants.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.config import CFG
from src.data.dataset import denormalize


def _renormalize(rgb_0_1: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CFG.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(CFG.imagenet_std).view(3, 1, 1)
    return (rgb_0_1 - mean) / std


def gaussian_blur(input_tensor: torch.Tensor, severity: float) -> torch.Tensor:
    """severity in [0, 1] -> kernel size / sigma scale."""
    rgb = denormalize(input_tensor[0])
    ksize = max(3, int(severity * 15) | 1)  # odd kernel size
    sigma = max(0.1, severity * 5)
    blurred = TF.gaussian_blur(rgb, kernel_size=ksize, sigma=sigma)
    return _renormalize(blurred).unsqueeze(0)


def brightness_shift(input_tensor: torch.Tensor, severity: float) -> torch.Tensor:
    """severity in [-1, 1]: negative darkens, positive brightens."""
    rgb = denormalize(input_tensor[0])
    shifted = (rgb + severity).clamp(0, 1)
    return _renormalize(shifted).unsqueeze(0)


def gaussian_noise(input_tensor: torch.Tensor, severity: float) -> torch.Tensor:
    """severity in [0, 1] -> noise std."""
    rgb = denormalize(input_tensor[0])
    noisy = (rgb + torch.randn_like(rgb) * severity).clamp(0, 1)
    return _renormalize(noisy).unsqueeze(0)


def random_occlusion(input_tensor: torch.Tensor, severity: float,
                      seed: int | None = None) -> torch.Tensor:
    """severity in [0, 1] -> fraction of image area occluded with a gray
    box (simulates a leaf partially blocked)."""
    if seed is not None:
        random.seed(seed)
    rgb = denormalize(input_tensor[0]).clone()
    _, h, w = rgb.shape
    box_frac = severity
    bh, bw = int(h * box_frac ** 0.5), int(w * box_frac ** 0.5)
    if bh > 0 and bw > 0:
        top = random.randint(0, max(0, h - bh))
        left = random.randint(0, max(0, w - bw))
        rgb[:, top:top + bh, left:left + bw] = 0.5  # mid-gray occluder
    return _renormalize(rgb).unsqueeze(0)


PERTURBATIONS = {
    "blur": (gaussian_blur, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    "brightness": (brightness_shift, [-0.6, -0.3, 0.0, 0.3, 0.6]),
    "noise": (gaussian_noise, [0.0, 0.05, 0.1, 0.2, 0.3]),
    "occlusion": (random_occlusion, [0.0, 0.1, 0.2, 0.3, 0.4]),
}
