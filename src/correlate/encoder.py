"""Siamese-style shared-weight encoder for explanation maps.

The same small CNN is applied to a Grad-CAM, a SHAP, or a LIME map -- it
never sees which method produced its input, only the map itself. That's
the "Siamese" part: one network, multiple inputs, a shared embedding space.
Trained with SupCon (see losses.py), it learns to place explanation maps
that justify the *same predicted class* close together, regardless of
which of the three methods produced them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplanationEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 1, S, S] -> L2-normalized embeddings [N, embed_dim]."""
        feats = self.conv(x).flatten(1)
        z = self.projection(feats)
        return F.normalize(z, dim=1)
