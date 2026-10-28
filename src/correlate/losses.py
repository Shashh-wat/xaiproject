"""Supervised Contrastive (SupCon) loss -- Khosla et al., 2020.

Standard formulation, applied here across explanation maps rather than raw
images: for anchor i with label y_i (the predicted class its explanation
map justifies), positives are every other map in the batch sharing y_i
-- whether it came from Grad-CAM, SHAP, or LIME -- and negatives are every
map with a different label. Pulling same-class maps together regardless of
method is exactly what makes the resulting embedding space measure
cross-method agreement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """embeddings: [N, D] L2-normalized. labels: [N] long tensor."""
        device = embeddings.device
        n = embeddings.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        sim = embeddings @ embeddings.T / self.temperature  # [N, N]
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stability

        labels = labels.view(-1, 1)
        same_label = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(n, device=device)
        pos_mask = same_label - self_mask  # exclude self as its own positive
        logit_mask = 1.0 - self_mask       # exclude self from the denominator

        exp_sim = torch.exp(sim) * logit_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        per_anchor_loss = -(pos_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]
        return per_anchor_loss.mean()
