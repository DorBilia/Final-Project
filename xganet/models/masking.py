"""Entropy-driven adaptive masking (Equations 23–25)."""

from __future__ import annotations

import torch
from torch import nn


class EntropyMask(nn.Module):
    def __init__(self, threshold: float, alpha: float) -> None:
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(
        self, xda_output: torch.Tensor, attention_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale tokens by m_i = σ(α (τ − H_i)).

        Returns (masked output, entropy, mask weights).
        """
        probs = attention_weights.clamp(min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        mask = torch.sigmoid(self.alpha * (self.threshold - entropy))
        masked = mask.unsqueeze(-1) * xda_output
        return masked, entropy, mask
