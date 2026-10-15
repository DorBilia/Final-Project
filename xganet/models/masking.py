"""Entropy-driven adaptive masking (Equations 23–25)."""

from __future__ import annotations

import math

import torch
from torch import nn


class EntropyMask(nn.Module):
    def __init__(self, threshold: float, alpha: float, relative: bool = True) -> None:
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.relative = relative

    def forward(
        self, xda_output: torch.Tensor, attention_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale tokens by m_i = σ(α (τ − H_i)).

        When ``relative=True``, entropy is divided by ln(B) so that τ lives on
        a [0, 1] scale independent of the batch / window size.

        Returns (masked output, entropy, mask weights).
        """
        probs = attention_weights.clamp(min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        if self.relative:
            max_entropy = math.log(max(probs.size(-1), 2))
            entropy = entropy / max_entropy
        mask = torch.sigmoid(self.alpha * (self.threshold - entropy))
        masked = mask.unsqueeze(-1) * xda_output
        return masked, entropy, mask
