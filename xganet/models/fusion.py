"""Adaptive gated fusion (Equations 32–33) and classifier (Equation 34)."""

from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, embedding_dim: int, fusion_bias: float) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            self.gate.bias.fill_(fusion_bias)

    def forward(
        self, feature_emb: torch.Tensor, structure_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([feature_emb, structure_emb], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        fused = gate * feature_emb + (1.0 - gate) * structure_emb
        return fused, gate


class ClassifierHead(nn.Module):
    """Two-layer FFN with ReLU, logits for softmax (Equation 34)."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        hidden = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)
