"""Learnable Co-occurrence Matrix refinement (Equation 31)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LearnableCooccurrence(nn.Module):
    """Bilinear feature–structure co-occurrence used as LCM(E)."""

    def __init__(self, embedding_dim: int, init_scale: float) -> None:
        super().__init__()
        self.w_a = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_b = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.mix = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.w_a.weight, mean=0.0, std=init_scale)
        nn.init.normal_(self.w_b.weight, mean=0.0, std=init_scale)

    def forward(
        self, feature_emb: torch.Tensor, structure_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = feature_emb.size(-1) ** 0.5
        left = self.w_a(feature_emb)
        right = self.w_b(structure_emb)
        cooccur = torch.softmax(left @ right.transpose(0, 1) / scale, dim=-1)
        lcm_feature = cooccur @ structure_emb
        lcm_structure = cooccur.transpose(0, 1) @ feature_emb
        alpha = torch.sigmoid(self.mix)
        feature_out = F.relu(alpha * lcm_feature + (1.0 - alpha) * feature_emb)
        structure_out = F.relu(alpha * lcm_structure + (1.0 - alpha) * structure_emb)
        return feature_out, structure_out
