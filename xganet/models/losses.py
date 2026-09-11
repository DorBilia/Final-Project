"""Categorical cross-entropy plus NT-Xent contrastive alignment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NTXentLoss(nn.Module):
    """InfoNCE between paired feature and structure embeddings in a batch."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, feature_emb: torch.Tensor, structure_emb: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(feature_emb, dim=-1)
        struct = F.normalize(structure_emb, dim=-1)
        logits = feat @ struct.transpose(0, 1) / self.temperature
        labels = torch.arange(feat.size(0), device=feat.device)
        loss_f = F.cross_entropy(logits, labels)
        loss_s = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_f + loss_s)


class XGANetCriterion(nn.Module):
    def __init__(self, temperature: float, lambda_contrast: float) -> None:
        super().__init__()
        self.classification = nn.CrossEntropyLoss()
        self.contrastive = NTXentLoss(temperature)
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        feature_emb: torch.Tensor,
        structure_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = self.classification(logits, labels)
        con_loss = self.contrastive(feature_emb, structure_emb)
        total = cls_loss + self.lambda_contrast * con_loss
        return total, cls_loss, con_loss
