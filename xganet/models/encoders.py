"""Feature encoder E_f and two-layer GCN structure encoder E_s."""

from __future__ import annotations

import torch
from torch import nn


class FeatureEncoder(nn.Module):
    """Projects raw flow features into the shared latent space of dimension d."""

    def __init__(self, in_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GCN(nn.Module):
    """Two-layer GCN: H' = Â ReLU(Â X W1) W2."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, node_features: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(adj_hat @ self.w1(node_features))
        return adj_hat @ self.w2(hidden)


class StructureEncoder(nn.Module):
    """GNN on Â[B] followed by a linear projection into dimension d."""

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.gnn = GCN(in_dim, hidden_dim, embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, node_features: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        raw = self.gnn(node_features, adj_hat)
        return self.proj(raw)
