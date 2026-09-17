"""Multi-head self-attention and Cross-Diffused Attention (Algorithms 2–3, Eq. 27–28)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def diffusion_delta(feature_emb: torch.Tensor, structure_emb: torch.Tensor) -> torch.Tensor:
    """Equation (28) diffusion term Δ(F, S) of shape (B, B)."""
    eps = 1e-8
    feat_norm = F.normalize(feature_emb, dim=-1, eps=eps)
    struct_norm = F.normalize(structure_emb, dim=-1, eps=eps)

    cos_f = feat_norm @ feat_norm.transpose(0, 1)
    cos_s = struct_norm @ struct_norm.transpose(0, 1)

    # log_den calculates log(sum(exp(cos_s)))
    log_den = torch.logsumexp(cos_s, dim=0, keepdim=True)

    # torch.exp(cos_f - log_den) calculates (exp(cos_f) / sum(exp(cos_s)))
    # torch.log1p(x) calculates log(1 + x) safely
    return torch.log1p(torch.exp(cos_f - log_den))


class MultiHeadSelfAttention(nn.Module):
    """Equations (12)–(15): LN(F + MultiHead(F, F, F))."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, dim = tokens.shape
        query = self.w_q(tokens).view(batch, self.num_heads, self.head_dim)
        key = self.w_k(tokens).view(batch, self.num_heads, self.head_dim)
        value = self.w_v(tokens).view(batch, self.num_heads, self.head_dim)
        scale = math.sqrt(self.head_dim)
        scores = torch.einsum("bhd,chd->hbc", query, key) / scale
        weights = torch.softmax(scores, dim=-1)
        heads = torch.einsum("hbc,chd->bhd", weights, value)
        combined = heads.reshape(batch, dim)
        return self.norm(tokens + self.w_o(combined))


class CrossDiffusedAttention(nn.Module):
    """XDA: queries from one view, keys/values from the other, plus Δ(F, S)."""

    def __init__(self, embedding_dim: int, num_heads: int, diffusion_scale: float) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.diffusion_scale = diffusion_scale
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        query_src: torch.Tensor,
        key_src: torch.Tensor,
        feature_for_delta: torch.Tensor,
        structure_for_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (raw XDA output O, mean attention weights)."""
        batch, dim = query_src.shape
        query = self.w_q(query_src).view(batch, self.num_heads, self.head_dim)
        key = self.w_k(key_src).view(batch, self.num_heads, self.head_dim)
        value = self.w_v(key_src).view(batch, self.num_heads, self.head_dim)
        scale = math.sqrt(self.head_dim)
        logits = torch.einsum("bhd,chd->hbc", query, key) / scale
        delta = diffusion_delta(feature_for_delta, structure_for_delta)
        logits = logits + self.diffusion_scale * delta.unsqueeze(0)
        weights = torch.softmax(logits, dim=-1)
        heads = torch.einsum("hbc,chd->bhd", weights, value)
        combined = heads.reshape(batch, dim)
        xda_out = self.w_o(combined)
        mean_weights = weights.mean(dim=0)
        return xda_out, mean_weights
