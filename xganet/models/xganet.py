"""Full X-GANet module following Algorithm A1."""

from __future__ import annotations

import torch
from torch import nn

from xganet.config import XGANetConfig
from xganet.data.graph import batch_adjacency, normalized_adjacency
from xganet.models.attention import CrossDiffusedAttention, MultiHeadSelfAttention
from xganet.models.encoders import FeatureEncoder, StructureEncoder
from xganet.models.fusion import ClassifierHead, GatedFusion
from xganet.models.lcm import LearnableCooccurrence
from xganet.models.masking import EntropyMask


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, multiplier: int) -> None:
        super().__init__()
        hidden = embedding_dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(tokens + self.net(tokens))


class XDABlock(nn.Module):
    def __init__(self, config: XGANetConfig) -> None:
        super().__init__()
        d = config.embedding_dim
        h = config.attention_heads
        self.self_f = MultiHeadSelfAttention(d, h)
        self.self_s = MultiHeadSelfAttention(d, h)
        self.xda_f2s = CrossDiffusedAttention(d, h, config.diffusion_scale)
        self.xda_s2f = CrossDiffusedAttention(d, h, config.diffusion_scale)
        self.lcm = LearnableCooccurrence(d, config.lcm_init_scale)
        self.mask = EntropyMask(config.entropy_threshold, config.entropy_alpha)
        self.norm_f = nn.LayerNorm(d)
        self.norm_s = nn.LayerNorm(d)
        self.ffn_f = FeedForward(d, config.ffn_multiplier)
        self.ffn_s = FeedForward(d, config.ffn_multiplier)

    def forward(
        self, feature_emb: torch.Tensor, structure_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        feature_emb = self.self_f(feature_emb)
        structure_emb = self.self_s(structure_emb)

        xda_f, attn_fs = self.xda_f2s(
            feature_emb, structure_emb, feature_emb, structure_emb
        )
        xda_s, attn_sf = self.xda_s2f(
            structure_emb, feature_emb, structure_emb, feature_emb
        )
        masked_f, entropy_f, mask_f = self.mask(xda_f, attn_fs)
        masked_s, entropy_s, mask_s = self.mask(xda_s, attn_sf)
        feature_emb = self.norm_f(feature_emb + masked_f)
        structure_emb = self.norm_s(structure_emb + masked_s)
        feature_emb, structure_emb = self.lcm(feature_emb, structure_emb)
        feature_emb = self.ffn_f(feature_emb)
        structure_emb = self.ffn_s(structure_emb)
        extras = {
            "attn_fs": attn_fs,
            "attn_sf": attn_sf,
            "entropy_f": entropy_f,
            "entropy_s": entropy_s,
            "mask_f": mask_f,
            "mask_s": mask_s,
        }
        return feature_emb, structure_emb, extras


class XGANet(nn.Module):
    def __init__(self, config: XGANetConfig, num_classes: int) -> None:
        super().__init__()
        self.config = config
        self.feature_encoder = FeatureEncoder(config.feature_dim, config.embedding_dim)
        self.structure_encoder = StructureEncoder(
            config.structure_in_dim, config.gcn_hidden, config.embedding_dim
        )
        self.blocks = nn.ModuleList(XDABlock(config) for _ in range(config.xda_layers))
        self.fusion = GatedFusion(config.embedding_dim, config.fusion_bias)
        self.classifier = ClassifierHead(config.embedding_dim, num_classes)

    def encode(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch["features"]
        structure_extra = batch["structure"]
        adj = batch_adjacency(
            batch["src_ids"],
            batch["dst_ids"],
            batch["time_index"],
            self.config.delta_t,
        )
        adj_hat = normalized_adjacency(adj)
        feature_emb = self.feature_encoder(features)
        struct_input = torch.cat([features, structure_extra], dim=-1)
        structure_emb = self.structure_encoder(struct_input, adj_hat)
        return feature_emb, structure_emb, adj_hat

    def forward(self, batch: dict[str, torch.Tensor], return_explain: bool = False):
        feature_emb, structure_emb, adj_hat = self.encode(batch)
        aligned_f, aligned_s = feature_emb, structure_emb
        last_extras: dict[str, torch.Tensor] = {}
        for block in self.blocks:
            aligned_f, aligned_s, last_extras = block(aligned_f, aligned_s)
        fused, gate = self.fusion(aligned_f, aligned_s)
        logits = self.classifier(fused)
        outputs = {
            "logits": logits,
            "feature_emb": feature_emb,
            "structure_emb": structure_emb,
            "fused": fused,
            "gate": gate,
        }
        if return_explain:
            outputs.update(last_extras)
            outputs["adj_hat"] = adj_hat
        return outputs
