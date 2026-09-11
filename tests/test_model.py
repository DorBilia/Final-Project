"""CPU tests for XDA, entropy mask, fusion, losses, and full model shapes."""

from __future__ import annotations

import torch

from xganet.config import XGANetConfig
from xganet.models.attention import CrossDiffusedAttention, diffusion_delta
from xganet.models.fusion import GatedFusion
from xganet.models.losses import NTXentLoss, XGANetCriterion
from xganet.models.masking import EntropyMask
from xganet.models.xganet import XGANet


def _tiny_config(num_heads: int = 4) -> XGANetConfig:
    return XGANetConfig(
        embedding_dim=32,
        xda_layers=2,
        attention_heads=num_heads,
        gcn_hidden=32,
        batch_size=8,
        ffn_multiplier=2,
    )


def _synthetic_batch(n: int = 8, num_classes: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "features": torch.rand(n, 11),
        "structure": torch.rand(n, 4),
        "src_ids": torch.randint(0, 4, (n,)),
        "dst_ids": torch.randint(0, 4, (n,)),
        "time_index": torch.arange(n),
        "y_attack": torch.randint(0, num_classes, (n,)),
        "y_binary": torch.randint(0, 2, (n,)),
    }


def test_diffusion_delta_shape() -> None:
    feat = torch.randn(6, 16)
    struct = torch.randn(6, 16)
    delta = diffusion_delta(feat, struct)
    assert delta.shape == (6, 6)
    assert torch.isfinite(delta).all()


def test_xda_shapes() -> None:
    xda = CrossDiffusedAttention(embedding_dim=32, num_heads=4, diffusion_scale=0.5)
    query = torch.randn(8, 32)
    key = torch.randn(8, 32)
    out, weights = xda(query, key, query, key)
    assert out.shape == (8, 32)
    assert weights.shape == (8, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_entropy_mask_range() -> None:
    masker = EntropyMask(threshold=1.2, alpha=1.0)
    tokens = torch.randn(5, 8)
    attn = torch.softmax(torch.randn(5, 5), dim=-1)
    masked, entropy, mask = masker(tokens, attn)
    assert masked.shape == tokens.shape
    assert torch.all((mask >= 0) & (mask <= 1))
    assert torch.all(entropy >= 0)


def test_gated_fusion() -> None:
    fusion = GatedFusion(embedding_dim=16, fusion_bias=0.0)
    feat = torch.randn(7, 16)
    struct = torch.randn(7, 16)
    fused, gate = fusion(feat, struct)
    assert fused.shape == (7, 16)
    assert gate.shape == (7, 16)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_ntxent_and_total_loss() -> None:
    feat = torch.randn(10, 16)
    struct = feat + 0.01 * torch.randn(10, 16)
    ntx = NTXentLoss(temperature=0.1)
    loss = ntx(feat, struct)
    assert torch.isfinite(loss)
    criterion = XGANetCriterion(temperature=0.1, lambda_contrast=0.1)
    logits = torch.randn(10, 4)
    labels = torch.arange(10) % 4
    total, cls_loss, con_loss = criterion(logits, labels, feat, struct)
    assert torch.isfinite(total)
    assert total.item() >= 0
    assert cls_loss.item() >= 0
    assert con_loss.item() >= 0


def test_xganet_forward_shapes() -> None:
    config = _tiny_config()
    num_classes = 3
    model = XGANet(config, num_classes=num_classes)
    batch = _synthetic_batch(n=8, num_classes=num_classes)
    outputs = model(batch, return_explain=True)
    assert outputs["logits"].shape == (8, num_classes)
    assert outputs["feature_emb"].shape == (8, 32)
    assert outputs["structure_emb"].shape == (8, 32)
    assert outputs["gate"].shape == (8, 32)
    assert outputs["mask_f"].shape == (8,)
    assert outputs["attn_fs"].shape == (8, 8)
    assert torch.isfinite(outputs["logits"]).all()
