"""Tiny synthetic overfit smoke test for Algorithm A1."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from xganet.config import XGANetConfig
from xganet.data.dataset import FlowGraphDataset, collate_flows
from xganet.data.load_nf import FlowArrays
from xganet.models.losses import XGANetCriterion
from xganet.models.xganet import XGANet
from xganet.utils import seed_everything


def _toy_arrays(n: int = 32, num_classes: int = 2) -> FlowArrays:
    rng = torch.Generator().manual_seed(0)
    features = torch.rand(n, 11, generator=rng).numpy().astype("float32")
    labels = (torch.arange(n) % num_classes).numpy().astype("int64")
    features[:, 0] = labels.astype("float32")
    structure = torch.rand(n, 4, generator=rng).numpy().astype("float32")
    src = (torch.arange(n) % 5).numpy().astype("int64")
    dst = ((torch.arange(n) + 1) % 5).numpy().astype("int64")
    times = torch.arange(n).numpy().astype("int64")
    return FlowArrays(
        features=features,
        structure=structure,
        src_ids=src,
        dst_ids=dst,
        time_index=times,
        y_attack=labels,
        y_binary=labels,
        class_names=("a", "b"),
        feat_min=features.min(axis=0),
        feat_max=features.max(axis=0),
    )


def test_tiny_overfit_loss_drops() -> None:
    seed_everything(0)
    config = XGANetConfig(
        embedding_dim=32,
        xda_layers=1,
        attention_heads=4,
        gcn_hidden=32,
        batch_size=16,
        ffn_multiplier=2,
        learning_rate=1e-3,
        lambda_contrast=0.05,
        epochs=40,
        delta_t=4,
    )
    arrays = _toy_arrays()
    loader = DataLoader(
        FlowGraphDataset(arrays),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_flows,
    )
    model = XGANet(config, num_classes=2)
    criterion = XGANetCriterion(config.contrastive_temperature, config.lambda_contrast)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def _epoch_stats() -> tuple[float, float]:
        model.train()
        total = 0.0
        correct = 0
        seen = 0
        n_batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _cls, _con = criterion(
                outputs["logits"],
                batch["y_attack"],
                outputs["feature_emb"],
                outputs["structure_emb"],
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
            pred = outputs["logits"].argmax(dim=-1)
            correct += int((pred == batch["y_attack"]).sum().item())
            seen += int(batch["y_attack"].numel())
        return total / max(n_batches, 1), correct / max(seen, 1)

    first_loss, _first_acc = _epoch_stats()
    last_loss = first_loss
    last_acc = 0.0
    for _ in range(config.epochs - 1):
        last_loss, last_acc = _epoch_stats()
    assert last_loss < first_loss
    assert last_acc >= 0.8
    assert last_loss < 1.0
