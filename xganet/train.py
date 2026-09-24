"""Train X-GANet with Algorithm A1, Adam, early stopping, and optional bf16 AMP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xganet.config import XGANetConfig
from xganet.data.dataset import FlowGraphDataset, collate_flows
from xganet.data.load_nf import load_flow_arrays
from xganet.metrics import compute_metrics
from xganet.models.losses import XGANetCriterion
from xganet.models.xganet import XGANet
from xganet.utils import autocast_context, configure_runtime, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train X-GANet on NF-BoT-IoT-v2")
    parser.add_argument("--data", type=str, default="packet_dataset/NF-BoT-IoT-v2.csv")  # Path to the source CSV dataset
    parser.add_argument("--sample-size", type=int, default=90600)  # Number of flows to subsample from the dataset
    parser.add_argument("--device", type=str, default="cuda")  # Device to run training on (e.g., 'cuda' or 'cpu')
    parser.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "none"])  # Automatic Mixed Precision data type
    parser.add_argument("--batch-size", type=int, default=128)  # Number of samples per training batch
    parser.add_argument("--epochs", type=int, default=100)  # Total number of training epochs
    parser.add_argument("--delta-t", type=int, default=100)  # Temporal window size for time-based graph edges
    parser.add_argument("--seed", type=int, default=42)  # Random seed for reproducibility
    parser.add_argument("--num-workers", type=int, default=None)  # Number of worker threads for data loading
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")  # Directory to save model checkpoints
    parser.add_argument("--cache", type=str, default="packet_dataset/nf_botiot_v2_sample.parquet")  # Path for caching the subsampled dataset
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> XGANetConfig:
    workers = args.num_workers
    if workers is None:
        workers = 0 if os.name == "nt" else 4
    return XGANetConfig(
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        delta_t=args.delta_t,
        seed=args.seed,
        amp_dtype=args.amp,
        num_workers=workers,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate_loader(
    model: XGANet,
    loader: DataLoader,
    criterion: XGANetCriterion,
    device: torch.device,
    amp_dtype: str,
    class_names: tuple[str, ...],
) -> tuple[float, dict[str, object]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp_dtype):
            outputs = model(batch)
            loss, _cls, _con = criterion(
                outputs["logits"],
                batch["y_attack"],
                outputs["feature_emb"],
                outputs["structure_emb"],
            )
        total_loss += float(loss.item())
        n_batches += 1
        logits = outputs["logits"].float()
        y_true.append(batch["y_attack"].cpu().numpy())
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
        y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
        class_names,
    )
    return avg_loss, metrics


def train_one_epoch(
    model: XGANet,
    loader: DataLoader,
    criterion: XGANetCriterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_dtype: str,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            outputs = model(batch)
            loss, _cls, _con = criterion(
                outputs["logits"],
                batch["y_attack"],
                outputs["feature_emb"],
                outputs["structure_emb"],
            )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    path: Path,
    model: XGANet,
    config: XGANetConfig,
    class_names: tuple[str, ...],
    feat_min: np.ndarray,
    feat_max: np.ndarray,
    splits: dict[str, np.ndarray],
    metrics: dict[str, object],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial_metrics = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "class_names": class_names,
            "feat_min": feat_min,
            "feat_max": feat_max,
            "splits": splits,
            "metrics": serial_metrics,
            "epoch": epoch,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    seed_everything(config.seed)
    device = resolve_device(args.device)
    configure_runtime(device)
    print(f"Using device: {device}")

    train_arr, val_arr, test_arr, splits = load_flow_arrays(
        args.data, config, cache_path=args.cache
    )
    class_names = train_arr.class_names
    print(
        f"Loaded flows train/val/test: {len(train_arr.features)}/"
        f"{len(val_arr.features)}/{len(test_arr.features)} classes={class_names}"
    )

    train_ds = FlowGraphDataset(train_arr)
    val_ds = FlowGraphDataset(val_arr)
    test_ds = FlowGraphDataset(test_arr)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
        collate_fn=collate_flows,
        drop_last=len(train_ds) > config.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
        collate_fn=collate_flows,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
        collate_fn=collate_flows,
    )

    model = XGANet(config, num_classes=len(class_names)).to(device)
    criterion = XGANetCriterion(config.contrastive_temperature, config.lambda_contrast)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    ckpt_dir = Path(args.ckpt_dir)
    best_val = float("inf")
    stale = 0
    best_path = ckpt_dir / "xganet_best.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.amp_dtype
        )
        val_loss, val_metrics = evaluate_loader(
            model, val_loader, criterion, device, config.amp_dtype, class_names
        )
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            stale = 0
            save_checkpoint(
                best_path,
                model,
                config,
                class_names,
                train_arr.feat_min,
                train_arr.feat_max,
                splits,
                val_metrics,
                epoch,
            )
        else:
            stale += 1
            if stale >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_path.exists():
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
    test_loss, test_metrics = evaluate_loader(
        model, test_loader, criterion, device, config.amp_dtype, class_names
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_metrics['accuracy']:.4f}")
    print(test_metrics["report"])
    save_checkpoint(
        ckpt_dir / "xganet_last.pt",
        model,
        config,
        class_names,
        train_arr.feat_min,
        train_arr.feat_max,
        splits,
        test_metrics,
        epoch,
    )
    cm = test_metrics["confusion_matrix"]
    np.save(ckpt_dir / "confusion_matrix.npy", cm)


if __name__ == "__main__":
    main()
