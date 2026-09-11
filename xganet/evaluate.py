"""Evaluate a trained X-GANet checkpoint and dump explainability stats."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xganet.config import XGANetConfig
from xganet.data.dataset import FlowGraphDataset, collate_flows
from xganet.data.load_nf import load_sampled_frame, arrays_from_frame, split_flow_arrays
from xganet.metrics import compute_metrics
from xganet.models.xganet import XGANet
from xganet.utils import autocast_context, configure_runtime, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate X-GANet")
    parser.add_argument("--ckpt", type=str, default="checkpoints/xganet_best.pt")
    parser.add_argument("--data", type=str, default="packet_dataset/nf_botiot_v2_sample.parquet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


@torch.no_grad()
def run_eval(
    model: XGANet,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    class_names: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, float]]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast_context(device, amp_dtype):
            outputs = model(batch, return_explain=True)
        logits = outputs["logits"].float()
        y_true.append(batch["y_attack"].cpu().numpy())
        y_pred.append(logits.argmax(dim=-1).cpu().numpy())
        y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())
        gates.append(outputs["gate"].float().mean(dim=-1).cpu().numpy())
        masks.append(outputs["mask_f"].float().cpu().numpy())
        entropies.append(outputs["entropy_f"].float().cpu().numpy())
    metrics = compute_metrics(
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
        class_names,
    )
    explain = {
        "mean_gate": float(np.concatenate(gates).mean()),
        "mean_entropy_mask": float(np.concatenate(masks).mean()),
        "mean_attention_entropy": float(np.concatenate(entropies).mean()),
    }
    return metrics, explain


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    configure_runtime(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = XGANetConfig.from_dict(payload["config"])
    seed_everything(config.seed)
    class_names = tuple(payload["class_names"])

    frame = load_sampled_frame(args.data, config, cache_path=None)
    arrays_payload = arrays_from_frame(frame, config)
    train_arr, val_arr, test_arr, _splits = split_flow_arrays(arrays_payload, config)
    split_map = {"train": train_arr, "val": val_arr, "test": test_arr}
    dataset = FlowGraphDataset(split_map[args.split])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_flows,
    )
    model = XGANet(config, num_classes=len(class_names)).to(device)
    model.load_state_dict(payload["model"])
    metrics, explain = run_eval(model, loader, device, config.amp_dtype, class_names)
    print(f"split={args.split} accuracy={metrics['accuracy']:.4f} "
          f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
          f"f1={metrics['f1']:.4f} auc={metrics['auc']}")
    print(metrics["report"])
    print(
        "explainability: "
        f"mean_gate={explain['mean_gate']:.4f} "
        f"mean_entropy_mask={explain['mean_entropy_mask']:.4f} "
        f"mean_attention_entropy={explain['mean_attention_entropy']:.4f}"
    )
    out_dir = Path(args.ckpt).parent
    np.save(out_dir / "confusion_matrix.npy", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
