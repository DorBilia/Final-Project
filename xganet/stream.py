"""Replay a NetFlow file in order and emit alerts from a trained X-GANet checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from xganet.config import XGANetConfig
from xganet.data.replay import (
    alerts_from_probs,
    format_alert_line,
    iter_flow_records,
    iter_sliding_windows,
    resolve_benign_name,
    window_to_batch,
)
from xganet.models.xganet import XGANet
from xganet.utils import autocast_context, configure_runtime, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay NetFlow CSV/parquet and print X-GANet alerts"
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/xganet_best.pt")  # Path to the trained model checkpoint
    parser.add_argument("--data", type=str, default="packet_dataset/nf_botiot_v2_sample.parquet")  # Path to the input NetFlow file for streaming
    parser.add_argument("--window", type=int, default=256)  # Number of flows to include in each sliding window
    parser.add_argument("--stride", type=int, default=60)  # Step size for advancing the sliding window
    parser.add_argument("--threshold", type=float, default=0.5)  # Minimum prediction probability required to trigger an alert
    parser.add_argument("--rate", type=float, default=0.0, help="Ingest flows/sec; 0 = unlimited")  # Ingestion rate in flows/sec; 0 means unlimited
    parser.add_argument("--jsonl", type=str, default=None)  # Optional path to output alerts in JSONL format
    parser.add_argument("--max-windows", type=int, default=None)  # Maximum number of sliding windows to process before stopping
    parser.add_argument("--benign-class", type=str, default=None)  # Name of the normal (non-attack) class
    parser.add_argument("--device", type=str, default="cuda")  # Device to run streaming inference on (e.g., 'cuda' or 'cpu')
    parser.add_argument("--chunk-size", type=int, default=8192)  # Number of records to read at once from the file
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def score_window(
    model: XGANet,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_dtype: str,
) -> np.ndarray:
    batch = move_batch(batch, device)
    with autocast_context(device, amp_dtype):
        outputs = model(batch)
    logits = outputs["logits"].float()
    return torch.softmax(logits, dim=-1).cpu().numpy()


def _pace(started: float, n_ingested: int, rate: float) -> None:
    if rate <= 0 or n_ingested <= 0:
        return
    expected = n_ingested / rate
    remaining = expected - (time.perf_counter() - started)
    if remaining > 0:
        time.sleep(remaining)


def _write_jsonl(handle: TextIO, alert: dict[str, Any]) -> None:
    handle.write(json.dumps(alert, ensure_ascii=True) + "\n")
    handle.flush()


class _CountingIter:
    """Count records pulled from the file so --rate paces ingest, not overlaps."""

    def __init__(self, records: Any) -> None:
        self._records = records
        self.n = 0

    def __iter__(self) -> Any:
        for record in self._records:
            self.n += 1
            yield record


def run_stream(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    configure_runtime(device)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = XGANetConfig.from_dict(payload["config"])
    seed_everything(config.seed)
    class_names = tuple(payload["class_names"])
    feat_min = np.asarray(payload["feat_min"], dtype=np.float32)
    feat_max = np.asarray(payload["feat_max"], dtype=np.float32)

    model = XGANet(config, num_classes=len(class_names)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    record_iter = iter_flow_records(args.data, chunk_size=args.chunk_size)
    try:
        first = next(record_iter)
    except StopIteration:
        print("no flows in replay file", file=sys.stderr)
        return 0

    def _all_records():
        yield first
        yield from record_iter

    ingest = _CountingIter(_all_records())
    has_label = first.get("Label") is not None
    benign_label = resolve_benign_name(class_names, args.benign_class, has_label=has_label)

    jsonl_handle: TextIO | None = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    n_windows = 0
    n_alerts = 0
    started = time.perf_counter()
    print(f"Using device: {device}", file=sys.stderr)
    print(
        f"Streaming {args.data} window={args.window} stride={args.stride} "
        f"threshold={args.threshold} benign={benign_label}",
        file=sys.stderr,
    )
    
    y_true_map: dict[int, int] = {}
    y_pred_map: dict[int, int] = {}
    y_prob_map: dict[int, np.ndarray] = {}
    
    try:
        for window_id, records in enumerate(
            iter_sliding_windows(iter(ingest), args.window, args.stride),
            start=1,
        ):
            batch, meta = window_to_batch(records, feat_min, feat_max, config.delta_t)
            probs = score_window(model, batch, device, config.amp_dtype)
            alerts = alerts_from_probs(
                probs, class_names, meta, benign_label, args.threshold, window_id
            )
            for alert in alerts:
                print(format_alert_line(alert), flush=True)
                if jsonl_handle is not None:
                    _write_jsonl(jsonl_handle, alert)
                    
            # Track predictions for evaluation
            preds = probs.argmax(axis=-1)
            for i, row_idx in enumerate(meta.row_idx):
                row_idx = int(row_idx)
                gt = meta.attacks[i]
                if gt is not None and gt in class_names:
                    y_true_map[row_idx] = class_names.index(gt)
                    y_pred_map[row_idx] = preds[i]
                    y_prob_map[row_idx] = probs[i]
                    
            n_windows += 1
            n_alerts += len(alerts)
            print(
                f"[stream] window={window_id} size={len(records)} alerts={len(alerts)}",
                file=sys.stderr,
            )
            _pace(started, ingest.n, args.rate)
            if args.max_windows is not None and n_windows >= args.max_windows:
                break
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    elapsed = time.perf_counter() - started
    print(
        f"done windows={n_windows} alerts={n_alerts} elapsed_s={elapsed:.2f}",
        file=sys.stderr,
    )
    
    if y_true_map:
        from xganet.metrics import compute_metrics
        all_rows = sorted(y_true_map.keys())
        y_true = np.array([y_true_map[k] for k in all_rows])
        y_pred = np.array([y_pred_map[k] for k in all_rows])
        y_prob = np.array([y_prob_map[k] for k in all_rows])
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        print(f"\n--- Streaming Evaluation Metrics ({len(all_rows)} unique flows) ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}" if isinstance(metrics['auc'], float) else f"AUC:       {metrics['auc']}")
        print(metrics["report"])

    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run_stream(args))


if __name__ == "__main__":
    main()
