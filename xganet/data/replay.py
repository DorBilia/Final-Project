"""Ordered NetFlow replay: chunked file scan, sliding windows, frozen scaling."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from xganet.data.features import extract_flow_features, minmax_scale
from xganet.data.graph import encode_ip_ids, structure_features

FEATURE_REQUIRED = (
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
)

OPTIONAL_COLUMNS = ("Label", "Attack")

DEFAULT_BENIGN_NAME = "Benign"


@dataclass(frozen=True)
class WindowMeta:
    """Per-flow identifiers kept alongside the model batch."""

    src_ip: tuple[str, ...]
    dst_ip: tuple[str, ...]
    row_idx: np.ndarray
    attacks: tuple[str | None, ...]
    labels: np.ndarray | None


def sliding_windows(
    items: Sequence[Any],
    window: int,
    stride: int,
) -> Iterator[list[Any]]:
    """Yield ordered windows, then a leftover slice if it has at least two items."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    n = len(items)
    start = 0
    while start + window <= n:
        yield list(items[start : start + window])
        start += stride
    leftover = list(items[start:])
    if len(leftover) >= 2:
        yield leftover


def iter_sliding_windows(
    records: Iterator[dict[str, Any]],
    window: int,
    stride: int,
) -> Iterator[list[dict[str, Any]]]:
    """Streaming variant of ``sliding_windows`` over an unbounded record iterator."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    buf: list[dict[str, Any]] = []
    for record in records:
        buf.append(record)
        while len(buf) >= window:
            yield buf[:window]
            buf = buf[stride:]
    if len(buf) >= 2:
        yield buf


def _lazy_frame(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.scan_csv(path)
    raise ValueError(f"unsupported replay file type: {path.suffix}")


def _selected_columns(names: Sequence[str]) -> list[str]:
    missing = [col for col in FEATURE_REQUIRED if col not in names]
    if missing:
        raise ValueError(f"replay file missing required columns: {missing}")
    selected = list(FEATURE_REQUIRED)
    for col in OPTIONAL_COLUMNS:
        if col in names:
            selected.append(col)
    if "row_idx" in names:
        selected = ["row_idx", *selected]
    return selected


def iter_flow_chunks(path: str | Path, chunk_size: int = 8192) -> Iterator[pl.DataFrame]:
    """Yield ordered chunks from a CSV or parquet NetFlow file."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    file_path = Path(path)
    lazy = _lazy_frame(file_path)
    for chunk in lazy.collect_batches(chunk_size=chunk_size, maintain_order=True):
        if chunk.height == 0:
            continue
        yield chunk


def _normalize_chunk(chunk: pl.DataFrame, arrival_start: int) -> pl.DataFrame:
    selected = _selected_columns(chunk.columns)
    frame = chunk.select(selected)
    if "row_idx" not in frame.columns:
        frame = frame.with_row_index("row_idx", offset=arrival_start)
    feature_subset = [col for col in FEATURE_REQUIRED if col in frame.columns]
    return frame.drop_nulls(subset=feature_subset)


def iter_flow_records(path: str | Path, chunk_size: int = 8192) -> Iterator[dict[str, Any]]:
    """Yield one NetFlow record at a time in file order."""
    arrival = 0
    for chunk in iter_flow_chunks(path, chunk_size=chunk_size):
        frame = _normalize_chunk(chunk, arrival_start=arrival)
        arrival += chunk.height
        for record in frame.iter_rows(named=True):
            yield dict(record)


def resolve_benign_name(
    class_names: Sequence[str],
    benign_class: str | None = None,
    has_label: bool = False,
) -> str:
    """Pick the class name treated as non-alerting traffic."""
    names = tuple(class_names)
    if benign_class is not None:
        if benign_class not in names:
            raise ValueError(f"benign class {benign_class!r} not in class_names {names}")
        return benign_class
    if DEFAULT_BENIGN_NAME in names:
        return DEFAULT_BENIGN_NAME
    if has_label:
        return names[0]
    raise ValueError(
        f"{DEFAULT_BENIGN_NAME!r} is not in class_names {names}; pass --benign-class"
    )


def should_alert(pred_name: str, prob: float, benign_label: str, threshold: float) -> bool:
    """Alert when the predicted class is not benign and probability meets the cutoff."""
    return pred_name != benign_label and float(prob) >= threshold


def alerts_from_probs(
    probs: np.ndarray,
    class_names: Sequence[str],
    meta: WindowMeta,
    benign_label: str,
    threshold: float,
    window_id: int,
) -> list[dict[str, Any]]:
    """Build alert records for flows in one scored window."""
    if probs.ndim != 2 or probs.shape[0] != len(meta.src_ip):
        raise ValueError("probs must have shape (W, C) matching the window")
    pred_idx = probs.argmax(axis=-1)
    alerts: list[dict[str, Any]] = []
    for i, class_i in enumerate(pred_idx):
        name = str(class_names[int(class_i)])
        prob = float(probs[i, int(class_i)])
        if not should_alert(name, prob, benign_label, threshold):
            continue
        attack = meta.attacks[i]
        alerts.append(
            {
                "window_id": window_id,
                "row_idx": int(meta.row_idx[i]),
                "src_ip": meta.src_ip[i],
                "dst_ip": meta.dst_ip[i],
                "pred": name,
                "prob": prob,
                "attack": attack,
            }
        )
    return alerts


def _column_array(records: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([row[key] for row in records])


def window_to_batch(
    records: Sequence[dict[str, Any]],
    feat_min: np.ndarray,
    feat_max: np.ndarray,
    delta_t: int,
) -> tuple[dict[str, torch.Tensor], WindowMeta]:
    """Scale a window with frozen min–max and build window-local structure features."""
    if len(records) < 2:
        raise ValueError("a replay window must contain at least 2 flows")
    src_ip = tuple(str(row["IPV4_SRC_ADDR"]) for row in records)
    dst_ip = tuple(str(row["IPV4_DST_ADDR"]) for row in records)
    src_ids, dst_ids, _vocab = encode_ip_ids(np.array(src_ip), np.array(dst_ip))
    time_index = np.array([int(row["row_idx"]) for row in records], dtype=np.int64)
    raw_features = extract_flow_features(
        _column_array(records, "FLOW_DURATION_MILLISECONDS"),
        _column_array(records, "IN_BYTES"),
        _column_array(records, "OUT_BYTES"),
        _column_array(records, "IN_PKTS"),
        _column_array(records, "OUT_PKTS"),
        _column_array(records, "L4_SRC_PORT"),
        _column_array(records, "L4_DST_PORT"),
        _column_array(records, "PROTOCOL"),
    )
    features = minmax_scale(raw_features, feat_min, feat_max)
    structure = structure_features(src_ids, dst_ids, time_index, delta_t)
    batch = {
        "features": torch.from_numpy(np.ascontiguousarray(features)),
        "structure": torch.from_numpy(np.ascontiguousarray(structure)),
        "src_ids": torch.from_numpy(src_ids),
        "dst_ids": torch.from_numpy(dst_ids),
        "time_index": torch.from_numpy(time_index),
    }
    attacks = tuple(
        None if row.get("Attack") is None else str(row["Attack"]) for row in records
    )
    labels = None
    if any("Label" in row and row["Label"] is not None for row in records):
        labels = np.array(
            [int(row["Label"]) if row.get("Label") is not None else -1 for row in records],
            dtype=np.int64,
        )
    meta = WindowMeta(
        src_ip=src_ip,
        dst_ip=dst_ip,
        row_idx=time_index,
        attacks=attacks,
        labels=labels,
    )
    return batch, meta


def format_alert_line(alert: dict[str, Any]) -> str:
    """Human-readable stdout line for one alerting flow."""
    gt = alert.get("attack")
    gt_part = f" gt={gt}" if gt not in {None, "None"} else ""
    return (
        f"ALERT row={alert['row_idx']} {alert['src_ip']} -> {alert['dst_ip']} "
        f"pred={alert['pred']} p={alert['prob']:.4f}{gt_part}"
    )
