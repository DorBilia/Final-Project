"""CPU tests for NetFlow replay windows, frozen scaling, and alert rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from xganet.config import XGANetConfig
from xganet.data.features import extract_flow_features, minmax_fit, minmax_scale
from xganet.data.replay import (
    WindowMeta,
    alerts_from_probs,
    format_alert_line,
    iter_flow_records,
    iter_sliding_windows,
    resolve_benign_name,
    should_alert,
    sliding_windows,
    window_to_batch,
)
from xganet.models.xganet import XGANet


def _record(index: int, attack: str = "Benign", src: str | None = None, dst: str | None = None) -> dict:
    return {
        "row_idx": index,
        "IPV4_SRC_ADDR": src or f"10.0.0.{index % 5}",
        "IPV4_DST_ADDR": dst or f"10.0.1.{index % 3}",
        "L4_SRC_PORT": 1000 + index,
        "L4_DST_PORT": 80,
        "PROTOCOL": 6,
        "IN_BYTES": 100 + index,
        "IN_PKTS": 2 + (index % 3),
        "OUT_BYTES": 50 + index,
        "OUT_PKTS": 1 + (index % 2),
        "FLOW_DURATION_MILLISECONDS": 1000 + 10 * index,
        "Label": 0 if attack == "Benign" else 1,
        "Attack": attack,
    }


def test_sliding_windows_non_overlapping() -> None:
    items = list(range(20))
    windows = list(sliding_windows(items, window=8, stride=8))
    assert [len(w) for w in windows] == [8, 8, 4]
    assert windows[0] == list(range(0, 8))
    assert windows[1] == list(range(8, 16))
    assert windows[2] == list(range(16, 20))


def test_sliding_windows_overlapping_counts() -> None:
    items = list(range(20))
    windows = list(sliding_windows(items, window=8, stride=4))
    full = [w for w in windows if len(w) == 8]
    leftover = [w for w in windows if len(w) < 8]
    assert len(full) == 4
    assert full[0] == list(range(0, 8))
    assert full[1] == list(range(4, 12))
    assert leftover == [list(range(16, 20))]


def test_sliding_windows_skips_lone_leftover() -> None:
    windows = list(sliding_windows(list(range(9)), window=8, stride=8))
    assert len(windows) == 1
    assert windows[0] == list(range(8))


def test_iter_sliding_windows_matches_list() -> None:
    records = [_record(i) for i in range(20)]
    streamed = list(iter_sliding_windows(iter(records), window=8, stride=8))
    listed = list(sliding_windows(records, window=8, stride=8))
    assert streamed == listed


def test_stream_features_use_frozen_minmax() -> None:
    records = [_record(i, attack="DDoS" if i % 2 else "Benign") for i in range(8)]
    raw = extract_flow_features(
        np.array([row["FLOW_DURATION_MILLISECONDS"] for row in records]),
        np.array([row["IN_BYTES"] for row in records]),
        np.array([row["OUT_BYTES"] for row in records]),
        np.array([row["IN_PKTS"] for row in records]),
        np.array([row["OUT_PKTS"] for row in records]),
        np.array([row["L4_SRC_PORT"] for row in records]),
        np.array([row["L4_DST_PORT"] for row in records]),
        np.array([row["PROTOCOL"] for row in records]),
    )
    extra = extract_flow_features(
        np.array([2000.0, 50.0]),
        np.array([900.0, 10.0]),
        np.array([400.0, 5.0]),
        np.array([20.0, 1.0]),
        np.array([10.0, 1.0]),
        np.array([443.0, 22.0]),
        np.array([8080.0, 53.0]),
        np.array([6, 17]),
    )
    feat_min, feat_max = minmax_fit(np.concatenate([raw, extra], axis=0))
    expected = minmax_scale(raw, feat_min, feat_max)
    batch, meta = window_to_batch(records, feat_min, feat_max, delta_t=100)
    np.testing.assert_allclose(batch["features"].numpy(), expected, atol=1e-6)
    assert batch["structure"].shape == (8, 4)
    assert meta.row_idx.tolist() == list(range(8))
    assert batch["src_ids"].shape == (8,)


def test_alert_rule_benign_and_threshold() -> None:
    class_names = ("Benign", "DDoS")
    probs = np.array(
        [
            [0.91, 0.09],
            [0.10, 0.90],
            [0.60, 0.40],
        ],
        dtype=np.float32,
    )
    meta = WindowMeta(
        src_ip=("10.0.0.1", "10.0.0.2", "10.0.0.3"),
        dst_ip=("10.0.1.1", "10.0.1.2", "10.0.1.3"),
        row_idx=np.array([10, 11, 12], dtype=np.int64),
        attacks=("Benign", "DDoS", "DDoS"),
        labels=None,
    )
    assert should_alert("Benign", 0.99, "Benign", 0.5) is False
    assert should_alert("DDoS", 0.90, "Benign", 0.5) is True
    assert should_alert("DDoS", 0.40, "Benign", 0.5) is False
    alerts = alerts_from_probs(probs, class_names, meta, "Benign", 0.5, window_id=1)
    assert len(alerts) == 1
    assert alerts[0]["pred"] == "DDoS"
    assert alerts[0]["row_idx"] == 11
    assert "DDoS" in format_alert_line(alerts[0])


def test_resolve_benign_name() -> None:
    assert resolve_benign_name(("Benign", "DDoS")) == "Benign"
    assert resolve_benign_name(("Normal", "DDoS"), benign_class="Normal") == "Normal"
    assert resolve_benign_name(("Normal", "DDoS"), has_label=True) == "Normal"


def test_untrained_model_scores_replay_window() -> None:
    records = [_record(i) for i in range(8)]
    raw = extract_flow_features(
        np.array([row["FLOW_DURATION_MILLISECONDS"] for row in records]),
        np.array([row["IN_BYTES"] for row in records]),
        np.array([row["OUT_BYTES"] for row in records]),
        np.array([row["IN_PKTS"] for row in records]),
        np.array([row["OUT_PKTS"] for row in records]),
        np.array([row["L4_SRC_PORT"] for row in records]),
        np.array([row["L4_DST_PORT"] for row in records]),
        np.array([row["PROTOCOL"] for row in records]),
    )
    feat_min, feat_max = minmax_fit(raw)
    batch, _meta = window_to_batch(records, feat_min, feat_max, delta_t=100)
    config = XGANetConfig(
        embedding_dim=32,
        xda_layers=2,
        attention_heads=4,
        gcn_hidden=32,
        ffn_multiplier=2,
    )
    model = XGANet(config, num_classes=3)
    model.eval()
    with torch.no_grad():
        logits = model(batch)["logits"]
    assert logits.shape == (8, 3)
    assert torch.isfinite(logits).all()


def test_iter_flow_records_from_csv_and_parquet(tmp_path: Path) -> None:
    rows = [_record(i, attack="DDoS" if i == 3 else "Benign") for i in range(5)]
    frame = pl.DataFrame(rows).drop("row_idx")
    csv_path = tmp_path / "flows.csv"
    parquet_path = tmp_path / "flows.parquet"
    frame.write_csv(csv_path)
    frame.write_parquet(parquet_path)
    csv_records = list(iter_flow_records(csv_path, chunk_size=2))
    parquet_records = list(iter_flow_records(parquet_path, chunk_size=2))
    assert [row["IPV4_SRC_ADDR"] for row in parquet_records] == [
        row["IPV4_SRC_ADDR"] for row in csv_records
    ]


def test_window_to_batch_with_struct_scaling() -> None:
    records = [_record(i) for i in range(8)]
    feat_min = np.zeros(11, dtype=np.float32)
    feat_max = np.ones(11, dtype=np.float32) * 1000
    struct_min = np.zeros(4, dtype=np.float32)
    struct_max = np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float32)
    batch, meta = window_to_batch(
        records, feat_min, feat_max, delta_t=100, struct_min=struct_min, struct_max=struct_max
    )
    struct = batch["structure"].numpy()
    assert struct.shape == (8, 4)
    assert np.all(struct >= 0.0)
    assert np.all(struct <= 1.0)


def test_window_collate() -> None:
    from xganet.data.dataset import WindowCollate
    struct_min = np.zeros(4, dtype=np.float32)
    struct_max = np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float32)
    collate = WindowCollate(struct_min=struct_min, struct_max=struct_max, delta_t=100)
    items = [
        {
            "features": torch.rand(11),
            "src_ids": torch.tensor(i % 3),
            "dst_ids": torch.tensor((i + 1) % 3),
            "time_index": torch.tensor(i),
            "y_attack": torch.tensor(0),
            "y_binary": torch.tensor(0),
        }
        for i in range(6)
    ]
    batch = collate(items)
    assert batch["features"].shape == (6, 11)
    assert batch["structure"].shape == (6, 4)
    assert batch["src_ids"].shape == (6,)
    assert np.all(batch["structure"].numpy() >= 0.0)
    assert np.all(batch["structure"].numpy() <= 1.0)
