"""Load and subsample NF-BoT-IoT-v2 for X-GANet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from xganet.config import XGANetConfig
from xganet.data.features import extract_flow_features, minmax_fit, minmax_scale
from xganet.data.graph import encode_ip_ids, structure_features

REQUIRED_COLUMNS = (
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
    "Label",
    "Attack",
)

DEFAULT_ATTACK_ORDER = ("Benign", "DDoS", "DoS", "Reconnaissance", "Theft")


@dataclass
class FlowArrays:
    features: np.ndarray
    structure: np.ndarray
    src_ids: np.ndarray
    dst_ids: np.ndarray
    time_index: np.ndarray
    y_attack: np.ndarray
    y_binary: np.ndarray
    class_names: tuple[str, ...]
    feat_min: np.ndarray
    feat_max: np.ndarray


def _class_names(attacks: np.ndarray) -> tuple[str, ...]:
    present = {str(a) for a in np.unique(attacks)}
    ordered = [name for name in DEFAULT_ATTACK_ORDER if name in present]
    extras = sorted(present.difference(DEFAULT_ATTACK_ORDER))
    return tuple(ordered + extras)


def _encode_labels(attacks: np.ndarray, class_names: tuple[str, ...]) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(class_names)}
    return np.array([mapping[str(a)] for a in attacks], dtype=np.int64)


def allocate_class_counts(
    class_counts: dict[str, int],
    sample_size: int,
    keep_all: tuple[str, ...],
    min_per_class: int,
) -> dict[str, int]:
    """Keep rare classes fully, floor others, fill the rest proportionally."""
    alloc: dict[str, int] = {}
    remaining = sample_size
    for name in keep_all:
        take = min(class_counts.get(name, 0), remaining)
        alloc[name] = take
        remaining -= take

    others = [name for name in class_counts if name not in keep_all]
    for name in others:
        available = class_counts[name]
        floor = min(min_per_class, available, remaining)
        alloc[name] = floor
        remaining -= floor

    other_total = sum(class_counts[name] - alloc[name] for name in others)
    if remaining > 0 and other_total > 0:
        leftovers = remaining
        for i, name in enumerate(others):
            spare = class_counts[name] - alloc[name]
            if spare <= 0:
                continue
            if i == len(others) - 1:
                extra = min(spare, leftovers)
            else:
                extra = min(spare, int(round(remaining * spare / other_total)))
            alloc[name] += extra
            leftovers -= extra
        if leftovers > 0:
            for name in others:
                spare = class_counts[name] - alloc[name]
                take = min(spare, leftovers)
                alloc[name] += take
                leftovers -= take
                if leftovers == 0:
                    break
    return alloc


def stratified_sample_indices(
    attacks: np.ndarray,
    sample_size: int,
    seed: int,
    keep_all: tuple[str, ...],
    min_per_class: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    class_counts: dict[str, int] = {}
    groups: dict[str, np.ndarray] = {}
    for name in np.unique(attacks):
        key = str(name)
        idx = np.flatnonzero(attacks == name)
        groups[key] = idx
        class_counts[key] = int(idx.size)

    # Convert sizes to window chunks to preserve burst continuity
    window = 256
    target_chunks = max(1, sample_size // window)
    min_chunks = max(1, min_per_class // window)
    
    chunk_counts = {k: max(1, v // window) for k, v in class_counts.items()}
    
    alloc_chunks = allocate_class_counts(
        chunk_counts, target_chunks, keep_all, min_chunks
    )
    
    chosen_anchors: list[np.ndarray] = []
    for name, take in alloc_chunks.items():
        if take <= 0:
            continue
        pool = groups[name]
        if take >= pool.size:
            chosen_anchors.append(pool)
        else:
            pick = rng.choice(pool, size=take, replace=False)
            chosen_anchors.append(np.sort(pick))
            
    if not chosen_anchors:
        return np.empty(0, dtype=np.int64)
        
    anchors = np.concatenate(chosen_anchors)
    
    # Expand anchors into contiguous sequence bursts
    chosen_rows = set()
    n_total = len(attacks)
    
    for anchor in anchors:
        start = max(0, anchor - window // 2)
        end = min(n_total, start + window)
        for i in range(start, end):
            chosen_rows.add(i)
            
    # Truncate or pad slightly to match sample request exactly (optional, but good for exact sizes)
    sorted_rows = sorted(list(chosen_rows))
    if len(sorted_rows) > sample_size:
        # Take the first sample_size elements, protecting the bursts somewhat
        sorted_rows = sorted_rows[:sample_size]
        
    return np.array(sorted_rows, dtype=np.int64)


def _cache_path(data_path: Path, sample_size: int, seed: int) -> Path:
    del sample_size, seed
    return data_path.parent / "nf_botiot_v2_sample.parquet"


def _cache_meta_path(cache: Path) -> Path:
    return cache.with_suffix(".json")


def _load_cached(cache: Path, sample_size: int, seed: int) -> pl.DataFrame | None:
    meta_path = _cache_meta_path(cache)
    if not cache.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("sample_size", -1)) != sample_size or int(meta.get("seed", -1)) != seed:
        return None
    return pl.read_parquet(cache)


def _write_cache(frame: pl.DataFrame, cache: Path, sample_size: int, seed: int) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache)
    _cache_meta_path(cache).write_text(
        json.dumps({"sample_size": sample_size, "seed": seed, "rows": frame.height}),
        encoding="utf-8",
    )


def _sample_from_csv(
    csv_path: Path,
    sample_size: int,
    seed: int,
    keep_all: tuple[str, ...],
    min_per_class: int,
) -> pl.DataFrame:
    attack_table = (
        pl.scan_csv(csv_path)
        .select("Attack")
        .with_row_index("row_idx")
        .collect()
    )
    attacks = attack_table["Attack"].to_numpy()
    local_idx = stratified_sample_indices(
        attacks, sample_size, seed, keep_all, min_per_class
    )
    keep_rows = attack_table["row_idx"].to_numpy()[local_idx]
    keep_set = set(keep_rows.tolist())
    sampled = (
        pl.scan_csv(csv_path)
        .with_row_index("row_idx")
        .select(["row_idx", *REQUIRED_COLUMNS])
        .filter(pl.col("row_idx").is_in(list(keep_set)))
        .drop_nulls()
        .collect()
    )
    return sampled.sort("row_idx")


def load_sampled_frame(
    data_path: str | Path,
    config: XGANetConfig,
    cache_path: str | Path | None = None,
) -> pl.DataFrame:
    path = Path(data_path)
    cache = Path(cache_path) if cache_path is not None else _cache_path(
        path, config.sample_size, config.seed
    )
    if path.suffix.lower() == ".parquet" and path.exists() and cache_path is None:
        return pl.read_parquet(path)
    cached = _load_cached(cache, config.sample_size, config.seed)
    if cached is not None:
        return cached
    if path.suffix.lower() == ".parquet":
        frame = pl.read_parquet(path)
    else:
        frame = _sample_from_csv(
            path,
            config.sample_size,
            config.seed,
            config.keep_all_classes,
            config.min_per_class,
        )
    _write_cache(frame, cache, config.sample_size, config.seed)
    return frame


def arrays_from_frame(frame: pl.DataFrame, config: XGANetConfig) -> dict[str, np.ndarray | tuple[str, ...]]:
    if "row_idx" not in frame.columns:
        frame = frame.with_row_index("row_idx")
    src_ids, dst_ids, _vocab = encode_ip_ids(
        frame["IPV4_SRC_ADDR"].to_numpy(),
        frame["IPV4_DST_ADDR"].to_numpy(),
    )
    time_index = frame["row_idx"].to_numpy().astype(np.int64, copy=False)
    raw_features = extract_flow_features(
        frame["FLOW_DURATION_MILLISECONDS"].to_numpy(),
        frame["IN_BYTES"].to_numpy(),
        frame["OUT_BYTES"].to_numpy(),
        frame["IN_PKTS"].to_numpy(),
        frame["OUT_PKTS"].to_numpy(),
        frame["L4_SRC_PORT"].to_numpy(),
        frame["L4_DST_PORT"].to_numpy(),
        frame["PROTOCOL"].to_numpy(),
    )
    structure = structure_features(src_ids, dst_ids, time_index, config.delta_t)
    attacks = frame["Attack"].to_numpy()
    class_names = _class_names(attacks)
    y_attack = _encode_labels(attacks, class_names)
    y_binary = frame["Label"].to_numpy().astype(np.int64, copy=False)
    return {
        "raw_features": raw_features,
        "structure": structure,
        "src_ids": src_ids,
        "dst_ids": dst_ids,
        "time_index": time_index,
        "y_attack": y_attack,
        "y_binary": y_binary,
        "class_names": class_names,
    }


def split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_fraction_of_train: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split the data to preserve the sequential window integrity."""
    n = len(y)
    all_idx = np.arange(n)
    if n < 5:
        return all_idx, all_idx, all_idx

    # Chronologically split instead of train_test_split scatter
    train_end = int(n * train_ratio)
    val_end = train_end + int(train_end * val_fraction_of_train)
    
    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]

    return train_idx, val_idx, test_idx


def _subset(payload: dict[str, np.ndarray | tuple[str, ...]], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw_features": payload["raw_features"][idx],
        "structure": payload["structure"][idx],
        "src_ids": payload["src_ids"][idx],
        "dst_ids": payload["dst_ids"][idx],
        "time_index": payload["time_index"][idx],
        "y_attack": payload["y_attack"][idx],
        "y_binary": payload["y_binary"][idx],
    }


def split_flow_arrays(
    payload: dict[str, np.ndarray | tuple[str, ...]],
    config: XGANetConfig,
) -> tuple[FlowArrays, FlowArrays, FlowArrays, dict[str, np.ndarray]]:
    y = payload["y_attack"]
    train_idx, val_idx, test_idx = split_indices(
        y, config.train_ratio, config.val_fraction_of_train, config.seed
    )
    train_raw = payload["raw_features"][train_idx]
    feat_min, feat_max = minmax_fit(train_raw)
    class_names = payload["class_names"]

    def _make(idx: np.ndarray) -> FlowArrays:
        if len(idx) == 0:
            idx = np.array([0]) # Fallback to prevent crash on empty split
        part = _subset(payload, idx)
        return FlowArrays(
            features=minmax_scale(part["raw_features"], feat_min, feat_max),
            structure=part["structure"],
            src_ids=part["src_ids"],
            dst_ids=part["dst_ids"],
            time_index=part["time_index"],
            y_attack=part["y_attack"],
            y_binary=part["y_binary"],
            class_names=class_names,
            feat_min=feat_min,
            feat_max=feat_max,
        )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return _make(train_idx), _make(val_idx), _make(test_idx), splits


def load_flow_arrays(
    data_path: str | Path,
    config: XGANetConfig,
    cache_path: str | Path | None = None,
) -> tuple[FlowArrays, FlowArrays, FlowArrays, dict[str, np.ndarray]]:
    frame = load_sampled_frame(data_path, config, cache_path=cache_path)
    payload = arrays_from_frame(frame, config)
    return split_flow_arrays(payload, config)
