"""In-memory PyTorch dataset that rebuilds Â per mini-batch."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from xganet.data.features import minmax_scale
from xganet.data.graph import structure_features
from xganet.data.load_nf import FlowArrays


class FlowGraphDataset(Dataset):
    def __init__(self, arrays: FlowArrays) -> None:
        self.features = torch.from_numpy(arrays.features)
        self.structure = torch.from_numpy(arrays.structure)
        self.src_ids = torch.from_numpy(arrays.src_ids)
        self.dst_ids = torch.from_numpy(arrays.dst_ids)
        self.time_index = torch.from_numpy(arrays.time_index)
        self.y_attack = torch.from_numpy(arrays.y_attack)
        self.y_binary = torch.from_numpy(arrays.y_binary)
        self.class_names = arrays.class_names
        self.feat_min = arrays.feat_min
        self.feat_max = arrays.feat_max
        self.struct_min = getattr(arrays, "struct_min", None)
        self.struct_max = getattr(arrays, "struct_max", None)

    def __len__(self) -> int:
        return int(self.features.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "structure": self.structure[index],
            "src_ids": self.src_ids[index],
            "dst_ids": self.dst_ids[index],
            "time_index": self.time_index[index],
            "y_attack": self.y_attack[index],
            "y_binary": self.y_binary[index],
        }


def collate_flows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "structure": torch.stack([item["structure"] for item in batch], dim=0),
        "src_ids": torch.stack([item["src_ids"] for item in batch], dim=0),
        "dst_ids": torch.stack([item["dst_ids"] for item in batch], dim=0),
        "time_index": torch.stack([item["time_index"] for item in batch], dim=0),
        "y_attack": torch.stack([item["y_attack"] for item in batch], dim=0),
        "y_binary": torch.stack([item["y_binary"] for item in batch], dim=0),
    }


class WindowCollate:
    """Collate function that computes structure features dynamically per-batch.

    During training and evaluation, this replaces globally-precomputed structure
    features with window-local structure features (matching streaming replay)
    and applies frozen struct min–max scaling. This eliminates the train/stream
    distribution shift.
    """

    def __init__(
        self,
        struct_min: np.ndarray | None,
        struct_max: np.ndarray | None,
        delta_t: int,
    ) -> None:
        self.struct_min = struct_min
        self.struct_max = struct_max
        self.delta_t = delta_t

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        features = torch.stack([item["features"] for item in batch], dim=0)
        src_ids = torch.stack([item["src_ids"] for item in batch], dim=0)
        dst_ids = torch.stack([item["dst_ids"] for item in batch], dim=0)
        time_index = torch.stack([item["time_index"] for item in batch], dim=0)
        y_attack = torch.stack([item["y_attack"] for item in batch], dim=0)
        y_binary = torch.stack([item["y_binary"] for item in batch], dim=0)

        # Compute structure features window-locally on this batch
        struct_np = structure_features(
            src_ids.numpy(), dst_ids.numpy(), time_index.numpy(), self.delta_t
        )
        if self.struct_min is not None and self.struct_max is not None:
            struct_np = minmax_scale(struct_np, self.struct_min, self.struct_max)
        structure = torch.from_numpy(np.ascontiguousarray(struct_np))

        return {
            "features": features,
            "structure": structure,
            "src_ids": src_ids,
            "dst_ids": dst_ids,
            "time_index": time_index,
            "y_attack": y_attack,
            "y_binary": y_binary,
        }


class SlidingWindowBatchSampler(Sampler[list[int]]):
    """Yields batches of contiguous overlapping blocks instead of random individual items."""
    def __init__(self, data_size: int, window_size: int, stride: int, shuffle_windows: bool = True) -> None:
        self.data_size = data_size
        self.window_size = window_size
        self.stride = stride
        self.shuffle_windows = shuffle_windows
        self.windows = []
        
        # Precompute valid window index ranges
        start = 0
        while start + self.window_size <= self.data_size:
            self.windows.append(list(range(start, start + self.window_size)))
            start += self.stride
            
        # Handle the leftover chunk if desired (omitted for strict window sizing)
        
    def __iter__(self):
        if self.shuffle_windows:
            # Shuffle the order of the windows for the epoch
            order = torch.randperm(len(self.windows)).tolist()
            for idx in order:
                yield self.windows[idx]
        else:
            for window in self.windows:
                yield window
                
    def __len__(self) -> int:
        return len(self.windows)
