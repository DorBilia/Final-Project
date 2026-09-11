"""In-memory PyTorch dataset that rebuilds Â per mini-batch."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

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
