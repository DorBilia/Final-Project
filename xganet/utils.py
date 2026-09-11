"""Runtime helpers for seeding, device selection, and A100-friendly matmul."""

from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device(device_name)


def configure_runtime(device: torch.device) -> None:
    """Enable TF32 and cuDNN autotune on NVIDIA Ampere GPUs (A100)."""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype in {"none", "off", ""}:
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)
