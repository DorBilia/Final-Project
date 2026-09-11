"""Hyperparameters from X-GANet Table A1 plus locked plan defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class XGANetConfig:
    """Training and model settings matching Table A1 where specified."""

    embedding_dim: int = 128
    xda_layers: int = 4
    attention_heads: int = 8
    diffusion_scale: float = 0.5
    entropy_threshold: float = 1.2
    entropy_alpha: float = 1.0
    contrastive_temperature: float = 0.1
    lambda_contrast: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    batch_size: int = 128
    lcm_init_scale: float = 0.1
    fusion_bias: float = 0.0
    early_stop_patience: int = 10
    epochs: int = 100
    delta_t: int = 100
    sample_size: int = 90600
    min_per_class: int = 2000
    seed: int = 42
    gcn_hidden: int = 128
    ffn_multiplier: int = 4
    train_ratio: float = 0.8
    val_fraction_of_train: float = 0.1
    amp_dtype: str = "bf16"
    num_workers: int = 4
    feature_dim: int = 11
    structure_extra_dim: int = 4
    keep_all_classes: tuple[str, ...] = field(default_factory=lambda: ("Theft",))

    @property
    def structure_in_dim(self) -> int:
        return self.feature_dim + self.structure_extra_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XGANetConfig:
        keep = data.get("keep_all_classes", ("Theft",))
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        filtered["keep_all_classes"] = tuple(keep)
        return cls(**filtered)
