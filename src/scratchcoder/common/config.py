"""Typed, strictly validated local experiment configuration."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 512
    width: int = 256
    layers: int = 4
    heads: int = 4
    context: int = 256

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("head dimension must be even and divide width")
        if self.vocab_size < 256:
            raise ValueError("vocab_size must include all 256 bytes")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 5
    batch_size: int = 2
    sequence_length: int = 32
    learning_rate: float = 0.0006
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "sequence_length", "accumulation_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"train.{name} must be finite and positive")
        if not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.weight_decay < float("inf"):
            raise ValueError("invalid decay settings")


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_mix: dict[str, float] = field(default_factory=lambda: {"web": 1.0})
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.train.sequence_length > self.model.context:
            raise ValueError("sequence_length exceeds model context")
        if not self.data_mix or any(not 0 < weight <= 1 for weight in self.data_mix.values()):
            raise ValueError("data_mix shares must be positive")
        if abs(sum(self.data_mix.values()) - 1.0) > 1e-8:
            raise ValueError("data_mix shares must sum to one")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a mapping")
        raw = dict(raw)
        try:
            return cls(
                model=ModelConfig(**raw.pop("model", {})),
                train=TrainConfig(**raw.pop("train", {})),
                **raw,
            )
        except TypeError as error:
            raise ValueError(f"invalid configuration fields: {error}") from error

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True), encoding="utf-8")
