"""Typed, strictly validated local experiment configuration."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML that also rejects ambiguous duplicate configuration keys."""


def unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("configuration keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


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
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    precision: str = "float32"
    validation_interval: int = 0
    validation_documents: int = 2

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "batch_size",
            "sequence_length",
            "accumulation_steps",
            "validation_documents",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be a positive integer")
        if type(self.validation_interval) is not int or self.validation_interval < 0:
            raise ValueError("train.validation_interval must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        for name in ("learning_rate", "gradient_clip", "adam_epsilon"):
            value = finite_number(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"train.{name} must be finite and positive")
        if (
            not 0 <= finite_number(self.min_lr_ratio, "min_lr_ratio") <= 1
            or finite_number(self.weight_decay, "weight_decay") < 0
        ):
            raise ValueError("invalid decay settings")
        if not all(0 <= finite_number(value, "beta") < 1 for value in (self.beta1, self.beta2)):
            raise ValueError("Adam betas must be in [0,1)")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_mix: dict[str, float] = field(default_factory=lambda: {"web": 1.0})
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig) or not isinstance(self.train, TrainConfig):
            raise ValueError("model and train must be typed configurations")
        if self.train.sequence_length > self.model.context:
            raise ValueError("sequence_length exceeds model context")
        if (
            not isinstance(self.data_mix, dict)
            or not self.data_mix
            or any(
                not isinstance(name, str)
                or not name
                or not 0 < finite_number(weight, "source share") <= 1
                for name, weight in self.data_mix.items()
            )
        ):
            raise ValueError("data_mix shares must be positive")
        if abs(sum(self.data_mix.values()) - 1.0) > 1e-8:
            raise ValueError("data_mix shares must sum to one")
        if self.parent_checkpoint is not None and (
            not isinstance(self.parent_checkpoint, str) or not self.parent_checkpoint.strip()
        ):
            raise ValueError("parent_checkpoint must be a nonempty path or null")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        try:
            raw: Any = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
        except yaml.YAMLError as error:
            raise ValueError("invalid or unsafe YAML configuration") from error
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
