"""Pure compute estimates; inputs are assumptions, not live prices."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    gpu_hours: float
    wall_hours: float
    dollars: float


def estimate(
    tokens: int,
    tokens_per_second_per_gpu: float,
    dollars_per_gpu_hour: float,
    *,
    gpus: int = 1,
    slowdown: float = 1.0,
    layer_parameters: int | None = None,
    reference_parameters: int | None = None,
) -> Estimate:
    """Price total GPU work and wall time, optionally scaling a reference speed."""
    if tokens < 0 or gpus < 1:
        raise ValueError("tokens must be nonnegative and gpus positive")
    for value in (tokens_per_second_per_gpu, slowdown):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("throughput and slowdown must be finite and positive")
    if not math.isfinite(dollars_per_gpu_hour) or dollars_per_gpu_hour < 0:
        raise ValueError("price must be finite and nonnegative")
    ratio = 1.0
    if layer_parameters is not None or reference_parameters is not None:
        if not layer_parameters or not reference_parameters:
            raise ValueError("both positive parameter counts are required")
        if min(layer_parameters, reference_parameters) <= 0:
            raise ValueError("parameter counts must be positive")
        ratio = layer_parameters / reference_parameters
    hours = tokens / tokens_per_second_per_gpu / 3600 * slowdown * ratio
    return Estimate(hours, hours / gpus, hours * dollars_per_gpu_hour)
