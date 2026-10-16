"""Pure compute estimates; inputs are assumptions, not live prices."""

import math
from dataclasses import dataclass

from scratchcoder.common.config import finite_number


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
    if type(tokens) is not int or tokens < 0 or type(gpus) is not int or gpus < 1:
        raise ValueError("tokens must be nonnegative integers and gpus positive integers")
    throughput = finite_number(tokens_per_second_per_gpu, "throughput")
    slowdown = finite_number(slowdown, "slowdown")
    price = finite_number(dollars_per_gpu_hour, "price")
    if throughput <= 0 or slowdown <= 0:
        raise ValueError("throughput and slowdown must be finite and positive")
    if price < 0:
        raise ValueError("price must be finite and nonnegative")
    ratio = 1.0
    if layer_parameters is not None or reference_parameters is not None:
        if (
            type(layer_parameters) is not int
            or type(reference_parameters) is not int
            or layer_parameters <= 0
            or reference_parameters <= 0
        ):
            raise ValueError("both parameter counts must be positive integers")
        try:
            ratio = layer_parameters / reference_parameters
        except OverflowError as error:
            raise ValueError("parameter ratio must be finite and representable") from error
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("parameter ratio must be finite and representable")
    try:
        hours = tokens / throughput / 3600 * slowdown * ratio
        wall, dollars = hours / gpus, hours * price
    except OverflowError as error:
        raise ValueError("estimate must be finite and representable") from error
    if (
        not all(math.isfinite(value) for value in (hours, wall, dollars))
        or tokens > 0
        and (hours <= 0 or wall <= 0 or price > 0 and dollars <= 0)
    ):
        raise ValueError("estimate must be finite and representable")
    return Estimate(hours, wall, dollars)
