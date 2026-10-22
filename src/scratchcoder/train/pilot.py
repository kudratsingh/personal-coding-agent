"""Pure batch, throughput and stop-rule arithmetic for bounded local experiments."""

import math
from dataclasses import dataclass
from typing import Any

from scratchcoder.common.config import finite_number
from scratchcoder.train.trainer import TrainingStop


class PilotStop(TrainingStop):
    """A recorded safety rule ended an experiment; it must not be auto-retried."""


NON_SELECTING_PROBE_STATUSES = frozenset({"oom", "skipped", "incomplete"})


def accumulation_steps(microbatch: int, sequence_length: int, tokens_per_update: int) -> int:
    values = (microbatch, sequence_length, tokens_per_update)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("batch dimensions must be positive integers")
    denominator = microbatch * sequence_length
    if tokens_per_update % denominator:
        raise ValueError("effective token batch must be an exact microbatch multiple")
    return tokens_per_update // denominator


def throughput(metrics: list[dict[str, float]], *, warmup_steps: int = 0) -> dict[str, float]:
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup steps must be a nonnegative integer")
    selected = []
    previous = 0
    for row in metrics:
        step = row["step"]
        tokens = finite_number(row["tokens"], "tokens")
        seconds = finite_number(row["step_seconds"], "step seconds")
        if not math.isfinite(step) or step != int(step) or step <= previous:
            raise ValueError("metric steps must be strictly increasing positive integers")
        if tokens <= 0 or seconds <= 0:
            raise ValueError("measured tokens and seconds must be positive")
        previous = int(step)
        if step > warmup_steps:
            selected.append((tokens, seconds))
    if not selected:
        raise ValueError("no measured post-warmup updates")
    tokens = math.fsum(row[0] for row in selected)
    seconds = math.fsum(row[1] for row in selected)
    rate = tokens / seconds
    if not all(math.isfinite(value) and value > 0 for value in (tokens, seconds, rate)):
        raise ValueError("throughput is not representable")
    return {"tokens": tokens, "step_seconds": seconds, "tokens_per_second": rate}


def cost_per_billion(tokens_per_second: float, quoted_gpu_hourly: float) -> float:
    speed = finite_number(tokens_per_second, "throughput")
    price = finite_number(quoted_gpu_hourly, "GPU hourly quote")
    if speed <= 0 or price < 0:
        raise ValueError("positive throughput and nonnegative price required")
    result = 1_000_000_000 / speed / 3600 * price
    if not math.isfinite(result) or price > 0 and result <= 0:
        raise ValueError("cost per billion is not representable")
    return result


def choose_microbatch(
    probes: list[dict[str, Any]],
    sizes: tuple[int, ...],
    steps: int,
    *,
    max_peak_bytes: float | None = None,
) -> int:
    """Select the largest microbatch whose probe completed its fixed step budget.

    ``oom``, ``skipped`` (probe window expired before it started) and
    ``incomplete`` (probe window expired during it) rows never select; any
    other status is a protocol violation and stops the pilot. A completed
    probe whose reserved peak exceeds ``max_peak_bytes`` is kept in the table
    but marked ``exceeds_headroom`` and never selected: a probe that barely
    fits leaves no room for allocator fragmentation in a 500-update run.
    """
    if max_peak_bytes is not None and finite_number(max_peak_bytes, "peak budget") <= 0:
        raise ValueError("peak budget must be positive")
    if (
        not sizes
        or len(set(sizes)) != len(sizes)
        or any(type(size) is not int or size <= 0 for size in sizes)
        or type(steps) is not int
        or steps <= 0
    ):
        raise ValueError("invalid probe specification")
    if len(probes) != len(sizes) or {row.get("microbatch") for row in probes} != set(sizes):
        raise PilotStop("batch probe inventory incomplete")
    fitting: list[int] = []
    for row in probes:
        size = row["microbatch"]
        if type(size) is not int or size not in sizes:
            raise ValueError("probe microbatch must match the requested integer size")
        if row.get("status") in NON_SELECTING_PROBE_STATUSES:
            continue
        if row.get("status") != "completed" or row.get("updates") != steps:
            raise PilotStop("batch probe did not complete its fixed step budget")
        rate = finite_number(row.get("tokens_per_second"), "probe throughput")
        peak = finite_number(row.get("peak_allocated_bytes"), "probe peak memory")
        reserved = finite_number(row.get("peak_reserved_bytes", peak), "probe reserved memory")
        if rate <= 0 or peak < 0 or reserved < peak:
            raise ValueError("invalid measured probe")
        if max_peak_bytes is not None and reserved > max_peak_bytes:
            row["exceeds_headroom"] = True
            continue
        fitting.append(size)
    if not fitting:
        raise PilotStop("no measured microbatch fits")
    return max(fitting)


@dataclass(frozen=True)
class Limits:
    probe_seconds: float
    training_seconds: float
    pod_seconds: float
    quoted_total_hourly: float
    estimated_total_cost: float
    approved_cost_cap: float
    cost_multiplier: float
    minimum_speed_fraction: float
    speed_check_seconds: float
    idle_seconds: float
    rising_steps: int
    starting_cost_usd: float = 0.0
    finalization_reserve_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_number(getattr(self, name), name)
            if name in {"starting_cost_usd", "finalization_reserve_seconds"} and value == 0:
                continue
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.probe_seconds >= self.pod_seconds
            or self.training_seconds >= self.pod_seconds
            or self.probe_seconds + self.finalization_reserve_seconds >= self.pod_seconds
            or self.minimum_speed_fraction > 1
            or self.cost_multiplier < 1
            or type(self.rising_steps) is not int
            or self.rising_steps < 5
            or self.starting_cost_usd >= self.cost_stop
        ):
            raise ValueError("inconsistent experiment limits")

    @property
    def cost_stop(self) -> float:
        result = min(self.approved_cost_cap, self.estimated_total_cost * self.cost_multiplier)
        if not math.isfinite(result):
            raise ValueError("cost limit is not representable")
        return result

    def training_deadline(self, *, pod_created: float, training_started: float) -> float:
        """Absolute time by which training must stop.

        The earlier of the training budget and the moment at which only the
        finalization reserve (final save, resume test, export, transfer) is
        left before the pod wall limit.
        """
        started = finite_number(training_started, "training start")
        created = finite_number(pod_created, "pod creation")
        if started < created:
            raise ValueError("training cannot start before the pod exists")
        return min(
            started + self.training_seconds,
            created + self.pod_seconds - self.finalization_reserve_seconds,
        )

    def stop_reason(self, *, pod_elapsed: float, idle_elapsed: float = 0) -> str | None:
        elapsed = finite_number(pod_elapsed, "pod elapsed")
        idle = finite_number(idle_elapsed, "idle elapsed")
        if elapsed < 0 or idle < 0:
            raise ValueError("elapsed times cannot be negative")
        if elapsed >= self.pod_seconds:
            return "pod_wall_limit"
        exposure = self.starting_cost_usd + elapsed / 3600 * self.quoted_total_hourly
        if not math.isfinite(exposure) or exposure >= self.cost_stop:
            return "cost_limit"
        if idle >= self.idle_seconds:
            return "idle_gpu_limit"
        return None
