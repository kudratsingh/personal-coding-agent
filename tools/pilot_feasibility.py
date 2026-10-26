"""Pure-arithmetic feasibility calculator for the bounded single-A40 pilot.

Every number produced here is derived analytically from the pilot specification,
or from one CPU-measured allocator proxy. Nothing in this module is, or depends
on, a CUDA or A40 measurement. The report exists so that the readiness audit has
reproducible, clearly labelled figures before any paid run is approved. Running
this tool launches no compute and incurs no cost.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from scratchcoder.common.config import ModelConfig
from scratchcoder.model.gpt import expected_parameters
from tools import rehearsal_ops as d35

GPU_DISCLAIMER = (
    "All GPU numbers are analytical or CPU-proxy estimates; no A40 measurement exists yet"
)

GIB = 2**30
MIB = 2**20
GB = 1_000_000_000

# fp32 parameters, fp32 gradients, Adam first moment, Adam second moment.
STATIC_BYTES_PER_PARAMETER = 4 * 4
# Checkpoint carries fp32 parameters plus two fp32 Adam moments per parameter.
CHECKPOINT_BYTES_PER_PARAMETER = 4 * 3
# Small allowance for checkpoint headers, metadata and optimizer counters.
CHECKPOINT_OVERHEAD_BYTES = MIB
# Measured on this implementation (CPU save of an untrained pilot-S model);
# a file-size observation, not a GPU measurement.
MEASURED_UNTRAINED_CHECKPOINT_BYTES = 1_623_502_465
MEASURED_UNTRAINED_CHECKPOINT_PROVENANCE = (
    "untrained pilot-S checkpoint saved by this implementation on CPU: "
    "1,623,502,465 bytes = 1.5120 GiB; a file-size observation, not a GPU measurement"
)

DEFAULT_ACTIVATION_GIB_PER_SEQUENCE = 1.72
ACTIVATION_PROXY_PROVENANCE = (
    "CPU allocator proxy measured 2026-10-22 on this implementation with fused attention and "
    "the log-softmax cross-entropy: 1,760 MiB growth for one 1024-token sequence, 1,720 MiB "
    "for the second (reference attention: 2,617 / 2,467 MiB; original code: 2,919 / 2,871 MiB); "
    "not a CUDA measurement"
)
# bf16 logits, fp32 copy, shifted copy, exp: the analytical output-layer term.
OUTPUT_LAYER_BYTES_PER_LOGIT = 2 + 4 + 4 + 4

DEFAULT_GPU_MEMORY_GIB = 48.0
DEFAULT_USABLE_GPU_MEMORY_GIB = 44.5
MARGINAL_FRACTION = 0.85
OOM_FRACTION = 1.0
VERDICT_FITS = "likely fits"
VERDICT_MARGINAL = "marginal"
VERDICT_OOM = "likely OOM"

DEFAULT_THROUGHPUTS = (5000.0, 10000.0, 15000.0, 25000.0, 40000.0)
DEFAULT_STARTUP_SECONDS = 600.0  # image pull, bootstrap, data staging, preflight
DEFAULT_TRANSFER_MBPS = 20.0
# The runbook arms the in-Pod timer at ORIGIN + pod_seconds - 180, and the worker plans
# against that earlier deadline, so the usable lifetime is 180 s short of the wall.
DEFAULT_SELF_DESTRUCT_MARGIN_SECONDS = 180.0
DEFAULT_PEAK_TFLOPS = 149.7  # A40 dense bf16 tensor-core peak

CHECKPOINT_IO_BYTES_PER_SECOND = 1e9
FINAL_VALIDATION_WINDOWS = 32
FINAL_VALIDATION_SECONDS = 10.0
# Guards floor() against float rounding at exact break-even throughputs.
UPDATE_FLOOR_TOLERANCE = 1e-6

SHARD_TOKENS = 66_030_000
TOKEN_BYTES = 2
VALIDATION_TOKENS = 32 * 1025
TOKENIZER_BYTES = 3_000_000
DISK_HEADROOM = 1.25
DISK_ROUND_GB = 10
DISK_MINIMUM_GB = 20

NETWORK_VOLUME_GB = 30
NETWORK_VOLUME_DOLLARS_PER_GB_MONTH = 0.07
NETWORK_VOLUME_CONCLUSION = (
    "Not required: the ~135 MB dataset is staged into the container disk at startup and "
    "checkpoints are retrieved by Mac-initiated transfer before the pod is destroyed, so a "
    "network volume would only add a recurring charge billed hourly until deleted."
)


# ---------------------------------------------------------------------------
# Specification


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _count_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


@dataclass(frozen=True)
class PilotSpecification:
    """The subset of the pilot JSON specification that the calculator consumes."""

    model: ModelConfig
    steps: int
    sequence_length: int
    warmup_steps: int
    microbatches: tuple[int, ...]
    probe_steps: int
    tokens_per_update: int
    learning_rates: tuple[float, ...]
    screen_steps: int
    probe_seconds: float
    training_seconds: float
    pod_seconds: float
    quoted_total_hourly: float
    quoted_gpu_hourly: float
    estimated_total_cost: float
    approved_cost_cap: float
    finalization_reserve_seconds: float
    probe_accumulation_steps: int = 2
    compile_probe_steps: int = 0
    compile_probe_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.microbatches or not self.learning_rates:
            raise ValueError("microbatches and learning_rates must be nonempty")
        if self.screen_steps > self.steps:
            raise ValueError("screen_steps must not exceed steps")
        if self.sequence_length > self.model.context:
            raise ValueError("sequence_length exceeds model context")
        if self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must not exceed steps")

    @property
    def arms(self) -> int:
        """Number of learning-rate screening arms."""
        return len(self.learning_rates)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PilotSpecification:
        spec = _mapping(raw, "specification")
        config = _mapping(spec.get("config"), "config")
        model = _mapping(config.get("model"), "config.model")
        train = _mapping(config.get("train"), "config.train")
        limits = _mapping(spec.get("limits"), "limits")
        microbatches = spec.get("microbatches")
        learning_rates = spec.get("learning_rates")
        if not isinstance(microbatches, list) or not isinstance(learning_rates, list):
            raise ValueError("microbatches and learning_rates must be lists")
        try:
            model_config = ModelConfig(**model)
        except TypeError as error:
            raise ValueError(f"invalid config.model fields: {error}") from error
        return cls(
            model=model_config,
            steps=_integer(train.get("steps"), "config.train.steps"),
            sequence_length=_integer(train.get("sequence_length"), "config.train.sequence_length"),
            warmup_steps=_integer(train.get("warmup_steps"), "config.train.warmup_steps"),
            microbatches=tuple(_integer(item, "microbatches[]") for item in microbatches),
            probe_steps=_integer(spec.get("probe_steps"), "probe_steps"),
            tokens_per_update=_integer(spec.get("tokens_per_update"), "tokens_per_update"),
            learning_rates=tuple(_number(item, "learning_rates[]") for item in learning_rates),
            screen_steps=_integer(spec.get("screen_steps"), "screen_steps"),
            probe_seconds=_number(limits.get("probe_seconds"), "limits.probe_seconds"),
            training_seconds=_number(limits.get("training_seconds"), "limits.training_seconds"),
            pod_seconds=_number(limits.get("pod_seconds"), "limits.pod_seconds"),
            quoted_total_hourly=_number(
                limits.get("quoted_total_hourly"), "limits.quoted_total_hourly"
            ),
            quoted_gpu_hourly=_number(spec.get("quoted_gpu_hourly"), "quoted_gpu_hourly"),
            estimated_total_cost=_number(
                limits.get("estimated_total_cost"), "limits.estimated_total_cost"
            ),
            approved_cost_cap=_number(limits.get("approved_cost_cap"), "limits.approved_cost_cap"),
            finalization_reserve_seconds=_number(
                limits.get("finalization_reserve_seconds"), "limits.finalization_reserve_seconds"
            ),
            probe_accumulation_steps=_integer(
                spec.get("probe_accumulation_steps", 2), "probe_accumulation_steps"
            ),
            compile_probe_steps=_count(spec.get("compile_probe_steps", 0), "compile_probe_steps"),
            compile_probe_seconds=_count_number(
                spec.get("compile_probe_seconds", 0), "compile_probe_seconds"
            ),
        )

    @classmethod
    def load(cls, path: Path) -> PilotSpecification:
        with path.open(encoding="utf-8") as handle:
            raw: object = json.load(handle)
        return cls.from_mapping(_mapping(raw, "specification"))


# ---------------------------------------------------------------------------
# Static model arithmetic


def model_parameters(spec: PilotSpecification) -> int:
    """Parameter count from the shared model formula (no weights are built)."""
    return expected_parameters(spec.model)


def static_bytes(parameters: int) -> int:
    """Resident optimizer state: fp32 params, grads, Adam m and Adam v (16 B/param)."""
    return parameters * STATIC_BYTES_PER_PARAMETER


def checkpoint_bytes_estimate(
    parameters: int, overhead_bytes: int = CHECKPOINT_OVERHEAD_BYTES
) -> int:
    """Analytical checkpoint size: 12 B/param (params + two Adam moments) plus overhead."""
    return parameters * CHECKPOINT_BYTES_PER_PARAMETER + overhead_bytes


def flops_per_token(parameters: int, layers: int, context: int, width: int) -> int:
    """Training FLOPs per token: 6N for weights plus 12·L·T·D for attention scores."""
    return 6 * parameters + 12 * layers * context * width


def total_updates(spec: PilotSpecification) -> int:
    """Optimizer updates in one pilot: every arm's screen plus the winner's continuation."""
    return spec.arms * spec.screen_steps + (spec.steps - spec.screen_steps)


# ---------------------------------------------------------------------------
# Memory


def accumulation_steps(tokens_per_update: int, microbatch: int, sequence_length: int) -> int:
    """Microbatches per update; the update must be an exact multiple of one microbatch."""
    tokens_per_microbatch = microbatch * sequence_length
    if tokens_per_microbatch <= 0 or tokens_per_update % tokens_per_microbatch:
        raise ValueError(
            f"tokens_per_update {tokens_per_update} is not a multiple of "
            f"microbatch {microbatch} x sequence_length {sequence_length}"
        )
    return tokens_per_update // tokens_per_microbatch


def activation_bytes(microbatch: int, activation_gib_per_sequence: float) -> float:
    """CPU-proxy activation footprint scaled linearly with the microbatch."""
    return microbatch * activation_gib_per_sequence * GIB


def output_layer_bytes(microbatch: int, sequence_length: int, vocab_size: int) -> int:
    """Analytical logits/cross-entropy term: B·T·V·(2 + 4 + 4 + 4) bytes.

    This is reported separately. It is already partly inside the CPU proxy, so it is
    not added to the fit decision.
    """
    return microbatch * sequence_length * vocab_size * OUTPUT_LAYER_BYTES_PER_LOGIT


def memory_verdict(total_bytes: float, usable_bytes: float) -> str:
    """Three-way verdict at 85% and 100% of usable device memory."""
    fraction = total_bytes / usable_bytes
    if fraction < MARGINAL_FRACTION:
        return VERDICT_FITS
    if fraction <= OOM_FRACTION:
        return VERDICT_MARGINAL
    return VERDICT_OOM


@dataclass(frozen=True)
class MicrobatchEstimate:
    microbatch: int
    accumulation_steps: int
    static_bytes: int
    activation_proxy_bytes: float
    proxy_total_bytes: float
    output_layer_analytical_bytes: int
    usable_bytes: float
    fraction_of_usable: float
    fits: bool
    verdict: str


def microbatch_table(
    spec: PilotSpecification,
    parameters: int,
    activation_gib_per_sequence: float,
    usable_gpu_memory_gib: float,
) -> list[MicrobatchEstimate]:
    """Per-microbatch memory table; fits = static + activation proxy <= usable."""
    usable = usable_gpu_memory_gib * GIB
    resident = static_bytes(parameters)
    rows: list[MicrobatchEstimate] = []
    for microbatch in spec.microbatches:
        activation = activation_bytes(microbatch, activation_gib_per_sequence)
        total = resident + activation
        rows.append(
            MicrobatchEstimate(
                microbatch=microbatch,
                accumulation_steps=accumulation_steps(
                    spec.tokens_per_update, microbatch, spec.sequence_length
                ),
                static_bytes=resident,
                activation_proxy_bytes=activation,
                proxy_total_bytes=total,
                output_layer_analytical_bytes=output_layer_bytes(
                    microbatch, spec.sequence_length, spec.model.vocab_size
                ),
                usable_bytes=usable,
                fraction_of_usable=total / usable,
                fits=total <= usable,
                verdict=memory_verdict(total, usable),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Throughput


def required_throughput(tokens_per_update: int, updates: int, training_seconds: float) -> float:
    """Tokens per second needed to finish `updates` within the training budget."""
    return tokens_per_update * updates / training_seconds


def implied_tflops(throughput: float, flops_token: int) -> float:
    """Achieved TFLOPS implied by a token throughput."""
    return throughput * flops_token / 1e12


def model_flops_utilization(throughput: float, flops_token: int, peak_tflops: float) -> float:
    """Fraction of dense peak that a token throughput would represent."""
    return implied_tflops(throughput, flops_token) / peak_tflops


@dataclass(frozen=True)
class ThroughputRequirement:
    label: str
    updates: int
    throughput: float
    implied_tflops: float
    mfu: float


# ---------------------------------------------------------------------------
# Schedule


@dataclass(frozen=True)
class Schedule:
    throughput: float
    implied_tflops: float
    mfu: float
    seconds_per_update: float
    probe_microbatches: tuple[int, ...]
    probe_seconds: float
    probe_within_limit: bool
    compile_probe_seconds: float
    pod_lifetime_seconds: float
    screens_seconds: float
    continuation_seconds: float
    training_deadline_seconds: float
    updates_total: int
    updates_reached: int
    screens_completed: int
    outcome: str
    training_seconds_used: float
    finalization_seconds: float
    transfer_seconds: float
    startup_seconds: float
    total_pod_seconds: float
    cost_dollars: float
    fits_pod: bool
    fits_cost_cap: bool


def finalization_seconds(checkpoint_bytes: int) -> float:
    """Checkpoint save + reload at 1 GB/s plus a fixed final validation pass."""
    return 2 * checkpoint_bytes / CHECKPOINT_IO_BYTES_PER_SECOND + FINAL_VALIDATION_SECONDS


def transfer_seconds(checkpoint_bytes: int, transfer_mbps: float) -> float:
    """Time to pull one checkpoint over the quoted megabit-per-second link."""
    return checkpoint_bytes * 8 / (transfer_mbps * 1e6)


def outcome_label(
    updates_reached: int, arms: int, screen_steps: int, steps: int, updates_total: int
) -> str:
    """Describe how far one pilot gets given the updates it can complete."""
    screen_updates = arms * screen_steps
    if updates_reached < screen_updates:
        return "no complete screen"
    if updates_reached == screen_updates:
        return "screens only (no winner continuation)"
    if updates_reached < updates_total:
        return f"partial continuation to step {screen_steps + updates_reached - screen_updates}"
    return f"complete {steps}"


def schedule(
    spec: PilotSpecification,
    throughput: float,
    *,
    probe_microbatches: Sequence[int],
    flops_token: int,
    peak_tflops: float = DEFAULT_PEAK_TFLOPS,
    startup_seconds: float = DEFAULT_STARTUP_SECONDS,
    transfer_mbps: float = DEFAULT_TRANSFER_MBPS,
    checkpoint_bytes: int = MEASURED_UNTRAINED_CHECKPOINT_BYTES,
    self_destruct_margin_seconds: float = DEFAULT_SELF_DESTRUCT_MARGIN_SECONDS,
) -> Schedule:
    """Wall-clock, pod and cost arithmetic for one assumed token throughput.

    `probe_microbatches` lists the probes expected to run; probes already judged
    likely-OOM are omitted by the caller because they fail before doing work.
    Each probe update accumulates `probe_accumulation_steps` microbatches. An
    enabled compile probe is charged its whole window, because compilation time
    is unknown until the A40 measures it. The usable pod lifetime is the wall
    minus the armed self-destruct margin, which the worker plans against.
    """
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("throughput must be a finite positive number")
    margin = self_destruct_margin_seconds
    if not math.isfinite(margin) or not 0 <= margin < spec.pod_seconds:
        raise ValueError("self-destruct margin must be finite and inside the pod wall")
    lifetime = spec.pod_seconds - margin
    per_update = spec.tokens_per_update / throughput
    probe = sum(
        spec.probe_steps * spec.probe_accumulation_steps * b * spec.sequence_length / throughput
        for b in probe_microbatches
    )
    compile_probe = spec.compile_probe_seconds if spec.compile_probe_steps else 0.0
    screens = spec.arms * spec.screen_steps * per_update
    continuation = (spec.steps - spec.screen_steps) * per_update
    deadline = min(
        spec.training_seconds,
        lifetime - startup_seconds - probe - compile_probe - spec.finalization_reserve_seconds,
    )
    deadline = max(deadline, 0.0)
    updates_all = total_updates(spec)
    reached = min(updates_all, math.floor(deadline / per_update + UPDATE_FLOOR_TOLERANCE))
    used = reached * per_update
    final = finalization_seconds(checkpoint_bytes)
    transfer = transfer_seconds(checkpoint_bytes, transfer_mbps)
    total = startup_seconds + probe + compile_probe + used + final + transfer
    cost = total / 3600 * spec.quoted_total_hourly
    return Schedule(
        throughput=throughput,
        implied_tflops=implied_tflops(throughput, flops_token),
        mfu=model_flops_utilization(throughput, flops_token, peak_tflops),
        seconds_per_update=per_update,
        probe_microbatches=tuple(probe_microbatches),
        probe_seconds=probe,
        probe_within_limit=probe <= spec.probe_seconds,
        compile_probe_seconds=compile_probe,
        pod_lifetime_seconds=lifetime,
        screens_seconds=screens,
        continuation_seconds=continuation,
        training_deadline_seconds=deadline,
        updates_total=updates_all,
        updates_reached=reached,
        screens_completed=min(spec.arms, reached // spec.screen_steps),
        outcome=outcome_label(reached, spec.arms, spec.screen_steps, spec.steps, updates_all),
        training_seconds_used=used,
        finalization_seconds=final,
        transfer_seconds=transfer,
        startup_seconds=startup_seconds,
        total_pod_seconds=total,
        cost_dollars=cost,
        fits_pod=total <= lifetime,
        fits_cost_cap=cost <= spec.approved_cost_cap,
    )


# ---------------------------------------------------------------------------
# Disk and storage


@dataclass(frozen=True)
class DiskEstimate:
    shard_bytes: int
    validation_bytes: int
    tokenizer_bytes: int
    dataset_bytes: int
    checkpoint_bytes: int
    checkpoint_copies: int
    checkpoint_total_bytes: int
    export_bytes: int
    total_bytes: int
    with_headroom_bytes: float
    recommended_container_disk_gb: int


def recommended_container_disk_gb(total_bytes: int) -> int:
    """Round total + 25% headroom up to a 10 GB multiple, never below 20 GB."""
    gigabytes = total_bytes * DISK_HEADROOM / GB
    rounded = math.ceil(gigabytes / DISK_ROUND_GB) * DISK_ROUND_GB
    return max(rounded, DISK_MINIMUM_GB)


def disk_estimate(
    arms: int, checkpoint_bytes: int = MEASURED_UNTRAINED_CHECKPOINT_BYTES
) -> DiskEstimate:
    """Container-disk demand: dataset plus one checkpoint per arm, continuation and staging."""
    shard = SHARD_TOKENS * TOKEN_BYTES
    validation = VALIDATION_TOKENS * TOKEN_BYTES
    dataset = shard + validation + TOKENIZER_BYTES
    copies = arms + 1 + 1  # screening arms, winner continuation, one staging temp during save
    checkpoints = copies * checkpoint_bytes
    total = dataset + checkpoints
    return DiskEstimate(
        shard_bytes=shard,
        validation_bytes=validation,
        tokenizer_bytes=TOKENIZER_BYTES,
        dataset_bytes=dataset,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_copies=copies,
        checkpoint_total_bytes=checkpoints,
        export_bytes=0,  # exports are hard links to saved checkpoints
        total_bytes=total,
        with_headroom_bytes=total * DISK_HEADROOM,
        recommended_container_disk_gb=recommended_container_disk_gb(total),
    )


@dataclass(frozen=True)
class NetworkVolumeDecision:
    size_gb: int
    dollars_per_gb_month: float
    monthly_cost_dollars: float
    billing: str
    required: bool
    conclusion: str


def network_volume_decision(
    size_gb: int = NETWORK_VOLUME_GB,
    dollars_per_gb_month: float = NETWORK_VOLUME_DOLLARS_PER_GB_MONTH,
) -> NetworkVolumeDecision:
    """Cost of an optional persistent volume and whether the pilot needs one."""
    return NetworkVolumeDecision(
        size_gb=size_gb,
        dollars_per_gb_month=dollars_per_gb_month,
        monthly_cost_dollars=round(size_gb * dollars_per_gb_month, 4),
        billing="billed hourly until deleted",
        required=False,
        conclusion=NETWORK_VOLUME_CONCLUSION,
    )


# ---------------------------------------------------------------------------
# Report


@dataclass(frozen=True)
class FeasibilityOptions:
    activation_gib_per_sequence: float = DEFAULT_ACTIVATION_GIB_PER_SEQUENCE
    gpu_memory_gib: float = DEFAULT_GPU_MEMORY_GIB
    usable_gpu_memory_gib: float = DEFAULT_USABLE_GPU_MEMORY_GIB
    throughputs: tuple[float, ...] = DEFAULT_THROUGHPUTS
    startup_seconds: float = DEFAULT_STARTUP_SECONDS
    transfer_mbps: float = DEFAULT_TRANSFER_MBPS
    peak_tflops: float = DEFAULT_PEAK_TFLOPS
    self_destruct_margin_seconds: float = DEFAULT_SELF_DESTRUCT_MARGIN_SECONDS

    def __post_init__(self) -> None:
        for name in ("activation_gib_per_sequence", "transfer_mbps", "peak_tflops"):
            _number(getattr(self, name), name)
        _number(self.gpu_memory_gib, "gpu_memory_gib")
        usable = _number(self.usable_gpu_memory_gib, "usable_gpu_memory_gib")
        if usable > self.gpu_memory_gib:
            raise ValueError("usable_gpu_memory_gib must not exceed gpu_memory_gib")
        if not math.isfinite(self.startup_seconds) or self.startup_seconds < 0:
            raise ValueError("startup_seconds must be a finite nonnegative number")
        margin = self.self_destruct_margin_seconds
        if isinstance(margin, bool) or not math.isfinite(margin) or margin < 0:
            raise ValueError("self_destruct_margin_seconds must be a finite nonnegative number")
        if not self.throughputs:
            raise ValueError("at least one throughput scenario is required")
        for value in self.throughputs:
            _number(value, "throughputs[]")


def build_report(
    spec: PilotSpecification, options: FeasibilityOptions, specification_path: str
) -> dict[str, Any]:
    """Assemble every estimate into one JSON-serializable mapping."""
    parameters = model_parameters(spec)
    flops_token = flops_per_token(
        parameters, spec.model.layers, spec.model.context, spec.model.width
    )
    table = microbatch_table(
        spec, parameters, options.activation_gib_per_sequence, options.usable_gpu_memory_gib
    )
    probe_microbatches = [row.microbatch for row in table if row.verdict != VERDICT_OOM]
    estimate = checkpoint_bytes_estimate(parameters)
    requirements = [
        ThroughputRequirement(
            label=label,
            updates=updates,
            throughput=rate,
            implied_tflops=implied_tflops(rate, flops_token),
            mfu=model_flops_utilization(rate, flops_token, options.peak_tflops),
        )
        for label, updates in (
            (f"{spec.arms} screens", spec.arms * spec.screen_steps),
            ("screens + continuation", total_updates(spec)),
        )
        for rate in (required_throughput(spec.tokens_per_update, updates, spec.training_seconds),)
    ]

    def plan(rate: float, plan_spec: PilotSpecification = spec) -> Schedule:
        return schedule(
            plan_spec,
            rate,
            probe_microbatches=probe_microbatches,
            flops_token=flops_token,
            peak_tflops=options.peak_tflops,
            startup_seconds=options.startup_seconds,
            transfer_mbps=options.transfer_mbps,
            self_destruct_margin_seconds=options.self_destruct_margin_seconds,
        )

    without_compile = replace(spec, compile_probe_steps=0, compile_probe_seconds=0.0)
    compile_cost = [
        {
            "throughput": rate,
            "updates_with_compile_probe": plan(rate).updates_reached,
            "updates_without_compile_probe": plan(rate, without_compile).updates_reached,
            "updates_lost": plan(rate, without_compile).updates_reached
            - plan(rate).updates_reached,
        }
        for rate in (requirements[-1].throughput, *options.throughputs)
    ]

    return {
        "disclaimer": GPU_DISCLAIMER,
        "specification": specification_path,
        "options": asdict(options),
        "model": {
            **asdict(spec.model),
            "parameters": parameters,
            "flops_per_token": flops_token,
            "updates_total": total_updates(spec),
            "arms": spec.arms,
            "steps": spec.steps,
            "screen_steps": spec.screen_steps,
            "tokens_per_update": spec.tokens_per_update,
        },
        "static_memory": {
            "static_bytes": static_bytes(parameters),
            "static_gib": static_bytes(parameters) / GIB,
            "checkpoint_bytes_estimate": estimate,
            "measured_untrained_checkpoint_bytes": MEASURED_UNTRAINED_CHECKPOINT_BYTES,
            "measured_untrained_checkpoint_provenance": MEASURED_UNTRAINED_CHECKPOINT_PROVENANCE,
            "checkpoint_estimate_relative_error": (
                estimate / MEASURED_UNTRAINED_CHECKPOINT_BYTES - 1
            ),
            "gpu_memory_gib": options.gpu_memory_gib,
            "usable_gpu_memory_gib": options.usable_gpu_memory_gib,
            "cuda_reserve_gib": options.gpu_memory_gib - options.usable_gpu_memory_gib,
            "activation_gib_per_sequence": options.activation_gib_per_sequence,
            "activation_proxy_provenance": ACTIVATION_PROXY_PROVENANCE,
        },
        "microbatches": [asdict(row) for row in table],
        "probe_microbatches": probe_microbatches,
        "compile_probe_seconds": spec.compile_probe_seconds if spec.compile_probe_steps else 0.0,
        "pod_lifetime_seconds": spec.pod_seconds - options.self_destruct_margin_seconds,
        "compile_probe_cost": compile_cost,
        "requirements": [
            {**asdict(item), "schedule": asdict(plan(item.throughput))} for item in requirements
        ],
        "scenarios": [asdict(plan(rate)) for rate in options.throughputs],
        "disk": asdict(disk_estimate(spec.arms)),
        "network_volume": asdict(network_volume_decision()),
    }


def _gib(value: float) -> str:
    return f"{value / GIB:,.2f} GiB"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:,.1f} min"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable summary of `build_report` output."""
    model = report["model"]
    static = report["static_memory"]
    options = report["options"]
    disk = report["disk"]
    volume = report["network_volume"]
    lines = [
        "# Pilot feasibility (analytical)",
        "",
        f"**{report['disclaimer']}.**",
        "",
        f"Specification: `{report['specification']}`; model {model['width']}w x "
        f"{model['layers']}L x {model['heads']}h, context {model['context']}, vocabulary "
        f"{model['vocab_size']:,}; {model['parameters']:,} parameters; "
        f"{model['arms']} arms x {model['screen_steps']} screen steps + "
        f"{model['steps'] - model['screen_steps']} continuation steps = "
        f"{model['updates_total']} updates of {model['tokens_per_update']:,} tokens.",
        "",
        "## Static memory",
        "",
        f"- Resident optimizer state (fp32 params + grads + Adam m + Adam v, 16 B/param): "
        f"{static['static_bytes']:,} bytes = {_gib(static['static_bytes'])}.",
        f"- Checkpoint estimate (12 B/param + {CHECKPOINT_OVERHEAD_BYTES:,} B overhead): "
        f"{static['checkpoint_bytes_estimate']:,} bytes; measured untrained checkpoint "
        f"{static['measured_untrained_checkpoint_bytes']:,} bytes = "
        f"{_gib(static['measured_untrained_checkpoint_bytes'])} "
        f"({static['checkpoint_estimate_relative_error'] * 100:+.3f}% estimate error). "
        f"Provenance: {static['measured_untrained_checkpoint_provenance']}.",
        f"- Device memory: {static['gpu_memory_gib']:.1f} GiB total, "
        f"{static['usable_gpu_memory_gib']:.1f} GiB usable after a "
        f"{static['cuda_reserve_gib']:.1f} GiB CUDA context/driver reserve.",
        f"- Activation proxy: {static['activation_gib_per_sequence']:.2f} GiB per sequence. "
        f"{static['activation_proxy_provenance']}.",
        "",
        "## Microbatch table",
        "",
        "Fit decision uses static + activation proxy only; the analytical output-layer term "
        "is listed for reference because the CPU proxy already contains part of it.",
        "",
        "| B | accumulation | activation proxy | total (static + activation) | of usable | "
        "verdict | output-layer term (analytical) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["microbatches"]:
        lines.append(
            f"| {row['microbatch']} | {row['accumulation_steps']} | "
            f"{_gib(row['activation_proxy_bytes'])} | {_gib(row['proxy_total_bytes'])} | "
            f"{row['fraction_of_usable'] * 100:.0f}% | {row['verdict']} | "
            f"{_gib(row['output_layer_analytical_bytes'])} |"
        )
    lines += [
        "",
        f"Probes assumed to run: {', '.join(f'B={b}' for b in report['probe_microbatches'])} "
        "(likely-OOM probes fail before doing work and are not timed).",
        "",
        "## Throughput requirements",
        "",
        f"FLOPs per token = 6N + 12·L·T·D = {model['flops_per_token']:,}; peak "
        f"{options['peak_tflops']} TFLOPS dense bf16.",
        "",
        "| target | updates | required tok/s | implied TFLOPS | MFU |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["requirements"]:
        lines.append(
            f"| {item['label']} | {item['updates']} | {item['throughput']:,.2f} | "
            f"{item['implied_tflops']:.2f} | {item['mfu'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## Scenario table",
        "",
        f"Usable pod lifetime {report['pod_lifetime_seconds']:.0f} s (the wall minus the "
        f"{options['self_destruct_margin_seconds']:.0f} s self-destruct margin). Startup "
        f"{options['startup_seconds']:.0f} s (image pull, bootstrap, data staging, "
        f"preflight); compile probe charged its full {report['compile_probe_seconds']:.0f} s "
        f"window; finalization = 2 x checkpoint at 1 GB/s + {FINAL_VALIDATION_WINDOWS} "
        f"validation windows ({FINAL_VALIDATION_SECONDS:.0f} s); transfer at "
        f"{options['transfer_mbps']:.0f} Mbps; cost at the quoted total hourly rate.",
        "",
        "| tok/s | MFU | s/update | probes | updates reached | outcome | training used | "
        "transfer | total pod | cost | fits pod | fits cap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["scenarios"]:
        lines.append(
            f"| {item['throughput']:,.0f} | {item['mfu'] * 100:.1f}% | "
            f"{item['seconds_per_update']:.2f} | {item['probe_seconds']:.0f} s | "
            f"{item['updates_reached']}/{item['updates_total']} | {item['outcome']} | "
            f"{_minutes(item['training_seconds_used'])} | {_minutes(item['transfer_seconds'])} | "
            f"{_minutes(item['total_pod_seconds'])} | ${item['cost_dollars']:.2f} | "
            f"{'yes' if item['fits_pod'] else 'no'} | {'yes' if item['fits_cost_cap'] else 'no'} |"
        )
    if report["compile_probe_seconds"]:
        required = report["compile_probe_cost"][0]
        lines += [
            "",
            "## Compile probe cost",
            "",
            f"At the required {required['throughput']:,.0f} tok/s the "
            f"{report['compile_probe_seconds']:.0f} s compile window, if fully used, costs "
            f"{required['updates_lost']} of {report['model']['updates_total']} updates "
            f"({required['updates_with_compile_probe']} reachable with it, "
            f"{required['updates_without_compile_probe']} without). A compile that finishes "
            "early returns the unused time to training.",
            "",
            "| tok/s | updates with compile probe | updates without | updates lost |",
            "| --- | --- | --- | --- |",
        ]
        for item in report["compile_probe_cost"][1:]:
            lines.append(
                f"| {item['throughput']:,.0f} | {item['updates_with_compile_probe']} | "
                f"{item['updates_without_compile_probe']} | {item['updates_lost']} |"
            )
    lines += [
        "",
        "## Disk",
        "",
        f"- Dataset: shards {disk['shard_bytes']:,} B + validation {disk['validation_bytes']:,} B "
        f"+ tokenizer {disk['tokenizer_bytes']:,} B = {disk['dataset_bytes'] / 1e6:,.1f} MB.",
        f"- Checkpoints: {disk['checkpoint_copies']} copies x {disk['checkpoint_bytes']:,} B "
        f"({model['arms']} arms + continuation + one staging temp during save) = "
        f"{disk['checkpoint_total_bytes'] / GB:,.2f} GB; exports are hard links "
        f"({disk['export_bytes']} extra bytes).",
        f"- Total {disk['total_bytes'] / GB:,.2f} GB; with {DISK_HEADROOM:.0%} headroom "
        f"{disk['with_headroom_bytes'] / GB:,.2f} GB; recommended container disk "
        f"**{disk['recommended_container_disk_gb']} GB** (rounded up to {DISK_ROUND_GB} GB, "
        f"minimum {DISK_MINIMUM_GB} GB). The pod image supplies torch; nothing else is "
        "installed on the container disk.",
        "",
        "## Storage decision",
        "",
        f"- Network volume of {volume['size_gb']} GB at ${volume['dollars_per_gb_month']:.2f}/GB-"
        f"month = ${volume['monthly_cost_dollars']:.2f}/month, {volume['billing']}.",
        f"- Required: {'yes' if volume['required'] else 'no'}. {volume['conclusion']}",
        "",
        f"**{report['disclaimer']}.**",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rehearsal (D-35): two concurrent single-A40 runs, A (stage1 then stage2) and B (mixed)

REHEARSAL_DISCLAIMER = (
    "Rehearsal figures are projections from one measured pilot (35,974 tok/s eager, 1.57x "
    "compiled) and stated assumptions; no rehearsal run has been measured"
)
# The S build's recorded counts (docs/rehearsal-data.md), used when no binding is given.
S_BUILD_STAGE_TOKENS: dict[str, tuple[tuple[str, int], ...]] = {
    "A": (("stage1", 1_000_000_000), ("stage2", 1_000_000_000)),
    "B": (("mixed", 2_000_000_000),),
}
S_BUILD_VALIDATION_TOKENS = 6_948_391 + 5_334_710
S_BUILD_DATA_BYTES = 2 * (2_000_000_000 + S_BUILD_VALIDATION_TOKENS)
DEFAULT_REHEARSAL_STARTUP_SECONDS = 1_200.0
"""Create request to first update: image pull, bootstrap, 4 GB fetch, tests, preflight."""
DEFAULT_COMPILE_SECONDS_PER_STAGE = 120.0
"""The pilot measured a one-off compile of about 48 s; each stage is allowed 120 s."""
DEFAULT_VALIDATION_SPEED_FACTOR = 2.0
"""Forward-only validation tokens per second relative to training (about 3x in FLOPs)."""
DEFAULT_CHECKPOINT_SAVE_SECONDS = 15.0
"""One 1.6 GB save plus its SHA-256 on the Pod; uploads to R2 run in the background."""
DEFAULT_KEEP_ROLLING = 2
CONTAINER_DISK_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730.0
COMPILE_CACHE_BYTES = 2 * GB


@dataclass(frozen=True)
class RehearsalPlan:
    """One run's stages with their planned optimizer updates."""

    run: str
    stages: tuple[tuple[str, int], ...]
    validation_targets: int

    def __post_init__(self) -> None:
        if not self.stages or any(_integer(updates, "updates") <= 0 for _, updates in self.stages):
            raise ValueError("a rehearsal plan needs stages with positive planned updates")
        _integer(self.validation_targets, "validation targets")

    @property
    def updates(self) -> int:
        return sum(updates for _, updates in self.stages)


def planned_updates(tokens: int, *, sequence_length: int = 1024, per_update: int = 128) -> int:
    """The binding's packing: every token is a target; the last window and update are partial."""
    windows = -(-_integer(tokens, "tokens") // sequence_length)
    return -(-windows // per_update)


def default_rehearsal_plans() -> dict[str, RehearsalPlan]:
    return {
        run: RehearsalPlan(
            run,
            tuple((name, planned_updates(tokens)) for name, tokens in stages),
            S_BUILD_VALIDATION_TOKENS,
        )
        for run, stages in S_BUILD_STAGE_TOKENS.items()
    }


def rehearsal_plans_from_binding(binding: Mapping[str, Any]) -> dict[str, RehearsalPlan]:
    """Each run's stages and planned updates, and the validation size, from a binding."""
    from tools.rehearsal_binding import RUNS, VALIDATION_STREAMS, check_binding

    checked = check_binding(binding)
    validation = sum(checked["streams"][name]["tokens"] for name in VALIDATION_STREAMS)
    return {
        run: RehearsalPlan(
            run,
            tuple(
                (stage["name"], stage["planned_updates"])
                for stage in checked["runs"][run]["stages"]
            ),
            validation,
        )
        for run in RUNS
    }


def binding_data_bytes(binding: Mapping[str, Any]) -> int:
    """Bytes of every bound shard, manifest and the tokenizer (what R2 holds per build)."""
    total = int(binding["tokenizer"]["size_bytes"])
    for stream in binding["streams"].values():
        for shard in stream["shards"]:
            total += int(shard["size_bytes"]) + int(shard["manifest_size_bytes"])
    return total


@dataclass(frozen=True)
class RehearsalOptions:
    """The D-35 limits (from ``tools.rehearsal_ops``) and the projection's assumptions."""

    reference_tokens_per_second: float = d35.REFERENCE_TOKENS_PER_SECOND
    minimum_speed_fraction: float = d35.MINIMUM_SPEED_FRACTION
    startup_seconds: float = DEFAULT_REHEARSAL_STARTUP_SECONDS
    compile_seconds_per_stage: float = DEFAULT_COMPILE_SECONDS_PER_STAGE
    validation_speed_factor: float = DEFAULT_VALIDATION_SPEED_FACTOR
    checkpoint_save_seconds: float = DEFAULT_CHECKPOINT_SAVE_SECONDS
    quoted_gpu_hourly: float = d35.QUOTED_GPU_HOURLY
    container_disk_gb: float = d35.CONTAINER_DISK_GB
    lifetime_seconds: float = d35.POD_LIFETIME_SECONDS
    self_destruct_margin_seconds: float = d35.SELF_DESTRUCT_MARGIN_SECONDS
    finalization_reserve_seconds: float = d35.FINALIZATION_RESERVE_SECONDS
    cost_stop_usd: float = d35.COST_STOP_USD
    validation_interval_updates: int = d35.VALIDATION_INTERVAL_UPDATES
    subset_targets_per_domain: int = d35.VALIDATION_TARGETS_PER_DOMAIN
    checkpoint_interval_seconds: float = d35.CHECKPOINT_SECONDS
    tokens_per_update: int = d35.TOKENS_PER_UPDATE

    def __post_init__(self) -> None:
        for name in (
            "reference_tokens_per_second",
            "validation_speed_factor",
            "quoted_gpu_hourly",
            "lifetime_seconds",
            "cost_stop_usd",
            "checkpoint_interval_seconds",
        ):
            _number(getattr(self, name), name)
        for name in (
            "startup_seconds",
            "compile_seconds_per_stage",
            "checkpoint_save_seconds",
            "container_disk_gb",
            "self_destruct_margin_seconds",
            "finalization_reserve_seconds",
        ):
            _count_number(getattr(self, name), name)
        for name in (
            "validation_interval_updates",
            "subset_targets_per_domain",
            "tokens_per_update",
        ):
            _integer(getattr(self, name), name)
        if not 0 < _number(self.minimum_speed_fraction, "minimum_speed_fraction") <= 1:
            raise ValueError("minimum_speed_fraction must be in (0, 1]")
        if self.self_destruct_margin_seconds >= self.lifetime_seconds:
            raise ValueError("the self-destruct margin must be shorter than the lifetime")

    @property
    def floor_tokens_per_second(self) -> float:
        return self.reference_tokens_per_second * self.minimum_speed_fraction


def rehearsal_total_hourly(gpu_hourly: float, container_disk_gb: float) -> float:
    """GPU quote plus the container disk's hourly share, rounded up to the cent."""
    raw = _number(gpu_hourly, "GPU hourly") + _count_number(container_disk_gb, "disk") * (
        CONTAINER_DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH
    )
    return math.ceil(round(raw * 100, 6)) / 100


def rehearsal_time_budget(
    plan: RehearsalPlan,
    options: RehearsalOptions,
    tokens_per_second: float,
    *,
    startup_seconds: float | None = None,
) -> dict[str, Any]:
    """Seconds from the create request to the end of finalization at one training speed.

    Startup, then training, interval validations every 500 global updates on 1,000,000
    targets per domain, a full validation of both streams at each quarter and end of
    every stage, one compile per stage, a checkpoint every 30 minutes plus one per stage
    end, then the finalization reserve. It fits when the total is within the lifetime
    less the self-destruct margin.
    """
    from tools.rehearsal_run import quarter_updates

    rate = _number(tokens_per_second, "tokens per second")
    startup = options.startup_seconds if startup_seconds is None else startup_seconds
    _count_number(startup, "startup seconds")
    stages = len(plan.stages)
    validation_rate = rate * options.validation_speed_factor
    training = plan.updates * options.tokens_per_update / rate
    interval_validations = plan.updates // options.validation_interval_updates
    interval_seconds = (
        interval_validations * 2 * options.subset_targets_per_domain / validation_rate
    )
    full_validations = sum(len(quarter_updates(updates)) + 1 for _, updates in plan.stages)
    full_seconds = full_validations * plan.validation_targets / validation_rate
    compile_seconds = stages * options.compile_seconds_per_stage
    running = training + interval_seconds + full_seconds + compile_seconds
    checkpoints = math.floor(running / options.checkpoint_interval_seconds) + stages
    checkpoint_seconds = checkpoints * options.checkpoint_save_seconds
    after_startup = running + checkpoint_seconds + options.finalization_reserve_seconds
    total = startup + after_startup
    usable = options.lifetime_seconds - options.self_destruct_margin_seconds
    return {
        "tokens_per_second": rate,
        "startup_seconds": startup,
        "training_seconds": training,
        "interval_validations": interval_validations,
        "interval_validation_seconds": interval_seconds,
        "full_validations": full_validations,
        "full_validation_seconds": full_seconds,
        "compile_seconds": compile_seconds,
        "checkpoints": checkpoints,
        "checkpoint_seconds": checkpoint_seconds,
        "finalization_reserve_seconds": options.finalization_reserve_seconds,
        "total_seconds": total,
        "usable_lifetime_seconds": usable,
        "margin_seconds": usable - total,
        "latest_startup_seconds": usable - after_startup,
        "fits": total <= usable,
    }


def build_rehearsal_report(
    plans: Mapping[str, RehearsalPlan],
    options: RehearsalOptions,
    *,
    source: str,
    data_bytes: int = S_BUILD_DATA_BYTES,
) -> dict[str, Any]:
    """Time, cost, R2 storage and container-disk fit for each rehearsal run."""
    from tools.r2_store import monthly_storage_usd

    hourly = rehearsal_total_hourly(options.quoted_gpu_hourly, options.container_disk_gb)
    wall_cost = options.lifetime_seconds * hourly / 3600
    model = ModelConfig(vocab_size=65_536, width=768, layers=12, heads=12, context=1024)
    parameters = expected_parameters(model)
    initialization_bytes = parameters * 4 + CHECKPOINT_OVERHEAD_BYTES
    checkpoint = MEASURED_UNTRAINED_CHECKPOINT_BYTES
    runs: dict[str, Any] = {}
    stored = data_bytes + initialization_bytes
    for run, plan in sorted(plans.items()):
        reference = rehearsal_time_budget(plan, options, options.reference_tokens_per_second)
        floor = rehearsal_time_budget(plan, options, options.floor_tokens_per_second)
        # A run that would outlast the wall is cut there, so the wall bounds its spend.
        planned_cost = min(floor["total_seconds"], options.lifetime_seconds) * hourly / 3600
        r2_checkpoints = len(plan.stages) + DEFAULT_KEEP_ROLLING
        stored += r2_checkpoints * checkpoint
        # Both runs read every bound stream, so each Pod holds all the data. Local
        # checkpoints: the rolling pair, each stage's final and one save in progress.
        disk = (
            data_bytes
            + initialization_bytes
            + (DEFAULT_KEEP_ROLLING + len(plan.stages) + 1) * checkpoint
            + COMPILE_CACHE_BYTES
        )
        runs[run] = {
            "stages": [{"name": name, "planned_updates": updates} for name, updates in plan.stages],
            "updates": plan.updates,
            "tokens_per_update": options.tokens_per_update,
            "reference": reference,
            "floor": floor,
            "time_fit_at_floor": floor["fits"],
            "cost": {
                "planned_at_floor_usd": planned_cost,
                "wall_bound_usd": wall_cost,
                "cost_stop_usd": options.cost_stop_usd,
                "fits": planned_cost <= options.cost_stop_usd,
                "wall_within_cost_stop": wall_cost <= options.cost_stop_usd,
            },
            "r2_checkpoints_kept": r2_checkpoints,
            "container_disk": {
                "estimated_bytes": disk,
                "with_headroom_bytes": int(disk * DISK_HEADROOM),
                "container_disk_gb": options.container_disk_gb,
                "fits": disk * DISK_HEADROOM <= options.container_disk_gb * GB,
            },
        }
    return {
        "schema_version": 1,
        "kind": "rehearsal-feasibility",
        "source": source,
        "options": asdict(options),
        "floor_tokens_per_second": options.floor_tokens_per_second,
        "quoted_total_hourly": hourly,
        "runs": runs,
        "two_runs_wall_bound_usd": 2 * wall_cost,
        "r2_upload": {
            "checkpoint_bytes": d35.S_LABEL_CHECKPOINT_BYTES,
            "labels_in_one_drain": 1,
            "drain_seconds": d35.DRAIN_SECONDS,
            "pull_reserve_seconds": d35.PULL_RESERVE_SECONDS,
            "safety_factor": 2,
            "required_upload_bytes_per_second": d35.REQUIRED_UPLOAD_BYTES_PER_SECOND,
            "gate": "the readiness receipt refuses a Pod whose measured R2 round-trip upload "
            "rate is lower",
        },
        "r2_storage": {
            "bytes": stored,
            "monthly_usd": monthly_storage_usd(stored),
            "assumption": "data, initialization, every stage-final milestone and two rolling "
            "checkpoints per run, at the measured 1,623,502,465-byte checkpoint size",
        },
        "disclaimer": REHEARSAL_DISCLAIMER,
    }


def render_rehearsal_markdown(report: Mapping[str, Any]) -> str:
    options = report["options"]
    lines = [
        "# Rehearsal feasibility (D-35)",
        "",
        f"Source: {report['source']}. Reference {options['reference_tokens_per_second']:,.0f} "
        f"tok/s; the speed floor is {options['minimum_speed_fraction']:.0%} of it, "
        f"{report['floor_tokens_per_second']:,.0f} tok/s. Quote ${options['quoted_gpu_hourly']:.2f}"
        f"/h GPU, ${report['quoted_total_hourly']:.2f}/h with the "
        f"{options['container_disk_gb']:g} GB container disk.",
        "",
        "| Run | Updates | Speed | Training | Validation | Checkpoints | Total | Lifetime "
        "left | Fits | Latest startup |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for run, row in report["runs"].items():
        for label in ("reference", "floor"):
            budget = row[label]
            validation = budget["interval_validation_seconds"] + budget["full_validation_seconds"]
            lines.append(
                f"| {run} | {row['updates']:,} | {label} {budget['tokens_per_second']:,.0f} | "
                f"{_minutes(budget['training_seconds'])} | {_minutes(validation)} | "
                f"{budget['checkpoints']} ({_minutes(budget['checkpoint_seconds'])}) | "
                f"{_minutes(budget['total_seconds'])} | "
                f"{_minutes(budget['usable_lifetime_seconds'])} | "
                f"{'yes' if budget['fits'] else '**no**'} | "
                f"{_minutes(budget['latest_startup_seconds'])} |"
            )
    lines += ["", "## Cost and storage", ""]
    for run, row in report["runs"].items():
        cost = row["cost"]
        disk = row["container_disk"]
        lines.append(
            f"- Run {run}: ${cost['planned_at_floor_usd']:.2f} planned at the floor, "
            f"${cost['wall_bound_usd']:.2f} at the 45,000-second wall, against the "
            f"${cost['cost_stop_usd']:.2f} stop: {'fits' if cost['fits'] else '**does not fit**'}"
            f"; container disk {disk['with_headroom_bytes'] / GB:,.1f} GB with headroom of "
            f"{disk['container_disk_gb']:g} GB: {'fits' if disk['fits'] else '**does not fit**'}."
        )
    storage = report["r2_storage"]
    upload = report["r2_upload"]
    lines += [
        f"- Both runs at their walls: ${report['two_runs_wall_bound_usd']:.2f}.",
        f"- R2 upload: {upload['labels_in_one_drain']} label of "
        f"{upload['checkpoint_bytes'] / 1e9:.2f} GB in the {upload['drain_seconds']} s drain "
        f"with a {upload['safety_factor']}x margin needs "
        f"{upload['required_upload_bytes_per_second'] / 1e6:.2f} MB/s; {upload['gate']}. "
        f"The last {upload['pull_reserve_seconds']} s of the transfer window are kept for "
        "the Mac's SSH pull of a milestone R2 could not take.",
        f"- R2 storage {storage['bytes'] / GB:,.1f} GB, ${storage['monthly_usd']:.2f}/month "
        f"({storage['assumption']}).",
        "",
        f"**{report['disclaimer']}.**",
        "",
    ]
    return "\n".join(lines)


def build_rehearsal_parser() -> argparse.ArgumentParser:
    defaults = RehearsalOptions()
    parser = argparse.ArgumentParser(
        prog="python -m tools.pilot_feasibility rehearsal",
        description="Time, cost and storage fit of the two D-35 rehearsal runs.",
    )
    parser.add_argument("--binding", type=Path, help="the rehearsal data binding (else S counts)")
    parser.add_argument("--binding-sha256", help="pin the binding to this SHA-256")
    parser.add_argument("--quoted-gpu-hourly", type=float, default=defaults.quoted_gpu_hourly)
    parser.add_argument("--startup-seconds", type=float, default=defaults.startup_seconds)
    parser.add_argument(
        "--reference-tokens-per-second", type=float, default=defaults.reference_tokens_per_second
    )
    parser.add_argument(
        "--validation-speed-factor", type=float, default=defaults.validation_speed_factor
    )
    parser.add_argument(
        "--compile-seconds-per-stage", type=float, default=defaults.compile_seconds_per_stage
    )
    parser.add_argument(
        "--checkpoint-save-seconds", type=float, default=defaults.checkpoint_save_seconds
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def rehearsal_main(argv: Sequence[str]) -> int:
    parser = build_rehearsal_parser()
    args = parser.parse_args(argv)
    try:
        options = RehearsalOptions(
            reference_tokens_per_second=args.reference_tokens_per_second,
            startup_seconds=args.startup_seconds,
            compile_seconds_per_stage=args.compile_seconds_per_stage,
            validation_speed_factor=args.validation_speed_factor,
            checkpoint_save_seconds=args.checkpoint_save_seconds,
            quoted_gpu_hourly=args.quoted_gpu_hourly,
        )
        if args.binding is None:
            if args.binding_sha256 is not None:
                parser.error("--binding-sha256 needs --binding")
            report = build_rehearsal_report(
                default_rehearsal_plans(), options, source="the S build's recorded counts"
            )
        else:
            from tools.rehearsal_binding import check_binding, load_binding

            if args.binding_sha256 is None:
                binding = check_binding(json.loads(args.binding.read_text(encoding="utf-8")))
            else:
                binding = load_binding(args.binding, expected_sha256=args.binding_sha256)
            report = build_rehearsal_report(
                rehearsal_plans_from_binding(binding),
                options,
                source=f"binding {args.binding}",
                data_bytes=binding_data_bytes(binding),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(render_rehearsal_markdown(report), end="")
    return 0


# ---------------------------------------------------------------------------
# CLI


def default_specification_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "pilot-a40-s.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=default_specification_path())
    parser.add_argument(
        "--activation-gib-per-sequence",
        type=float,
        default=DEFAULT_ACTIVATION_GIB_PER_SEQUENCE,
        help=ACTIVATION_PROXY_PROVENANCE,
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=float,
        default=DEFAULT_GPU_MEMORY_GIB,
        help="total device memory in GiB (A40: 48)",
    )
    parser.add_argument(
        "--usable-gpu-memory-gib",
        type=float,
        default=DEFAULT_USABLE_GPU_MEMORY_GIB,
        help="memory left after CUDA context/driver reserve; used for every fit decision",
    )
    parser.add_argument(
        "--throughputs",
        type=float,
        nargs="+",
        default=list(DEFAULT_THROUGHPUTS),
        help="assumed tokens/second scenarios (analytical, not measured)",
    )
    parser.add_argument(
        "--startup-seconds",
        type=float,
        default=DEFAULT_STARTUP_SECONDS,
        help="image pull, bootstrap, data staging and preflight before the first probe",
    )
    parser.add_argument("--transfer-mbps", type=float, default=DEFAULT_TRANSFER_MBPS)
    parser.add_argument(
        "--self-destruct-margin-seconds",
        type=float,
        default=DEFAULT_SELF_DESTRUCT_MARGIN_SECONDS,
        help="seconds the armed in-Pod timer fires before the wall (runbook: 180)",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS,
        help="dense bf16 peak used for the MFU fraction (A40: 149.7)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["rehearsal"]:
        return rehearsal_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        spec = PilotSpecification.load(args.specification)
        options = FeasibilityOptions(
            activation_gib_per_sequence=args.activation_gib_per_sequence,
            gpu_memory_gib=args.gpu_memory_gib,
            usable_gpu_memory_gib=args.usable_gpu_memory_gib,
            throughputs=tuple(args.throughputs),
            startup_seconds=args.startup_seconds,
            transfer_mbps=args.transfer_mbps,
            peak_tflops=args.peak_tflops,
            self_destruct_margin_seconds=args.self_destruct_margin_seconds,
        )
        report = build_report(spec, options, str(args.specification))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
