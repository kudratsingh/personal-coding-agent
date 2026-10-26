"""Rehearsal pretraining worker for runs A and B (D-35) on one existing Pod; never provisions.

``run`` (Pod)
    Check the readiness receipt that ``python -m tools.pilot_readiness_receipt rehearsal``
    wrote, the shipped code revision against it, the A40 preflight (with its compile
    check), the data binding and the shared initialization. Then train each stage of one
    run over every bound document exactly once, validate on a fixed subset every
    ``interval`` global updates and on both full validation streams at each quarter and
    the end of every stage, and send checkpoints to R2 in the background. The heartbeat
    reads ``running`` throughout, across run A's stage boundary too, ``uploading`` while
    the last checkpoints drain to R2 with the GPU idle, and ``awaiting_transfer`` only
    once every milestone and ``report.json`` are verified in R2. A milestone that cannot
    be verified ends the run ``failed`` (``r2_milestones_unverified``) with its local
    export listed in ``finalization.json`` for the supervisor to pull over SSH.
``smoke`` (CPU)
    Both runs on tiny authored shards with an in-memory bucket; nothing touches a network.
``write-initialization`` (Mac)
    Write the shared seed-0 initialization both runs start from and print its SHA-256.

R2 credentials come only from the environment and are never printed or written. See
``docs/walkthroughs/rehearsal-run.md``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import batched
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import Request
from xml.sax.saxutils import escape

import numpy as np
import numpy.typing as npt
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig, finite_number
from scratchcoder.common.logging import sanitize
from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data.onepass import (
    OnePassLoader,
    combine_digests,
    document_sha256,
    multiset_digest,
    order_digest,
    pass_order,
)
from scratchcoder.data.shards import TokenShard, file_sha256, write_shard
from scratchcoder.model.gpt import GPT, expected_parameters
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.pilot import Limits, PilotStop
from scratchcoder.train.trainer import Trainer, synchronized_time, training_device
from scratchcoder.train.validation import (
    ValidationWindow,
    document_windows,
    iter_document_windows,
    selection_sha256,
    validate_batched,
)
from tools.gpu_pilot import check_termination_guarantee, termination_window
from tools.pilot_preflight import hardware_description
from tools.r2_store import (
    LABEL,
    MILESTONE_RETRY_MAX_SECONDS,
    MILESTONE_RETRY_SECONDS,
    SHA256_HEX,
    CheckpointUploader,
    Credentials,
    Store,
    bounded_store,
    transient_error,
)
from tools.rehearsal_binding import (
    ORDER_SEED,
    RUNS,
    VALIDATION_STREAMS,
    load_binding,
    r2_credentials,
    verify_binding,
)
from tools.rehearsal_ops import (
    PULL_RESERVE_SECONDS,
    READINESS_GATES,
    READINESS_KIND,
    UPLOADING_STATUS,
    run_prefix,
)
from tools.runpod_watchdog import MAX_TRANSFER_SECONDS, MODE_WALL_SECONDS, PodIdentity, Policy

SCHEMA_VERSION = 1
SPECIFICATION_KIND = "rehearsal-run-specification"
REPORT_KIND = "rehearsal-run-report"
INITIALIZATION_RUN_ID = "rehearsal-init"
R2_ROOT = "rehearsal-runs/"
MAX_REPORT_BYTES = 1_000_000
MAX_HEARTBEAT_BYTES = 65_536  # The watchdog treats a larger heartbeat as malformed.
MAX_POD_SECONDS = MODE_WALL_SECONDS["rehearsal"]
MAX_IDLE_SECONDS = 600
MAX_APPROVED_COST_USD = 10.0
HEARTBEAT_REFRESH_SECONDS = 5.0
FULL_VALIDATION_SORT_WINDOWS = 512
FINALIZE_MARGIN_SECONDS = 120.0
"""After the final drain: the HEAD checks, the ``report.json`` upload and the final heartbeat.
The heartbeat is due ``PULL_RESERVE_SECONDS`` before the watchdog's transfer window ends,
which leaves the Mac supervisor that long to pull an unverified milestone over SSH."""
LAST_CALL_SECONDS = 45.0
"""No R2 check after the drain starts later than this before the handover, except the HEAD
that verifies a ``report.json`` PUT that succeeded."""
FINAL_WRITE_SECONDS = 10.0
"""Kept after every R2 check for closing the uploader, ``finalization.json`` and the final
heartbeat: each check's socket timeout is sized to end this long before the handover."""
CHECK_TIMEOUT_SECONDS = 20.0
"""The longest socket timeout of one R2 check attempt after the drain."""
CHECK_ATTEMPTS = 6
"""A check after the drain retries transient failures up to this many times while time lasts."""
ROLLING_SKIP_SECONDS = 900.0
"""The D-35 minimum for ``rolling_skip_seconds``: no rolling save starts when the stage's
remaining updates at the measured rate, or the training deadline, end within this long,
so a stage-final milestone never queues behind a fresh rolling upload."""
SETTLE_ROUNDS = 3
"""Finalization resubmits unverified milestones and drains at most this many times."""
RETRY_GROWTH = MILESTONE_RETRY_MAX_SECONDS / MILESTONE_RETRY_SECONDS
"""A failing milestone's retry wait doubles up to this multiple of the first (15 s to 60 s)."""
DEFAULT_REVISION = Path("/workspace/rehearsal/revision.json")
"""Where the runbook ships the Mac's ``revision.json`` on the Pod."""
REVISION_KEYS = frozenset({"git_commit", "clean", "archive_sha256"})
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
HARD_STOPS = frozenset({"pod_wall_limit", "cost_limit", "idle_gpu_limit"})
NO_FINALIZATION_STOPS = frozenset({"pod_wall_limit", "cost_limit"})
"""Stops that never wait for uploads: finalization must not extend a wall or cost limit."""
REHEARSAL_GATES = READINESS_GATES
"""Receipt gates, each an explicit ``true``, as the receipt builder sets them. The pilot's
``ssh_transfer_tested`` is replaced by ``r2_round_trip_tested``: checkpoints leave the Pod
through R2, not SSH."""
LAUNCH_RULE = (
    "at launch the Pod lifetime left, to the first of the wall, the armed in-Pod deadline, "
    "the watchdog's wall and either cost stop, must hold the whole planned run at the "
    "reference speed: every update at expected_tokens_per_second, every interval subset "
    "validation and the full validation of both streams at each quarter and end of every "
    "stage at validation_tokens_per_second, startup_seconds per stage, the planned "
    "checkpoint saves and the finalization reserve"
)
SPEED_CHECK_RULE = (
    "after speed_check_seconds of training the run stops if its post-compile throughput is "
    "below minimum_speed_fraction x expected_tokens_per_second, or if its projected finish "
    "at the measured rates (remaining updates, interval and quarter and stage-end full "
    "validations, recompilation, checkpoint saves and the finalization reserve) passes the "
    "end of the Pod lifetime; the second rule binds even when the throughput passes"
)
SECRET_VARIABLES = (
    "R2_POD_ACCESS_KEY_ID",
    "R2_POD_SECRET_ACCESS_KEY",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "RUNPOD_API_KEY",
    "HF_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)
EXPECTED_TICKET = (
    "documents",
    "tokens",
    "targets",
    "windows",
    "planned_updates",
    "multiset_sha256",
    "order_sha256",
)
SPECIFICATION_KEYS = {
    "schema_version",
    "kind",
    "run",
    "model",
    "train",
    "warmup_fraction",
    "stages",
    "validation",
    "checkpoint_seconds",
    "checkpoint_save_seconds",
    "expected_tokens_per_second",
    "validation_tokens_per_second",
    "speed_check_warmup_updates",
    "quoted_gpu_hourly",
    "limits",
    "r2",
}
MODEL_KEYS = {"vocab_size", "width", "layers", "heads", "context", "qk_norm"}
TRAIN_KEYS = {
    "seed",
    "batch_size",
    "sequence_length",
    "accumulation_steps",
    "min_lr_ratio",
    "weight_decay",
    "gradient_clip",
    "beta1",
    "beta2",
    "adam_epsilon",
    "precision",
    "attention",
    "optimizer_foreach",
    "z_loss",
    "schedule",
    "compile",
}
"""The shared training settings; each stage supplies its own steps, warmup and peak."""
STAGE_KEYS = {"name", "streams", "learning_rate", "updates", "warmup_updates"}
VALIDATION_KEYS = {
    "interval_updates",
    "subset_target_tokens",
    "batch_size",
    "autocast_bf16",
    "streams",
}
LIMIT_KEYS = {
    "pod_seconds",
    "startup_seconds",
    "finalization_reserve_seconds",
    "quoted_total_hourly",
    "estimated_total_cost",
    "approved_cost_cap",
    "cost_multiplier",
    "minimum_speed_fraction",
    "speed_check_seconds",
    "idle_seconds",
    "rising_steps",
    "starting_cost_usd",
}
R2_KEYS = {
    "keep_rolling",
    "storage_cap_usd_per_month",
    "rolling_skip_seconds",
    "max_transfer_seconds",
}


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly the keys {sorted(keys)}")
    return value


def rehearsal_limits(
    *,
    pod_seconds: float,
    startup_seconds: float,
    finalization_reserve_seconds: float,
    quoted_total_hourly: float,
    estimated_total_cost: float,
    approved_cost_cap: float,
    cost_multiplier: float,
    minimum_speed_fraction: float,
    speed_check_seconds: float,
    idle_seconds: float,
    rising_steps: int,
    starting_cost_usd: float = 0.0,
) -> Limits:
    """The pilot's ``Limits`` for a rehearsal run, which has no probes.

    ``startup_seconds`` takes the probe window's place: the launch plan charges it once
    per stage, for the model build, the parent check and compilation (stage 1 also checks
    the data). Training may run until only the finalization reserve is left before the
    Pod lifetime ends, so the training budget is the lifetime minus that reserve.
    """
    return Limits(
        probe_seconds=startup_seconds,
        training_seconds=pod_seconds - finalization_reserve_seconds,
        pod_seconds=pod_seconds,
        quoted_total_hourly=quoted_total_hourly,
        estimated_total_cost=estimated_total_cost,
        approved_cost_cap=approved_cost_cap,
        cost_multiplier=cost_multiplier,
        minimum_speed_fraction=minimum_speed_fraction,
        speed_check_seconds=speed_check_seconds,
        idle_seconds=idle_seconds,
        rising_steps=rising_steps,
        starting_cost_usd=starting_cost_usd,
        finalization_reserve_seconds=finalization_reserve_seconds,
    )


def limit_values(limits: Limits) -> dict[str, Any]:
    """``limits`` under the specification file's names."""
    values = asdict(limits)
    values["startup_seconds"] = values.pop("probe_seconds")
    del values["training_seconds"]
    return values


@dataclass(frozen=True)
class StagePlan:
    name: str
    streams: tuple[str, ...]
    learning_rate: float
    updates: int
    warmup_updates: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not LABEL.fullmatch(self.name)
            or not isinstance(self.streams, tuple)
            or not self.streams
            or any(not isinstance(stream, str) for stream in self.streams)
        ):
            raise ValueError("a stage needs a plain name and a tuple of stream names")
        if finite_number(self.learning_rate, "stage learning rate") <= 0:
            raise ValueError("stage learning rates must be positive")
        for name in ("updates", "warmup_updates"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"stage {name} must be an integer")
        if not 0 < self.warmup_updates < self.updates:
            raise ValueError(f"{self.name}: warmup must be positive and shorter than the stage")


@dataclass(frozen=True)
class ValidationPlan:
    interval_updates: int
    subset_target_tokens: int
    batch_size: int
    autocast_bf16: bool
    streams: tuple[tuple[str, str], ...]
    """(label, validation stream) pairs, e.g. ("web", "validation-web")."""

    def __post_init__(self) -> None:
        for name in ("interval_updates", "subset_target_tokens", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"validation {name} must be a positive integer")
        if type(self.autocast_bf16) is not bool:
            raise ValueError("validation autocast_bf16 must be a boolean")
        if (
            not isinstance(self.streams, tuple)
            or tuple(stream for _, stream in self.streams) != VALIDATION_STREAMS
            or len({label for label, _ in self.streams}) != len(self.streams)
            or any(not isinstance(label, str) or not label for label, _ in self.streams)
        ):
            raise ValueError(
                f"validation must read {list(VALIDATION_STREAMS)} under distinct labels"
            )


@dataclass(frozen=True)
class Specification:
    run: str
    model: ModelConfig
    train: TrainConfig
    """Stage 1's settings; every stage replaces steps, warmup and the peak rate."""
    stages: tuple[StagePlan, ...]
    validation: ValidationPlan
    limits: Limits
    warmup_fraction: float
    checkpoint_seconds: float
    checkpoint_save_seconds: float
    """A planning bound for one save, its SHA-256 and export; the speed check uses the
    measured save time once a save has happened."""
    expected_tokens_per_second: float
    """The reference speed: the launch rule's basis and the speed floor's reference."""
    validation_tokens_per_second: float
    """A planning rate for launch fit only; the speed check uses the measured rate."""
    speed_check_warmup_updates: int
    quoted_gpu_hourly: float
    keep_rolling: int
    storage_cap_usd_per_month: float
    rolling_skip_seconds: float
    """No rolling save starts when the stage's remaining updates at the measured rate, or
    the training deadline, end within this long (``ROLLING_SKIP_SECONDS`` for D-35)."""
    max_transfer_seconds: float
    """The watchdog policy's transfer window: an idle Pod may be held this long after the
    last GPU work. The final drain uses it except the ``FINALIZE_MARGIN_SECONDS`` the checks
    after it need and the ``PULL_RESERVE_SECONDS`` kept for the Mac's SSH pull."""

    def __post_init__(self) -> None:
        if not isinstance(self.run, str) or self.run not in RUNS:
            raise ValueError(f"unknown run {self.run!r}; the runs are {sorted(RUNS)}")
        if not isinstance(self.stages, tuple) or not all(
            isinstance(stage, StagePlan) for stage in self.stages
        ):
            raise ValueError("stages must be a tuple of stage plans")
        if tuple((stage.name, stage.streams) for stage in self.stages) != RUNS[self.run]:
            raise ValueError(f"run {self.run}'s stages and streams differ from the run design")
        RunConfig(model=self.model, train=self.train)  # Context, field and type rules.
        first = self.stages[0]
        if (self.train.steps, self.train.warmup_steps, self.train.learning_rate) != (
            first.updates,
            first.warmup_updates,
            first.learning_rate,
        ):
            raise ValueError("the shared training settings must carry stage 1's values")
        if self.train.validation_interval != 0:
            raise ValueError("the worker validates itself; train.validation_interval must be 0")
        if self.train.z_loss != 0:
            raise ValueError("the masked final update does not support z-loss")
        if not 0 < finite_number(self.warmup_fraction, "warmup fraction") < 1:
            raise ValueError("warmup_fraction must be in (0, 1)")
        for stage in self.stages:
            if stage.warmup_updates != round(self.warmup_fraction * stage.updates):
                raise ValueError(f"{stage.name}: warmup must be round(warmup_fraction x updates)")
        if not isinstance(self.validation, ValidationPlan) or not isinstance(self.limits, Limits):
            raise ValueError("validation and limits must be typed plans")
        if self.limits.finalization_reserve_seconds <= 0 or self.limits.training_seconds != (
            self.limits.pod_seconds - self.limits.finalization_reserve_seconds
        ):
            raise ValueError("rehearsal limits must come from rehearsal_limits")
        for name in (
            "checkpoint_seconds",
            "checkpoint_save_seconds",
            "expected_tokens_per_second",
            "validation_tokens_per_second",
            "quoted_gpu_hourly",
            "storage_cap_usd_per_month",
            "max_transfer_seconds",
        ):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if finite_number(self.rolling_skip_seconds, "rolling_skip_seconds") < 0:
            raise ValueError("rolling_skip_seconds must not be negative")
        if self.quoted_gpu_hourly > self.limits.quoted_total_hourly:
            raise ValueError("GPU price exceeds total price")
        warmup = self.speed_check_warmup_updates
        if type(warmup) is not int or not 0 <= warmup < min(s.updates for s in self.stages):
            raise ValueError("speed-check warmup must be a nonnegative integer below every stage")
        if type(self.keep_rolling) is not int or self.keep_rolling < 1:
            raise ValueError("keep_rolling must be a positive integer")
        if self.max_transfer_seconds > MAX_TRANSFER_SECONDS:
            raise ValueError(f"max_transfer_seconds exceeds the {MAX_TRANSFER_SECONDS} s ceiling")
        if self.max_transfer_seconds <= FINALIZE_MARGIN_SECONDS + PULL_RESERVE_SECONDS:
            # A final drain waits without GPU work: it, the checks after it and the Mac's
            # SSH pull must all end inside the watchdog's transfer window.
            raise ValueError(
                "max_transfer_seconds must exceed the finalize margin and the pull reserve"
            )

    @property
    def tokens_per_update(self) -> int:
        train = self.train
        return train.batch_size * train.sequence_length * train.accumulation_steps

    @property
    def total_updates(self) -> int:
        return sum(stage.updates for stage in self.stages)

    def stage_train(self, index: int) -> TrainConfig:
        stage = self.stages[index]
        return replace(
            self.train,
            steps=stage.updates,
            warmup_steps=stage.warmup_updates,
            learning_rate=stage.learning_rate,
        )

    def with_limits(self, **changes: Any) -> Specification:
        """This specification with some limits changed, under the file's names."""
        return replace(self, limits=rehearsal_limits(**{**limit_values(self.limits), **changes}))

    def to_dict(self) -> dict[str, Any]:
        """The specification file's form; ``from_dict`` reads it back unchanged."""
        value = asdict(self)
        train = value.pop("train")
        value["train"] = {key: train[key] for key in sorted(TRAIN_KEYS)}
        value["stages"] = [
            {**stage, "streams": list(stage["streams"])} for stage in value["stages"]
        ]
        value["limits"] = limit_values(self.limits)
        value["validation"]["streams"] = dict(self.validation.streams)
        value["r2"] = {key: value.pop(key) for key in sorted(R2_KEYS)}
        return {"schema_version": SCHEMA_VERSION, "kind": SPECIFICATION_KIND, **value}

    @classmethod
    def load(cls, path: Path) -> Specification:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: Any) -> Specification:
        top = _exact(raw, SPECIFICATION_KEYS, "the specification")
        if top["schema_version"] != SCHEMA_VERSION or top["kind"] != SPECIFICATION_KIND:
            raise ValueError("not a rehearsal run specification of this schema")
        if not isinstance(top["stages"], list) or not top["stages"]:
            raise ValueError("stages must be a nonempty list")
        stages = []
        for item in top["stages"]:
            stage = _exact(item, STAGE_KEYS, "a stage")
            if not isinstance(stage["streams"], list):
                raise ValueError("stage streams must be a list")
            stages.append(StagePlan(**{**stage, "streams": tuple(stage["streams"])}))
        first = stages[0]
        train = TrainConfig(
            **_exact(top["train"], TRAIN_KEYS, "train"),
            steps=first.updates,
            warmup_steps=first.warmup_updates,
            learning_rate=first.learning_rate,
        )
        validation = _exact(top["validation"], VALIDATION_KEYS, "validation")
        streams = validation["streams"]
        if not isinstance(streams, dict) or any(
            not isinstance(value, str) for value in streams.values()
        ):
            raise ValueError("validation streams must map labels to stream names")
        # A JSON object has no order, and sorted-key writers (atomic_json) reorder it:
        # the streams are read in the binding's validation order whatever the file's order.
        rank = {name: index for index, name in enumerate(VALIDATION_STREAMS)}
        ordered = sorted(streams.items(), key=lambda item: rank.get(item[1], len(rank)))
        r2 = _exact(top["r2"], R2_KEYS, "r2")
        return cls(
            run=top["run"],
            model=ModelConfig(**_exact(top["model"], MODEL_KEYS, "model")),
            train=train,
            stages=tuple(stages),
            validation=ValidationPlan(**{**validation, "streams": tuple(ordered)}),
            limits=rehearsal_limits(**_exact(top["limits"], LIMIT_KEYS, "limits")),
            warmup_fraction=top["warmup_fraction"],
            checkpoint_seconds=top["checkpoint_seconds"],
            checkpoint_save_seconds=top["checkpoint_save_seconds"],
            expected_tokens_per_second=top["expected_tokens_per_second"],
            validation_tokens_per_second=top["validation_tokens_per_second"],
            speed_check_warmup_updates=top["speed_check_warmup_updates"],
            quoted_gpu_hourly=top["quoted_gpu_hourly"],
            keep_rolling=r2["keep_rolling"],
            storage_cap_usd_per_month=r2["storage_cap_usd_per_month"],
            rolling_skip_seconds=r2["rolling_skip_seconds"],
            max_transfer_seconds=r2["max_transfer_seconds"],
        )


def quarter_updates(updates: int) -> tuple[tuple[int, int], ...]:
    """(quarter k, stage update) for the full validations inside a stage of ``updates``.

    The updates are ``round(k * updates / 4)`` for k = 1, 2, 3 (Python's rounding), each
    once and strictly inside the stage; the stage end has its own full validation.
    """
    found: dict[int, int] = {}
    for quarter in (1, 2, 3):
        update = round(quarter * updates / 4)
        if 0 < update < updates:
            found.setdefault(update, quarter)
    return tuple((quarter, update) for update, quarter in found.items())


def full_validations(stages: Iterable[StagePlan]) -> int:
    """Full validations of both streams these stages plan: each quarter and stage end."""
    return sum(len(quarter_updates(stage.updates)) + 1 for stage in stages)


def planned_seconds(
    spec: Specification,
    *,
    stage_targets: Sequence[int],
    full_validation_targets: int,
    tokens_per_second: float | None = None,
) -> dict[str, float]:
    """Pod seconds the whole run needs from the worker's start at one training speed, by part.

    ``tokens_per_second`` defaults to the reference speed, ``expected_tokens_per_second``:
    the launch rule's basis. Validation counts every interval subset (global update 0
    included) and every full validation (``full_validations``: each quarter and end of
    every stage) at the planning validation rate. Startup is charged once per stage.
    Checkpoints are one save per ``checkpoint_seconds`` of training and validation plus
    one per stage end, each ``checkpoint_save_seconds`` long. The finalization reserve
    comes last.
    """
    rate = finite_number(
        spec.expected_tokens_per_second if tokens_per_second is None else tokens_per_second,
        "training tokens per second",
    )
    if rate <= 0 or len(stage_targets) != len(spec.stages):
        raise ValueError("the plan needs a positive speed and one target count per stage")
    targets = sum(stage_targets)
    interval = spec.validation.interval_updates
    validations = len(range(0, spec.total_updates + 1, interval))
    subset = spec.validation.subset_target_tokens * len(spec.validation.streams)
    full = full_validations(spec.stages)
    validation_tokens = validations * subset + full * full_validation_targets
    training = targets / rate
    validation = validation_tokens / spec.validation_tokens_per_second
    saves = math.floor((training + validation) / spec.checkpoint_seconds) + len(spec.stages)
    parts = {
        "training_seconds": training,
        "validation_seconds": validation,
        "startup_seconds": len(spec.stages) * spec.limits.probe_seconds,
        "checkpoint_seconds": saves * spec.checkpoint_save_seconds,
        "finalization_reserve_seconds": spec.limits.finalization_reserve_seconds,
    }
    return {
        "tokens_per_second": rate,
        "training_targets": float(targets),
        "interval_validations": float(validations),
        "full_validations": float(full),
        "validation_tokens": float(validation_tokens),
        "validation_tokens_per_second": spec.validation_tokens_per_second,
        "checkpoint_saves": float(saves),
        **parts,
        "total_seconds": math.fsum(parts.values()),
    }


def binding_targets(spec: Specification, binding: Mapping[str, Any]) -> tuple[list[int], int]:
    """Each stage's targets and both full validation streams' targets, from a checked binding.

    The binding's stages, streams and planned updates must be the specification's.
    """
    stages = binding["runs"][spec.run]["stages"]
    bound = [(stage["name"], tuple(stage["streams"]), stage["planned_updates"]) for stage in stages]
    if bound != [(plan.name, plan.streams, plan.updates) for plan in spec.stages]:
        raise ValueError("the binding's stages, streams or planned updates differ from the spec's")
    full = sum(
        int(binding["streams"][stream]["tokens"]) - int(binding["streams"][stream]["documents"])
        for _, stream in spec.validation.streams
    )  # Every document's tokens after its BOS.
    return [int(stage["targets"]) for stage in stages], full


def lifetime_end(
    policy: Policy,
    *,
    origin: float,
    deadline: float,
    pod_seconds: float,
    cost_stop_usd: float,
    quoted_total_hourly: float,
    starting_cost_usd: float = 0.0,
) -> float:
    """The first moment a guard ends the Pod: the wall, the armed in-Pod deadline, the
    watchdog's wall, or the worker's or the watchdog's cost stop at its quoted rate.

    The worker passes its limits; the receipt builder passes the same D-35 values."""
    return min(
        origin + pod_seconds,
        deadline,
        origin + policy.max_wall_seconds,
        origin + (cost_stop_usd - starting_cost_usd) / quoted_total_hourly * 3600,
        origin
        + (policy.max_cost_usd - policy.starting_cost_usd) / policy.quoted_total_hourly * 3600,
    )


def require_fit(remaining_seconds: float, needed_seconds: float) -> None:
    """The launch rule's one comparison, shared by the worker and the receipt builder."""
    remaining = finite_number(remaining_seconds, "remaining seconds")
    needed = finite_number(needed_seconds, "reference seconds needed")
    if needed <= 0 or remaining < needed:
        raise ValueError(
            f"{remaining:,.0f} s of Pod lifetime left cannot hold the run at the reference "
            f"speed ({needed:,.0f} s needed)"
        )


def launch_fit(
    spec: Specification, binding: Mapping[str, Any], *, remaining_seconds: float
) -> dict[str, Any]:
    """The launch rule's record: the plan at the reference speed against the lifetime left.

    The plan at the speed floor is recorded too, for information: a run that only fits
    at the reference speed may be stopped by the speed check's projected-finish rule.
    """
    stage_targets, full = binding_targets(spec, binding)
    reference = planned_seconds(spec, stage_targets=stage_targets, full_validation_targets=full)
    floor = planned_seconds(
        spec,
        stage_targets=stage_targets,
        full_validation_targets=full,
        tokens_per_second=spec.limits.minimum_speed_fraction * spec.expected_tokens_per_second,
    )
    remaining = finite_number(remaining_seconds, "remaining seconds")
    return {
        "rule": LAUNCH_RULE,
        "speed_check_rule": SPEED_CHECK_RULE,
        "remaining_seconds": remaining,
        "reference_tokens_per_second": reference["tokens_per_second"],
        "reference_seconds_needed": reference["total_seconds"],
        "floor_tokens_per_second": floor["tokens_per_second"],
        "floor_seconds_needed": floor["total_seconds"],
        "fits_at_reference": 0 < reference["total_seconds"] <= remaining,
        "fits_at_floor": 0 < floor["total_seconds"] <= remaining,
        "reference": reference,
        "floor": floor,
    }


def check_receipt_identity(
    receipt: Mapping[str, Any],
    *,
    run: str,
    specification_sha256: str,
    binding_sha256: str,
    preflight_sha256: str,
    initialization_sha256: str,
) -> None:
    """The receipt must be a rehearsal readiness receipt for this run and these files."""
    if receipt.get("kind") != READINESS_KIND or receipt.get("run") != run:
        raise ValueError(f"the receipt is not a {READINESS_KIND} receipt for run {run}")
    for name, value, message in (
        ("specification_sha256", specification_sha256, "specification differs"),
        ("binding_sha256", binding_sha256, "binding identity differs"),
        ("preflight_sha256", preflight_sha256, "preflight report differs"),
        ("initialization_sha256", initialization_sha256, "initialization identity differs"),
    ):
        if receipt.get(name) != value:
            raise ValueError(f"{message} from the readiness receipt")


def check_revision(receipt: Mapping[str, Any], revision: Any) -> dict[str, Any]:
    """The shipped ``revision.json`` must equal the receipt's ``revision``; return it.

    The Mac writes both from one clean tree: ``git_commit`` (40 hex digits), ``clean``
    (``true``) and ``archive_sha256``, the SHA-256 of the archive it shipped. The Pod's
    copy of the code has no ``.git``, so this is the record of which commit trains.
    """
    expected = receipt.get("revision")
    if (
        not isinstance(expected, dict)
        or set(expected) != REVISION_KEYS
        or not isinstance(expected["git_commit"], str)
        or not GIT_COMMIT.fullmatch(expected["git_commit"])
        or expected["clean"] is not True
        or not isinstance(expected["archive_sha256"], str)
        or not SHA256_HEX.fullmatch(expected["archive_sha256"])
    ):
        raise ValueError(
            "the readiness receipt must record the revision of a clean tree: "
            f"exactly {sorted(REVISION_KEYS)}"
        )
    if revision != expected:
        raise ValueError("the shipped code revision differs from the readiness receipt's")
    return dict(expected)


@dataclass(frozen=True)
class Launch:
    """What a checked receipt fixes for the run."""

    origin: float
    """The conservative billing origin: the earlier of the create request and the Pod's
    provider creation time."""
    end: float
    """``lifetime_end``: the first moment any guard ends the Pod."""
    pod_id: str
    """From the receipt's captured identity; ``RUNPOD_POD_ID`` is often absent over SSH."""
    spec: Specification
    """The specification, its Pod lifetime shortened to ``end - origin`` when that is less."""
    fit: dict[str, Any]


def check_readiness(
    receipt: dict[str, Any],
    *,
    now: float,
    spec: Specification,
    binding: Mapping[str, Any],
    specification_sha256: str,
    binding_sha256: str,
    preflight_sha256: str,
    initialization_sha256: str,
) -> Launch:
    """Accept exactly a receipt from ``pilot_readiness_receipt rehearsal`` for this launch.

    The receipt's kind, run and the specification, binding, preflight and initialization
    SHA-256s must match; every gate must be ``true``; the termination guarantee, Pod
    identity, watchdog policy (``mode: rehearsal``, this run's transfer window) and armed
    deadline are checked as in the pilot, and the R2 prefix must be this Pod's. The
    specification must skip rolling saves at least ``ROLLING_SKIP_SECONDS`` before a
    stage ends. Then the launch rule: the lifetime left must hold the whole run at the
    reference speed. ``run`` also checks the shipped code revision (``check_revision``).
    """
    check_receipt_identity(
        receipt,
        run=spec.run,
        specification_sha256=specification_sha256,
        binding_sha256=binding_sha256,
        preflight_sha256=preflight_sha256,
        initialization_sha256=initialization_sha256,
    )
    missing = [name for name in REHEARSAL_GATES if receipt.get(name) is not True]
    if missing:
        raise ValueError("rehearsal launch readiness gates are unresolved: " + ", ".join(missing))
    check_termination_guarantee(receipt)
    limits = spec.limits
    if (
        limits.pod_seconds > MAX_POD_SECONDS
        or limits.idle_seconds > MAX_IDLE_SECONDS
        or limits.approved_cost_cap > MAX_APPROVED_COST_USD
    ):
        raise ValueError("specification exceeds the rehearsal wall, idle or cost ceiling")
    if spec.rolling_skip_seconds < ROLLING_SKIP_SECONDS:
        raise ValueError(
            f"rolling saves must stop {ROLLING_SKIP_SECONDS:g} s before a stage ends, so the "
            "stage-final milestone never queues behind one"
        )
    origin, deadline = termination_window(receipt, now=now, limits=limits)
    policy = Policy(**receipt["watchdog_policy"])
    if policy.mode != "rehearsal":
        raise ValueError('the watchdog policy must say "mode": "rehearsal"')
    if policy.max_transfer_seconds != spec.max_transfer_seconds:
        raise ValueError(
            f"the watchdog allows {policy.max_transfer_seconds:g} s of uploading after GPU "
            f"work; the worker plans {spec.max_transfer_seconds:g} s"
        )
    pod_id = PodIdentity.from_dict(receipt["pod_identity"]).pod_id
    if receipt.get("r2_run_prefix") != run_prefix(spec.run, pod_id):
        raise ValueError("the receipt's R2 prefix is not this run's Pod prefix")
    end = lifetime_end(
        policy,
        origin=origin,
        deadline=deadline,
        pod_seconds=limits.pod_seconds,
        cost_stop_usd=limits.cost_stop,
        quoted_total_hourly=limits.quoted_total_hourly,
        starting_cost_usd=limits.starting_cost_usd,
    )
    fit = launch_fit(spec, binding, remaining_seconds=end - now)
    require_fit(fit["remaining_seconds"], fit["reference_seconds_needed"])
    if end - origin < limits.pod_seconds:
        # The in-pod timer (or a guard) ends the Pod before the wall; every deadline plans
        # against that moment.
        spec = spec.with_limits(pod_seconds=end - origin)
    return Launch(origin, end, pod_id, spec, fit)


def check_preflight(
    preflight: Mapping[str, Any], *, hardware: Mapping[str, Any], spec: Specification
) -> None:
    """Require a completed A40 preflight with the fused-attention and compile checks."""
    if (
        preflight.get("status") != "completed"
        or preflight.get("hardware_precision_and_resume_checked") is not True
        or preflight.get("device") != "cuda:0"
        or preflight.get("hardware") != hardware
    ):
        raise ValueError("a matching A40 hardware precision/resume preflight is required")
    for name, needed in (
        ("fused_attention", spec.train.attention == "fused"),
        ("compile_equivalence", spec.train.compile),
    ):
        check = preflight.get(name)
        if needed and (not isinstance(check, dict) or check.get("status") != "passed"):
            raise ValueError(
                f"the rehearsal needs the on-device {name.replace('_', ' ')} check to have "
                "passed in the preflight (tools.pilot_preflight --compile-check)"
            )


def _write_new_bytes(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` whole, and only if nothing is there yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # Fails if anything already holds the name.
    finally:
        temporary.unlink()


def _replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_initialization(
    specs: Sequence[Specification], tokenizer_identity: str, output: Path
) -> str:
    """Write the shared initialization in the parent-checkpoint format; return its SHA-256.

    The weights are what a trainer seeded with the specifications' common seed builds
    before its first update. Every specification must name the same model and seed.
    """
    if not specs or len({(spec.model, spec.train.seed) for spec in specs}) != 1:
        raise ValueError("the runs must share one model and seed to share an initialization")
    if not re.fullmatch(r"[0-9a-f]{64}", tokenizer_identity):
        raise ValueError("the tokenizer identity must be a SHA-256 hex digest")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists; an initialization is never rewritten")
    torch.manual_seed(specs[0].train.seed)
    model = GPT(specs[0].model)
    state = {
        "config": {"model": asdict(specs[0].model)},
        "tokenizer_hash": tokenizer_identity,
        "model": model.state_dict(),
        "run_id": INITIALIZATION_RUN_ID,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink()
    return file_sha256(output)


def verify_initialization(
    path: Path, *, expected_sha256: str, model: ModelConfig, tokenizer_identity: str
) -> None:
    """Check the initialization's bytes and format before any trainer reads it."""
    if file_sha256(path) != expected_sha256:
        raise ValueError("the initialization's SHA-256 differs from the expected value")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"config", "tokenizer_hash", "model", "run_id"}:
        raise ValueError("the initialization is not in the parent-checkpoint format")
    if state["config"] != {"model": asdict(model)} or state["run_id"] != INITIALIZATION_RUN_ID:
        raise ValueError("the initialization was written for another model")
    if state["tokenizer_hash"] != tokenizer_identity:
        raise ValueError("the initialization was written for another tokenizer")


@dataclass(frozen=True, eq=False)
class StageData:
    """One stage's verified shards in order, its document order and the binding's ticket."""

    plan: StagePlan
    shards: tuple[tuple[str, TokenShard, str], ...]
    order: npt.NDArray[np.void]
    expected: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class RunData:
    stages: tuple[StageData, ...]
    validation: tuple[tuple[str, TokenShard], ...]
    tokenizer: ByteBPE
    binding_sha256: str
    multiset_sha256: str
    data_kind: str


def load_run_data(
    binding_path: Path, root: Path, *, expected_sha256: str, spec: Specification
) -> RunData:
    """Verify every file the run reads under ``root`` and pair it with the specification."""
    verified = verify_binding(binding_path, root, expected_sha256=expected_sha256, run=spec.run)
    train = spec.train
    packing = {
        "sequence_length": train.sequence_length,
        "microbatch": train.batch_size,
        "accumulation": train.accumulation_steps,
        "windows_per_update": train.batch_size * train.accumulation_steps,
        "tokens_per_update": spec.tokens_per_update,
    }
    if verified["packing"] != packing:
        raise ValueError("the specification packs windows differently from the binding")
    tokenizer: ByteBPE = verified["tokenizer"]
    if tokenizer.vocab_size != spec.model.vocab_size:
        raise ValueError("the model vocabulary differs from the bound tokenizer")
    stages = []
    for plan, stage in zip(spec.stages, verified["stages"][spec.run], strict=True):
        if stage["name"] != plan.name or tuple(stage["streams"]) != plan.streams:
            raise ValueError(f"{plan.name}: the binding's stage differs from the specification")
        if stage["planned_updates"] != plan.updates:
            raise ValueError(
                f"{plan.name}: the specification plans {plan.updates} updates, "
                f"the binding {stage['planned_updates']}"
            )
        shards = tuple(
            (stream, shard, sha256)
            for stream in plan.streams
            for sha256, shard in verified["streams"][stream]
        )
        if [sha256 for _, _, sha256 in shards] != [sha256 for sha256, _ in stage["shards"]]:
            raise ValueError(f"{plan.name}: shard order differs from the bound order")
        expected = {key: stage[key] for key in EXPECTED_TICKET}
        stages.append(StageData(plan, shards, stage["order"], expected))
    validation = []
    for label, stream in spec.validation.streams:
        held = verified["streams"][stream]
        if len(held) != 1:
            raise ValueError(f"{stream}: the subset rule reads exactly one shard per stream")
        validation.append((label, held[0][1]))
    return RunData(
        tuple(stages),
        tuple(validation),
        tokenizer,
        expected_sha256,
        combine_digests(*(stage.expected["multiset_sha256"] for stage in stages)),
        "hash-bound-rehearsal-build",
    )


def length_sorted(windows: Iterable[ValidationWindow], buffer: int) -> Iterator[ValidationWindow]:
    """Windows in runs of ``buffer``, each run sorted by length so batches pad little.

    Only the batching changes: every window is still scored exactly once.
    """
    for run in batched(windows, buffer):
        yield from sorted(run, key=lambda window: len(window.tokens))


def _redact(value: Any, secrets: Sequence[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_redact(key, secrets): _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    return value


def environment_secrets(environ: Mapping[str, str] = os.environ) -> list[str]:
    """Values of the known credential variables that are set; never logged themselves."""
    return [environ[name] for name in SECRET_VARIABLES if environ.get(name)]


class HandoverDue(TimeoutError):
    """An R2 check after the drain that could not start before the handover's limits.

    Unlike a socket timeout, it says nothing about R2: the label was simply not checked.
    """


@dataclass
class _Active:
    """The stage being trained: its data, global offset and per-update timing."""

    index: int
    data: StageData
    offset: int
    record: dict[str, Any]
    rolling: Path
    timing: list[tuple[int, float, float]] = field(default_factory=list)
    """(stage update, tokens, step seconds) for every update."""

    @property
    def quarters(self) -> dict[int, int]:
        """Stage update -> quarter k, for the full validations inside this stage."""
        return {update: quarter for quarter, update in quarter_updates(self.data.plan.updates)}


class Worker:
    """One rehearsal run on one machine; checkpoints leave through R2.

    ``pod_created`` is the conservative billing origin, and ``spec.limits.pod_seconds``
    already ends at the armed in-pod deadline when that comes first. A milestone that
    fails to upload is retried for the rest of the run, first after
    ``upload_retry_seconds`` and then doubling up to ``RETRY_GROWTH`` times that.
    """

    def __init__(
        self,
        spec: Specification,
        root: Path,
        data: RunData,
        *,
        initialization: Path,
        initialization_sha256: str,
        store: Store,
        pod_id: str,
        pod_created: float,
        device: str = "cpu",
        clock: Callable[[], float] = time.time,
        secrets: Iterable[str] = (),
        extra: Mapping[str, Any] | None = None,
        upload_retry_seconds: float = MILESTONE_RETRY_SECONDS,
        check_store_for: Callable[[float], Store] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if len(data.stages) != len(spec.stages) or any(
            stage.plan != plan or stage.expected["planned_updates"] != plan.updates
            for stage, plan in zip(data.stages, spec.stages, strict=False)
        ):
            raise ValueError("the run data's stages differ from the specification's")
        if data.tokenizer.vocab_size != spec.model.vocab_size:
            raise ValueError("the model vocabulary differs from the tokenizer")
        if [label for label, _ in data.validation] != [
            label for label, _ in spec.validation.streams
        ]:
            raise ValueError("the validation shards differ from the specification's streams")
        if not isinstance(pod_id, str) or not LABEL.fullmatch(pod_id):
            raise ValueError("the Pod ID must be a plain path component")
        if root.exists() or root.is_symlink():
            raise FileExistsError("rehearsal output must be a new directory")
        self.spec, self.root, self.data, self.store = spec, root, data, store
        # The R2 checks after the final drain take a store with the given socket timeout:
        # a single-attempt bounded store on the Pod (``run``), the main store in tests.
        self.check_store_for = (
            (lambda _timeout: store) if check_store_for is None else check_store_for
        )
        self.sleep = sleep
        self.handover: float | None = None
        """When finalization starts: the time its final heartbeat is due."""
        self.initialization, self.initialization_sha256 = initialization, initialization_sha256
        self.device = training_device(device)
        self.clock, self.pod_created = clock, pod_created
        self.pod_end = pod_created + spec.limits.pod_seconds
        # Longest first, so a secret that contains another is redacted whole.
        self.secrets = tuple(sorted({s for s in secrets if len(s) >= 4}, key=len, reverse=True))
        self.pod_id = pod_id
        self.prefix = run_prefix(spec.run, pod_id)  # The layout the Mac side verifies.
        self.last_activity = clock()
        self.last_heartbeat = 0.0
        self.last_saved = 0.0
        self.training_started: float | None = None
        self.training_deadline: float | None = None
        self.stop_reason: str | None = None
        self.uploading = False
        """True while finalization resubmits, drains, checks and uploads to R2, so while
        any milestone is still pending or in flight: the heartbeat then reads
        ``uploading``, which the watchdog allows for ``max_transfer_seconds``."""
        self.phase = "initializing"
        self.stage_name = spec.stages[0].name
        self.global_update = 0
        self.tokens_processed = 0
        self.rising = 0
        self.previous_loss: float | None = None
        self.validation_seconds = 0.0
        self.validation_tokens = 0.0
        self.save_seconds: list[float] = []
        self.active: _Active | None = None
        self.timings: list[list[tuple[int, float, float]]] = []
        self.labels: list[str] = []
        self.milestones: list[str] = []
        self.label_files: dict[str, list[tuple[str, Path, str]]] = {}
        self.checkpoints: list[dict[str, Any]] = []
        self.r2_final: dict[str, Any] | None = None
        self.verification_errors: dict[str, str] = {}
        self.total_targets = sum(int(stage.expected["targets"]) for stage in data.stages)
        estimate = 12 * expected_parameters(spec.model)  # fp32 weights and both moments.
        needed = (len(spec.stages) + 3) * estimate
        existing = root.parent
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free < needed:
            raise ValueError(f"{free:,} free bytes cannot hold {needed:,} bytes of checkpoints")
        sequence = spec.train.sequence_length
        self.subset: dict[str, tuple[ValidationWindow, ...]] = {}
        identity: dict[str, Any] = {}
        self.full_targets = 0
        for label, shard in data.validation:
            windows = document_windows(
                shard,
                data.tokenizer,
                sequence_length=sequence,
                target_tokens=spec.validation.subset_target_tokens,
            )
            self.subset[label] = tuple(sorted(windows, key=lambda window: len(window.tokens)))
            # Each document's tokens after its BOS; windows of only special tokens are skipped.
            full = sum(length - 1 for _, length in shard.offsets)
            self.full_targets += full
            identity[label] = {
                "shard_sha256": file_sha256(shard.path),
                "subset_windows": len(windows),
                "subset_target_tokens": sum(len(window.tokens) - 1 for window in windows),
                "subset_target_bytes": sum(window.target_bytes for window in windows),
                "subset_selection_sha256": selection_sha256(windows),
                "full_targets_upper_bound": full,
            }
        self.subset_tokens = sum(row["subset_target_tokens"] for row in identity.values())
        commit, dirty = git_state()
        revision = (extra or {}).get("code_revision")
        if isinstance(revision, Mapping):
            # The Pod's copy of the code has no .git: the checked revision.json names it.
            commit, dirty = revision["git_commit"], revision["clean"] is not True
        self.stage_records: list[dict[str, Any]] = [
            {
                "name": stage.plan.name,
                "streams": list(stage.plan.streams),
                "status": "pending",
                "planned_updates": stage.plan.updates,
                "updates": 0,
                "tokens": 0,
                "planned_targets": stage.expected["targets"],
                "learning_rate": stage.plan.learning_rate,
                "warmup_updates": stage.plan.warmup_updates,
                "optimizer": "fresh AdamW; the schedule restarts with this stage's warmup",
                "parent_checkpoint_sha256": None,
                "ticket": None,
                "expected_ticket": dict(stage.expected),
                "expected_ticket_match": None,
                "final_checkpoint_sha256": None,
                "r2_labels": [],
                "validations": [],
                "planned_quarter_updates": [u for _, u in quarter_updates(stage.plan.updates)],
                "quarter_validations": [],
                "full_validation": None,
                "rolling_saves_skipped_from_update": None,
            }
            for stage in data.stages
        ]
        limits = spec.limits
        self.report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "run": spec.run,
            "status": "running",
            "created_at": utc_now(),
            "git_commit": commit,
            "dirty": dirty,
            "specification": spec.to_dict(),
            "device": str(self.device),
            "data_kind": data.data_kind,
            "binding_sha256": data.binding_sha256,
            "tokenizer_identity": data.tokenizer.sha256,
            "initial_weights_sha256": initialization_sha256,
            "pod_id": pod_id,
            "validation_identity": identity,
            "stages": self.stage_records,
            "run_ticket": None,
            "training_tokens_processed": 0,
            "checkpoints": self.checkpoints,
            "limits": {
                **limit_values(limits),
                "cost_stop_usd": limits.cost_stop,
                "pod_end_unix": self.pod_end,
                "training_deadline_unix": None,
            },
            "speed_check": {
                "status": "pending",
                "rule": SPEED_CHECK_RULE,
                "after_seconds": limits.speed_check_seconds,
                "floor_tokens_per_second": limits.minimum_speed_fraction
                * spec.expected_tokens_per_second,
            },
            "cost": None,
            "r2": {"prefix": self.prefix, "bucket": store.credentials.bucket},
            "timings": {"pod_created_unix": pod_created, "worker_started_unix": clock()},
            "disk": {"free_bytes": free, "checkpoint_bytes_estimate": estimate},
            "semantics": (
                "every bound document of each stage is a training target exactly once; "
                "validation runs on the fixed subset at every multiple of the interval in "
                "global updates (0 included) and on both full validation streams at stage "
                "updates round(k x stage updates / 4) for k = 1, 2, 3 and at each stage "
                "end; a stage-1 end is never run completion; training stops at the Pod end "
                "minus the finalization reserve, and wall, cost and idle limits are never "
                "extended"
            ),
            "limitations": [
                "worker time checks cannot terminate a provider resource during an API outage",
                "quoted-rate costs are estimates until reconciled with provider billing",
                "one training seed per run; the comparison also differs by the stage-2 "
                "optimizer and schedule reset",
            ],
            **dict(extra or {}),
        }
        root.mkdir(parents=True)
        (root / "updates.jsonl").touch()
        self.uploader = CheckpointUploader(
            store,
            prefix=self.prefix,
            cap_usd=spec.storage_cap_usd_per_month,
            receipt_dir=root / "r2-receipts",
            keep_rolling=spec.keep_rolling,
            retry_seconds=upload_retry_seconds,
            retry_max_seconds=upload_retry_seconds * RETRY_GROWTH,
        )
        self.publish()
        self.heartbeat()

    # Records ---------------------------------------------------------------------------

    def scrub(self, value: Any) -> Any:
        """JSON-safe and free of every known secret value."""
        return _redact(sanitize(value), self.secrets)

    def write_json(self, path: Path, value: Any, *, limit: int = MAX_REPORT_BYTES) -> str:
        data = (
            json.dumps(self.scrub(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
        if len(data) >= limit:
            raise ValueError(f"{path.name} would be {len(data):,} bytes, over {limit:,}")
        _replace_bytes(path, data)
        return hashlib.sha256(data).hexdigest()

    def publish(self) -> str:
        self.report["training_tokens_processed"] = self.tokens_processed
        return self.write_json(self.root / "report.json", self.report)

    def heartbeat(self) -> None:
        status = self.report["status"]
        if self.uploading:
            # Finalization: no GPU work, the last checkpoints and the report go to R2. The
            # watchdog allows this for max_transfer_seconds; the supervisor never tears down.
            status = UPLOADING_STATUS
        elif status in {"completed", "time_limited"}:
            # Training is over; the run is done only once its durable copy is verified.
            status = "awaiting_transfer" if self.r2_final is not None else "running"
        value: dict[str, Any] = {
            "schema_version": 1,
            "updated_at_unix": self.clock(),
            "last_gpu_activity_unix": self.last_activity,
            "phase": self.phase,
            "step": self.global_update,
            "training_tokens_processed": self.tokens_processed,
            "compile_probe_active": False,
            "status": status,
            "run": self.spec.run,
            "pod_id": self.pod_id,
            "stage": self.stage_name,
        }
        if self.r2_final is not None:
            value["r2_final"] = self.r2_final
        if not self.uploading and "failure" in self.report:
            value["failure"] = self.report["failure"]
        self.write_json(self.root / "heartbeat.json", value, limit=MAX_HEARTBEAT_BYTES)
        self.last_heartbeat = self.clock()

    # Limits ----------------------------------------------------------------------------

    def stopped(self) -> bool:
        now = self.clock()
        reason = self.spec.limits.stop_reason(
            pod_elapsed=max(0, now - self.pod_created),
            idle_elapsed=max(0, now - self.last_activity),
        )
        if reason:
            self.stop_reason = reason
        elif self.training_deadline is not None and now >= self.training_deadline:
            self.stop_reason = "finalization_reserve"
        return self.stop_reason is not None

    def require_hard_limits(self) -> None:
        """Wall, cost and idle limits; the finalization reserve still permits final saves."""
        now = self.clock()
        reason = self.spec.limits.stop_reason(
            pod_elapsed=max(0, now - self.pod_created),
            idle_elapsed=max(0, now - self.last_activity),
        )
        if reason:
            self.stop_reason = reason
            raise PilotStop(reason)

    def tick(self) -> None:
        """GPU work in progress outside an update (validation): refresh activity, keep limits."""
        now = self.clock()
        self.last_activity = now
        reason = self.spec.limits.stop_reason(pod_elapsed=max(0, now - self.pod_created))
        if reason:
            self.stop_reason = reason
            raise PilotStop(reason)
        if now - self.last_heartbeat >= HEARTBEAT_REFRESH_SECONDS:
            self.heartbeat()

    # Device ----------------------------------------------------------------------------

    def peak(self) -> dict[str, int]:
        cuda = self.device.type == "cuda"
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else 0,
        }

    def release(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # Validation ------------------------------------------------------------------------

    def _validate(self, model: GPT, windows: Iterable[ValidationWindow]) -> dict[str, float]:
        return validate_batched(
            model,
            windows,
            batch_size=self.spec.validation.batch_size,
            autocast_bf16=self.spec.validation.autocast_bf16,
            on_batch=self.tick,
        )

    def validate_subset(self, model: GPT, update: int) -> None:
        self.phase = f"{self.stage_name}-validation"
        started = synchronized_time(self.device)
        results = {label: self._validate(model, windows) for label, windows in self.subset.items()}
        seconds = synchronized_time(self.device) - started
        assert self.active is not None
        self.active.record["validations"].append(
            {"global_update": update, **results, "seconds": seconds}
        )
        self.validation_seconds += seconds
        self.validation_tokens += math.fsum(result["tokens"] for result in results.values())
        self.last_activity = self.clock()
        self.phase = self.stage_name
        self.publish()

    def validate_full(self, model: GPT) -> dict[str, Any]:
        self.phase = f"{self.stage_name}-full-validation"
        self.heartbeat()
        started = synchronized_time(self.device)
        results: dict[str, Any] = {}
        for label, shard in self.data.validation:
            windows = iter_document_windows(
                shard,
                self.data.tokenizer,
                sequence_length=self.spec.train.sequence_length,
                target_tokens=None,
            )
            results[label] = self._validate(
                model, length_sorted(windows, FULL_VALIDATION_SORT_WINDOWS)
            )
        seconds = synchronized_time(self.device) - started
        self.validation_seconds += seconds
        self.validation_tokens += math.fsum(result["tokens"] for result in results.values())
        self.last_activity = self.clock()
        return {**results, "seconds": seconds, "global_update": self.global_update}

    def validate_quarter(self, model: GPT, quarter: int, stage_update: int) -> None:
        """Both full streams at quarter ``quarter`` of the stage, kept in its record."""
        result = self.validate_full(model)
        assert self.active is not None
        self.active.record["quarter_validations"].append(
            {"quarter": quarter, "stage_update": stage_update, **result}
        )
        self.phase = self.stage_name
        self.publish()

    # Checkpoints -----------------------------------------------------------------------

    def save(
        self,
        trainer: Trainer,
        path: Path,
        label: str,
        *,
        milestone: bool,
        ticket: dict[str, Any] | None = None,
    ) -> str:
        """Save, export and queue one checkpoint for R2; returns its SHA-256."""
        self.require_hard_limits()
        self.phase = f"{self.stage_name}-checkpoint"
        started = synchronized_time(self.device)
        trainer.checkpoint(path)
        seconds = synchronized_time(self.device) - started
        # Copying the state off the device is the run's last GPU work before a final
        # drain: the transfer window then counts from the end of the final save.
        self.last_activity = self.clock()
        digest = file_sha256(path)
        size = path.stat().st_size
        self.write_json(
            path.with_name(path.name + ".sha256.json"),
            {"sha256": digest, "size_bytes": size, "step": trainer.step},
        )
        self.save_seconds.append(seconds)
        self.last_saved = self.clock()
        self.checkpoints.append(
            {
                "label": label,
                "stage": self.stage_name,
                "global_update": self.global_update,
                "stage_update": trainer.step,
                "milestone": milestone,
                "path": str(path.relative_to(self.root)),
                "save_seconds": seconds,
                "size_bytes": size,
                "sha256": digest,
                "training_tokens_processed": self.tokens_processed,
            }
        )
        self.export(path, digest, label, milestone=milestone, ticket=ticket)
        self.phase = self.stage_name
        return digest

    def export(
        self,
        checkpoint: Path,
        digest: str,
        label: str,
        *,
        milestone: bool,
        ticket: dict[str, Any] | None,
    ) -> None:
        """Publish an immutable label directory and submit it to the uploader."""
        directory = self.root / "exports" / label
        staging = directory.with_name("." + label)
        staging.mkdir(parents=True, exist_ok=False)
        try:
            os.link(checkpoint, staging / "checkpoint.pt")
        except OSError:
            shutil.copyfile(checkpoint, staging / "checkpoint.pt")
        (staging / "checkpoint.pt.sha256").write_text(f"{digest}  checkpoint.pt\n")
        shutil.copyfile(self.root / "updates.jsonl", staging / "updates.jsonl")
        names = ["checkpoint.pt", "checkpoint.pt.sha256", "updates.jsonl", "record.json"]
        if ticket is not None:
            self.write_json(staging / "ticket.json", ticket)
            names.append("ticket.json")
        assert self.active is not None
        self.active.record["r2_labels"].append(label)
        self.labels.append(label)
        if milestone:
            self.milestones.append(label)
        self.report["latest_export"] = f"exports/{label}"
        self.write_json(staging / "record.json", self.report)
        staging.replace(directory)
        files = [
            (
                name,
                directory / name,
                digest if name == "checkpoint.pt" else file_sha256(directory / name),
            )
            for name in names
        ]
        self.label_files[label] = files
        self.uploader.submit(label, files, milestone=milestone)
        self.prune_exports()
        self.publish()

    def prune_exports(self) -> None:
        """Remove rolling label directories the uploader no longer needs, keeping the newest.

        Milestones stay; a pending or in-flight label is never touched.
        """
        status = self.uploader.status()
        busy = set(status["pending"]) | {status["in_flight"]}
        rolling = [label for label in self.labels if label not in self.milestones]
        for label in rolling[:-1]:
            directory = self.root / "exports" / label
            if label not in busy and directory.is_dir():
                shutil.rmtree(directory)

    # Training --------------------------------------------------------------------------

    def on_step(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        active = self.active
        assert active is not None
        row = {**metrics, "step": float(trainer.step)}
        if any(not math.isfinite(value) for value in row.values()):
            raise FloatingPointError("nonfinite rehearsal metric")
        self.last_activity = self.clock()
        update = active.offset + trainer.step
        self.global_update = update
        tokens = int(metrics["tokens"])
        self.tokens_processed += tokens
        active.record["updates"] = trainer.step
        active.record["tokens"] += tokens
        active.record["global_update_end"] = update
        active.timing.append((trainer.step, float(tokens), float(metrics["step_seconds"])))
        line = {
            "run": self.spec.run,
            "stage": self.stage_name,
            "update": update,
            "stage_update": trainer.step,
            **metrics,
        }
        with (self.root / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, allow_nan=False) + "\n")
            stream.flush()
        self.heartbeat()
        loss = metrics["loss"]
        self.rising = (
            self.rising + 1 if self.previous_loss is not None and loss > self.previous_loss else 0
        )
        self.previous_loss = loss
        if self.rising >= self.spec.limits.rising_steps:
            self.stop_reason = "loss_rising"
            raise PilotStop(self.stop_reason)
        if self.stopped():
            return  # The trainer stops before its next update; a limit names the reason.
        if update % self.spec.validation.interval_updates == 0:
            self.validate_subset(trainer.model, update)
        quarter = active.quarters.get(trainer.step)
        if quarter is not None:
            self.validate_quarter(trainer.model, quarter, trainer.step)
        self.speed_check()
        if (
            trainer.step < active.data.plan.updates
            and not self.stopped()
            and self.clock() - self.last_saved >= self.spec.checkpoint_seconds
        ):
            if self.near_stage_end(trainer.step):
                if active.record["rolling_saves_skipped_from_update"] is None:
                    active.record["rolling_saves_skipped_from_update"] = update
                    self.publish()
            else:
                self.save(
                    trainer, active.rolling, f"{self.stage_name}-step{update}", milestone=False
                )

    def near_stage_end(self, stage_update: int) -> bool:
        """Whether the stage ends within ``rolling_skip_seconds``: a rolling save is skipped.

        The stage's remaining updates count at its measured seconds per update (after
        the compile warmup when there are such updates; the reference speed before any
        update); the training deadline, where a time-limited stage ends, counts too.
        """
        assert self.active is not None
        warmup = self.spec.speed_check_warmup_updates
        timing = self.active.timing
        steady = [seconds for step, _, seconds in timing if step > warmup]
        measured = steady or [seconds for _, _, seconds in timing]
        per_update = (
            math.fsum(measured) / len(measured)
            if measured
            else self.spec.tokens_per_update / self.spec.expected_tokens_per_second
        )
        left = (self.active.data.plan.updates - stage_update) * per_update
        if self.training_deadline is not None:
            left = min(left, self.training_deadline - self.clock())
        return left <= self.spec.rolling_skip_seconds

    def _post_compile(self) -> list[tuple[float, float]]:
        warmup = self.spec.speed_check_warmup_updates
        return [
            (tokens, seconds)
            for timing in self.timings
            for step, tokens, seconds in timing
            if step > warmup
        ]

    def _compile_excess(self) -> float:
        """Seconds a stage's first updates spend above its steady state (compilation)."""
        warmup = self.spec.speed_check_warmup_updates
        for timing in self.timings:
            steady = [seconds for step, _, seconds in timing if step > warmup]
            first = [seconds for step, _, seconds in timing if step <= warmup]
            if steady and len(first) == warmup:
                return max(0.0, math.fsum(first) - warmup * math.fsum(steady) / len(steady))
        return 0.0

    def remaining_validation_tokens(self, update: int) -> float:
        """Subset tokens of every interval validation after global ``update``, plus every
        full validation after it: this stage's later quarters and end, and each later
        stage's quarters and end."""
        active = self.active
        assert active is not None
        interval = self.spec.validation.interval_updates
        following = (update // interval + 1) * interval
        count = len(range(following, self.spec.total_updates + 1, interval))
        stage_update = update - active.offset
        full = sum(1 for later in active.quarters if later > stage_update) + 1
        full += full_validations(self.spec.stages[active.index + 1 :])
        return float(count * self.subset_tokens + full * self.full_targets)

    def speed_check(self) -> None:
        """Once, after ``speed_check_seconds`` of training: speed floor and projected finish.

        Throughput counts only updates after each stage's first
        ``speed_check_warmup_updates`` (compilation). The projected finish adds, at the
        measured rates, the remaining targets, every remaining validation (interval
        subsets, and the full streams at each later quarter and stage end), a
        recompilation per later stage, the remaining checkpoint saves (at the measured
        save time, or ``checkpoint_save_seconds`` before the first save) and the
        finalization reserve; it must not pass the Pod end, the armed deadline when that
        comes first. Either failure stops the run (``SPEED_CHECK_RULE``).
        """
        record = self.report["speed_check"]
        if record["status"] != "pending" or self.training_started is None:
            return
        now = self.clock()
        if now - self.training_started < self.spec.limits.speed_check_seconds:
            return
        measured = self._post_compile()
        if not measured:
            return  # Waits for the first update after compilation.
        assert self.active is not None
        tokens = math.fsum(row[0] for row in measured)
        seconds = math.fsum(row[1] for row in measured)
        rate = tokens / seconds
        validation_rate = (
            self.validation_tokens / self.validation_seconds
            if self.validation_seconds > 0
            else self.spec.validation_tokens_per_second
        )
        remaining_targets = max(0, self.total_targets - self.tokens_processed)
        validation_tokens = self.remaining_validation_tokens(self.global_update)
        training_seconds = remaining_targets / rate
        later_stages = len(self.spec.stages) - self.active.index - 1
        save = (
            math.fsum(self.save_seconds) / len(self.save_seconds)
            if self.save_seconds
            else self.spec.checkpoint_save_seconds
        )
        saves = int(training_seconds // self.spec.checkpoint_seconds) + later_stages + 1
        parts = {
            "training_seconds": training_seconds,
            "validation_seconds": validation_tokens / validation_rate,
            "recompile_seconds": later_stages * self._compile_excess(),
            "checkpoint_seconds": saves * save,
            "finalization_reserve_seconds": self.spec.limits.finalization_reserve_seconds,
        }
        finish = now + math.fsum(parts.values())
        floor = self.spec.limits.minimum_speed_fraction * self.spec.expected_tokens_per_second
        record.update(
            checked_at_unix=now,
            training_elapsed_seconds=now - self.training_started,
            global_update=self.global_update,
            post_compile_updates=len(measured),
            measured_tokens_per_second=rate,
            expected_tokens_per_second=self.spec.expected_tokens_per_second,
            floor_tokens_per_second=floor,
            measured_validation_tokens_per_second=validation_rate,
            remaining_targets=remaining_targets,
            remaining_validation_tokens=validation_tokens,
            checkpoint_save_seconds_basis=(
                "measured" if self.save_seconds else "planned checkpoint_save_seconds"
            ),
            projected_seconds=parts,
            projected_finish_unix=finish,
            pod_end_unix=self.pod_end,
        )
        reason = None
        if rate < floor:
            reason = "throughput_below_limit"
        elif finish > self.pod_end:
            reason = "projected_finish_after_deadline"
        record["status"] = "passed" if reason is None else "failed"
        if reason is not None:
            record["reason"] = reason
            self.stop_reason = reason
            raise PilotStop(reason)
        self.publish()

    def train_stage(self, index: int, parent: Path, parent_sha256: str) -> tuple[Path, str] | None:
        """Train one stage; returns its final checkpoint, or None when time-limited.

        The parent's bytes are checked against the SHA-256 recorded when it was saved
        (or the expected initialization hash) before a trainer reads them; the trainer
        then loads its weights only, with a fresh optimizer and schedule.
        """
        stage = self.data.stages[index]
        plan = stage.plan
        record = self.stage_records[index]
        offset = sum(previous.updates for previous in self.spec.stages[:index])
        self.stage_name = plan.name
        self.active = _Active(
            index, stage, offset, record, self.root / "checkpoints" / plan.name / "last.pt"
        )
        self.timings.append(self.active.timing)
        self.rising, self.previous_loss = 0, None
        record.update(status="running", started_unix=self.clock(), global_update_start=offset)
        if file_sha256(parent) != parent_sha256:
            raise ValueError(
                f"{plan.name}: parent checkpoint bytes differ from the recorded SHA-256"
            )
        record["parent_checkpoint_sha256"] = parent_sha256
        train = self.spec.stage_train(index)
        loader = OnePassLoader(
            stage.shards,
            stage.order,
            sequence_length=train.sequence_length,
            batch_size=train.batch_size,
            accumulation_steps=train.accumulation_steps,
        )
        if (
            loader.planned_updates != plan.updates
            or loader.total_targets != stage.expected["targets"]
            or loader.order_sha256 != stage.expected["order_sha256"]
        ):
            raise ValueError(f"{plan.name}: the loader's plan differs from the binding")
        tokens = {name: 0 for name in plan.streams}
        for name, shard, _ in stage.shards:
            tokens[name] += sum(length for _, length in shard.offsets)
        total = sum(tokens.values())
        config = RunConfig(
            model=self.spec.model,
            train=train,
            data_mix={name: count / total for name, count in tokens.items()},
            parent_checkpoint=str(parent),
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        trainer = Trainer(
            config,
            loader,
            run_id=f"rehearsal-{self.spec.run}-{plan.name}",
            run_root=self.root / "runs",
            data_hash=self.data.binding_sha256,
            tokenizer_hash=self.data.tokenizer.sha256,
            device=str(self.device),
        )
        self.last_saved = self.clock()  # The checkpoint cadence restarts with each stage.
        self.publish()
        if index == 0:
            self.validate_subset(trainer.model, 0)  # Global update 0: the shared initialization.
        self.phase = plan.name
        trainer.train(should_stop=self.stopped, on_step=self.on_step)
        reason = self.stop_reason
        if reason is not None and reason != "finalization_reserve":
            raise PilotStop(reason)
        complete = trainer.step == plan.updates and loader.exhausted
        if not complete and reason is None:
            raise PilotStop("incomplete_stage")
        # A stage that ends inside the reserve, even on its last update, gets its final
        # save and uploads but no full validation and no further stage.
        late = reason is not None or (
            self.training_deadline is not None and self.clock() >= self.training_deadline
        )
        if late:
            self.report["training_stop_reason"] = "finalization_reserve"
            self.training_deadline = None
            self.stop_reason = None
        ticket = loader.ticket()
        expected = dict(stage.expected)
        match = (
            ticket["complete"] is True
            and ticket["updates"] == expected["planned_updates"]
            and all(ticket[key] == expected[key] for key in ("documents", "targets", "windows"))
            and ticket["multiset_sha256"] == expected["multiset_sha256"]
            and ticket["order_sha256"] == expected["order_sha256"]
        )
        record.update(ticket=ticket, expected_ticket_match=match, peak=self.peak())
        path = self.root / "checkpoints" / plan.name / "final.pt"
        digest = self.save(
            trainer,
            path,
            f"{plan.name}-final",
            milestone=True,
            ticket={
                "run": self.spec.run,
                "stage": plan.name,
                "global_update": self.global_update,
                "binding_sha256": self.data.binding_sha256,
                "ticket": ticket,
                "expected": expected,
                "expected_ticket_match": match,
            },
        )
        record["final_checkpoint_sha256"] = digest
        trainer.finish(success=True)
        self.active.rolling.unlink(missing_ok=True)  # Exports keep any linked bytes they need.
        if late:
            record.update(
                status="completed" if complete else "time_limited",
                ended_unix=self.clock(),
                full_validation={"status": "skipped", "reason": "finalization_reserve"},
            )
            self.publish()
            if complete and not match:
                raise PilotStop("ticket_mismatch")
            return None
        record["full_validation"] = self.validate_full(trainer.model)
        record.update(status="completed", ended_unix=self.clock())
        self.publish()
        if not match:
            raise PilotStop("ticket_mismatch")
        del trainer
        self.release()
        if self.spec.train.compile:
            torch._dynamo.reset()  # The next stage compiles its own model.
        return path, digest

    def execute(self) -> dict[str, Any]:
        try:
            self.training_started = self.clock()
            self.training_deadline = self.spec.limits.training_deadline(
                pod_created=self.pod_created, training_started=self.training_started
            )
            self.report["timings"]["training_started_unix"] = self.training_started
            self.report["limits"]["training_deadline_unix"] = self.training_deadline
            parent, parent_sha256 = self.initialization, self.initialization_sha256
            status = "completed"
            for index in range(len(self.spec.stages)):
                if self.stopped():
                    if self.stop_reason != "finalization_reserve":
                        raise PilotStop(str(self.stop_reason))
                    self.report["training_stop_reason"] = self.stop_reason
                    self.stop_reason = None
                    status = "time_limited"
                    break
                final = self.train_stage(index, parent, parent_sha256)
                if final is None:
                    status = "time_limited"
                    break
                parent, parent_sha256 = final
            self.report["status"] = status
        except PilotStop as error:
            self.report.update(status="stopped", stop_reason=str(error))
        except BaseException as error:
            self.report.update(status="failed", error_type=type(error).__name__)
            self.report["error"] = str(sanitize(str(error)))[:500]
            raise
        finally:
            self.account()
            self.finalize()
            self.active = None
            self.release()
        return self.report

    def run_ticket(self) -> dict[str, Any]:
        tickets = [record["ticket"] for record in self.stage_records if record["ticket"]]
        combined = combine_digests(*(ticket["multiset_sha256"] for ticket in tickets))
        return {
            "multiset_sha256": combined,
            "expected_multiset_sha256": self.data.multiset_sha256,
            "documents": sum(ticket["documents"] for ticket in tickets),
            "targets": sum(ticket["targets"] for ticket in tickets),
            "match": len(tickets) == len(self.stage_records)
            and all(record["expected_ticket_match"] for record in self.stage_records)
            and combined == self.data.multiset_sha256,
        }

    def account(self) -> None:
        self.report["run_ticket"] = self.run_ticket()
        now = self.clock()
        elapsed = max(0.0, now - self.pod_created)
        limits = self.spec.limits
        self.report["cost"] = {
            "quoted_total_hourly": limits.quoted_total_hourly,
            "quoted_gpu_hourly": self.spec.quoted_gpu_hourly,
            "starting_cost_usd": limits.starting_cost_usd,
            "pod_elapsed_seconds": elapsed,
            "quoted_accrued_cost_usd": limits.starting_cost_usd
            + elapsed / 3600 * limits.quoted_total_hourly,
            "cost_stop_usd": limits.cost_stop,
            "approved_cost_cap_usd": limits.approved_cost_cap,
            "basis": "quoted rate times Pod lifetime; reconcile with provider billing",
        }
        measured = self._post_compile()
        if measured:
            tokens = math.fsum(row[0] for row in measured)
            seconds = math.fsum(row[1] for row in measured)
            self.report["training_throughput"] = {
                "post_compile_tokens": tokens,
                "post_compile_step_seconds": seconds,
                "tokens_per_second": tokens / seconds,
                "excluded_warmup_updates_per_stage": self.spec.speed_check_warmup_updates,
            }
        self.report["validation_throughput"] = {
            "tokens": self.validation_tokens,
            "seconds": self.validation_seconds,
        }
        if self.report["speed_check"]["status"] == "pending":
            self.report["speed_check"]["status"] = "not_reached"
        self.report["training_peak"] = self.peak()
        self.report["timings"].update(
            ended_unix=now,
            validation_seconds=self.validation_seconds,
            checkpoint_save_seconds=math.fsum(self.save_seconds),
        )
        self.report["ended_at"] = utc_now()

    # Finalization ----------------------------------------------------------------------

    def handover_due(self) -> float:
        """When the final heartbeat is due: ``PULL_RESERVE_SECONDS`` before the watchdog's
        transfer window ends, and never past the Pod end less a minute.

        The window is the watchdog's: ``max_transfer_seconds`` after the last GPU work,
        which is the end of the final save or validation, so the start of finalization
        for every ending that saves a milestone. Handing over then, 480 s after the last
        GPU work for D-35, is before the watchdog's 600 s idle limit: the Mac supervisor
        can start an SSH pull, whose growing bytes keep the Pod, with the rest of the
        window still to run.
        """
        return min(
            self.last_activity + self.spec.max_transfer_seconds - PULL_RESERVE_SECONDS,
            self.pod_end - 60,
        )

    def drain_budget(self) -> float:
        """Seconds the final drain may wait: until ``FINALIZE_MARGIN_SECONDS`` before the
        handover (360 s after the last GPU work for D-35)."""
        return max(0.0, self.handover_due() - FINALIZE_MARGIN_SECONDS - self.clock())

    def _checked[T](self, action: Callable[[Store], T], what: str, *, late: bool = False) -> T:
        """One R2 check after the drain, retried on transient failures while time lasts.

        It starts only until ``LAST_CALL_SECONDS`` before the handover (``late``: until
        ``FINAL_WRITE_SECONDS`` before it, for the HEAD that verifies a PUT that
        succeeded); one that cannot start raises ``HandoverDue``, and one that runs out of
        time after a failed attempt raises that attempt's own error. Each attempt's socket
        timeout is at most ``CHECK_TIMEOUT_SECONDS`` and a quarter of the time left until
        ``FINAL_WRITE_SECONDS`` before the handover, so an endpoint that resolves to up to
        four addresses still answers or fails in time; name resolution is not bounded.
        """
        assert self.handover is not None, "R2 checks after the drain need a handover time"
        end = self.handover - FINAL_WRITE_SECONDS
        start_by = end if late else self.handover - LAST_CALL_SECONDS
        failed: Exception | None = None
        for attempt in range(1, CHECK_ATTEMPTS + 1):
            now = self.clock()
            if now >= start_by:
                if failed is not None:
                    raise failed  # R2 did answer, or failed to: keep that, not "no time".
                raise HandoverDue(f"no time left before the handover to {what}")
            store = self.check_store_for(min(CHECK_TIMEOUT_SECONDS, (end - now) / 4))
            try:
                return action(store)
            except Exception as error:
                if attempt == CHECK_ATTEMPTS or not transient_error(error):
                    raise
                failed = error
            self.sleep(min(2.0**attempt, max(0.0, start_by - self.clock()) / 4))
        raise RuntimeError("unreachable")

    def _stored(self, key: str, *, late: bool = False) -> str | None:
        """The SHA-256 R2 stores for ``key``, checked within the handover's time limits."""
        return self._checked(lambda store: store.stored_sha256(key), f"check {key}", late=late)

    def kept_labels(self) -> set[str]:
        """Labels the uploader verified and has not deleted (no request is made)."""
        status = self.uploader.status()
        return set(status["verified"]) - set(status["deleted"])

    def milestone_order(self) -> list[str]:
        """The milestones with exports, the newest first: the order they are checked in."""
        return [*reversed([label for label in self.milestones if label in self.label_files])]

    def verified_labels(
        self, labels: Iterable[str], *, unchecked: list[str] | None = None
    ) -> dict[str, bool]:
        """Each of ``labels``, in order: verified and kept by the uploader, and every file
        checked again by HEAD.

        A HEAD that still fails after its retries, or one there is no time left for, counts
        as unverified; its error is kept in ``verification_errors``. Given ``unchecked``, a
        label whose HEADs could not all start in time (``HandoverDue``) is appended there
        instead of being counted either way.
        """
        kept = self.kept_labels()
        result: dict[str, bool] = {}
        for label in labels:
            files = self.label_files[label]
            try:
                result[label] = label in kept and all(
                    self._stored(self.prefix + label + "/" + name) == digest
                    for name, _, digest in files
                )
            except Exception as error:  # recorded; the label is not verified
                self.verification_errors[label] = f"{type(error).__name__}: {error}"
                if unchecked is not None and isinstance(error, HandoverDue):
                    unchecked.append(label)
                else:
                    result[label] = False
        return result

    def record_rolling(self, labels: dict[str, bool], receipt: dict[str, Any]) -> None:
        """Check the rolling labels by HEAD while ``_checked`` allows, for the record only:
        no ending depends on them.

        Each joins ``labels`` (the receipt's ``labels_verified``), except one whose checks
        could not all start before the handover: ``labels_unchecked`` names it, since it
        may still be whole in R2 for the owner to resume from. A run without ``r2_final``
        also records its newest verified label (``last_verified_label``).
        """
        unchecked: list[str] = []
        rolling = [label for label in self.label_files if label not in self.milestones]
        labels.update(self.verified_labels(rolling, unchecked=unchecked))
        receipt.update(labels_verified=labels, labels_unchecked=unchecked)
        if self.r2_final is None:
            last = [label for label in self.labels if labels.get(label)]
            receipt["last_verified_label"] = last[-1] if last else None

    def export_listing(self, label: str) -> dict[str, Any]:
        """A label's local export: its directory, and each file's path, SHA-256 and size."""
        files = []
        for name, path, digest in self.label_files[label]:
            try:
                size: int | None = path.stat().st_size
            except OSError:
                size = None
            files.append({"name": name, "path": str(path), "sha256": digest, "size_bytes": size})
        checkpoint = next(row for row in files if row["name"] == "checkpoint.pt")
        return {
            "label": label,
            "export": f"exports/{label}",
            "export_path": str(self.root / "exports" / label),
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "files": files,
        }

    def drain(self, budget: float) -> bool:
        """Wait up to ``budget`` seconds for queued uploads, keeping the heartbeat fresh.

        Only ``updated_at_unix`` moves: no GPU work happens, so the idle clock keeps running.
        """
        deadline = time.monotonic() + budget
        while True:
            left = deadline - time.monotonic()
            if self.uploader.drain(max(0.0, min(left, HEARTBEAT_REFRESH_SECONDS))):
                return True
            if left <= HEARTBEAT_REFRESH_SECONDS:
                return False
            self.heartbeat()

    def settle(self, budget: float, receipt: dict[str, Any]) -> dict[str, bool]:
        """Resubmit every unverified milestone and drain until all verify or ``budget`` ends.

        The first round resubmits each milestone the uploader has not verified (one
        waiting to retry becomes due at once; one uploading is left to finish). Each
        round drains for the rest of the budget and checks every milestone by HEAD, the
        newest first; one whose HEAD check fails is resubmitted in the next round, at most
        ``SETTLE_ROUNDS`` rounds in all. Returns the HEAD-checked milestones.
        """
        deadline = time.monotonic() + budget
        labels: dict[str, bool] = {}
        rounds: list[dict[str, Any]] = []
        drained = False
        for index in range(SETTLE_ROUNDS):
            if index == 0:
                kept = self.kept_labels()
                missing = [label for label in self.milestones if label not in kept]
            else:
                missing = [label for label in self.milestones if not labels.get(label)]
            for label in missing:
                self.uploader.submit(label, self.label_files[label], milestone=True)
            drained = self.drain(max(0.0, deadline - time.monotonic()))
            labels = self.verified_labels(self.milestone_order())
            rounds.append({"resubmitted": missing, "drained": drained})
            if all(labels.get(label) for label in self.milestones):
                break
            if time.monotonic() >= deadline:
                break
        receipt.update(drained=drained, settle_rounds=rounds)
        return labels

    def finalize(self) -> None:
        """Resubmit and drain uploads, verify them, upload ``report.json``, then the last
        heartbeat.

        The heartbeat reads ``uploading`` until this ends, whatever the training outcome,
        so the Mac side never reads a final status while a milestone is still pending or
        in flight inside the transfer window. A wall or cost stop never waits for uploads.
        Otherwise every unverified milestone is resubmitted and the drain may run until
        ``FINALIZE_MARGIN_SECONDS`` before the handover (``drain_budget``); the bounded
        checks after it end by the handover (``handover_due``). A run with a milestone still not
        verified in R2 then ends ``failed`` with ``failure: r2_milestones_unverified``;
        ``finalization.json`` and ``report.json`` (``r2.unverified_milestones``) list each
        such milestone's local export, path and SHA-256 for the supervisor to pull over
        SSH. A completed or time-limited run whose report cannot be verified in R2 ends
        ``failed`` (``r2_finalization``). ``r2_final`` is published only when a verified
        stage-final milestone closes the run and every milestone is verified; a stopped
        run without one gets none, and the supervisor archives diagnostics. The rolling
        labels are checked last, for the record, after a failed ``report.json`` upload too;
        ``finalization.json`` names any that could not be checked in time
        (``labels_unchecked``).
        """
        self.phase = "finalizing"
        trained = self.report["status"]
        hurry = self.report.get("stop_reason") in NO_FINALIZATION_STOPS
        self.handover = self.handover_due()
        receipt: dict[str, Any] = {
            "training_status": trained,
            "started_unix": self.clock(),
            "transfer_window_end_unix": self.last_activity + self.spec.max_transfer_seconds,
            "handover_due_unix": self.handover,
        }
        report_key = self.prefix + "report.json"
        unverified: list[dict[str, Any]] | None = None
        labels: dict[str, bool] = {}
        self.uploading = not hurry
        try:
            self.heartbeat()
            if hurry:
                kept = self.kept_labels()
                labels = {label: label in kept for label in self.label_files}
                receipt.update(drain_budget_seconds=0.0, drained=self.drain(0.0))
            else:
                budget = self.drain_budget()
                receipt["drain_budget_seconds"] = budget
                labels = self.settle(budget, receipt)
            self.prune_exports()
            milestones = {label: labels.get(label, False) for label in self.milestones}
            receipt["labels_verified"] = labels  # Rolling labels are added at the end.
            unverified = [self.export_listing(label) for label, ok in milestones.items() if not ok]
            durable = bool(milestones) and not unverified
            if unverified and not hurry:
                self.report.update(
                    status="failed", failure="r2_milestones_unverified", training_status=trained
                )
            self.report["r2"].update(
                uploader=self.uploader.status(),
                milestones_verified=milestones,
                unverified_milestones=unverified,
                verification_errors=dict(self.verification_errors),
                report_key=report_key,
            )
            report_sha256 = self.publish()
            if hurry:
                return
            path = self.root / "report.json"
            size = path.stat().st_size
            self._checked(
                lambda store: store.put_known(report_key, path, report_sha256, size),
                "upload report.json",
            )
            # The PUT succeeded: its verifying HEAD may still start until the final writes.
            if self._stored(report_key, late=True) != report_sha256:
                raise RuntimeError("report.json differs in R2 after upload")
            receipt["report_sha256"] = report_sha256
            if durable:
                # The last milestone, a stage-final label, is what the supervisor verifies;
                # a rolling label never closes a run.
                final = self.milestones[-1]
                files = {name: digest for name, _, digest in self.label_files[final]}
                self.r2_final = {
                    "report_key": report_key,
                    "report_sha256": report_sha256,
                    "checkpoint_key": f"{self.prefix}{final}/checkpoint.pt",
                    "checkpoint_sha256": files["checkpoint.pt"],
                }
            # Rolling labels last, for the record only: no ending depends on them.
            self.record_rolling(labels, receipt)
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            if unverified is None:
                # Failed before the check: every milestone the uploader has not verified
                # may exist only on this Pod.
                kept = self.kept_labels()
                unverified = [
                    self.export_listing(label) for label in self.milestones if label not in kept
                ]
                self.report["r2"]["unverified_milestones"] = unverified
                if unverified and not hurry:
                    self.report.update(
                        status="failed",
                        failure="r2_milestones_unverified",
                        training_status=trained,
                    )
            if self.report["status"] in {"completed", "time_limited"}:
                self.report.update(
                    status="failed", failure="r2_finalization", training_status=trained
                )
            self.publish()
            if not hurry and "labels_unchecked" not in receipt:
                # The report upload (or a step before it) failed, not the rolling labels:
                # check them while time lasts, or name them unchecked, so the owner can
                # still find one to resume from. The checks end before the final writes.
                self.record_rolling(labels, receipt)
        finally:
            receipt["unverified_milestones"] = unverified or []
            receipt["verification_errors"] = dict(self.verification_errors)
            receipt["uploader_closed"] = self.uploader.close(timeout=5)
            receipt["r2_final"] = self.r2_final
            receipt["status"] = self.report["status"]
            if "failure" in self.report:
                receipt["failure"] = self.report["failure"]
            self.write_json(self.root / "finalization.json", receipt)
            self.uploading = False
            self.heartbeat()


# Smoke: tiny authored shards and an in-memory bucket -----------------------------------

SMOKE_CREDENTIALS = Credentials(
    "https://smoke.r2.cloudflarestorage.com", "smoke-key-id", "smoke-secret-key", "smoke-bucket"
)
SMOKE_STREAMS = {
    "stage1-web": (24, 1),
    "stage2-web": (6, 1),
    "stage2-python": (18, 2),
    "validation-web": (6, 1),
    "validation-python": (6, 1),
}
"""Each authored stream's document and shard counts."""


class _Reply:
    """A GET answer in the shape ``Store.download``'s opener returns."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status, self._body, self._offset = status, body, 0

    def read(self, size: int = -1) -> bytes:
        end = len(self._body) if size < 0 else self._offset + size
        block = self._body[self._offset : end]
        self._offset += len(block)
        return block

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class MemoryBucket:
    """One R2 bucket held in memory and served through ``Store``'s transport, for smokes.

    It answers like R2 where the Mac side depends on it: HEAD carries ``content-length``
    and the SHA-256 metadata, and GET returns an object's bytes (``Store.download``).
    ``fault`` may answer a request instead of the bucket: given (method, key) it returns
    (status, body) or None.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.digests: dict[str, str] = {}
        self.fault: Callable[[str, str], tuple[int, bytes] | None] | None = None
        self.lock = threading.Lock()

    def __call__(self, request: Request) -> tuple[int, Mapping[str, str], bytes]:
        method = request.get_method()
        path = urlsplit(request.full_url).path
        key = unquote(path.split("/", 2)[2]) if path.count("/") >= 2 else ""
        headers = {name.lower(): value for name, value in request.header_items()}
        if "authorization" not in headers:
            return 403, {}, b"unsigned"
        answer = self.fault(method, key) if self.fault is not None else None
        if answer is not None:
            return answer[0], {}, answer[1]
        with self.lock:
            if method == "GET" and not key:
                contents = "".join(
                    f"<Contents><Key>{escape(name)}</Key><Size>{len(data)}</Size></Contents>"
                    for name, data in sorted(self.objects.items())
                )
                body = (
                    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    f"{contents}<IsTruncated>false</IsTruncated></ListBucketResult>"
                )
                return 200, {}, body.encode()
            if method == "HEAD":
                if key not in self.objects:
                    return 404, {}, b""
                return (
                    200,
                    {
                        "x-amz-meta-sha256": self.digests[key],
                        "content-length": str(len(self.objects[key])),
                    },
                    b"",
                )
            if method == "GET":
                if key not in self.objects:
                    return 404, {}, b"NoSuchKey"
                return 200, {"content-length": str(len(self.objects[key]))}, self.objects[key]
            if method == "DELETE":
                self.objects.pop(key, None)
                self.digests.pop(key, None)
                return 204, {}, b""
            if method == "PUT":
                payload: Any = request.data
                data = payload.read() if hasattr(payload, "read") else bytes(payload or b"")
                if hashlib.sha256(data).hexdigest() != headers.get("x-amz-content-sha256"):
                    return 400, {}, b"XAmzContentSHA256Mismatch"
                self.objects[key] = data
                self.digests[key] = headers.get("x-amz-meta-sha256", "")
                return 200, {}, b""
        return 405, {}, b"unsupported"

    def open(self, request: Request) -> _Reply:
        """``Store.download``'s opener: the same answers, nothing leaves the process."""
        status, _, body = self(request)
        return _Reply(status, body)

    def store(self, credentials: Credentials = SMOKE_CREDENTIALS) -> Store:
        return Store(credentials, transport=self, opener=self.open, sleep=lambda _: None)


def author_streams(root: Path, tokenizer: ByteBPE) -> dict[str, list[tuple[str, TokenShard]]]:
    """Deterministic tiny shards for every bound stream, at shards/<stream>/<n>.bin."""
    streams: dict[str, list[tuple[str, TokenShard]]] = {}
    for name, (documents, count) in SMOKE_STREAMS.items():
        rng = random.Random(name)
        rows = [
            [rng.randrange(1, 256) for _ in range(rng.randrange(4, 17))] for _ in range(documents)
        ]
        shards = []
        for part in range(count):
            path = root / "shards" / name / f"{name}-{part:05d}.bin"
            write_shard(
                path,
                rows[part::count],
                vocab_size=tokenizer.vocab_size,
                tokenizer_hash=tokenizer.sha256,
                bos_id=tokenizer.bos_id,
            )
            shard = TokenShard(
                path,
                tokenizer_hash=tokenizer.sha256,
                vocab_size=tokenizer.vocab_size,
                bos_id=tokenizer.bos_id,
            )
            shards.append((file_sha256(path), shard))
        streams[name] = shards
    return streams


SMOKE_TRAIN: dict[str, Any] = {
    "seed": 0,
    "batch_size": 2,
    "sequence_length": 8,
    "accumulation_steps": 2,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "precision": "float32",
    "attention": "fused",
    "optimizer_foreach": True,
    "z_loss": 0.0,
    "schedule": "cosine",
    "compile": False,
}
SMOKE_LEARNING_RATES = {"stage1": 0.0006, "stage2": 0.0003, "mixed": 0.0006}


def _stage_tokens(streams: Mapping[str, list[tuple[str, TokenShard]]], names: Iterable[str]) -> int:
    return sum(
        length for name in names for _, shard in streams[name] for _, length in shard.offsets
    )


def smoke_specification(
    run: str, streams: Mapping[str, list[tuple[str, TokenShard]]]
) -> Specification:
    """A tiny CPU specification of ``run`` whose stages fit the authored streams."""
    train = SMOKE_TRAIN
    rows = train["batch_size"] * train["accumulation_steps"]
    stages = []
    for name, names in RUNS[run]:
        windows = -(-_stage_tokens(streams, names) // train["sequence_length"])
        updates = -(-windows // rows)
        stages.append(
            {
                "name": name,
                "streams": list(names),
                "learning_rate": SMOKE_LEARNING_RATES[name],
                "updates": updates,
                "warmup_updates": round(0.2 * updates),
            }
        )
    return Specification.from_dict(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": SPECIFICATION_KIND,
            "run": run,
            "model": {
                "vocab_size": 272,
                "width": 16,
                "layers": 1,
                "heads": 2,
                "context": 8,
                "qk_norm": False,
            },
            "train": dict(train),
            "warmup_fraction": 0.2,
            "stages": stages,
            "validation": {
                "interval_updates": 4,
                "subset_target_tokens": 20,
                "batch_size": 4,
                "autocast_bf16": False,
                "streams": {"web": "validation-web", "python": "validation-python"},
            },
            "checkpoint_seconds": 0.05,
            "checkpoint_save_seconds": 1.0,
            "expected_tokens_per_second": 1.0,
            "validation_tokens_per_second": 1.0,
            "speed_check_warmup_updates": 2,
            "quoted_gpu_hourly": 0.59,
            "limits": {
                "pod_seconds": 3600,
                "startup_seconds": 60,
                "finalization_reserve_seconds": 120,
                "quoted_total_hourly": 0.6,
                "estimated_total_cost": 1.0,
                "approved_cost_cap": 10.0,
                "cost_multiplier": 1.2,
                "minimum_speed_fraction": 0.85,
                "speed_check_seconds": 0.01,
                "idle_seconds": 600,
                "rising_steps": 25,
                "starting_cost_usd": 0.0,
            },
            "r2": {
                "keep_rolling": 2,
                "storage_cap_usd_per_month": 5.0,
                "rolling_skip_seconds": 0.0,  # Tiny stages: every cadence save happens.
                "max_transfer_seconds": 900,
            },
        }
    )


def authored_run_data(
    spec: Specification, streams: Mapping[str, list[tuple[str, TokenShard]]], tokenizer: ByteBPE
) -> RunData:
    """``RunData`` for authored streams, each ticket computed from the shards themselves."""
    train = spec.train
    rows = train.batch_size * train.accumulation_steps
    stages = []
    for plan in spec.stages:
        shards = tuple((name, shard, sha) for name in plan.streams for sha, shard in streams[name])
        order = pass_order([(sha, len(shard)) for _, shard, sha in shards], ORDER_SEED)
        tokens = _stage_tokens(streams, plan.streams)
        windows = -(-tokens // train.sequence_length)
        multiset = multiset_digest(
            document_sha256(np.asarray(shard[index], dtype=np.uint16))
            for _, shard, _ in shards
            for index in range(len(shard))
        )
        expected = {
            "documents": sum(len(shard) for _, shard, _ in shards),
            "tokens": tokens,
            "targets": tokens,
            "windows": windows,
            "planned_updates": -(-windows // rows),
            "multiset_sha256": multiset,
            "order_sha256": order_digest(order),
        }
        stages.append(StageData(plan, shards, order, expected))
    identity = {name: [sha for sha, _ in shards] for name, shards in sorted(streams.items())}
    return RunData(
        tuple(stages),
        tuple((label, streams[stream][0][1]) for label, stream in spec.validation.streams),
        tokenizer,
        hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        combine_digests(*(stage.expected["multiset_sha256"] for stage in stages)),
        "authored-tiny-CPU-only",
    )


def smoke(output: Path, *, bucket: MemoryBucket | None = None) -> dict[str, Any]:
    """Runs A and B from one shared initialization on authored shards, on CPU, offline.

    Both runs write to one in-memory ``bucket`` (a new one unless given), as the two Pods
    share one R2 bucket, under their own ``rehearsal-runs/<run>/smoke-<run>/`` prefixes.
    """
    torch.set_num_threads(1)
    if output.exists() or output.is_symlink():
        raise FileExistsError("smoke output must be a new directory")
    tokenizer = ByteBPE()
    streams = author_streams(output / "data", tokenizer)
    specs = {run: smoke_specification(run, streams) for run in RUNS}
    initialization = output / "initialization.pt"
    initial_sha256 = write_initialization(list(specs.values()), tokenizer.sha256, initialization)
    bucket = MemoryBucket() if bucket is None else bucket
    reports = {}
    for run, spec in specs.items():
        reports[run] = Worker(
            spec,
            output / f"run-{run}",
            authored_run_data(spec, streams, tokenizer),
            initialization=initialization,
            initialization_sha256=initial_sha256,
            store=bucket.store(),
            pod_id=f"smoke-{run.lower()}",
            pod_created=time.time(),
        ).execute()
    tickets = {run: report["run_ticket"] for run, report in reports.items()}
    return {
        "initialization_sha256": initial_sha256,
        "status": {run: report["status"] for run, report in reports.items()},
        "multiset_sha256": {run: ticket["multiset_sha256"] for run, ticket in tickets.items()},
        "same_multiset": len({ticket["multiset_sha256"] for ticket in tickets.values()}) == 1,
        "tickets_match": all(ticket["match"] for ticket in tickets.values()),
        "bucket_objects": len(bucket.objects),
    }


# Pod run -------------------------------------------------------------------------------


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _revision(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the shipped code revision {path}: {error}") from error


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Every launch gate before any training, then one run on ``cuda:0``.

    The shipped ``--revision`` must equal the receipt's ``revision``; the report records
    it as ``code_revision``.
    """
    spec = Specification.load(args.specification)
    if spec.train.precision != "bfloat16":
        raise ValueError("the A40 rehearsal specification must use bfloat16 autocast")
    receipt = _object(args.readiness)
    identity = {
        "specification_sha256": file_sha256(args.specification),
        "binding_sha256": args.binding_sha256,
        "preflight_sha256": file_sha256(args.preflight),
        "initialization_sha256": args.initialization_sha256,
    }
    check_receipt_identity(receipt, run=spec.run, **identity)  # Before reading anything large.
    binding = load_binding(args.binding, expected_sha256=args.binding_sha256)
    launch = check_readiness(receipt, now=time.time(), spec=spec, binding=binding, **identity)
    spec = launch.spec
    hardware = hardware_description(training_device("cuda:0"))
    check_preflight(_object(args.preflight), hardware=hardware, spec=spec)
    # The worker's SSH launch does not source /etc/rp_environment, so RUNPOD_POD_ID is
    # usually absent: the Pod ID is the receipt's captured identity, cross-checked if set.
    pod_id = launch.pod_id
    if os.environ.get("RUNPOD_POD_ID") not in (None, "", pod_id):
        raise ValueError("this Pod is not the one the receipt identifies")
    credentials = r2_credentials(os.environ, key_prefix="R2_POD_")
    revision = check_revision(receipt, _revision(args.revision))
    data = load_run_data(args.binding, args.root, expected_sha256=args.binding_sha256, spec=spec)
    verify_initialization(
        args.initialization,
        expected_sha256=args.initialization_sha256,
        model=spec.model,
        tokenizer_identity=data.tokenizer.sha256,
    )
    worker = Worker(
        spec,
        args.output,
        data,
        initialization=args.initialization,
        initialization_sha256=args.initialization_sha256,
        store=Store(credentials),
        check_store_for=lambda timeout: bounded_store(credentials, timeout=timeout, attempts=1),
        pod_id=pod_id,
        pod_created=launch.origin,
        device="cuda:0",
        secrets=[
            credentials.access_key_id,
            credentials.secret_access_key,
            *environment_secrets(),
        ],
        extra={
            "code_revision": revision,
            "hardware": hardware,
            "readiness_receipt": sanitize(receipt),
            "specification_sha256": identity["specification_sha256"],
            "preflight_sha256": identity["preflight_sha256"],
            "pod_id_source": "readiness receipt pod_identity",
            "launch_fit": launch.fit,
            "pod_lifetime_end_unix": launch.end,
        },
    )
    return worker.execute()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="train one rehearsal run on this Pod")
    for name in ("specification", "binding", "root", "initialization", "readiness", "preflight"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--binding-sha256", required=True)
    run_parser.add_argument("--initialization-sha256", required=True)
    run_parser.add_argument(
        "--revision",
        type=Path,
        default=DEFAULT_REVISION,
        help="the Mac's revision.json shipped with the code; must equal the receipt's",
    )
    smoke_parser = commands.add_parser("smoke", help="both runs on tiny authored data, CPU")
    smoke_parser.add_argument("--output", type=Path, required=True)
    initial = commands.add_parser("write-initialization", help="the shared initialization (Mac)")
    initial.add_argument(
        "--specification",
        type=Path,
        action="append",
        required=True,
        help="give it once per run (A and B): both must share the model and seed",
    )
    initial.add_argument("--binding", type=Path, required=True)
    initial.add_argument("--binding-sha256", required=True)
    initial.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(smoke(args.output), indent=1, sort_keys=True))
    elif args.command == "write-initialization":
        specs = [Specification.load(path) for path in args.specification]
        if sorted(spec.run for spec in specs) != sorted(RUNS):
            parser.error(f"--specification must name one specification per run {sorted(RUNS)}")
        binding = load_binding(args.binding, expected_sha256=args.binding_sha256)
        identity = binding["tokenizer"]["identity"]
        digest = write_initialization(specs, identity, args.output)
        print(
            json.dumps(
                {
                    "initialization": str(args.output),
                    "sha256": digest,
                    "tokenizer_identity": identity,
                    "parameters": expected_parameters(specs[0].model),
                    "seed": specs[0].train.seed,
                },
                indent=1,
            )
        )
    else:
        report = run(args)
        print(f"Rehearsal run {report['run']} {report['status']}: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
