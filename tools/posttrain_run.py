"""Post-training worker for rehearsal runs 1 and 2 on one existing Pod; never provisions.

``run`` (Pod)
    Check the readiness receipt that ``python -m tools.pilot_readiness_receipt posttrain``
    wrote, the shipped code revision, the A40 preflight and the input binding, then run the
    specification's stages in order, uploading every artifact to R2 under
    ``posttrain-runs/<run>/<pod id>/`` as soon as it exists:

    1. ``probe``: generation speed at batch 1, 64, 128 and 256 on real prompt lengths, top-p
       on, bf16. Below the planning speeds the run stops here (``generation_below_planning_
       speed``) with the measured rates in ``report.json``.
    2. ``arc-base``: ARC-Easy label log-likelihood on run A's two stage finals and run B.
    3. ``eval-base``: base-B completions for every suite and split (the base protocols).
    4. ``sft``: one pass over the SFT rows; dev loss at each quarter, the lowest kept as
       weights only; a speed check after 50 updates.
    5. ``arc-sft``: ARC-Easy on the selected SFT weights.
    6. ``eval-sft``: SFT completions for every suite and split (chat-v1).
    7. ``screen``: RL screening attempts from the SFT weights, for ``rl-screen`` on the Mac.

    Then the pretraining rehearsal's finalization: drain, HEAD checks, ``report.json`` and
    ``r2_final``.

    Run 1b (the SFT re-run) runs ``probe``, ``sft`` (two passes, the special-token rows
    redrawn and trained at 10x), ``format-gate`` (the SFT format gate on the selected
    weights, on the GPU: below its bars the run stops with ``format_gate_failed`` after the
    gate record and the weights are in R2, and nothing paid follows), ``arc-sft``,
    ``eval-sft`` and ``screen``; run 1's base results stand, so no base stage runs.

    Run 2 (RL with the Mac in the loop, the more-SFT control and their evaluation) runs:

    1. ``rl``: ``run_rl`` from run 1's selected SFT weights on the withheld kept tasks, its
       rewards and dev verdicts from the Mac through the reward bridge
       (``BridgeRewardSource``, ``BridgeDevEvaluator``), each dev check's weights uploaded
       at once. A reward-wait timeout, a host failure, a lost bridge or a margin stop ends
       RL's training, not the run: the stages that need no Mac wait follow
       (``tools/posttrain_run2.py``, ``ARM_STOP_RULE``); collapse, patience, leakage or
       completion is a result.
    2. ``more-sft``: ``run_more_sft`` from the same SFT weights on the kept tasks' references,
       its GPU-busy budget RL's measured busy seconds, its dev checks through the bridge.
    3. ``arc-post``: ARC-Easy on the RL-selected and the control-selected checkpoints.
    4. ``eval-rl``, ``eval-more-sft``: the seven completions files of each (chat-v1).

    An arm whose selected weights are the SFT weights, bit for bit, is not evaluated again:
    run 1's SFT results, pinned in ``sft_results``, stand for it (``SAME_WEIGHTS_RULE``).

    While the bridge client waits the heartbeat reads ``awaiting_rewards``, and a Mac lease
    renewed on the Pod counts as progress for the in-Pod idle rule (``RUN2_LIVENESS``).
``smoke`` (CPU)
    Run 1 (or, with ``--run 1b``, run 1b with its format gate; with ``--run 2``, run 2 with a
    Mac bridge server thread) end to end on a tiny model, fixture tasks and an in-memory
    bucket.

Every stage start checks the wall, cost and idle limits and applies the projected-finish
rule (``PROJECTED_FINISH_RULE``); during a stage the worker stops at the finalization reserve
(``RESERVE_STOP``), ``finalization_reserve_seconds`` before the Pod lifetime ends, so the
drain, ``report.json`` and the handover always have their time. The stage machinery is a
registry of kinds (``STAGE_KINDS``): run 2 adds ``rl``, ``more-sft`` and their evaluation,
and its reward waits report ``awaiting_rewards``. Hidden tests never reach the Pod: every pool
task file it reads is a Pod export (``scratchcoder.eval.export``). R2 credentials come only
from the environment; every write goes through the R2 prefix guard
(``posttrain_ops.GuardedStore``). See ``docs/walkthroughs/posttrain-run.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import batched
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig, finite_number
from scratchcoder.common.logging import sanitize
from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data import chat
from scratchcoder.data.sft import SFTLoader, SFTRow, read_rows
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.benchmark import benchmark_generation
from scratchcoder.engine.generation import Completion, generate_batch
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.choice_command import checkpoint_choice_scorer
from scratchcoder.eval.choices import evaluate_choices, load_choices
from scratchcoder.eval.completions import Sampling
from scratchcoder.eval.format_gate import BENCHMARK_SMOKE_SUITES, gate_record
from scratchcoder.eval.format_gate import summary as gate_summary
from scratchcoder.eval.generate import generate_completions
from scratchcoder.eval.protocols import (
    BENCHMARK_SPLIT,
    BENCHMARK_SUITES,
    CHAT,
    POOL,
    POOL_SPLITS,
    load_suite_tasks,
    protocol_for,
)
from scratchcoder.eval.tasks import load_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.attempts import ScreeningHeader, write_attempts
from scratchcoder.rl.checkpoint import save_weights
from scratchcoder.rl.control import ControlConfig
from scratchcoder.rl.loop import RLConfig
from scratchcoder.rl.rollout import DEV_TOKENS, screening_rollouts
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder
from scratchcoder.train.pilot import PilotStop
from scratchcoder.train.trainer import Trainer, synchronized_time, training_device
from scratchcoder.train.validation import assistant_loss, token_weighted_loss
from tools import posttrain_binding, posttrain_ops
from tools.gpu_pilot import check_termination_guarantee
from tools.pilot_preflight import hardware_description
from tools.r2_store import MILESTONE_RETRY_SECONDS, CheckpointUploader, Store, bounded_store
from tools.rehearsal_binding import r2_credentials
from tools.rehearsal_run import (
    HEARTBEAT_REFRESH_SECONDS,
    MAX_HEARTBEAT_BYTES,
    RETRY_GROWTH,
    Worker,
    check_revision,
    environment_secrets,
    lifetime_end,
    quarter_updates,
    require_fit,
)
from tools.runpod_watchdog import CLOCK_SKEW_SECONDS, PodIdentity, Policy, parse_timestamp

SCHEMA_VERSION = 1
SPECIFICATION_KIND = "posttrain-run-specification"
REPORT_KIND = posttrain_ops.REPORT_KIND
DEFAULT_REVISION = Path("/workspace/posttrain/revision.json")
STAGE_KINDS = ("probe", "arc", "eval", "sft", "screen", "format-gate", "rl", "more-sft")
"""The stage kinds, each run by ``PosttrainWorker.run_<kind>`` (``-`` read as ``_``): run 1's
first five, run 1b's ``format-gate``, and run 2's ``rl`` and ``more-sft`` (its ARC and
evaluation stages are ``arc`` and ``eval`` stages of the selected checkpoints)."""
CHECKPOINT_KINDS = frozenset({"checkpoint"})
SFT_REFERENCE = "sft"
"""A stage's checkpoint reference to the weights the ``sft`` stage selected."""
RL_REFERENCE = "rl"
CONTROL_REFERENCE = "more-sft"
PRODUCED_REFERENCES = {"sft": SFT_REFERENCE, "rl": RL_REFERENCE, "more-sft": CONTROL_REFERENCE}
"""The checkpoint reference each producing stage kind selects: a later stage names it in
place of an input (the SFT stage's lowest dev loss; RL's and the control's selected check)."""
PROBE_STOP = "generation_below_planning_speed"
SPEED_STOP = "sft_throughput_below_limit"
PROJECTED_STOP = "projected_finish_after_deadline"
FORMAT_GATE_STOP = "format_gate_failed"
"""Run 1b: the selected SFT weights failed the SFT format gate. The gate record and the
weights are already in R2; the run finalizes in full and no later stage (nothing paid)
starts from those weights."""
FORMAT_GATE_RULE = (
    "the format gate runs on the dev-loss-selected SFT checkpoint only, after the sft stage "
    "and before any stage that reads the SFT weights; it judges the first assistant token, "
    "the block's close, the separation of the three output specials (largest cosine "
    "strictly below 0.9) and a greedy smoke of pool dev prompts, at "
    "scratchcoder.eval.format_gate's thresholds; the benchmark smoke is recorded, not judged"
)
WAIT_MARGIN_STOP = "wait_margin"
"""Run 2: the next bridge wait would pass the cost stop, the wall or the finalization reserve.
``should_stop`` is not read during a wait, so the worker stops before one while the time and
money for one full wait (``BridgeTimeouts.wait_margin_seconds``) are not left; unlike the wall
and cost stops it finalizes in full."""
BRIDGE_LOST_STOP = "bridge_lost"
"""Run 2: a wait whose Mac lease stayed stale for ``lost_bridge_seconds`` was given up, before
the in-Pod idle rule deletes the Pod, so the latest weights and the report still reach R2."""
RESERVE_STOP = "finalization_reserve"
"""A stage still running when only the finalization reserve is left stops here. Unlike the
wall and cost stops it finalizes in full: the drain, the HEAD checks, ``report.json`` and the
handover all fit the reserve, which is the watchdog's transfer window."""
LAUNCH_RULE = (
    "at launch the Pod lifetime left, to the first of the wall, the armed in-Pod deadline, the "
    "watchdog's wall and either cost stop, must hold every stage at the planning rates "
    "(generation at the planning speed of the generation batch, pool attempts priced at their "
    "token cap, SFT at the reference slot rate, ARC and checkpoint loads at their planning "
    "seconds) plus the finalization reserve"
)
PROJECTED_FINISH_RULE = (
    "at the start of every stage the run stops before starting it if now plus the work "
    "remaining (this and every later stage, generation at the probe's measured rate at the "
    "generation batch size, or the evaluation files' observed rate once lower, with pool "
    "attempts priced at their token cap, SFT at its measured slot rate once measured, ARC at "
    "its measured seconds per checkpoint once measured) plus the finalization reserve passes "
    "the end of the Pod lifetime; a stage still running at the reserve stops there"
)
SECRET_VARIABLES = (
    "R2_POD_ACCESS_KEY_ID",
    "R2_POD_SECRET_ACCESS_KEY",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "RUNPOD_API_KEY",
)
TOP_KEYS = {
    "schema_version",
    "kind",
    "run",
    "model",
    "limits",
    "planning",
    "probe",
    "generation",
    "sft",
    "screening",
    "stages",
    "inputs",
    "r2",
}
PLANNING_KEYS = {
    "generation_tokens_per_second",
    "training_slots_per_second",
    "forward_tokens_per_second",
    "probe_seconds",
    "arc_seconds_per_checkpoint",
    "checkpoint_load_seconds",
    "compile_seconds",
    "weights_save_seconds",
}
GATE_PLANNING_KEY = "format_gate_seconds"
"""Run 1b's planning: the format gate's seconds on the GPU, load excluded (an estimate)."""
PROBE_KEYS = {
    "batch_sizes",
    "prompt_lengths",
    "max_new_tokens",
    "temperature",
    "top_p",
    "precision",
    "seed",
    "repeats",
    "minimum_tokens_per_second",
}
GENERATION_KEYS = {"batch_size", "precision"}
SFT_KEYS = {
    "train_rows",
    "planned_dev_tokens",
    "learning_rate",
    "warmup_fraction",
    "min_lr_ratio",
    "schedule",
    "beta1",
    "beta2",
    "adam_epsilon",
    "weight_decay",
    "gradient_clip",
    "batch_size",
    "accumulation_steps",
    "sequence_length",
    "seed",
    "passes",
    "precision",
    "attention",
    "optimizer_foreach",
    "compile",
    "dev_batch_size",
    "speed_check_updates",
    "speed_check_warmup_updates",
    "reference_slots_per_second",
    "minimum_speed_fraction",
}
SFT_SPECIAL_KEYS = {"special_rows_init", "special_rows_seed", "special_rows_lr_scale"}
"""Run 1b's SFT adds these (``TrainConfig.special_rows_*``); run 1's has none and trains the
embedding as it always did."""
SCREENING_KEYS = {
    "samples",
    "temperature",
    "top_p",
    "max_new_tokens",
    "seed_base",
    "planned_tasks",
    "chunk_tasks",
}
EVAL_FILE_KEYS = {
    "name",
    "suite",
    "split",
    "tasks",
    "protocol",
    "samples",
    "temperature",
    "top_p",
    "seed",
    "planned_tasks",
}
INPUT_PIN_KEYS = {"kind", "key", "size_bytes", "sha256"}
STAGE_KEYS = {
    "probe": {"name", "kind", "checkpoint", "prompts"},
    "arc": {"name", "kind", "checkpoints", "questions"},
    "eval": {"name", "kind", "checkpoint", "files"},
    "sft": {"name", "kind", "parent", "train", "dev"},
    "screen": {"name", "kind", "checkpoint", "tasks", "candidates"},
    "format-gate": {
        "name",
        "kind",
        "checkpoint",
        "rows",
        "tasks",
        "benchmarks",
        "row_count",
        "smoke_prompts",
        "benchmark_prompts",
    },
}
RUN2_TOP_KEYS = {
    "schema_version",
    "kind",
    "run",
    "model",
    "limits",
    "planning",
    "generation",
    "rl",
    "control",
    "bridge",
    "stages",
    "inputs",
    "sft_results",
    "sft_gate",
    "r2",
}
RUN2_PLANNING_KEYS = {
    "generation_tokens_per_second",
    "training_slots_per_second",
    "arc_seconds_per_checkpoint",
    "checkpoint_load_seconds",
    "weights_save_seconds",
    "reward_wait_seconds_per_update",
    "dev_wait_seconds_per_check",
}
"""Run 2's planning rates: generation by batch size (the evaluation batch, RL's rollout batch
and its dev-check batch), training slots, ARC, loads and saves, and the Mac's time: one
update's reward wait and one dev check's wait."""
RL_CONFIG_KEYS = tuple(name for name in RLConfig.__dataclass_fields__)
RL_KEYS = {*RL_CONFIG_KEYS, "attention", "dev_tasks", "dev_samples", "dev_batch_size", "min_tasks"}
"""The RL protocol: every ``RLConfig`` setting, the trained model's attention kernel, the dev
subset (its size, samples and generation batch) and the fewest kept tasks the run accepts."""
CONTROL_CONFIG_KEYS = tuple(
    name
    for name in ControlConfig.__dataclass_fields__
    if name not in {"attention", "sequence_length"}
)
CONTROL_KEYS = set(CONTROL_CONFIG_KEYS)
"""The control's ``ControlConfig`` settings; its attention kernel is RL's (``RLResult.attention``)
and its rows use the model's whole context."""
BRIDGE_STAGE_KEYS = {"name", "kind", "parent", "tasks", "dev", "facts"}
RUN2_STAGE_KEYS = {
    "rl": BRIDGE_STAGE_KEYS,
    "more-sft": BRIDGE_STAGE_KEYS,
    "arc": STAGE_KEYS["arc"],
    "eval": STAGE_KEYS["eval"],
}
RUN2_GATE_FIELDS: tuple[tuple[str, ...], ...] = (
    ("limits", "ceiling_usd"),
    ("limits", "gate_estimate_usd"),
    ("limits", "cost_limit_usd"),
    ("inputs", "sft-weights", "key"),
    ("inputs", "sft-weights", "size_bytes"),
    ("inputs", "sft-weights", "sha256"),
    ("inputs", "rl-tasks", "size_bytes"),
    ("inputs", "rl-tasks", "sha256"),
    ("inputs", "rl-dev", "size_bytes"),
    ("inputs", "rl-dev", "sha256"),
    ("inputs", "rl-facts", "size_bytes"),
    ("inputs", "rl-facts", "sha256"),
    ("inputs", "pool-test", "size_bytes"),
    ("inputs", "pool-test", "sha256"),
)
"""Run 2's values fixed at the owner's gate, ``null`` in the tracked specification: the ceiling
(re-priced from run 1's measured rates), the gate's estimate (the run 2 projection at run 1's
measured rates) and the cost limit it derives (the lower of the ceiling and 120% of the
estimate, ``posttrain_ops.estimate_limit_usd``), run 1's selected SFT weights (key, size,
SHA-256), the three ``rl-pod-export`` files and the pool test export, and every pin of
``sft_results`` (``SFT_RESULTS``). The worker and the readiness receipt refuse a
specification in which any is still ``null``."""
SFT_RESULT_FILES = (
    "pool-dev-greedy",
    "pool-dev-n8",
    "pool-test-greedy",
    "pool-test-n8",
    "humaneval-greedy",
    "humaneval-plus-greedy",
    "mbpp-plus-greedy",
)
"""The seven completions files of every chat-v1 evaluation stage, run 1's ``eval-sft`` and run
2's ``eval-rl`` and ``eval-more-sft`` alike."""
SFT_ARC = "arc-sft-sft/arc.json"
SFT_RESULTS = (SFT_ARC, *(f"eval-sft-{name}/completions.jsonl" for name in SFT_RESULT_FILES))
"""Run 2's ``sft_results``: run 1's artifacts of the SFT weights' evaluation (its ARC report and
its seven completions files, ``<label>/<name>`` in run 1's prefix), each pinned at the gate by
R2 key, size and SHA-256 like an input. An arm whose selected weights are the SFT weights
reuses them instead of a duplicate evaluation (the owner's run 2 gate rulings); the pins make
that reuse verifiable. They never reach the Pod."""
SFT_GATE_ARTIFACT = "format-gate/gate.json"
SFT_GATE_KEYS = ("key", "size_bytes", "sha256", "checkpoint_sha256", "passed")
"""Run 2's ``sft_gate``: the passing format-gate record of its SFT weights, the SFT run's
``format-gate`` artifact beside them, pinned at the gate by R2 key, size and SHA-256, with
the weights' SHA-256 it measured and its verdict. Run 2 refuses SFT weights without one."""
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,47}\Z")


def gate_nulls(raw: Mapping[str, Any]) -> list[str]:
    """The run-2 gate fields (``RUN2_GATE_FIELDS`` and every ``sft_results`` pin) still
    ``null`` in ``raw``, dotted."""
    missing = []
    fields = [
        *RUN2_GATE_FIELDS,
        *(("sft_results", name, key) for name in SFT_RESULTS for key in sorted(INPUT_PIN_KEYS)),
        *(("sft_gate", key) for key in SFT_GATE_KEYS),
    ]
    for path in fields:
        if path[-1] == "kind":
            continue
        value: Any = raw
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if value is None:
            missing.append(".".join(path))
    return missing


def check_sft_results(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Run 2's ``sft_results``: exactly ``SFT_RESULTS``, each pin ``key``, ``size_bytes`` and
    ``sha256`` (``null`` before the gate); once filled, each key the artifact's own under run 1's
    prefix, the prefix of the ``sft-weights`` input."""
    pins = raw.get("sft_results")
    if not isinstance(pins, Mapping) or set(pins) != set(SFT_RESULTS):
        raise ValueError(f"sft_results must pin exactly {list(SFT_RESULTS)}")
    weights = ((raw.get("inputs") or {}).get("sft-weights") or {}).get("key")
    prefix = None
    if isinstance(weights, str) and weights.endswith("/sft-weights/checkpoint.pt"):
        prefix = weights.removesuffix("sft-weights/checkpoint.pt")
    checked = {}
    for name in SFT_RESULTS:
        pin = pins[name]
        if not isinstance(pin, Mapping) or set(pin) != {"key", "size_bytes", "sha256"}:
            raise ValueError(f"sft_results {name} gives exactly key, size_bytes and sha256")
        if all(pin[field] is None for field in pin):
            checked[name] = dict(pin)
            continue
        if prefix is None or pin["key"] != prefix + name:
            raise ValueError(f"sft_results {name} is not run 1's artifact beside its SFT weights")
        if type(pin["size_bytes"]) is not int or pin["size_bytes"] < 1:
            raise ValueError(f"sft_results {name} needs a positive size")
        digest = pin["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or digest.strip("0123456789abcdef"):
            raise ValueError(f"sft_results {name} needs a SHA-256")
        checked[name] = dict(pin)
    return checked


def check_sft_gate(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Run 2's ``sft_gate`` (``SFT_GATE_KEYS``, every value ``null`` before the gate); once
    filled, the SFT run's ``format-gate/gate.json`` beside the ``sft-weights`` input, with a
    size and SHA-256, the weights' own SHA-256 as the checkpoint it measured, and passed."""
    pin = raw.get("sft_gate")
    if not isinstance(pin, Mapping) or set(pin) != set(SFT_GATE_KEYS):
        raise ValueError(f"sft_gate must give exactly {list(SFT_GATE_KEYS)}")
    if all(pin[key] is None for key in SFT_GATE_KEYS):
        return dict(pin)
    weights = (raw.get("inputs") or {}).get("sft-weights") or {}
    key = weights.get("key")
    if not isinstance(key, str) or not key.endswith("/sft-weights/checkpoint.pt"):
        raise ValueError("sft_gate needs the sft-weights input's R2 key")
    if pin["key"] != key.removesuffix("sft-weights/checkpoint.pt") + SFT_GATE_ARTIFACT:
        raise ValueError("sft_gate is not the format-gate record beside the SFT weights")
    if type(pin["size_bytes"]) is not int or pin["size_bytes"] < 1:
        raise ValueError("sft_gate needs a positive size")
    for name in ("sha256", "checkpoint_sha256"):
        digest = pin[name]
        if not isinstance(digest, str) or len(digest) != 64 or digest.strip("0123456789abcdef"):
            raise ValueError(f"sft_gate {name} must be 64 lowercase hex digits")
    if pin["checkpoint_sha256"] != weights.get("sha256"):
        raise ValueError("the format gate measured other weights than run 2's sft-weights")
    if pin["passed"] is not True:
        raise ValueError("run 2's SFT weights did not pass the format gate")
    return dict(pin)


def rl_config_of(rl: Mapping[str, Any]) -> RLConfig:
    """``RLConfig`` from a specification's ``rl`` section (its rules are ``RLConfig``'s)."""
    values = {name: rl[name] for name in RL_CONFIG_KEYS}
    if not isinstance(values["dev_checks"], list | tuple):
        raise ValueError("rl.dev_checks must be a list of update numbers")
    values["dev_checks"] = tuple(values["dev_checks"])
    return RLConfig(**values)


def control_config_of(control: Mapping[str, Any], *, attention: str) -> ControlConfig:
    """``ControlConfig`` from a specification's ``control`` section and RL's attention kernel."""
    values = {name: control[name] for name in CONTROL_CONFIG_KEYS}
    if not isinstance(values["pass_checks"], list | tuple):
        raise ValueError("control.pass_checks must be a list of pass counts")
    values["pass_checks"] = tuple(values["pass_checks"])
    return ControlConfig(**values, attention=attention)


def _exact(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly the keys {sorted(keys)}")
    return value


def _positive(value: Any, name: str) -> float:
    number = finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rates(value: Any, name: str) -> dict[int, float]:
    """A ``{batch size: tokens per second}`` mapping written with string keys."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must map batch sizes to tokens per second")
    rates = {}
    for key, rate in value.items():
        if not isinstance(key, str) or not key.isdigit() or int(key) < 1:
            raise ValueError(f"{name} keys must be batch sizes")
        rates[int(key)] = _positive(rate, f"{name}[{key}]")
    return rates


@dataclass(frozen=True)
class EvalFile:
    """One completions file of an evaluation stage."""

    name: str
    suite: str
    split: str
    tasks: str
    protocol: str
    samples: int
    temperature: float
    top_p: float
    seed: int
    planned_tasks: int

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"evaluation file name {self.name!r} is not a plain name")
        if self.suite == POOL:
            if self.split not in POOL_SPLITS:
                raise ValueError(f"{self.name}: a pool file's split is dev or test")
        elif self.suite in BENCHMARK_SUITES:
            if self.split != BENCHMARK_SPLIT:
                raise ValueError(f"{self.name}: a benchmark file's split is eval")
        else:
            raise ValueError(f"{self.name}: unknown suite {self.suite!r}")
        protocol_for(self.protocol, self.suite)
        _ = self.sampling  # The sampling rules: greedy has one sample, and so on.
        _count(self.planned_tasks, f"{self.name} planned_tasks")

    @property
    def sampling(self) -> Sampling:
        return Sampling(
            samples=self.samples,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )

    @property
    def cap(self) -> int:
        return protocol_for(self.protocol, self.suite).cap

    @property
    def planned_tokens(self) -> int:
        """Generated tokens if every attempt spent its protocol's whole cap (an upper bound)."""
        return self.planned_tasks * self.samples * self.cap


@dataclass(frozen=True)
class StagePlan:
    name: str
    kind: str
    settings: Mapping[str, Any]

    @property
    def files(self) -> tuple[EvalFile, ...]:
        return tuple(self.settings.get("files", ()))

    def checkpoint_references(self) -> list[str]:
        if self.kind == "arc":
            return list(self.settings["checkpoints"])
        if self.kind in ("sft", "rl", "more-sft"):
            return [self.settings["parent"]]
        return [self.settings["checkpoint"]]


@dataclass(frozen=True)
class Specification:
    run: str
    model: ModelConfig
    limits: Mapping[str, Any]
    planning: Mapping[str, Any]
    probe: Mapping[str, Any]
    generation: Mapping[str, Any]
    sft: Mapping[str, Any]
    screening: Mapping[str, Any]
    stages: tuple[StagePlan, ...]
    inputs: Mapping[str, Mapping[str, Any]]
    r2: Mapping[str, Any]
    raw: Mapping[str, Any] = field(repr=False, compare=False, default_factory=dict)
    rl: Mapping[str, Any] = field(default_factory=dict)
    """Run 2's RL protocol (``RL_KEYS``); empty for run 1."""
    control: Mapping[str, Any] = field(default_factory=dict)
    """Run 2's more-SFT control settings (``CONTROL_KEYS``); empty for run 1."""
    bridge: Mapping[str, Any] = field(default_factory=dict)
    """Run 2's reward bridge (``posttrain_ops.BRIDGE_KEYS``); empty for run 1."""
    sft_results: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Run 2's pins of run 1's SFT evaluation artifacts (``SFT_RESULTS``); empty for run 1."""

    @property
    def ceiling(self) -> posttrain_ops.CeilingLimits:
        return posttrain_ops.specification_limits(self.raw)

    @property
    def bridge_timeouts(self) -> posttrain_ops.BridgeTimeouts | None:
        """Run 2's derived wait bounds; ``None`` for a run without the bridge."""
        return posttrain_ops.specification_bridge_timeouts(self.raw)

    def rl_config(self) -> RLConfig:
        """The ``RLConfig`` of run 2's protocol."""
        return rl_config_of(self.rl)

    def control_config(self, *, attention: str) -> ControlConfig:
        """The control's ``ControlConfig``, with RL's attention kernel."""
        return control_config_of(self.control, attention=attention)

    @property
    def max_transfer_seconds(self) -> float:
        return float(self.limits["max_transfer_seconds"])

    @property
    def finalization_reserve_seconds(self) -> float:
        return float(self.limits["finalization_reserve_seconds"])

    @property
    def generation_batch(self) -> int:
        return int(self.generation["batch_size"])

    @property
    def rows_per_update(self) -> int:
        return int(self.sft["batch_size"]) * int(self.sft["accumulation_steps"])

    @property
    def sft_updates(self) -> int:
        return -(-int(self.sft["train_rows"]) * int(self.sft["passes"]) // self.rows_per_update)

    @property
    def slots_per_update(self) -> int:
        return self.rows_per_update * int(self.sft["sequence_length"])

    def stage(self, name: str) -> StagePlan:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))  # type: ignore[no-any-return]

    @classmethod
    def load(cls, path: Path) -> Specification:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: Any, *, allow_gate_nulls: bool = False) -> Specification:
        """Parse a specification. Run 2's gate values must all be filled unless
        ``allow_gate_nulls`` (the cost projection of the tracked template only)."""
        if isinstance(raw, dict) and raw.get("run") == "2":
            return cls._from_run2(raw, allow_gate_nulls=allow_gate_nulls)
        top = _exact(raw, TOP_KEYS, "the specification")
        if top["schema_version"] != SCHEMA_VERSION or top["kind"] != SPECIFICATION_KIND:
            raise ValueError("not a post-training run specification of this schema")
        if top["run"] not in posttrain_ops.RUNS:
            raise ValueError(f"unknown post-training run {top['run']!r}")
        model = ModelConfig(**_exact(top["model"], set(ModelConfig.__dataclass_fields__), "model"))
        limits = posttrain_ops.read_limits(top["limits"])
        posttrain_ops.specification_limits(top)  # The derived limits must exist.
        gated = any(
            isinstance(item, dict) and item.get("kind") == "format-gate" for item in top["stages"]
        )
        planning_keys = PLANNING_KEYS | ({GATE_PLANNING_KEY} if gated else set())
        planning = _exact(top["planning"], planning_keys, "planning")
        rates = _rates(planning["generation_tokens_per_second"], "planning rates")
        for name in planning_keys - {"generation_tokens_per_second"}:
            _positive(planning[name], f"planning.{name}")
        probe = _exact(top["probe"], PROBE_KEYS, "probe")
        sizes = probe["batch_sizes"]
        if not isinstance(sizes, list) or [type(size) for size in sizes] != [int] * len(sizes):
            raise ValueError("probe.batch_sizes must be a list of integers")
        if not sizes or any(size < 1 for size in sizes) or sizes != sorted(set(sizes)):
            raise ValueError("probe.batch_sizes must be increasing positive integers")
        minimum = _rates(probe["minimum_tokens_per_second"], "probe minimum rates")
        if not set(minimum) <= set(sizes):
            raise ValueError("every probe threshold must be at a probed batch size")
        for name in ("prompt_lengths", "max_new_tokens", "repeats"):
            _count(probe[name], f"probe.{name}")
        if probe["precision"] not in ("float32", "bfloat16"):
            raise ValueError("probe.precision must be float32 or bfloat16")
        generation = _exact(top["generation"], GENERATION_KEYS, "generation")
        if generation["batch_size"] not in rates or generation["batch_size"] not in sizes:
            raise ValueError("the generation batch must be probed and have a planning rate")
        if generation["precision"] not in ("float32", "bfloat16"):
            raise ValueError("generation.precision must be float32 or bfloat16")
        special = isinstance(top["sft"], dict) and bool(SFT_SPECIAL_KEYS & set(top["sft"]))
        sft = _exact(top["sft"], SFT_KEYS | (SFT_SPECIAL_KEYS if special else set()), "sft")
        if special:
            # TrainConfig's rules: the initialization, a nonnegative seed, a positive scale.
            TrainConfig(**{key: sft[key] for key in SFT_SPECIAL_KEYS})
        for name in (
            "train_rows",
            "planned_dev_tokens",
            "batch_size",
            "accumulation_steps",
            "sequence_length",
            "passes",
            "dev_batch_size",
            "speed_check_updates",
        ):
            _count(sft[name], f"sft.{name}")
        if type(sft["speed_check_warmup_updates"]) is not int or not (
            0 <= sft["speed_check_warmup_updates"] < sft["speed_check_updates"]
        ):
            raise ValueError("sft.speed_check_warmup_updates must lie below the check")
        if not 0 < finite_number(sft["warmup_fraction"], "sft.warmup_fraction") < 1:
            raise ValueError("sft.warmup_fraction must be in (0, 1)")
        if not 0 < finite_number(sft["minimum_speed_fraction"], "sft speed fraction") <= 1:
            raise ValueError("sft.minimum_speed_fraction must be in (0, 1]")
        _positive(sft["reference_slots_per_second"], "sft.reference_slots_per_second")
        if sft["sequence_length"] > model.context:
            raise ValueError("SFT rows must fit the model context")
        screening = _exact(top["screening"], SCREENING_KEYS, "screening")
        for name in ("samples", "max_new_tokens", "planned_tasks", "chunk_tasks"):
            _count(screening[name], f"screening.{name}")
        inputs = top["inputs"]
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("the specification must list its inputs")
        for name, pin in inputs.items():
            _exact(pin, INPUT_PIN_KEYS, f"input {name}")
            if not posttrain_binding.NAME.fullmatch(name):
                raise ValueError(f"input name {name!r} is not plain")
            if pin["kind"] not in posttrain_binding.INPUT_KINDS:
                raise ValueError(f"input {name} has an unknown kind")
        stages = tuple(cls._stage(item, inputs) for item in top["stages"])
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names) or not stages:
            raise ValueError("stage names must be unique and the stage list nonempty")
        if stages[0].kind != "probe":
            raise ValueError("the probe is the first stage: every rate after it is measured")
        produced: set[str] = set()
        gate_seen = False
        for stage in stages:
            for reference in stage.checkpoint_references():
                if reference == SFT_REFERENCE and SFT_REFERENCE not in produced:
                    raise ValueError(f"{stage.name} reads the SFT weights before the sft stage")
                if reference == SFT_REFERENCE and gated and not gate_seen:
                    if stage.kind != "format-gate":
                        raise ValueError(
                            f"{stage.name} reads the SFT weights before the format gate passed"
                        )
            if stage.kind == "sft":
                produced.add(SFT_REFERENCE)
            if stage.kind == "format-gate":
                gate_seen = True
        if gated and [stage.kind for stage in stages].count("format-gate") != 1:
            raise ValueError("a specification gates its SFT weights with one format-gate stage")
        if top["run"] == "1b" and not gated:
            raise ValueError("run 1b's SFT weights go through the format gate before any use")
        r2 = _exact(top["r2"], {"storage_cap_usd_per_month"}, "r2")
        _positive(r2["storage_cap_usd_per_month"], "r2.storage_cap_usd_per_month")
        return cls(
            run=top["run"],
            model=model,
            limits=limits,
            planning=planning,
            probe=probe,
            generation=generation,
            sft=sft,
            screening=screening,
            stages=stages,
            inputs=inputs,
            r2=r2,
            raw=json.loads(json.dumps(top)),
        )

    @classmethod
    def _from_run2(cls, raw: dict[str, Any], *, allow_gate_nulls: bool) -> Specification:
        """Run 2: RL, the more-SFT control, ARC and evaluation of both selected checkpoints."""
        top = _exact(raw, RUN2_TOP_KEYS, "the run-2 specification")
        if top["schema_version"] != SCHEMA_VERSION or top["kind"] != SPECIFICATION_KIND:
            raise ValueError("not a post-training run specification of this schema")
        missing = gate_nulls(top)
        if missing and not allow_gate_nulls:
            raise ValueError(
                f"run 2's gate values are still null: {missing}; they are fixed at the owner's "
                "gate (python -m tools.posttrain_approved run2-gate)"
            )
        model = ModelConfig(**_exact(top["model"], set(ModelConfig.__dataclass_fields__), "model"))
        limits = posttrain_ops.read_limits(
            top["limits"],
            allow_null=posttrain_ops.RUN2_GATE_LIMITS,
            keys=posttrain_ops.RUN2_LIMIT_KEYS,
        )
        if all(limits[key] is not None for key in posttrain_ops.RUN2_GATE_LIMITS):
            # The limits exist, and the written cost limit is the lower of the ceiling and
            # 120% of the gate's estimate.
            posttrain_ops.specification_limits(top)
        planning = _exact(top["planning"], RUN2_PLANNING_KEYS, "planning")
        rates = _rates(planning["generation_tokens_per_second"], "planning rates")
        for name in RUN2_PLANNING_KEYS - {"generation_tokens_per_second"}:
            _positive(planning[name], f"planning.{name}")
        generation = _exact(top["generation"], GENERATION_KEYS, "generation")
        rl = _exact(top["rl"], RL_KEYS, "rl")
        control = _exact(top["control"], CONTROL_KEYS, "control")
        for name in ("dev_tasks", "dev_samples", "dev_batch_size", "min_tasks"):
            _count(rl[name], f"rl.{name}")
        if rl["attention"] not in ("reference", "fused"):
            raise ValueError("rl.attention must be reference or fused")
        rl_config_of(rl)  # Every RLConfig rule holds.
        control_config_of(control, attention=str(rl["attention"]))
        for batch in (generation["batch_size"], rl["generation_batch"], rl["dev_batch_size"]):
            if batch not in rates:
                raise ValueError(f"generation batch {batch} has no planning rate")
        if generation["precision"] not in ("float32", "bfloat16"):
            raise ValueError("generation.precision must be float32 or bfloat16")
        if rl["max_new_tokens"] >= model.context:
            raise ValueError("RL rollouts must leave room for a prompt in the model context")
        bridge = top["bridge"]
        posttrain_ops.specification_bridge_timeouts(top)  # Every bound equals its derivation.
        if not isinstance(bridge["root"], str) or not bridge["root"].startswith("/"):
            raise ValueError("bridge.root must be an absolute Pod path")
        inputs = top["inputs"]
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError("the specification must list its inputs")
        for name, pin in inputs.items():
            _exact(pin, INPUT_PIN_KEYS, f"input {name}")
            if not posttrain_binding.NAME.fullmatch(name):
                raise ValueError(f"input name {name!r} is not plain")
            if pin["kind"] not in posttrain_binding.INPUT_KINDS:
                raise ValueError(f"input {name} has an unknown kind")
        stages = tuple(cls._stage(item, inputs, RUN2_STAGE_KEYS) for item in top["stages"])
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names) or not stages:
            raise ValueError("stage names must be unique and the stage list nonempty")
        kinds = [stage.kind for stage in stages]
        if kinds.count("rl") != 1 or kinds[0] != "rl" or kinds.count("more-sft") > 1:
            raise ValueError("run 2 starts with its one rl stage and has at most one control")
        first = stages[0].settings
        produced: set[str] = set()
        for stage in stages:
            for reference in stage.checkpoint_references():
                if reference in PRODUCED_REFERENCES.values() and reference not in produced:
                    raise ValueError(
                        f"{stage.name} reads the {reference} weights before they exist"
                    )
            if stage.kind == "more-sft":
                same = {key: stage.settings[key] for key in ("parent", "tasks", "dev", "facts")}
                if same != {key: first[key] for key in same}:
                    raise ValueError(
                        "the control starts from RL's parent, tasks, dev set and facts"
                    )
            if stage.kind in PRODUCED_REFERENCES:
                produced.add(PRODUCED_REFERENCES[stage.kind])
            if stage.kind == "eval" and tuple(entry.name for entry in stage.files) != (
                SFT_RESULT_FILES
            ):
                raise ValueError(
                    f"{stage.name}: run 2 evaluates the seven files of run 1's eval-sft, "
                    "whose results an arm with the SFT weights reuses"
                )
        sft_results = check_sft_results(top)
        check_sft_gate(top)
        r2 = _exact(top["r2"], {"storage_cap_usd_per_month"}, "r2")
        _positive(r2["storage_cap_usd_per_month"], "r2.storage_cap_usd_per_month")
        return cls(
            run=top["run"],
            model=model,
            limits=limits,
            planning=planning,
            probe={},
            generation=generation,
            sft={},
            screening={},
            stages=stages,
            inputs=inputs,
            r2=r2,
            raw=json.loads(json.dumps(top)),
            rl=rl,
            control=control,
            bridge=bridge,
            sft_results=sft_results,
        )

    @staticmethod
    def _stage(
        item: Any,
        inputs: Mapping[str, Any],
        allowed: Mapping[str, set[str]] = STAGE_KEYS,
    ) -> StagePlan:
        if not isinstance(item, dict) or item.get("kind") not in allowed:
            raise ValueError(f"a stage needs a kind among {sorted(allowed)}")
        kind = item["kind"]
        settings = dict(_exact(item, allowed[kind], f"stage {item.get('name')}"))
        if not isinstance(settings["name"], str) or not _NAME.fullmatch(settings["name"]):
            raise ValueError("a stage needs a plain name")

        def input_of(reference: Any, kinds: Collection[str]) -> None:
            if reference not in inputs or inputs[reference]["kind"] not in kinds:
                raise ValueError(f"stage {settings['name']}: {reference!r} is not a {kinds} input")

        def checkpoint(reference: Any) -> None:
            produced = (
                (SFT_REFERENCE,) if allowed is STAGE_KEYS else (RL_REFERENCE, CONTROL_REFERENCE)
            )
            if reference not in produced:
                input_of(reference, CHECKPOINT_KINDS)

        if kind == "probe":
            checkpoint(settings["checkpoint"])
            input_of(settings["prompts"], {"pool-export"})
        elif kind == "arc":
            references = settings["checkpoints"]
            if not isinstance(references, list) or not references:
                raise ValueError("an ARC stage lists its checkpoints")
            for reference in references:
                checkpoint(reference)
            input_of(settings["questions"], {"arc-export"})
        elif kind == "eval":
            checkpoint(settings["checkpoint"])
            files = settings["files"]
            if not isinstance(files, list) or not files:
                raise ValueError("an evaluation stage lists its files")
            parsed = []
            for raw_file in files:
                entry = EvalFile(**_exact(raw_file, EVAL_FILE_KEYS, "an evaluation file"))
                input_of(entry.tasks, {"pool-export"} if entry.suite == POOL else {"benchmark"})
                parsed.append(entry)
            if len({entry.name for entry in parsed}) != len(parsed):
                raise ValueError("evaluation file names must be unique within a stage")
            settings["files"] = tuple(parsed)
        elif kind == "sft":
            input_of(settings["parent"], CHECKPOINT_KINDS)
            input_of(settings["train"], {"sft-rows"})
            dev = settings["dev"]
            if not isinstance(dev, list) or not dev:
                raise ValueError("the sft stage lists its dev sets")
            for reference in dev:
                input_of(reference, {"sft-rows"})
        elif kind == "format-gate":
            if settings["checkpoint"] != SFT_REFERENCE:
                raise ValueError("the format gate reads the SFT stage's selected weights")
            input_of(settings["rows"], {"sft-rows"})
            input_of(settings["tasks"], {"pool-export"})
            benchmarks = settings["benchmarks"]
            if not isinstance(benchmarks, list) or len(set(benchmarks)) != len(benchmarks):
                raise ValueError("the format gate lists its benchmark smoke inputs once each")
            for reference in benchmarks:
                input_of(reference, {"benchmark"})
                if reference not in BENCHMARK_SMOKE_SUITES:
                    raise ValueError(
                        f"the format gate's benchmark smoke reads {BENCHMARK_SMOKE_SUITES}"
                    )
            for name in ("row_count", "smoke_prompts", "benchmark_prompts"):
                _count(settings[name], f"format-gate {name}")
        elif kind in ("rl", "more-sft"):
            input_of(settings["parent"], CHECKPOINT_KINDS)
            input_of(settings["tasks"], {"rl-tasks"})
            input_of(settings["dev"], {"rl-dev-tasks"})
            input_of(settings["facts"], {"rl-facts"})
        else:
            checkpoint(settings["checkpoint"])
            input_of(settings["tasks"], {"candidates-export"})
            input_of(settings["candidates"], {"candidate-list"})
        return StagePlan(settings["name"], kind, settings)


# Planning -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rates:
    """The rates a plan is priced at: planning values, or measured ones once known."""

    generation_tokens_per_second: float
    training_slots_per_second: float
    forward_tokens_per_second: float
    arc_seconds_per_checkpoint: float
    checkpoint_load_seconds: float
    probe_seconds: float
    compile_seconds: float
    weights_save_seconds: float
    basis: Mapping[str, str] = field(default_factory=dict)
    rollout_tokens_per_second: float | None = None
    """Run 2: RL rollout generation at its batch (256), priced at the token cap."""
    dev_generation_tokens_per_second: float | None = None
    """Run 2: dev-check generation at its batch, priced at the token cap."""
    reward_wait_seconds_per_update: float | None = None
    dev_wait_seconds_per_check: float | None = None
    """Run 2: the Mac's time for one update's rewards and one dev check's verdicts."""
    rl_busy_seconds: float | None = None
    """Run 2: RL's measured GPU-busy seconds, the control's budget, once RL has run."""
    format_gate_seconds: float | None = None
    """Run 1b: the format gate on the GPU (planned, then measured), load excluded."""

    def recorded(self) -> dict[str, float]:
        """The rates as a report records them (run 1's keys only for run 1)."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "basis" and value is not None
        }

    @classmethod
    def planning(cls, spec: Specification) -> Rates:
        plan = spec.planning
        generation = _rates(plan["generation_tokens_per_second"], "planning rates")
        if spec.run == "2":
            return cls(
                generation_tokens_per_second=generation[spec.generation_batch],
                training_slots_per_second=float(plan["training_slots_per_second"]),
                forward_tokens_per_second=0.0,
                arc_seconds_per_checkpoint=float(plan["arc_seconds_per_checkpoint"]),
                checkpoint_load_seconds=float(plan["checkpoint_load_seconds"]),
                probe_seconds=0.0,
                compile_seconds=0.0,
                weights_save_seconds=float(plan["weights_save_seconds"]),
                basis={
                    name: "planning"
                    for name in ("generation", "training", "arc", "rollout", "dev", "mac")
                },
                rollout_tokens_per_second=generation[int(spec.rl["generation_batch"])],
                dev_generation_tokens_per_second=generation[int(spec.rl["dev_batch_size"])],
                reward_wait_seconds_per_update=float(plan["reward_wait_seconds_per_update"]),
                dev_wait_seconds_per_check=float(plan["dev_wait_seconds_per_check"]),
            )
        return cls(
            generation_tokens_per_second=generation[spec.generation_batch],
            training_slots_per_second=float(plan["training_slots_per_second"]),
            forward_tokens_per_second=float(plan["forward_tokens_per_second"]),
            arc_seconds_per_checkpoint=float(plan["arc_seconds_per_checkpoint"]),
            checkpoint_load_seconds=float(plan["checkpoint_load_seconds"]),
            probe_seconds=float(plan["probe_seconds"]),
            compile_seconds=float(plan["compile_seconds"]),
            weights_save_seconds=float(plan["weights_save_seconds"]),
            basis={name: "planning" for name in ("generation", "training", "arc")},
            format_gate_seconds=(
                float(plan[GATE_PLANNING_KEY]) if GATE_PLANNING_KEY in plan else None
            ),
        )


def rl_parts(spec: Specification, rates: Rates) -> dict[str, float]:
    """Run 2's RL stage at ``rates``, part by part (the proposal's rows), at the token caps.

    Rollouts: updates x tasks x G attempts at the rollout cap; training: the same attempts as
    whole-context slots; reward waits: one per update; dev checks: each the subset's greedy and
    sampled programs at the 512-token cap, its Mac wait and its weights save."""
    rl = spec.rl
    attempts = int(rl["updates"]) * int(rl["tasks_per_update"]) * int(rl["group_size"])
    checks = len(rl["dev_checks"])
    assert rates.rollout_tokens_per_second and rates.reward_wait_seconds_per_update is not None
    return {
        "load": rates.checkpoint_load_seconds,
        "rollouts": attempts * int(rl["max_new_tokens"]) / rates.rollout_tokens_per_second,
        "training": attempts * spec.model.context / rates.training_slots_per_second,
        "reward_waits": int(rl["updates"]) * rates.reward_wait_seconds_per_update,
        "dev_generation": checks * dev_check_generation_seconds(spec, rates),
        "dev_waits": checks * float(rates.dev_wait_seconds_per_check or 0.0),
        "saves": checks * rates.weights_save_seconds,
    }


def dev_check_generation_seconds(spec: Specification, rates: Rates) -> float:
    """One dev check's generation: greedy plus the samples of every subset task at the cap."""
    programs = int(spec.rl["dev_tasks"]) * (int(spec.rl["dev_samples"]) + 1)
    assert rates.dev_generation_tokens_per_second
    return programs * DEV_TOKENS / rates.dev_generation_tokens_per_second


def control_parts(spec: Specification, rates: Rates) -> dict[str, float]:
    """Run 2's control at ``rates``: its budget (RL's busy seconds, planned or measured) and
    its checks after 1, 2, 4 and 8 passes and at the budget's end (the start is RL's)."""
    parts = rl_parts(spec, rates)
    busy = (
        parts["rollouts"] + parts["training"]
        if rates.rl_busy_seconds is None
        else rates.rl_busy_seconds
    )
    checks = len(spec.control["pass_checks"]) + 1
    return {
        "load": rates.checkpoint_load_seconds,
        "training": busy,
        "dev_generation": checks * dev_check_generation_seconds(spec, rates),
        "dev_waits": checks * float(rates.dev_wait_seconds_per_check or 0.0),
        "saves": checks * rates.weights_save_seconds,
    }


def stage_seconds(spec: Specification, stage: StagePlan, rates: Rates) -> float:
    """One stage's planned seconds at ``rates``; generation is priced at the token caps."""
    load = rates.checkpoint_load_seconds
    if stage.kind == "rl":
        return math.fsum(rl_parts(spec, rates).values())
    if stage.kind == "more-sft":
        return math.fsum(control_parts(spec, rates).values())
    if stage.kind == "probe":
        return rates.probe_seconds
    if stage.kind == "format-gate":
        assert rates.format_gate_seconds is not None
        return load + rates.format_gate_seconds
    if stage.kind == "arc":
        return len(stage.settings["checkpoints"]) * (load + rates.arc_seconds_per_checkpoint)
    if stage.kind == "eval":
        tokens = sum(entry.planned_tokens for entry in stage.files)
        return len(stage.files) * load + tokens / rates.generation_tokens_per_second
    if stage.kind == "sft":
        quarters = len(quarter_updates(spec.sft_updates)) + 1
        return (
            load
            + rates.compile_seconds
            + spec.sft_updates * spec.slots_per_update / rates.training_slots_per_second
            + quarters * int(spec.sft["planned_dev_tokens"]) / rates.forward_tokens_per_second
            + quarters * rates.weights_save_seconds
        )
    screening = spec.screening
    tokens = (
        int(screening["planned_tasks"])
        * int(screening["samples"])
        * int(screening["max_new_tokens"])
    )
    return load + tokens / rates.generation_tokens_per_second


def planned_seconds(
    spec: Specification, rates: Rates, *, stages: Sequence[StagePlan] | None = None
) -> dict[str, Any]:
    """Seconds of ``stages`` (default: all) at ``rates`` plus the finalization reserve."""
    selected = spec.stages if stages is None else stages
    parts = {stage.name: stage_seconds(spec, stage, rates) for stage in selected}
    reserve = spec.finalization_reserve_seconds
    return {
        "stages": parts,
        "finalization_reserve_seconds": reserve,
        "total_seconds": math.fsum(parts.values()) + reserve,
        "rates": rates.recorded(),
        "basis": dict(rates.basis),
    }


def launch_fit(spec: Specification, *, remaining_seconds: float) -> dict[str, Any]:
    """The launch rule's record: the whole run at the planning rates against the time left."""
    plan = planned_seconds(spec, Rates.planning(spec))
    remaining = finite_number(remaining_seconds, "remaining seconds")
    return {
        "rule": LAUNCH_RULE,
        "projected_finish_rule": PROJECTED_FINISH_RULE,
        "remaining_seconds": remaining,
        "planning_seconds_needed": plan["total_seconds"],
        "fits_at_planning": 0 < plan["total_seconds"] <= remaining,
        "plan": plan,
    }


# Readiness ------------------------------------------------------------------------------


def check_receipt_identity(
    receipt: Mapping[str, Any],
    *,
    run: str,
    specification_sha256: str,
    binding_sha256: str,
    preflight_sha256: str,
) -> None:
    if receipt.get("kind") != posttrain_ops.READINESS_KIND or receipt.get("run") != run:
        raise ValueError(f"the receipt is not a post-training readiness receipt for run {run}")
    for name, value in (
        ("specification_sha256", specification_sha256),
        ("binding_sha256", binding_sha256),
        ("preflight_sha256", preflight_sha256),
    ):
        if receipt.get(name) != value:
            raise ValueError(f"the {name.removesuffix('_sha256')} differs from the receipt")


@dataclass(frozen=True)
class Launch:
    origin: float
    end: float
    pod_id: str
    fit: dict[str, Any]


def check_policy(policy: Policy, spec: Specification) -> None:
    """The watchdog policy carries the limits ``ceiling_limits`` derives, or stricter ones;
    run 2's wall and cost stop are exactly the ones derived from its cost limit."""
    derived = spec.ceiling
    if (
        policy.mode != "posttrain"
        or policy.max_wall_seconds > derived.wall_seconds
        or policy.max_cost_usd > derived.watchdog_cost_stop_usd
        or policy.max_idle_seconds > spec.limits["mac_idle_seconds"]
        or policy.quoted_total_hourly < derived.quoted_total_hourly
    ):
        raise ValueError("the watchdog policy is weaker than the ceiling-derived limits")
    if spec.run == "2" and (
        policy.max_wall_seconds != derived.wall_seconds
        or policy.max_cost_usd != derived.watchdog_cost_stop_usd
    ):
        raise ValueError(
            "the watchdog policy's wall or cost stop differs from the limits derived from run "
            "2's cost limit (posttrain_ops limits --policy-output writes them)"
        )
    if policy.max_transfer_seconds != spec.max_transfer_seconds:
        raise ValueError("the watchdog's transfer window differs from the specification's")
    # Run 2's bounded idle extension while the Pod awaits the Mac is exactly the derived
    # bound (the Pod's own timeout for that wait plus the slack): more would let a hung worker
    # idle longer, less would delete the Pod during a legitimate wait. Run 1 waits on nothing
    # and has none.
    bridge = spec.bridge_timeouts
    for kind, granted in (
        ("reward", policy.awaiting_reward_extension_seconds),
        ("dev", policy.awaiting_dev_extension_seconds),
    ):
        bound = 0.0 if bridge is None else bridge.extension(kind)
        if granted != bound:
            raise ValueError(
                f"the watchdog's {kind}-wait idle extension ({granted:g} s) is not the derived "
                f"bound ({bound:g} s)"
            )


def check_self_destruct(receipt: Mapping[str, Any], spec: Specification, *, heartbeat: str) -> None:
    """The armed timer is live (not a dry run) and watches the worker's heartbeat with the
    specification's idle limit."""
    if receipt.get("self_destruct_dry_run") is not False:
        raise ValueError("the in-Pod timer is a dry run (or unrecorded): it never deletes the Pod")
    idle = receipt.get("self_destruct_idle")
    if not isinstance(idle, dict):
        raise ValueError("the receipt does not record the in-Pod idle self-destruct")
    if idle.get("idle_seconds") != spec.limits["pod_idle_seconds"]:
        raise ValueError("the in-Pod timer's idle limit differs from the specification's")
    files = idle.get("activity_files")
    if not isinstance(files, list) or heartbeat not in files:
        raise ValueError(f"the in-Pod timer does not watch the worker's heartbeat {heartbeat}")


def check_readiness(
    receipt: Mapping[str, Any],
    *,
    now: float,
    spec: Specification,
    specification_sha256: str,
    binding_sha256: str,
    preflight_sha256: str,
    heartbeat: str,
) -> Launch:
    """Accept exactly a receipt from ``pilot_readiness_receipt posttrain`` for this launch.

    Its kind, run and SHA-256s must match, every gate be ``true``, the termination guarantee
    honest, the policy at least as strict as ``ceiling_limits``, the armed deadline inside the
    derived timer lifetime, the in-Pod idle timer watching this worker's heartbeat, and the R2
    prefix this Pod's. Then the launch rule: the lifetime left must hold every stage at the
    planning rates.
    """
    check_receipt_identity(
        receipt,
        run=spec.run,
        specification_sha256=specification_sha256,
        binding_sha256=binding_sha256,
        preflight_sha256=preflight_sha256,
    )
    missing = [gate for gate in posttrain_ops.READINESS_GATES if receipt.get(gate) is not True]
    if missing:
        raise ValueError("post-training readiness gates are unresolved: " + ", ".join(missing))
    check_termination_guarantee(dict(receipt))
    policy = Policy(**receipt["watchdog_policy"])
    check_policy(policy, spec)
    check_self_destruct(receipt, spec, heartbeat=heartbeat)
    identity = PodIdentity.from_dict(receipt["pod_identity"])
    created = parse_timestamp(identity.created_at)
    origin = policy.billing_started_at_unix
    if origin is None or not 0 < origin <= created + CLOCK_SKEW_SECONDS:
        raise ValueError("a conservative create-request billing origin is required")
    billing = origin
    origin = min(origin, created)
    derived = spec.ceiling
    deadline = finite_number(receipt.get("self_destruct_deadline_unix"), "self-destruct deadline")
    # The runbook arms the timer at the create request's time (the policy's billing origin)
    # plus the derived timer lifetime, so that is the bound, exactly: the 180 s margin inside
    # the wall is enforced. The origin is never later than the provider's creation time plus
    # the clock skew, so the timer fires at most that skew past creation plus 17,820 s.
    if not now < deadline <= billing + derived.self_destruct_seconds:
        raise ValueError("the armed in-Pod deadline lies outside the ceiling-derived lifetime")
    if receipt.get("r2_run_prefix") != posttrain_ops.run_prefix(spec.run, identity.pod_id):
        raise ValueError("the receipt's R2 prefix is not this run's Pod prefix")
    end = lifetime_end(
        policy,
        origin=origin,
        deadline=deadline,
        pod_seconds=derived.wall_seconds,
        cost_stop_usd=derived.worker_cost_stop_usd,
        quoted_total_hourly=derived.quoted_total_hourly,
    )
    fit = launch_fit(spec, remaining_seconds=end - now)
    require_fit(fit["remaining_seconds"], fit["planning_seconds_needed"])
    return Launch(origin, end, identity.pod_id, fit)


def check_preflight(preflight: Mapping[str, Any], *, hardware: Mapping[str, Any]) -> None:
    """A completed A40 preflight with the fused-attention and compile checks."""
    if (
        preflight.get("status") != "completed"
        or preflight.get("hardware_precision_and_resume_checked") is not True
        or preflight.get("device") != "cuda:0"
        or preflight.get("hardware") != hardware
    ):
        raise ValueError("a matching A40 hardware precision/resume preflight is required")
    for name in ("fused_attention", "compile_equivalence"):
        check = preflight.get(name)
        if not isinstance(check, dict) or check.get("status") != "passed":
            raise ValueError(f"the preflight's {name} check has not passed")


# Generation with a heartbeat ------------------------------------------------------------


def ticking_generate(
    tick: Callable[[int], None],
) -> Callable[..., list[Completion]]:
    """``generate_many`` with ``tick(generated tokens)`` after every batch.

    The batches are ``generate_many``'s exactly: prompts sorted longest first (ties by
    input index), cut into batches of ``batch_size``, each one ``generate_batch`` call, so
    every row gets the same completion; ``tick`` keeps the heartbeat and the limits live
    through an evaluation stage's generation.
    """

    def generate(
        model: GPT,
        prompts: Sequence[Sequence[int]],
        *,
        batch_size: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        seeds: Sequence[int] | None = None,
        stops: Sequence[Collection[int]] | None = None,
        log_probs: bool = False,
        precision: str = "float32",
        prefill_chunk: int | None = None,
        stop_check_interval: int = 8,
    ) -> list[Completion]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not prompts:
            raise ValueError("prompts must be nonempty")
        for name, values in (("seeds", seeds), ("stops", stops)):
            if values is not None and len(values) != len(prompts):
                raise ValueError(f"{name} need one entry per prompt")
        order = sorted(range(len(prompts)), key=lambda index: (-len(prompts[index]), index))
        results: dict[int, Completion] = {}
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            completions = generate_batch(
                model,
                [prompts[index] for index in chunk],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seeds=None if seeds is None else [seeds[index] for index in chunk],
                stops=None if stops is None else [stops[index] for index in chunk],
                log_probs=log_probs,
                precision=precision,
                prefill_chunk=prefill_chunk,
                stop_check_interval=stop_check_interval,
            )
            results.update(zip(chunk, completions, strict=True))
            tick(sum(len(completion.tokens) for completion in completions))
        return [results[index] for index in range(len(prompts))]

    return generate


def ticking_scorer(
    score: Callable[[str, list[str]], list[float]],
    tick: Callable[[], None],
    *,
    every: int = 64,
) -> Callable[[str, list[str]], list[float]]:
    """``score`` with ``tick()`` after every ``every`` ARC questions."""
    count = 0

    def scored(prompt: str, labels: list[str]) -> list[float]:
        nonlocal count
        values = score(prompt, labels)
        count += 1
        if count % every == 0:
            tick()
        return values

    return scored


@dataclass(frozen=True)
class RunLimits:
    """The worker's own wall, cost and idle stops, from ``ceiling_limits``."""

    pod_seconds: float
    quoted_total_hourly: float
    cost_stop_usd: float
    idle_seconds: float
    starting_cost_usd: float = 0.0

    def stop_reason(self, *, pod_elapsed: float, idle_elapsed: float = 0.0) -> str | None:
        if pod_elapsed >= self.pod_seconds:
            return "pod_wall_limit"
        cost = self.starting_cost_usd + pod_elapsed / 3600 * self.quoted_total_hourly
        if not math.isfinite(cost) or cost >= self.cost_stop_usd:
            return "cost_limit"
        if idle_elapsed >= self.idle_seconds:
            return "idle_gpu_limit"
        return None

    @classmethod
    def of(cls, spec: Specification, *, pod_seconds: float | None = None) -> RunLimits:
        derived = spec.ceiling
        return cls(
            pod_seconds=derived.wall_seconds if pod_seconds is None else pod_seconds,
            quoted_total_hourly=derived.quoted_total_hourly,
            cost_stop_usd=derived.worker_cost_stop_usd,
            idle_seconds=float(spec.limits["mac_idle_seconds"]),
        )


# The worker -----------------------------------------------------------------------------


class PosttrainWorker(Worker):
    """One post-training run on one machine; every artifact leaves through R2.

    It reuses the pretraining rehearsal worker's R2 finalization (``Worker.finalize`` and the
    methods it calls: the drain, the settle rounds, the bounded HEAD checks, the
    ``report.json`` upload, the handover) unchanged. It sets every attribute those methods read
    here and does not call ``Worker.__init__``, which builds a pretraining run. Every label is
    a milestone: nothing is pruned or rolled, and the run is durable once every label verified.
    """

    def __init__(
        self,
        spec: Specification,
        root: Path,
        *,
        inputs: Mapping[str, Path],
        store: Store,
        pod_id: str,
        pod_created: float,
        limits: RunLimits | None = None,
        device: str = "cpu",
        clock: Callable[[], float] = time.time,
        secrets: Iterable[str] = (),
        extra: Mapping[str, Any] | None = None,
        upload_retry_seconds: float = MILESTONE_RETRY_SECONDS,
        check_store_for: Callable[[float], Store] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        tokenizer: Path | None = None,
    ) -> None:
        if set(inputs) != set(spec.inputs):
            raise ValueError("the inputs differ from the specification's")
        for name, path in inputs.items():
            # Hidden tests never reach the Pod: every task file it reads is a Pod export.
            posttrain_binding.require_export(name, str(spec.inputs[name]["kind"]), path)
        if root.exists() or root.is_symlink():
            raise FileExistsError("post-training output must be a new directory")
        self.plan, self.root, self.inputs = spec, root, dict(inputs)
        self.prefix = posttrain_ops.run_prefix(spec.run, pod_id)
        self.store = posttrain_ops.GuardedStore(store, self.prefix)
        bounded = check_store_for
        self.check_store_for = (
            (lambda _timeout: self.store)
            if bounded is None
            else (lambda timeout: posttrain_ops.GuardedStore(bounded(timeout), self.prefix))
        )
        self.sleep, self.clock = sleep, clock
        self.handover: float | None = None
        self.pod_id, self.pod_created = pod_id, pod_created
        self.limits = RunLimits.of(spec) if limits is None else limits
        self.pod_end = pod_created + self.limits.pod_seconds
        self.reserve_unix = self.pod_end - spec.finalization_reserve_seconds
        """No stage runs past this: the rest of the lifetime is the finalization's."""
        self.device = training_device(device)
        self.tokenizer_path = tokenizer or next(
            path for name, path in self.inputs.items() if spec.inputs[name]["kind"] == "tokenizer"
        )
        self.tokenizer = ByteBPE.load(self.tokenizer_path)
        if self.tokenizer.vocab_size != spec.model.vocab_size:
            raise ValueError("the model vocabulary differs from the tokenizer")
        # Every prompt the worker encodes goes through the cached encoder, which gives the
        # reference encoder's IDs exactly: the reference applies all 65,000 merges to every
        # chunk, about a second per pool prompt, which would idle the GPU past the 600 s limit.
        self.encoder = FastEncoder(self.tokenizer)
        self.secrets = tuple(sorted({s for s in secrets if len(s) >= 4}, key=len, reverse=True))
        self.last_activity = clock()
        self.last_heartbeat = 0.0
        self.stop_reason: str | None = None
        self.uploading = False
        self.phase = "initializing"
        self.stage_name = spec.stages[0].name
        self.progress = 0
        self.generated_tokens = 0
        self.labels: list[str] = []
        self.milestones: list[str] = []
        self.label_files: dict[str, list[tuple[str, Path, str]]] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.r2_final: dict[str, Any] | None = None
        self.verification_errors: dict[str, str] = {}
        self.rates = Rates.planning(spec)
        self.sft_weights: Path | None = None
        self.probe_rates: dict[int, float] = {}
        self.current: StagePlan | None = None
        self.observed: dict[str, float] = {"cap_tokens": 0.0, "generation_seconds": 0.0}
        """The completed evaluation files' generation: tokens priced at their caps, and the
        seconds they took. Their ratio re-bases the projections when below the probe's rate."""
        commit, dirty = git_state()
        revision = (extra or {}).get("code_revision")
        if isinstance(revision, Mapping):
            commit, dirty = revision["git_commit"], revision["clean"] is not True
        self.stage_by_name: dict[str, dict[str, Any]] = {
            stage.name: {
                "name": stage.name,
                "kind": stage.kind,
                "status": "pending",
                "outputs": [],
                "seconds": None,
            }
            for stage in spec.stages
        }
        derived = spec.ceiling
        self.report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "run": spec.run,
            "pod_id": pod_id,
            "status": "running",
            "created_at": utc_now(),
            "git_commit": commit,
            "dirty": dirty,
            "specification": spec.to_dict(),
            "device": str(self.device),
            "tokenizer_identity": self.tokenizer.sha256,
            "inputs": {name: file_sha256(path) for name, path in sorted(self.inputs.items())},
            "stages": list(self.stage_by_name.values()),
            "artifacts": self.artifacts,
            "projections": [],
            "probe": None,
            "sft": None,
            "format_gate": None,
            "limits": {
                **asdict(derived),
                "pod_seconds": self.limits.pod_seconds,
                "cost_stop_usd": self.limits.cost_stop_usd,
                "idle_seconds": self.limits.idle_seconds,
                "pod_end_unix": self.pod_end,
                "finalization_reserve_seconds": spec.finalization_reserve_seconds,
                "finalization_reserve_unix": self.reserve_unix,
            },
            "cost": None,
            "r2": {"prefix": self.prefix, "bucket": store.credentials.bucket},
            "timings": {"pod_created_unix": pod_created, "worker_started_unix": clock()},
            "semantics": (
                "stages run in order; every artifact is uploaded as soon as it exists; the "
                "probe stops the run below the planning speeds; each stage starts only if the "
                "projected finish fits the Pod lifetime, and a stage still running at the "
                "finalization reserve stops there; SFT stops below the speed floor; wall, cost "
                "and idle limits are never extended"
            ),
            "limitations": [
                "worker time checks cannot terminate a provider resource during an API outage",
                "quoted-rate costs are estimates until reconciled with provider billing",
                "generation token counts are priced at the protocol caps in every projection",
            ],
            **dict(extra or {}),
        }
        root.mkdir(parents=True)
        self.uploader = CheckpointUploader(
            self.store,
            prefix=self.prefix,
            cap_usd=float(spec.r2["storage_cap_usd_per_month"]),
            receipt_dir=root / "r2-receipts",
            keep_rolling=1,
            retry_seconds=upload_retry_seconds,
            retry_max_seconds=upload_retry_seconds * RETRY_GROWTH,
        )
        self.publish()
        self.heartbeat()

    # The finalization's view of the specification and the closing record --------------

    @property
    def transfer_window_seconds(self) -> float:
        return self.plan.max_transfer_seconds

    def closing_record(self, report_key: str, report_sha256: str) -> dict[str, Any]:
        return {
            "report_key": report_key,
            "report_sha256": report_sha256,
            "artifact_count": len(self.artifacts),
        }

    def export_listing(self, label: str) -> dict[str, Any]:
        files = []
        for name, path, digest in self.label_files[label]:
            try:
                size: int | None = path.stat().st_size
            except OSError:
                size = None
            files.append({"name": name, "path": str(path), "sha256": digest, "size_bytes": size})
        listing: dict[str, Any] = {
            "label": label,
            "export": f"exports/{label}",
            "export_path": str(self.root / "exports" / label),
            "files": files,
        }
        weights = [row for row in files if row["name"] == "checkpoint.pt"]
        if weights:
            listing.update(
                checkpoint_sha256=weights[0]["sha256"],
                checkpoint_size_bytes=weights[0]["size_bytes"],
            )
        return listing

    # Records ----------------------------------------------------------------------------

    def publish(self) -> str:
        return self.write_json(self.root / "report.json", self.report)

    def heartbeat(self) -> None:
        status = self.report["status"]
        if self.uploading:
            status = posttrain_ops.UPLOADING_STATUS
        elif status in posttrain_ops.SUCCESS_STATUSES:
            status = "awaiting_transfer" if self.r2_final is not None else "running"
        value: dict[str, Any] = {
            "schema_version": 1,
            "updated_at_unix": self.clock(),
            "last_gpu_activity_unix": self.last_activity,
            "phase": self.phase,
            "step": self.progress,
            "generated_tokens": self.generated_tokens,
            "status": status,
            "run": self.plan.run,
            "pod_id": self.pod_id,
            "stage": self.stage_name,
        }
        if self.r2_final is not None:
            value["r2_final"] = self.r2_final
        if not self.uploading and "failure" in self.report:
            value["failure"] = self.report["failure"]
        self.heartbeat_fields(value)
        self.write_json(self.root / "heartbeat.json", value, limit=MAX_HEARTBEAT_BYTES)
        self.last_heartbeat = self.clock()

    def heartbeat_fields(self, value: dict[str, Any]) -> None:
        """A hook for a run's own heartbeat fields (run 2's ``awaiting_rewards``); run 1 adds
        none."""

    def before_finalize(self) -> None:
        """A hook for labels made just before the finalization (run 2's bridge directories);
        run 1 has none."""

    def after_stages(self) -> None:
        """A hook run after the last stage, before the run is recorded completed (run 2 raises
        ``PilotStop`` there when an arm's training ended early); run 1 has none."""

    # Limits -----------------------------------------------------------------------------

    def stopped(self) -> bool:
        now = self.clock()
        reason = self.limits.stop_reason(
            pod_elapsed=max(0.0, now - self.pod_created),
            idle_elapsed=max(0.0, now - self.last_activity),
        )
        if reason:
            self.stop_reason = reason
        elif self.stop_reason is None and now >= self.reserve_unix:
            self.stop_reason = RESERVE_STOP
        return self.stop_reason is not None

    def tick(self, tokens: int = 0) -> None:
        """GPU work in progress: refresh activity and the heartbeat, keep wall and cost, and
        stop at the finalization reserve (``RESERVE_STOP``), which finalizes in full."""
        now = self.clock()
        self.last_activity = now
        self.progress += 1
        self.generated_tokens += tokens
        reason = self.limits.stop_reason(pod_elapsed=max(0.0, now - self.pod_created))
        if reason is None and now >= self.reserve_unix:
            reason = RESERVE_STOP
        if reason:
            self.stop_reason = reason
            raise PilotStop(reason)
        if now - self.last_heartbeat >= HEARTBEAT_REFRESH_SECONDS:
            self.heartbeat()

    def release(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # Artifacts --------------------------------------------------------------------------

    def export_label(self, label: str, files: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
        """Publish ``exports/<label>/`` with these files and ``record.json``; upload it now."""
        directory = self.root / "exports" / label
        staging = directory.with_name("." + label)
        staging.mkdir(parents=True, exist_ok=False)
        names = []
        for name, source in files:
            target = staging / name
            try:
                os.link(source, target)
            except OSError:
                target.write_bytes(source.read_bytes())
            names.append(name)
        self.report["latest_export"] = f"exports/{label}"
        self.write_json(staging / "record.json", self.report)
        names.append("record.json")
        staging.replace(directory)
        rows = [(name, directory / name, file_sha256(directory / name)) for name in names]
        self.label_files[label] = rows
        self.labels.append(label)
        self.milestones.append(label)
        outputs = []
        for name, path, digest in rows:
            artifact = {
                "label": label,
                "name": name,
                "key": f"{self.prefix}{label}/{name}",
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
            self.artifacts.append(artifact)
            outputs.append({key: artifact[key] for key in ("label", "name", "sha256", "bytes")})
        if self.current is not None:
            # The stage's record lists each output as it is made, so a stage that stops or
            # fails later still names what it left in R2.
            self.stage_by_name[self.current.name]["outputs"].extend(outputs)
        self.uploader.submit(label, rows, milestone=True)
        self.publish()
        return outputs

    def artifact_path(self, stage: StagePlan, name: str) -> Path:
        path = self.root / "artifacts" / stage.name / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_artifact(self, stage: StagePlan, name: str, value: Any) -> Path:
        path = self.artifact_path(stage, name)
        self.write_json(path, value)
        return path

    def checkpoint_path(self, reference: str) -> Path:
        if reference == SFT_REFERENCE:
            if self.sft_weights is None:
                raise RuntimeError("the SFT weights were read before the sft stage selected them")
            return self.sft_weights
        return self.inputs[reference]

    def load(self, reference: str) -> tuple[GPT, ByteBPE]:
        self.phase = f"{self.stage_name}-load"
        model, tokenizer = load_checkpoint(
            self.checkpoint_path(reference), self.tokenizer_path, device=str(self.device)
        )
        self.last_activity = self.clock()
        return model, tokenizer

    # Projections ------------------------------------------------------------------------

    def remaining_stages(self, stage: StagePlan) -> list[StagePlan]:
        index = self.plan.stages.index(stage)
        return list(self.plan.stages[index:])

    def projected_finish(self, stage: StagePlan) -> None:
        """The projected-finish rule (``PROJECTED_FINISH_RULE``) before every stage."""
        now = self.clock()
        plan = planned_seconds(self.plan, self.rates, stages=self.remaining_stages(stage))
        finish = now + plan["total_seconds"]
        record = {
            "stage": stage.name,
            "checked_at_unix": now,
            "projected_finish_unix": finish,
            "pod_end_unix": self.pod_end,
            "fits": finish <= self.pod_end,
            **plan,
        }
        self.report["projections"].append(record)
        self.publish()
        if finish > self.pod_end:
            self.stop_reason = PROJECTED_STOP
            raise PilotStop(PROJECTED_STOP)

    # Stages -----------------------------------------------------------------------------

    def run_probe(self, stage: StagePlan) -> list[dict[str, Any]]:
        probe = self.plan.probe
        model, _ = self.load(stage.settings["checkpoint"])
        tasks = list(load_tasks(self.inputs[stage.settings["prompts"]]))
        lengths = []
        for index, task in enumerate(tasks, 1):
            lengths.append(len(chat.task_prompt(self.encoder, task)))
            if index % 256 == 0:
                self.tick()  # CPU work before the first batch keeps the heartbeat live too.
        lengths.sort(reverse=True)
        window = int(probe["prompt_lengths"])
        middle = len(lengths) // 2
        start = max(0, min(middle - window // 2, len(lengths) - window))
        chosen = lengths[start : start + window]
        self.phase = stage.name
        results = []
        for size in probe["batch_sizes"]:
            self.tick()
            results += benchmark_generation(
                model,
                batch_sizes=[size],
                prompt_lengths=chosen,
                max_new_tokens=int(probe["max_new_tokens"]),
                temperature=float(probe["temperature"]),
                top_k=None,
                top_p=float(probe["top_p"]),
                seed=int(probe["seed"]),
                precision=str(probe["precision"]),
                repeats=int(probe["repeats"]),
            )
            self.tick(int(results[-1]["generated_tokens"]))
        self.probe_rates = {
            int(row["batch_size"]): float(row["tokens_per_second"]) for row in results
        }
        minimum = _rates(probe["minimum_tokens_per_second"], "probe minimum rates")
        below = {size: rate for size, rate in minimum.items() if self.probe_rates[size] < rate}
        record = {
            "rule": "the run stops after the probe if any probed rate is below its planning speed",
            "measured_tokens_per_second": {str(k): v for k, v in self.probe_rates.items()},
            "minimum_tokens_per_second": {str(k): v for k, v in minimum.items()},
            "below_planning_speed": {str(k): v for k, v in below.items()},
            "prompt_lengths": {
                "pool_prompts": len(lengths),
                "median": lengths[middle] if lengths else None,
                "window": chosen,
                "protocol": CHAT,
            },
            "passed": not below,
            "results": results,
        }
        self.report["probe"] = {key: value for key, value in record.items() if key != "results"}
        path = self.write_artifact(stage, "generation-benchmark.json", record)
        outputs = self.export_label(stage.name, [("generation-benchmark.json", path)])
        del model
        self.release()
        if below:
            self.stop_reason = PROBE_STOP
            raise PilotStop(PROBE_STOP)
        measured = self.probe_rates[self.plan.generation_batch]
        self.rates = replace(
            self.rates,
            generation_tokens_per_second=measured,
            basis={**self.rates.basis, "generation": "measured"},
        )
        return outputs

    def run_arc(self, stage: StagePlan) -> list[dict[str, Any]]:
        questions = list(load_choices(self.inputs[stage.settings["questions"]]))
        outputs, seconds = [], []
        for reference in stage.settings["checkpoints"]:
            model, tokenizer = self.load(reference)
            started = self.clock()
            self.phase = f"{stage.name}-{reference}"
            report = evaluate_choices(
                questions,
                ticking_scorer(checkpoint_choice_scorer(model, tokenizer), self.tick),
                checkpoint_hash=file_sha256(self.checkpoint_path(reference)),
                tokenizer_hash=tokenizer.sha256,
            )
            self.tick()
            report.update(
                precision="float32",
                device=str(self.device),
                benchmark={
                    "suite": "ARC-Easy",
                    "source_split": "test",
                    "task_file_sha256": file_sha256(self.inputs[stage.settings["questions"]]),
                    "model_context": model.config.context,
                    "checkpoint": reference,
                },
            )
            label = f"{stage.name}-{reference}"
            path = self.write_artifact(stage, f"{reference}-arc.json", report)
            outputs += self.export_label(label, [("arc.json", path)])
            seconds.append(self.clock() - started)
            del model
            self.release()
        self.rates = replace(
            self.rates,
            arc_seconds_per_checkpoint=max(seconds)
            if seconds
            else self.rates.arc_seconds_per_checkpoint,
            basis={**self.rates.basis, "arc": "measured"},
        )
        return outputs

    def run_eval(self, stage: StagePlan) -> list[dict[str, Any]]:
        checkpoint = self.checkpoint_path(stage.settings["checkpoint"])
        generation = self.plan.generation
        outputs, files = [], []
        for entry in stage.files:
            self.phase = f"{stage.name}-{entry.name}"
            self.heartbeat()
            path = self.artifact_path(stage, f"{entry.name}.completions.jsonl")
            header, digest = generate_completions(
                checkpoint=checkpoint,
                tokenizer_path=self.tokenizer_path,
                suite=entry.suite,
                split=entry.split,
                tasks_path=self.inputs[entry.tasks],
                protocol=entry.protocol,
                output=path,
                sampling=entry.sampling,
                device=str(self.device),
                precision=str(generation["precision"]),
                batch_size=int(generation["batch_size"]),
                generate=ticking_generate(self.tick),
            )
            self.last_activity = self.clock()
            if header.task_count != entry.planned_tasks:
                raise ValueError(
                    f"{entry.name}: {header.task_count} tasks, the specification plans "
                    f"{entry.planned_tasks}"
                )
            files.append(
                {
                    "name": entry.name,
                    "completions_sha256": digest,
                    "suite": header.suite,
                    "split": header.split,
                    "protocol": header.protocol,
                    "task_file_sha256": header.task_file_sha256,
                    "tasks": header.task_count,
                    "planned_tasks": entry.planned_tasks,
                    "rows": header.rows,
                    "generation_seconds": header.generation_seconds,
                }
            )
            self.stage_by_name[stage.name]["files"] = files
            outputs += self.export_label(
                f"{stage.name}-{entry.name}", [("completions.jsonl", path)]
            )
            self.observe_generation(entry, header.generation_seconds)
            self.release()
        return outputs

    def observe_generation(self, entry: EvalFile, seconds: float) -> None:
        """Re-base the projections on the evaluation files generated so far.

        Their rate is priced as the projections price them, every attempt at its token cap,
        so it predicts a like file's seconds exactly. The probe measures random prompts at
        256 new tokens; real files with longer contexts may be slower. The projection rate
        becomes the lower of the probe's and the files' so far, so it only grows stricter.
        """
        if not math.isfinite(seconds) or seconds <= 0:
            return
        self.observed["cap_tokens"] += entry.planned_tokens
        self.observed["generation_seconds"] += seconds
        observed = self.observed["cap_tokens"] / self.observed["generation_seconds"]
        probe = self.probe_rates.get(self.plan.generation_batch, observed)
        rate = min(probe, observed)
        self.rates = replace(
            self.rates,
            generation_tokens_per_second=rate,
            basis={
                **self.rates.basis,
                "generation": "measured" if rate == probe else "observed-evaluation",
            },
        )
        self.report["generation_observed"] = {
            **self.observed,
            "cap_tokens_per_second": observed,
            "probe_tokens_per_second": probe,
            "projection_tokens_per_second": rate,
        }

    def run_sft(self, stage: StagePlan) -> list[dict[str, Any]]:
        settings = self.plan.sft
        rows = list(read_rows(self.inputs[stage.settings["train"]]))
        loader = SFTLoader(
            rows,
            sequence_length=int(settings["sequence_length"]),
            batch_size=int(settings["batch_size"]),
            accumulation_steps=int(settings["accumulation_steps"]),
            seed=int(settings["seed"]),
            passes=int(settings["passes"]),
        )
        if len(rows) != settings["train_rows"] or loader.dropped:
            raise ValueError(
                f"the SFT rows ({len(rows)}, dropped {loader.dropped}) differ from the "
                f"specification's {settings['train_rows']}"
            )
        updates = loader.planned_updates
        if updates != self.plan.sft_updates:
            raise ValueError("the SFT loader plans another number of updates")
        dev: dict[str, list[SFTRow]] = {
            name: list(read_rows(self.inputs[name])) for name in stage.settings["dev"]
        }
        warmup = max(1, round(float(settings["warmup_fraction"]) * updates))
        parent = self.checkpoint_path(stage.settings["parent"])
        config = RunConfig(
            model=self.plan.model,
            train=TrainConfig(
                seed=int(settings["seed"]),
                steps=updates,
                batch_size=int(settings["batch_size"]),
                sequence_length=int(settings["sequence_length"]),
                learning_rate=float(settings["learning_rate"]),
                min_lr_ratio=float(settings["min_lr_ratio"]),
                warmup_steps=warmup,
                weight_decay=float(settings["weight_decay"]),
                gradient_clip=float(settings["gradient_clip"]),
                accumulation_steps=int(settings["accumulation_steps"]),
                beta1=float(settings["beta1"]),
                beta2=float(settings["beta2"]),
                adam_epsilon=float(settings["adam_epsilon"]),
                precision=str(settings["precision"]),
                attention=str(settings["attention"]),
                optimizer_foreach=bool(settings["optimizer_foreach"]),
                schedule=str(settings["schedule"]),
                compile=bool(settings["compile"]),
                loss_normalization="update_targets",
                **{key: settings[key] for key in SFT_SPECIAL_KEYS if key in settings},
            ),
            data_mix={"sft": 1.0},
            parent_checkpoint=str(parent),
        )
        self.phase = f"{stage.name}-build"
        trainer = Trainer(
            config,
            loader,
            run_id=f"posttrain-{self.plan.run}-sft",
            run_root=self.root / "runs",
            data_hash=file_sha256(self.inputs[stage.settings["train"]]),
            tokenizer_hash=self.tokenizer.sha256,
            device=str(self.device),
        )
        self.last_activity = self.clock()
        checks = {update: quarter for quarter, update in quarter_updates(updates)}
        checks[updates] = 4
        record: dict[str, Any] = {
            "parent": stage.settings["parent"],
            "parent_sha256": file_sha256(parent),
            "planned_updates": updates,
            "warmup_updates": warmup,
            "rows_per_update": self.plan.rows_per_update,
            "slots_per_update": self.plan.slots_per_update,
            "loader_signature": loader.signature,
            "order_sha256": loader.order_sha256,
            "quarters": [],
            "speed_check": {"status": "pending", "after_updates": settings["speed_check_updates"]},
            "special_rows": trainer.special_rows,
            "special_rows_lr_scale": float(settings.get("special_rows_lr_scale", 1.0)),
            "selected": None,
        }
        self.report["sft"] = record
        timing: list[tuple[int, float]] = []
        best: dict[str, Any] = {}
        updates_path = self.artifact_path(stage, "updates.jsonl")
        updates_path.touch()
        speed_after = min(int(settings["speed_check_updates"]), updates)
        speed_warmup = int(settings["speed_check_warmup_updates"])

        def on_step(trainer: Trainer, metrics: dict[str, float]) -> None:
            if any(not math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("nonfinite SFT metric")
            self.tick()
            self.stage_name = stage.name
            self.phase = stage.name
            timing.append((trainer.step, float(metrics["step_seconds"])))
            with updates_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"update": trainer.step, **metrics}, allow_nan=False) + "\n"
                )
            if trainer.step == speed_after:
                self.sft_speed_check(record, timing, warmup=speed_warmup)
            quarter = checks.get(trainer.step)
            if quarter is not None:
                self.sft_quarter(stage, trainer, record, dev, quarter, best)
            if self.stopped():
                raise PilotStop(str(self.stop_reason))

        self.phase = stage.name
        trainer.train(on_step=on_step)
        if trainer.step != updates or not loader.exhausted:
            raise PilotStop("incomplete_sft")
        record["ticket"] = loader.ticket()
        trainer.finish(success=True)
        steady = [seconds for step, seconds in timing if step > speed_warmup] or [
            seconds for _, seconds in timing
        ]
        measured = len(steady) * self.plan.slots_per_update / math.fsum(steady)
        record["slots_per_second"] = measured
        self.rates = replace(
            self.rates,
            training_slots_per_second=measured,
            basis={**self.rates.basis, "training": "measured"},
        )
        selected = best["path"]
        for path in (self.root / "artifacts" / stage.name).glob("quarter-*.pt"):
            if path != selected:
                path.unlink()
        self.sft_weights = selected
        record["selected"] = {
            key: best[key] for key in ("quarter", "update", "dev_loss", "sha256", "bytes")
        }
        summary = self.write_artifact(stage, "sft.json", record)
        outputs = self.export_label(
            stage.name, [("sft.json", summary), ("updates.jsonl", updates_path)]
        )
        outputs += self.export_label(f"{stage.name}-weights", [("checkpoint.pt", selected)])
        if bool(settings["compile"]):
            torch._dynamo.reset()
        del trainer
        self.release()
        return outputs

    def sft_speed_check(
        self, record: dict[str, Any], timing: Sequence[tuple[int, float]], *, warmup: int
    ) -> None:
        """Slot throughput over the updates after the compile warmup, step seconds only."""
        settings = self.plan.sft
        steady = [seconds for step, seconds in timing if step > warmup]
        if not steady:
            steady = [seconds for _, seconds in timing]
        rate = len(steady) * self.plan.slots_per_update / math.fsum(steady)
        floor = float(settings["minimum_speed_fraction"]) * float(
            settings["reference_slots_per_second"]
        )
        record["speed_check"].update(
            status="passed" if rate >= floor else "failed",
            measured_slots_per_second=rate,
            floor_slots_per_second=floor,
            reference_slots_per_second=settings["reference_slots_per_second"],
            excluded_warmup_updates=warmup,
            counted_updates=len(steady),
            rule="step seconds only, after the warmup updates; below the floor the run stops",
        )
        self.publish()
        if rate < floor:
            self.stop_reason = SPEED_STOP
            raise PilotStop(SPEED_STOP)

    def sft_quarter(
        self,
        stage: StagePlan,
        trainer: Trainer,
        record: dict[str, Any],
        dev: Mapping[str, Sequence[SFTRow]],
        quarter: int,
        best: dict[str, Any],
    ) -> None:
        """Assistant-token dev loss at a quarter pass; the lowest so far is saved as weights."""
        settings = self.plan.sft
        self.phase = f"{stage.name}-dev-loss"
        started = synchronized_time(self.device)
        results = {
            name: assistant_loss(
                trainer.model,
                rows,
                batch_size=int(settings["dev_batch_size"]),
                autocast_bf16=settings["precision"] == "bfloat16",
                on_batch=self.tick,
            )
            for name, rows in dev.items()
        }
        loss = token_weighted_loss(results.values())
        entry: dict[str, Any] = {
            "quarter": quarter,
            "update": trainer.step,
            "dev_loss": loss,
            "by_set": results,
            "seconds": synchronized_time(self.device) - started,
        }
        if not best or loss < best["dev_loss"]:
            path = self.artifact_path(stage, f"quarter-{quarter}.pt")
            digest = save_weights(
                path,
                trainer.model,
                tokenizer_hash=self.tokenizer.sha256,
                run_id=f"posttrain-{self.plan.run}-sft",
                step=trainer.step,
                metadata={"quarter": quarter, "dev_loss": loss},
            )
            self.last_activity = self.clock()
            best.clear()
            best.update(
                quarter=quarter,
                update=trainer.step,
                dev_loss=loss,
                path=path,
                sha256=digest,
                bytes=path.stat().st_size,
            )
            entry["saved"] = True
        record["quarters"].append(entry)
        self.phase = stage.name
        self.publish()

    def run_format_gate(self, stage: StagePlan) -> list[dict[str, Any]]:
        """The SFT format gate on the dev-loss-selected weights (``FORMAT_GATE_RULE``).

        The gate reads the code dev rows, the pool dev export and the benchmark releases on
        the Pod, runs on this worker's device and runs no generated code. Its record is
        uploaded at once; below its bars the run stops (``FORMAT_GATE_STOP``) and finalizes,
        so the weights (uploaded by the sft stage) and the record are in R2 and nothing
        paid starts from those weights."""
        settings = stage.settings
        weights = self.checkpoint_path(stage.settings["checkpoint"])
        model, tokenizer = self.load(stage.settings["checkpoint"])
        self.phase = stage.name
        self.tick()
        started = synchronized_time(self.device)
        record = gate_record(
            model,
            tokenizer,
            checkpoint=weights,
            tokenizer_path=self.tokenizer_path,
            rows_path=self.inputs[settings["rows"]],
            tasks_path=self.inputs[settings["tasks"]],
            benchmarks={name: self.inputs[name] for name in settings["benchmarks"]},
            device=str(self.device),
            row_count=int(settings["row_count"]),
            smoke_prompts=int(settings["smoke_prompts"]),
            benchmark_prompts=int(settings["benchmark_prompts"]),
        )
        seconds = synchronized_time(self.device) - started
        self.tick()
        if record["inputs"]["checkpoint_sha256"] != file_sha256(weights):
            raise RuntimeError("the format gate measured other weights than the selected ones")
        record["rule"] = FORMAT_GATE_RULE
        path = self.write_artifact(stage, "gate.json", record)
        brief = gate_summary(record)
        self.report["format_gate"] = {
            "passed": record["passed"],
            "failed_checks": record["failed_checks"],
            "checkpoint_sha256": record["inputs"]["checkpoint_sha256"],
            "rows_without_public_tests_skipped": len(record["rows_without_public_tests_skipped"]),
            "checks": brief["checks"],
            "benchmark_smoke": {
                suite: {key: entry[key] for key in ("prompts", "opened", "closed", "programs")}
                for suite, entry in record["benchmark_smoke"].items()
            },
            "seconds": seconds,
            "rule": FORMAT_GATE_RULE,
        }
        outputs = self.export_label(stage.name, [("gate.json", path)])
        self.rates = replace(
            self.rates,
            format_gate_seconds=seconds,
            basis={**self.rates.basis, "format_gate": "measured"},
        )
        del model
        self.release()
        if not record["passed"]:
            self.stop_reason = FORMAT_GATE_STOP
            raise PilotStop(FORMAT_GATE_STOP)
        return outputs

    def run_screen(self, stage: StagePlan) -> list[dict[str, Any]]:
        settings = self.plan.screening
        tasks_path = self.inputs[stage.settings["tasks"]]
        candidates_path = self.inputs[stage.settings["candidates"]]
        tasks = list(load_tasks(tasks_path))
        listed = candidate_ids(candidates_path)
        if [task.id for task in tasks] != listed:
            raise ValueError("the candidates export is not in candidate-file order")
        model, tokenizer = self.load(stage.settings["checkpoint"])
        self.phase = stage.name
        generation = self.plan.generation
        started = synchronized_time(self.device)
        attempts = []
        for chunk in batched(tasks, int(settings["chunk_tasks"])):
            self.tick()
            rollouts = screening_rollouts(
                model,
                list(chunk),
                self.encoder,
                tokenizer,
                batch_size=int(generation["batch_size"]),
                precision=str(generation["precision"]),
                samples=int(settings["samples"]),
                seed=int(settings["seed_base"]),
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                max_new_tokens=int(settings["max_new_tokens"]),
            )
            attempts += [rollout.attempt for rollout in rollouts]
            self.tick(sum(len(rollout.tokens) for rollout in rollouts))
        seconds = synchronized_time(self.device) - started
        commit, dirty = self.report["git_commit"], self.report["dirty"]
        header = ScreeningHeader(
            task_file_sha256=file_sha256(tasks_path),
            candidates_sha256=file_sha256(candidates_path),
            checkpoint_sha256=file_sha256(self.checkpoint_path(stage.settings["checkpoint"])),
            tokenizer_sha256=tokenizer.sha256,
            tasks=len(tasks),
            samples=int(settings["samples"]),
            attempts=len(attempts),
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            seed_base=int(settings["seed_base"]),
            precision=str(generation["precision"]),
            device=str(self.device),
            batch_size=int(generation["batch_size"]),
            generation_seconds=seconds,
            created_at=utc_now(),
            git_commit=commit,
            dirty=dirty,
        )
        path = self.artifact_path(stage, "attempts.jsonl")
        digest = write_attempts(path, header, attempts)
        tokens = [attempt.completion_tokens for attempt in attempts]
        self.stage_by_name[stage.name]["screening"] = {
            "attempts_sha256": digest,
            "tasks": len(tasks),
            "attempts": len(attempts),
            "generated_tokens": sum(tokens),
            "mean_completion_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
            "with_program": sum(attempt.code is not None for attempt in attempts),
            "generation_seconds": seconds,
        }
        outputs = self.export_label(stage.name, [("attempts.jsonl", path)])
        del model
        self.release()
        return outputs

    # The run ----------------------------------------------------------------------------

    def run_stage(self, stage: StagePlan) -> None:
        record = self.stage_by_name[stage.name]
        self.current = stage  # A stop from here on, even before the stage starts, is its own.
        if self.stopped():
            raise PilotStop(str(self.stop_reason))
        self.stage_name, self.phase = stage.name, stage.name
        self.projected_finish(stage)
        started = self.clock()
        record.update(status="running", started_unix=started)
        self.publish()
        self.heartbeat()
        runner: Callable[[StagePlan], list[dict[str, Any]]] = getattr(
            self, "run_" + stage.kind.replace("-", "_")
        )
        outputs = runner(stage)
        if outputs != record["outputs"]:
            raise RuntimeError(f"stage {stage.name}'s outputs differ from its exported labels")
        self.current = None
        record.update(status="completed", ended_unix=self.clock(), seconds=self.clock() - started)
        self.publish()

    def check_planned_tasks(self) -> None:
        """Every evaluation file and the screening hold the tasks the specification prices.

        The projections price each file at its planned tasks, so a task file of another size
        would misprice every time rule; checked before any stage starts.
        """
        counts: dict[tuple[str, str, Path], int] = {}
        for stage in self.plan.stages:
            for entry in stage.files:
                key = (entry.suite, entry.split, self.inputs[entry.tasks])
                if key not in counts:
                    counts[key] = len(load_suite_tasks(*key)[0])
                if counts[key] != entry.planned_tasks:
                    raise ValueError(
                        f"{stage.name} {entry.name}: the task file holds {counts[key]} tasks, "
                        f"the specification plans {entry.planned_tasks}"
                    )
            if stage.kind == "screen":
                listed = len(candidate_ids(self.inputs[stage.settings["candidates"]]))
                if listed != int(self.plan.screening["planned_tasks"]):
                    raise ValueError(
                        f"{stage.name}: {listed} candidates, the specification plans "
                        f"{self.plan.screening['planned_tasks']}"
                    )

    def close_stage(self, status: str, **fields: Any) -> None:
        """Record a stopped or failed stage's status, end and seconds; its outputs are listed."""
        if self.current is None:
            return
        record = self.stage_by_name[self.current.name]
        ended = self.clock()
        record.update(status=status, ended_unix=ended, **fields)
        started = record.get("started_unix")
        if isinstance(started, int | float):
            record["seconds"] = ended - started
        self.current = None

    def execute(self) -> dict[str, Any]:
        try:
            self.report["timings"]["stages_started_unix"] = self.clock()
            self.check_planned_tasks()
            for stage in self.plan.stages:
                self.run_stage(stage)
            self.after_stages()
            self.report["status"] = "completed"
        except PilotStop as error:
            self.report.update(status="stopped", stop_reason=str(error))
            self.close_stage("stopped", stop_reason=str(error))
        except BaseException as error:
            self.report.update(status="failed", error_type=type(error).__name__)
            self.report["error"] = str(sanitize(str(error)))[:500]
            self.close_stage("failed")
            raise
        finally:
            for record in self.stage_by_name.values():
                if record["status"] == "pending":
                    record["status"] = "skipped"
            self.account()
            self.before_finalize()
            self.finalize()
            self.release()
        return self.report

    def account(self) -> None:
        now = self.clock()
        elapsed = max(0.0, now - self.pod_created)
        self.report["cost"] = {
            "quoted_total_hourly": self.limits.quoted_total_hourly,
            "pod_elapsed_seconds": elapsed,
            "quoted_accrued_cost_usd": self.limits.starting_cost_usd
            + elapsed / 3600 * self.limits.quoted_total_hourly,
            "cost_stop_usd": self.limits.cost_stop_usd,
            "ceiling_usd": self.plan.ceiling.ceiling_usd,
            "basis": "quoted rate times Pod lifetime; reconcile with provider billing",
        }
        self.report["rates"] = {**self.rates.recorded(), "basis": dict(self.rates.basis)}
        self.report["generated_tokens"] = self.generated_tokens
        self.report["timings"].update(ended_unix=now)
        self.report["ended_at"] = utc_now()


def candidate_ids(path: Path) -> list[str]:
    """The task IDs of an RL candidate list, in file order."""
    from scratchcoder.eval.export import read_candidate_list

    return [identifier for identifier, _ in read_candidate_list(path)]


# The Pod run ----------------------------------------------------------------------------


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Every launch gate before any GPU work, then the run's stages on ``cuda:0``."""
    spec = Specification.load(args.specification)
    receipt = _object(args.readiness)
    identity = {
        "specification_sha256": file_sha256(args.specification),
        "binding_sha256": args.binding_sha256,
        "preflight_sha256": file_sha256(args.preflight),
    }
    check_receipt_identity(receipt, run=spec.run, **identity)
    binding = posttrain_binding.load_binding(args.binding, expected_sha256=args.binding_sha256)
    posttrain_binding.require_specification(binding, spec.raw)
    heartbeat = str((args.output / "heartbeat.json").absolute())
    launch = check_readiness(receipt, now=time.time(), spec=spec, heartbeat=heartbeat, **identity)
    hardware = hardware_description(training_device("cuda:0"))
    check_preflight(_object(args.preflight), hardware=hardware)
    if os.environ.get("RUNPOD_POD_ID") not in (None, "", launch.pod_id):
        raise ValueError("this Pod is not the one the receipt identifies")
    credentials = r2_credentials(os.environ, key_prefix="R2_POD_")
    try:
        revision_value = json.loads(args.revision.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the shipped code revision: {error}") from error
    revision = check_revision(receipt, revision_value)
    verified = posttrain_binding.verify(binding, args.root)
    inputs = {name: Path(path) for name, path in verified["paths"].items()}
    limits = RunLimits.of(spec, pod_seconds=launch.end - launch.origin)
    worker_class: type[PosttrainWorker] = PosttrainWorker
    if spec.run == "2":
        from tools.posttrain_run2 import Run2Worker

        worker_class = Run2Worker
    worker = worker_class(
        spec,
        args.output,
        inputs=inputs,
        store=Store(credentials),
        check_store_for=lambda timeout: bounded_store(credentials, timeout=timeout, attempts=1),
        pod_id=launch.pod_id,
        pod_created=launch.origin,
        limits=limits,
        device="cuda:0",
        secrets=[credentials.access_key_id, credentials.secret_access_key, *environment_secrets()],
        extra={
            "code_revision": revision,
            "hardware": hardware,
            "readiness_receipt": sanitize(receipt),
            "specification_sha256": identity["specification_sha256"],
            "binding_sha256": args.binding_sha256,
            "preflight_sha256": identity["preflight_sha256"],
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
    run_parser = commands.add_parser("run", help="run one post-training run on this Pod")
    for name in ("specification", "binding", "root", "readiness", "preflight", "output"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--binding-sha256", required=True)
    run_parser.add_argument("--revision", type=Path, default=DEFAULT_REVISION)
    smoke_parser = commands.add_parser("smoke", help="run 1 or 2 end to end on tiny fixtures, CPU")
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument(
        "--run", choices=posttrain_ops.RUNS, default="1", help="1 (default), 1b or 2"
    )
    args = parser.parse_args(argv)
    if args.command == "smoke":
        if args.run == "1b":
            from tools.posttrain_smoke import smoke_1b

            print(json.dumps(smoke_1b(args.output), indent=1, sort_keys=True))
            return 0
        if args.run == "2":
            from tools.posttrain_smoke2 import smoke as smoke2

            print(json.dumps(smoke2(args.output), indent=1, sort_keys=True))
            return 0
        from tools.posttrain_smoke import smoke

        print(json.dumps(smoke(args.output), indent=1, sort_keys=True))
        return 0
    report = run(args)
    print(f"Post-training run {report['run']} {report['status']}: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
