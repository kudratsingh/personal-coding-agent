"""Bounded pilot worker on an existing machine; never provisions compute.

The smoke command uses authored tiny data on CPU. The run command additionally
requires externally bound data and an operational receipt. A receipt documents
readiness; this worker cannot enforce a provider's resource lifetime.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig, finite_number
from scratchcoder.common.logging import sanitize
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.loader import Documents, MixedLoader
from scratchcoder.data.shards import TokenShard, file_sha256
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.train.pilot import (
    Limits,
    PilotStop,
    accumulation_steps,
    choose_microbatch,
    cost_per_billion,
    throughput,
)
from scratchcoder.train.trainer import (
    Trainer,
    TrainingStop,
    synchronized_time,
    training_device,
)
from scratchcoder.train.validation import ValidationWindow, shard_windows, validate
from tools.cpu_resume_check import assert_identical
from tools.pilot_data import verify_binding
from tools.pilot_preflight import hardware_description
from tools.runpod_watchdog import CLOCK_SKEW_SECONDS, PodIdentity, Policy, parse_timestamp

_CUDA_MEMORY_MARKERS = (
    "out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUBLAS_STATUS_NOT_INITIALIZED",
)


COMPILE_PROBE_PREFIX = "probe-compiled-"
"""Phase prefix of the measurement-only ``torch.compile`` probe (a probe for every other rule)."""
COMPILE_SOFT_STOP_SECONDS = 10.0
"""The child stops at an update boundary this long before the hard window, once it has data."""
COMPILE_CHILD_SECRETS = ("RUNPOD_API_KEY", "HF_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _is_cuda_memory_error(error: BaseException) -> bool:
    """Recognize allocation failures that CUDA libraries raise as plain RuntimeError."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error)
    return any(marker in message for marker in _CUDA_MEMORY_MARKERS)


def trainer_config(
    spec: Specification, size: int, rate: float, *, probe: bool, compiled: bool = False
) -> RunConfig:
    """The run configuration of one probe, arm or the compile probe's child trainer."""
    if compiled and not probe:
        raise ValueError("compiled training is measured only by the compile probe")
    probe_steps = spec.compile_probe_steps if compiled else spec.probe_steps
    return replace(
        spec.config,
        train=replace(
            spec.config.train,
            batch_size=size,
            accumulation_steps=spec.probe_accumulation_steps
            if probe
            else accumulation_steps(
                size, spec.config.train.sequence_length, spec.tokens_per_update
            ),
            learning_rate=rate,
            steps=probe_steps if probe else spec.config.train.steps,
            warmup_steps=min(spec.config.train.warmup_steps, probe_steps)
            if probe
            else spec.config.train.warmup_steps,
            compile=spec.config.train.compile or compiled,
        ),
    )


def document_source(documents: Documents) -> dict[str, Any]:
    """Serialize the training documents so a child process reads the identical stream."""
    if isinstance(documents, TokenShard):
        metadata = json.loads(
            documents.path.with_suffix(documents.path.suffix + ".json").read_text()
        )
        return {
            "kind": "shard",
            "path": str(documents.path.resolve()),
            "tokenizer_hash": metadata["tokenizer_hash"],
            "vocab_size": documents.vocab_size,
            "bos_id": documents.bos_id,
            "identity": documents.identity,
        }
    return {"kind": "inline", "documents": [list(documents[i]) for i in range(len(documents))]}


def load_document_source(source: dict[str, Any]) -> Documents:
    """Rebuild the parent's training documents inside the compile probe child."""
    if source.get("kind") == "shard":
        shard = TokenShard(
            Path(source["path"]),
            tokenizer_hash=source["tokenizer_hash"],
            vocab_size=source["vocab_size"],
            bos_id=source["bos_id"],
        )
        if shard.identity != source["identity"]:
            raise ValueError("compile probe child sees a different training shard")
        return shard
    if source.get("kind") == "inline":
        return [[int(token) for token in document] for document in source["documents"]]
    raise ValueError("unknown compile probe document source")


def run_compile_probe_child(request_path: Path) -> int:
    """Child entry point: train the compiled probe and record each update as it lands.

    One JSON row per completed update is appended and fsynced immediately, so a
    parent that kills this process at the hard window keeps every finished
    update. The child soft-stops at an update boundary shortly before the
    window once it has at least one update, and exits 1 after recording a failure.
    """
    request = json.loads(request_path.read_text())
    rows_path, result_path = Path(request["rows_path"]), Path(request["result_path"])
    torch.set_num_threads(int(request["num_threads"]))
    deadline = float(request["launched_unix"]) + float(request["window_seconds"])
    device = training_device(request["device"])
    result: dict[str, Any] = {"torch_version": torch.__version__, "device": str(device)}
    trainer: Trainer | None = None
    code = 1
    try:
        raw = request["config"]
        config = RunConfig(
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            data_mix=raw["data_mix"],
            parent_checkpoint=raw["parent_checkpoint"],
        )
        result["compile"] = config.train.compile
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        trainer = Trainer(
            config,
            MixedLoader(
                {"web": load_document_source(request["documents"])},
                {"web": 1.0},
                sequence_length=config.train.sequence_length,
                batch_size=config.train.batch_size,
                seed=config.train.seed,
            ),
            run_id=request["run_id"],
            run_root=Path(request["run_root"]),
            data_hash=request["data_hash"],
            tokenizer_hash=request["tokenizer_hash"],
            device=str(device),
        )
        current = trainer

        def soft_stop() -> bool:
            return current.step >= 1 and time.time() >= deadline - COMPILE_SOFT_STOP_SECONDS

        with rows_path.open("a", encoding="utf-8") as stream:

            def record(model: Trainer, metrics: dict[str, float]) -> None:
                row = {**metrics, "step": float(model.step)}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            trainer.train(should_stop=soft_stop, on_step=record)
        result.update(
            status="completed" if trainer.step == config.train.steps else "incomplete",
            updates=trainer.step,
        )
        code = 0
    except BaseException as error:
        memory = device.type == "cuda" and _is_cuda_memory_error(error)
        result.update(
            status="oom" if memory else "failed",
            updates=0 if trainer is None else trainer.step,
            error_type=type(error).__name__,
            error=str(sanitize(str(error)))[:500],
        )
    finally:
        cuda = device.type == "cuda"
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if cuda else 0
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if cuda else 0
        atomic_json(result_path, result)
    return code


def _kill_group(process: subprocess.Popen[bytes]) -> str | None:
    """Kill the child and every compile worker it started, reap the child, never raise.

    An exited leader is reaped first. ``killpg`` then reaches any surviving group
    member; it fails with ``ProcessLookupError`` when none is left and, on macOS,
    with ``PermissionError`` when only an unreaped zombie remains. Returns a
    description of anything unexpected instead of raising.
    """
    problem: str | None = None
    try:
        process.poll()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # Nothing left to signal.
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL ends a process.
            problem = "child did not exit after SIGKILL"
    except Exception as error:  # pragma: no cover - defensive; teardown must not raise.
        problem = f"{type(error).__name__}: {error}"
    return problem


def _completed_rows(path: Path) -> list[dict[str, float]]:
    """Rows the child finished, in order; a torn final line from a kill is ignored."""
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if (
            not isinstance(row, dict)
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in row.values()
            )
            or row.get("step") != len(rows) + 1
        ):
            break
        rows.append({key: float(value) for key, value in row.items()})
    return rows


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        result = value.copy()
        for key, item in value.items():
            result[key] = cpu_tree(item)
        return result
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


@dataclass(frozen=True)
class Specification:
    config: RunConfig
    limits: Limits
    microbatches: tuple[int, ...]
    probe_steps: int
    tokens_per_update: int
    learning_rates: tuple[float, ...]
    screen_steps: int
    checkpoint_seconds: float
    expected_tokens_per_second: float | None
    quoted_gpu_hourly: float
    probe_accumulation_steps: int = 2
    memory_headroom_fraction: float = 0.9
    # One extra measurement-only probe with ``torch.compile`` on, at the chosen
    # microbatch, after the eager probes and before the first real update. It never
    # selects the microbatch and never trains an arm; 0 steps disables it.
    compile_probe_steps: int = 0
    compile_probe_seconds: float = 0.0
    compile_probe_warmup_updates: int = 2

    def __post_init__(self) -> None:
        if type(self.probe_accumulation_steps) is not int or self.probe_accumulation_steps < 2:
            # One accumulated microbatch keeps the fp32 gradients resident during the
            # next forward pass, so the probe peak bounds the accumulated training peak.
            raise ValueError("probe_accumulation_steps must be an integer of at least 2")
        if not 0 < finite_number(self.memory_headroom_fraction, "memory headroom") <= 1:
            raise ValueError("memory_headroom_fraction must be in (0, 1]")
        for name in ("compile_probe_steps", "compile_probe_warmup_updates"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        seconds = finite_number(self.compile_probe_seconds, "compile probe seconds")
        if self.compile_probe_steps:
            if self.config.train.compile:
                raise ValueError(
                    "the compile probe compares against eager training; keep compile off"
                )
            if seconds <= 0:
                raise ValueError("an enabled compile probe needs a positive time window")
            if seconds > self.limits.idle_seconds:
                # The budgeted compile window is the only exception to the idle limit;
                # it can never hold an idle GPU longer than the idle limit itself.
                raise ValueError("the compile probe window must not exceed the idle limit")
            if not self.compile_probe_warmup_updates < min(
                self.compile_probe_steps, self.probe_steps
            ):
                raise ValueError("compile probe warmup must leave measured updates in both probes")
        elif seconds != 0:
            raise ValueError("a disabled compile probe must not reserve pod time")
        reserved = self.limits.probe_seconds + seconds + self.limits.finalization_reserve_seconds
        if reserved >= self.limits.pod_seconds:
            raise ValueError(
                "probe, compile probe and finalization windows exceed the pod lifetime"
            )
        if (
            not self.microbatches
            or len(set(self.microbatches)) != len(self.microbatches)
            or type(self.probe_steps) is not int
            or self.probe_steps < 1
            or type(self.screen_steps) is not int
            or not 0 < self.screen_steps < self.config.train.steps
            or not self.learning_rates
            or len(set(self.learning_rates)) != len(self.learning_rates)
            or self.config.parent_checkpoint is not None
            or self.config.train.validation_interval != 0
            or self.config.data_mix != {"web": 1.0}
        ):
            raise ValueError("invalid pilot specification")
        for size in self.microbatches:
            accumulation_steps(size, self.config.train.sequence_length, self.tokens_per_update)
        for rate in self.learning_rates:
            if finite_number(rate, "learning rate") <= 0:
                raise ValueError("learning rates must be positive")
        for name in ("checkpoint_seconds", "quoted_gpu_hourly"):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.expected_tokens_per_second is not None
            and finite_number(self.expected_tokens_per_second, "expected throughput") <= 0
        ):
            raise ValueError(
                "expected throughput must be positive or null for measured probe baseline"
            )
        if self.quoted_gpu_hourly > self.limits.quoted_total_hourly:
            raise ValueError("GPU price exceeds total price")
        if self.checkpoint_seconds >= self.limits.idle_seconds:
            raise ValueError("checkpoint interval must be shorter than idle timeout")

    @classmethod
    def load(cls, path: Path) -> Specification:
        raw = json.loads(path.read_text())
        raw["config"] = RunConfig(
            model=ModelConfig(**raw["config"]["model"]),
            train=TrainConfig(**raw["config"]["train"]),
            data_mix=raw["config"].get("data_mix", {"web": 1.0}),
            parent_checkpoint=raw["config"].get("parent_checkpoint"),
        )
        raw["limits"] = Limits(**raw["limits"])
        raw["microbatches"] = tuple(raw["microbatches"])
        raw["learning_rates"] = tuple(raw["learning_rates"])
        return cls(**raw)


TERMINATION_GUARANTEES = ("provider_enforced", "redundant_best_effort")


def check_readiness(receipt: dict[str, Any], *, now: float, spec: Specification) -> float:
    """Reject unresolved launch gates before model allocation or output creation.

    The receipt must state honestly how termination is enforced. Runpod offers
    no provider-enforced Pod lifetime, so a ``redundant_best_effort`` guarantee
    (in-pod self-destruct timer plus the Mac-side API watchdog plus the worker
    deadline) additionally requires the owner's explicit acceptance of that
    residual risk; it is never upgraded to a hard guarantee by this check.
    """
    required = (
        "owner_confirms_four_runpod_steps",
        "cloud_data_pre_staged",
        "ssh_transfer_tested",
        "external_watchdog_armed",
        "pod_self_destruct_armed",
        "hardware_preflight_passed",
        "full_console_quote_confirmed",
        "staging_storage_cost_bound",
    )
    if any(receipt.get(name) is not True for name in required):
        raise ValueError("launch readiness gates are unresolved")
    check_termination_guarantee(receipt)
    if spec.limits.pod_seconds > 5400 or spec.limits.idle_seconds > 600:
        raise ValueError("specification exceeds the pilot wall or idle ceiling")
    origin, termination = termination_window(receipt, now=now, limits=spec.limits)
    # The worker plans against the earlier of the wall and the armed in-pod deadline.
    remaining = min(origin + spec.limits.pod_seconds, termination) - now
    needed = (
        spec.limits.probe_seconds
        + spec.compile_probe_seconds
        + spec.limits.finalization_reserve_seconds
    )
    if remaining <= needed:
        raise ValueError("pod lifetime left is too short for probes and finalization")
    return origin


def check_termination_guarantee(receipt: dict[str, Any]) -> None:
    """Require an honest termination guarantee, and owner acceptance of a best-effort one."""
    guarantee = receipt.get("termination_guarantee")
    if guarantee not in TERMINATION_GUARANTEES:
        raise ValueError("termination_guarantee must be provider_enforced or redundant_best_effort")
    if (
        guarantee == "redundant_best_effort"
        and receipt.get("owner_accepts_best_effort_termination") is not True
    ):
        raise ValueError("best-effort termination requires explicit owner acceptance")


def termination_window(
    receipt: dict[str, Any], *, now: float, limits: Limits
) -> tuple[float, float]:
    """Check the receipt's Pod identity, watchdog policy and armed self-destruct deadline.

    Returns the conservative billing origin and the armed in-pod deadline. The policy
    must be at least as strict as ``limits`` and the deadline inside the approved Pod
    lifetime; a Pod that already exceeds a limit is refused.
    """
    identity = PodIdentity.from_dict(receipt.get("pod_identity"))
    policy = Policy(**receipt.get("watchdog_policy", {}))
    if (
        policy.max_wall_seconds > limits.pod_seconds
        or policy.max_idle_seconds > limits.idle_seconds
        or policy.max_cost_usd > limits.cost_stop
        or policy.quoted_total_hourly < limits.quoted_total_hourly
        or policy.starting_cost_usd < limits.starting_cost_usd
    ):
        raise ValueError("watchdog policy is weaker than the bound specification")
    if (
        not isinstance(receipt.get("termination_evidence"), str)
        or not receipt["termination_evidence"].strip()
    ):
        raise ValueError("termination evidence is required")
    created = finite_number(receipt.get("pod_created_at_unix"), "pod creation timestamp")
    if created != parse_timestamp(identity.created_at):
        raise ValueError("pod timestamp differs from captured provider identity")
    origin = policy.billing_started_at_unix
    if origin is None or not 0 < origin <= created + CLOCK_SKEW_SECONDS:
        raise ValueError("conservative create-request billing origin is required")
    origin = min(origin, created)  # A fast Mac clock never shortens the accounted lifetime.
    termination = finite_number(
        receipt.get("self_destruct_deadline_unix"), "self-destruct deadline"
    )
    if not now < termination <= origin + limits.pod_seconds:
        raise ValueError("self-destruct deadline is outside the approved pod lifetime")
    if created <= 0 or created > now:
        raise ValueError("pod creation timestamp is invalid")
    if limits.stop_reason(pod_elapsed=now - origin):
        raise ValueError("pod already exceeds an operating limit")
    return origin, termination


def check_preflight(
    preflight: dict[str, Any], *, hardware: dict[str, Any], spec: Specification
) -> None:
    """Require a completed on-device preflight, including the fused-attention check."""
    if (
        preflight.get("status") != "completed"
        or preflight.get("hardware_precision_and_resume_checked") is not True
        or preflight.get("device") != "cuda:0"
        or preflight.get("hardware") != hardware
    ):
        raise ValueError("a matching A40 hardware precision/resume preflight is required")
    fused = preflight.get("fused_attention")
    if spec.config.train.attention == "fused" and (
        not isinstance(fused, dict) or fused.get("status") != "passed"
    ):
        raise ValueError(
            "a fused-attention specification needs the on-device fused attention "
            "equivalence check to have passed in the preflight"
        )


class Worker:
    def __init__(
        self,
        spec: Specification,
        root: Path,
        documents: Documents,
        windows: tuple[ValidationWindow, ...],
        *,
        data_hash: str,
        tokenizer_hash: str,
        device: str,
        pod_created: float,
        clock: Callable[[], float] = time.time,
        data_kind: str = "hash-bound-reviewed-source",
        sleep: Callable[[float], None] = time.sleep,
        compile_probe_command: Sequence[str] | None = None,
        compile_poll_seconds: float = 2.0,
    ) -> None:
        self.spec, self.root, self.documents, self.windows = spec, root, documents, windows
        self.sleep = sleep
        self.compile_probe_command = tuple(
            compile_probe_command or (sys.executable, "-m", "tools.gpu_pilot", "compile-probe")
        )
        if finite_number(compile_poll_seconds, "compile poll seconds") <= 0:
            raise ValueError("compile poll interval must be positive")
        self.compile_poll_seconds = compile_poll_seconds
        self.compile_probe_active = False
        self.data_hash, self.tokenizer_hash = data_hash, tokenizer_hash
        self.device = training_device(device)
        if not windows:
            raise ValueError("fixed held-out windows are required")
        self.validation_identity = {
            "windows": len(windows),
            "target_tokens": sum(len(window.tokens) - 1 for window in windows),
            "target_bytes": sum(window.target_bytes for window in windows),
            "sha256": hashlib.sha256(
                json.dumps([asdict(window) for window in windows], sort_keys=True).encode()
            ).hexdigest(),
        }
        if root.exists():
            raise FileExistsError("pilot output must be a new directory")
        self.clock, self.pod_created = clock, pod_created
        self.probe_started: float | None = None
        self.compile_probe_started: float | None = None
        self.training_started: float | None = None
        self.training_deadline: float | None = None
        self.tokens_processed = 0
        self.last_activity = clock()
        self.stop_reason: str | None = None
        self.speed_reference = spec.expected_tokens_per_second
        self.phase = "initializing"
        self.save_label = "initializing"  # Checkpoint directory; differs from phase in Phase C.
        self.rows: dict[str, list[dict[str, float]]] = {}
        self.checkpoints: list[dict[str, Any]] = []
        self.last_saved: dict[str, float] = {}
        self.export_count = 0
        self.rising = 0
        self.previous_loss: float | None = None
        commit, dirty = git_state()
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "created_at": utc_now(),
            "git_commit": commit,
            "dirty": dirty,
            "specification": asdict(spec),
            "device": str(self.device),
            "data_kind": data_kind,
            "binding_sha256": data_hash,
            "tokenizer_identity": tokenizer_hash,
            "validation_identity": self.validation_identity,
            "probes": [],
            "probe_window_expired": False,
            "compile_probe": None,
            "arms": [],
            "training_tokens_processed": 0,
            "checkpoints": self.checkpoints,
            "deadline_semantics": (
                "probe window counts from the first probe; the training deadline is the "
                "earlier of the training budget and the pod wall minus the finalization "
                "reserve; wall, cost and idle limits are never extended"
            ),
            "shutdown_receipt": None,
            "durable_copy_receipt": None,
            "actual_provider_cost": None,
            "limitations": [
                "worker time checks cannot terminate a provider resource during an API outage",
                "allocator peaks exclude allocations owned by other processes and the CUDA driver",
                "quoted-rate costs are estimates until reconciled with provider billing",
            ],
        }
        root.mkdir(parents=True)
        self.publish()

    def publish(self) -> None:
        self.report["loss_curves"] = self.rows
        atomic_json(self.root / "report.json", self.report)

    def heartbeat(self, step: int) -> None:
        atomic_json(
            self.root / "heartbeat.json",
            {
                "schema_version": 1,
                "updated_at_unix": self.clock(),
                "last_gpu_activity_unix": self.last_activity,
                "phase": self.phase,
                "step": step,
                "training_tokens_processed": self.tokens_processed,
                # Budgeted compile-probe time, bounded by a hard kill at its window.
                "compile_probe_active": self.compile_probe_active,
                "status": "awaiting_transfer"
                if self.report["status"] in {"completed", "time_limited"}
                else self.report["status"],
            },
        )

    def stopped(self) -> bool:
        now = self.clock()
        reason = self.spec.limits.stop_reason(
            pod_elapsed=max(0, now - self.pod_created),
            idle_elapsed=max(0, now - self.last_activity),
        )
        if reason:
            self.stop_reason = reason
        elif (
            self.phase.startswith("probe")
            and not self.phase.startswith(COMPILE_PROBE_PREFIX)
            and self.probe_started is not None
            and now - self.probe_started >= self.spec.limits.probe_seconds
        ):
            # Soft eager-probe window; the compile probe has its own hard window.
            self.stop_reason = "probe_time_limit"
        elif (
            self.training_started is not None
            and self.training_deadline is not None
            and (now >= self.training_deadline)
        ):
            self.stop_reason = (
                "training_time_limit"
                if self.training_deadline
                == self.training_started + self.spec.limits.training_seconds
                else "finalization_reserve"
            )
        return self.stop_reason is not None

    def require_time(self) -> None:
        if self.stopped():
            raise PilotStop(str(self.stop_reason))

    def trainer(self, name: str, size: int, rate: float, *, probe: bool = False) -> Trainer:
        config = trainer_config(self.spec, size, rate, probe=probe)
        return Trainer(
            config,
            MixedLoader(
                {"web": self.documents},
                {"web": 1.0},
                sequence_length=config.train.sequence_length,
                batch_size=size,
                seed=config.train.seed,
            ),
            run_id=name,
            run_root=self.root,
            data_hash=self.data_hash,
            tokenizer_hash=self.tokenizer_hash,
            device=str(self.device),
        )

    def peak_budget_bytes(self) -> float | None:
        """Largest reserved probe peak accepted for training; None on CPU."""
        if self.device.type != "cuda":
            return None
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        return total * self.spec.memory_headroom_fraction

    def reset_peak(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self) -> dict[str, int]:
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
            if self.device.type == "cuda"
            else 0,
        }

    def release(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def save(self, trainer: Trainer, name: str, *, export: bool = True) -> Path:
        self.require_time()
        path = self.root / name / "last.pt"
        started = synchronized_time(self.device)
        trainer.checkpoint(path)
        seconds = synchronized_time(self.device) - started
        digest = file_sha256(path)
        atomic_json(
            path.with_name(path.name + ".sha256.json"),
            {"sha256": digest, "size_bytes": path.stat().st_size, "step": trainer.step},
        )
        self.checkpoints.append(
            {
                "arm": name,
                "step": trainer.step,
                "path": str(path.relative_to(self.root)),
                "save_seconds": seconds,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "training_tokens_processed": self.tokens_processed,
            }
        )
        self.last_saved[name] = self.clock()
        if hasattr(trainer.loader, "positions"):
            self.checkpoints[-1]["documents_consumed"] = dict(trainer.loader.positions)
        self.publish()
        if export:
            self.export(path, name, trainer.step)
        return path

    def export(self, checkpoint: Path, name: str, step: int) -> None:
        """Publish an immutable transfer bundle without duplicating checkpoint bytes."""
        self.export_count += 1
        directory = self.root / "exports" / f"{name}-step{step}-{self.export_count}"
        staging = directory.with_name("." + directory.name)
        staging.mkdir(parents=True, exist_ok=False)
        try:
            os.link(checkpoint, staging / "checkpoint.pt")
        except OSError:
            # Filesystems without hard links (some network mounts) get a byte copy.
            shutil.copyfile(checkpoint, staging / "checkpoint.pt")
            self.report["export_copy_mode"] = "copy"
        else:
            self.report.setdefault("export_copy_mode", "hard-link")
        shutil.copyfile(self.root / "updates.jsonl", staging / "training.jsonl")
        self.report["latest_transfer_bundle"] = str(directory.relative_to(self.root))
        atomic_json(staging / "record.json", self.report)
        staging.replace(directory)
        self.publish()
        # Keep the newest previous bundle so a pull in progress can finish; older
        # bundles for the same label are removed (hard links keep this cheap).
        previous = sorted(
            (
                path
                for path in directory.parent.glob(f"{name}-step*")
                if path != directory and path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        for stale in previous[:-1]:
            shutil.rmtree(stale)

    def on_step(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        row = {**metrics, "step": float(trainer.step)}
        if any(not math.isfinite(value) for value in row.values()):
            raise FloatingPointError("nonfinite pilot metric")
        self.rows.setdefault(self.phase, []).append(row)
        self.last_activity = self.clock()
        with (self.root / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"arm": self.phase, **row}, allow_nan=False) + "\n")
            stream.flush()
        if not self.phase.startswith("probe"):
            self.tokens_processed += int(metrics["tokens"])
            self.report["training_tokens_processed"] = self.tokens_processed
        self.heartbeat(trainer.step)
        if self.phase.startswith("probe"):
            return
        loss = metrics["loss"]
        self.rising = (
            self.rising + 1 if self.previous_loss is not None and loss > self.previous_loss else 0
        )
        self.previous_loss = loss
        if self.rising >= self.spec.limits.rising_steps:
            self.stop_reason = "loss_rising"
            raise PilotStop(self.stop_reason)
        assert self.training_started is not None
        if self.clock() - self.training_started >= self.spec.limits.speed_check_seconds:
            measured = [
                row
                for name, rows in self.rows.items()
                if not name.startswith("probe")
                for row in rows
            ]
            # Arm step numbers restart; throughput requires a monotonically numbered view.
            numbered = [{**row, "step": float(i + 1)} for i, row in enumerate(measured)]
            if throughput(numbered)["tokens_per_second"] < (
                float(self.speed_reference or 0) * self.spec.limits.minimum_speed_fraction
            ):
                self.stop_reason = "throughput_below_limit"
                raise PilotStop(self.stop_reason)
        if (
            not self.stopped()
            and self.clock() - self.last_saved.get(self.save_label, self.training_started)
            >= self.spec.checkpoint_seconds
        ):
            self.save(trainer, self.save_label)

    def probe(self) -> int:
        """Measure each candidate microbatch; the window counts from the first probe.

        When the probe window expires, the running probe is recorded as
        ``incomplete`` and the remaining sizes as ``skipped``; the pilot then
        continues with the largest completed probe rather than aborting. Wall,
        cost and idle limits still stop the pilot outright.
        """
        self.probe_started = self.clock()
        self.report["probe_started_at_unix"] = self.probe_started
        for size in self.spec.microbatches:
            self.phase = f"probe-{size}"
            if self.stopped():
                if self.stop_reason != "probe_time_limit":
                    raise PilotStop(str(self.stop_reason))
                self.report["probe_window_expired"] = True
                self.report["probes"].append(
                    {"microbatch": size, "status": "skipped", "updates": 0}
                )
                continue
            self.reset_peak()
            trainer: Trainer | None = None
            row: dict[str, Any] = {"microbatch": size}
            try:
                trainer = self.trainer(self.phase, size, self.spec.learning_rates[0], probe=True)
                trainer.train(should_stop=self.stopped, on_step=self.on_step)
                self.stopped()
                if self.stop_reason is not None and self.stop_reason != "probe_time_limit":
                    raise PilotStop(str(self.stop_reason))
                if trainer.step == self.spec.probe_steps:
                    row.update(
                        status="completed",
                        updates=trainer.step,
                        **throughput(self.rows[self.phase]),
                    )
                else:
                    row.update(status="incomplete", updates=trainer.step)
                if self.stop_reason == "probe_time_limit":
                    self.report["probe_window_expired"] = True
            except TrainingStop:
                raise  # A recorded safety stop is never an allocation failure.
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if self.device.type != "cuda" or not _is_cuda_memory_error(error):
                    raise
                row.update(
                    status="oom",
                    updates=0 if trainer is None else trainer.step,
                    error_type=type(error).__name__,
                )
            finally:
                row.update(self.peak())
                self.report["probes"].append(row)
                self.publish()
                del trainer
                self.release()
        if self.stop_reason == "probe_time_limit":
            self.stop_reason = None  # Soft limit: select among measured probes.
        size = choose_microbatch(
            self.report["probes"],
            self.spec.microbatches,
            self.spec.probe_steps,
            max_peak_bytes=self.peak_budget_bytes(),
        )
        self.report["chosen_microbatch"] = size
        self.report["accumulation_steps"] = accumulation_steps(
            size, self.spec.config.train.sequence_length, self.spec.tokens_per_update
        )
        selected = next(row for row in self.report["probes"] if row["microbatch"] == size)
        if self.speed_reference is None:
            self.speed_reference = float(selected["tokens_per_second"])
        self.report["speed_reference_tokens_per_second"] = self.speed_reference
        self.report["speed_reference_basis"] = (
            "explicit prelaunch estimate"
            if self.spec.expected_tokens_per_second is not None
            else "measured chosen-microbatch probe, before accumulated training"
        )
        self.report["chosen_probe_quoted_gpu_cost_per_billion_tokens"] = cost_per_billion(
            float(selected["tokens_per_second"]), self.spec.quoted_gpu_hourly
        )
        self.publish()  # The complete probe table is durable before the first real update.
        print(json.dumps({"chosen_microbatch": size, "probes": self.report["probes"]}), flush=True)
        return size

    def compile_probe(self, size: int) -> None:
        """Time ``torch.compile`` once at the chosen microbatch, in a child process.

        Measurement only: the row is recorded under ``compile_probe``, separate
        from the eager probe table, and never influences the microbatch, the
        learning-rate arms or the continuation. The parent never calls
        ``torch.compile``, so no compiled graph, guard or compile worker reaches
        the arms. The child rebuilds the eager probe's exact trainer with compile
        on (same microbatch, accumulation, seed, data order and attention kernel)
        and records each finished update immediately. The window
        (``compile_probe_seconds``) is a hard limit: the parent kills the child's
        whole process group when it expires, keeping every finished update.
        While the child runs inside its window the parent refreshes the idle
        clock and heartbeat, because that time is budgeted and bounded; wall and
        cost limits still stop the pilot. A failure is recorded and the pilot
        continues eagerly.
        """
        steps = self.spec.compile_probe_steps
        if steps == 0:
            self.report["compile_probe"] = {"status": "disabled"}
            self.publish()
            return
        warmup = self.spec.compile_probe_warmup_updates
        self.phase = COMPILE_PROBE_PREFIX + str(size)
        row: dict[str, Any] = {
            "microbatch": size,
            "accumulation_steps": self.spec.probe_accumulation_steps,
            "steps_requested": steps,
            "warmup_updates_excluded": warmup,
            "window_seconds": self.spec.compile_probe_seconds,
            "compile_mode": "torch.compile(model, dynamic=True), default backend",
            "isolation": "child process; whole process group killed at the window",
            "scope": (
                "measurement only: not used for microbatch choice, learning-rate arms, "
                "continuation or checkpoints; the pilot trains eagerly"
            ),
        }
        # Only the hard limits apply here: the eager probe window is soft and over.
        now = self.clock()
        reason = self.spec.limits.stop_reason(
            pod_elapsed=max(0, now - self.pod_created),
            idle_elapsed=max(0, now - self.last_activity),
        )
        if reason:
            self.stop_reason = reason
            raise PilotStop(reason)
        directory = self.root / "compile-probe"
        directory.mkdir()
        rows_path, result_path = directory / "rows.jsonl", directory / "result.json"
        request = {
            "schema_version": 1,
            "config": asdict(
                trainer_config(
                    self.spec, size, self.spec.learning_rates[0], probe=True, compiled=True
                )
            ),
            "documents": document_source(self.documents),
            "run_id": self.phase,
            "run_root": str(directory / "run"),
            "data_hash": self.data_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "device": str(self.device),
            "num_threads": torch.get_num_threads(),
            "launched_unix": time.time(),
            "window_seconds": self.spec.compile_probe_seconds,
            "rows_path": str(rows_path),
            "result_path": str(result_path),
        }
        atomic_json(directory / "request.json", request)
        environment = {
            key: value for key, value in os.environ.items() if key not in COMPILE_CHILD_SECRETS
        }
        started = self.compile_probe_started = self.clock()
        process: subprocess.Popen[bytes] | None = None
        outcome = "window"
        try:
            with (
                (directory / "child.stdout").open("wb") as out,
                (directory / "child.stderr").open("wb") as err,
            ):
                process = subprocess.Popen(
                    [*self.compile_probe_command, "--request", str(directory / "request.json")],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                self.compile_probe_active = True
                while True:
                    if process.poll() is not None:
                        outcome = "exited"
                        break
                    now = self.clock()
                    reason = self.spec.limits.stop_reason(
                        pod_elapsed=max(0, now - self.pod_created)
                    )
                    if reason:
                        self.stop_reason = reason
                        row.update(status="stopped", stop_reason=reason)
                        raise PilotStop(reason)
                    if now - started >= self.spec.compile_probe_seconds:
                        break
                    self.last_activity = now  # Budgeted, bounded by the hard window.
                    self.heartbeat(len(_completed_rows(rows_path)))
                    self.sleep(self.compile_poll_seconds)
            problem = _kill_group(process)  # Also ends compile workers left by a clean exit.
            if problem:
                row["kill_error"] = problem
            compiled_rows = _completed_rows(rows_path)
            self.rows[self.phase] = compiled_rows
            with (self.root / "updates.jsonl").open("a", encoding="utf-8") as stream:
                for item in compiled_rows:
                    stream.write(json.dumps({"arm": self.phase, **item}, allow_nan=False) + "\n")
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            row["updates"] = len(compiled_rows)
            row["exit_code"] = process.returncode if outcome == "exited" else None
            for key in ("peak_allocated_bytes", "peak_reserved_bytes", "torch_version", "compile"):
                row[key] = result.get(key)
            if compiled_rows:
                row["first_update_seconds"] = compiled_rows[0]["step_seconds"]
            crashed = outcome == "exited" and process.returncode != 0
            tail = (directory / "child.stderr").read_bytes()[-2000:].decode("utf-8", "replace")
            if len(compiled_rows) == steps and result.get("status") == "completed":
                # Durable evidence first: every row and the completed result are on disk.
                row["status"] = "completed"
                row.update(self._compile_comparison(size, warmup))
                if crashed:
                    row["teardown_error"] = {
                        "exit_code": process.returncode,
                        "stderr_tail": str(sanitize(tail))[-500:],
                    }
            elif crashed:
                status = result.get("status")
                row.update(
                    status="oom" if status == "oom" else "failed",
                    error_type=result.get("error_type", "ChildExit"),
                    # The child's exception message (its head) when it recorded one,
                    # otherwise the tail of its standard error.
                    error=str(sanitize(result["error"]))[:500]
                    if result.get("error")
                    else str(sanitize(tail))[-500:],
                    error_source="child exception" if result.get("error") else "stderr tail",
                )
            elif compiled_rows:
                row["status"] = "incomplete"
            elif outcome == "exited":
                row.update(
                    status="failed",
                    error_type="NoUpdates",
                    error="the child exited cleanly without recording an update",
                )
            else:
                row["status"] = "timeout"
                row["note"] = (
                    "child startup plus the first compiled update did not finish within "
                    "the window, so compilation and startup together exceed window_seconds"
                )
        except TrainingStop:
            row.setdefault("status", "stopped")
            raise  # A recorded safety stop is never a compiler failure.
        except Exception as error:
            row.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(sanitize(str(error)))[:500],
            )
        finally:
            if process is not None:
                problem = _kill_group(process)  # Never raises.
                if problem:
                    row.setdefault("kill_error", problem)
            row.setdefault("status", "interrupted")
            row.setdefault("updates", len(self.rows.get(self.phase, [])))
            row["elapsed_seconds"] = max(0.0, self.clock() - started)
            self.compile_probe_active = False
            self.last_activity = self.clock()  # The first arm never inherits a stale idle clock.
            self.report["compile_probe"] = row
            self.publish()
            self.heartbeat(0)
        print(json.dumps({"compile_probe": row}), flush=True)

    def _compile_comparison(self, size: int, warmup: int) -> dict[str, Any]:
        """Steady-state speedup and one-off compile cost against the eager probe."""
        compiled_rows = self.rows[self.phase]
        eager_rows = self.rows[f"probe-{size}"]
        compiled = throughput(compiled_rows, warmup_steps=warmup)
        eager = throughput(eager_rows, warmup_steps=warmup)
        compiled_mean = compiled["step_seconds"] / (len(compiled_rows) - warmup)
        eager_mean = eager["step_seconds"] / (len(eager_rows) - warmup)
        # Time spent above each probe's own steady state during its warmup updates;
        # the difference estimates what compilation costs once.
        compiled_excess = math.fsum(r["step_seconds"] for r in compiled_rows[:warmup]) - (
            warmup * compiled_mean
        )
        eager_excess = math.fsum(r["step_seconds"] for r in eager_rows[:warmup]) - (
            warmup * eager_mean
        )
        one_off = compiled_excess - eager_excess
        saving = eager_mean - compiled_mean
        break_even = max(0.0, one_off) / saving if saving > 0 else None
        tokens_per_probe_update = compiled_rows[0]["tokens"]
        tokens = None if break_even is None else break_even * tokens_per_probe_update
        return {
            "tokens_per_second": compiled["tokens_per_second"],
            "eager_tokens_per_second": eager["tokens_per_second"],
            "speedup": compiled["tokens_per_second"] / eager["tokens_per_second"],
            "tokens_per_probe_update": tokens_per_probe_update,
            "steady_probe_update_seconds": compiled_mean,
            "eager_steady_probe_update_seconds": eager_mean,
            "first_update_seconds": compiled_rows[0]["step_seconds"],
            "eager_first_update_seconds": eager_rows[0]["step_seconds"],
            "one_off_compile_seconds_estimate": one_off,
            "break_even_probe_updates": break_even,
            "break_even_tokens": tokens,
            "break_even_training_updates": None
            if tokens is None
            else tokens / self.spec.tokens_per_update,
            "comparison_basis": (
                "same microbatch, accumulation, seed and data order as the eager probe; "
                "both exclude the same warmup updates; the child's process start, torch "
                "import, CUDA context and model build happen before its first timed update "
                "and count only in elapsed_seconds; its first update carries compilation "
                "plus per-process library warmup"
            ),
        }

    def verify_checkpoint(self, checkpoint: Path) -> str:
        """Reject a checkpoint whose bytes differ from its recorded save identity."""
        sidecar = checkpoint.with_name(checkpoint.name + ".sha256.json")
        recorded = json.loads(sidecar.read_text())
        actual = file_sha256(checkpoint)
        if (
            recorded.get("sha256") != actual
            or recorded.get("size_bytes") != checkpoint.stat().st_size
            or not any(row["sha256"] == actual for row in self.checkpoints)
        ):
            raise ValueError("checkpoint bytes differ from the recorded save identity")
        return actual

    def resume_test(
        self, trainer: Trainer, name: str, checkpoint: Path, size: int, rate: float
    ) -> None:
        """Restore the final checkpoint into a fresh trainer and compare every state.

        A separate trainer keeps the continuation's own run record intact: its
        provenance still names the screen checkpoint it resumed from.
        """
        self.require_time()
        digest = self.verify_checkpoint(checkpoint)
        before = validate(trainer.model, self.windows)
        expected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = self.trainer(name + "-resume-check", size, rate)
        torch.manual_seed(999)  # Deliberately unrelated RNG position before the restore.
        restored.resume(checkpoint, expected_sha256=digest)
        after = validate(restored.model, self.windows)
        self.last_activity = self.clock()
        assert_identical(after, before)
        assert_identical(cpu_tree(restored.model.state_dict()), expected["model"])
        assert_identical(cpu_tree(restored.optimizer.state_dict()), expected["optimizer"])
        assert_identical(restored.loader.state_dict(), expected["rank_states"][0]["loader"])
        assert_identical(torch.get_rng_state(), expected["rank_states"][0]["rng_cpu"])
        if self.device.type == "cuda":
            assert_identical(torch.cuda.get_rng_state_all(), expected["rank_states"][0]["rng_cuda"])
        if restored.step != expected["step"] or restored.step != trainer.step:
            raise AssertionError("restored checkpoint step differs")
        self.report["last_checkpoint_resume"] = {
            "tested": True,
            "arm": name,
            "step": restored.step,
            "sha256": digest,
            "scope": (
                "fresh trainer restored numerical state, loader, RNG and identical "
                "held-out outputs; no extra training update"
            ),
        }
        del restored
        self.release()
        self.publish()

    def execute(self) -> dict[str, Any]:
        trainer: Trainer | None = None
        final_checkpoint: tuple[Path, str, int] | None = None
        try:
            size = self.probe()
            self.compile_probe(size)
            self.phase = "screen"
            self.training_started = self.clock()
            self.training_deadline = self.spec.limits.training_deadline(
                pod_created=self.pod_created, training_started=self.training_started
            )
            self.report["training_started_at_unix"] = self.training_started
            self.report["training_deadline_unix"] = self.training_deadline
            self.reset_peak()
            scores: list[tuple[float, float, str]] = []
            for i, rate in enumerate(self.spec.learning_rates):
                self.phase = self.save_label = f"lr-{i}"
                self.rising, self.previous_loss = 0, None
                trainer = self.trainer(self.phase, size, rate)
                trainer.train(
                    stop_after=self.spec.screen_steps,
                    should_stop=self.stopped,
                    on_step=self.on_step,
                )
                self.require_time()
                if trainer.step != self.spec.screen_steps:
                    raise PilotStop("incomplete_learning_rate_screen")
                heldout = validate(trainer.model, self.windows)
                path = self.save(trainer, self.phase)
                trainer.finish(success=True)
                self.report["arms"].append(
                    {
                        "name": self.phase,
                        "learning_rate": rate,
                        "validation": heldout,
                        "step": trainer.step,
                        "checkpoint_sha256": file_sha256(path),
                        "training_tokens": trainer.step * self.spec.tokens_per_update,
                    }
                )
                scores.append((heldout["loss"], rate, self.phase))
                del trainer
                trainer = None
                self.release()
            _, rate, name = min(scores)
            self.report["winner"] = name
            self.phase = name  # Loss curves continue under the winning arm's key.
            self.save_label = name + "-continued"  # The step-100 checkpoint stays untouched.
            trainer = self.trainer(self.save_label, size, rate)
            screen_checkpoint = self.root / name / "last.pt"
            trainer.resume(
                screen_checkpoint, expected_sha256=self.verify_checkpoint(screen_checkpoint)
            )
            self.report["continuation"] = {
                "arm": name,
                "resumed_from_step": trainer.step,
                "resumed_checkpoint_sha256": file_sha256(screen_checkpoint),
            }
            self.rising, self.previous_loss = 0, None
            self.last_saved[self.save_label] = self.clock()  # Cadence restarts at resume.
            trainer.train(should_stop=self.stopped, on_step=self.on_step)
            # A training deadline permits final save/resume inside the remaining pod
            # budget; cost, idle and pod deadlines never permit additional work.
            if self.stop_reason in {"training_time_limit", "finalization_reserve"}:
                self.report["training_stop_reason"] = self.stop_reason
                self.training_started = None
                self.training_deadline = None
                self.stop_reason = None
            self.phase = "finalizing"
            # The single final bundle is published after the status is known.
            path = self.save(trainer, self.save_label, export=False)
            trainer.finish(success=True)
            self.resume_test(trainer, self.save_label, path, size, rate)
            self.report["selected_step"] = trainer.step
            self.report["status"] = (
                "time_limited" if trainer.step < self.spec.config.train.steps else "completed"
            )
            final_checkpoint = (path, self.save_label, trainer.step)
        except PilotStop as error:
            self.report.update(status="stopped", stop_reason=str(error))
            # Only previously published whole-update checkpoints are promised on a
            # safety stop; never extend a pod/cost deadline for finalization.
        except BaseException as error:
            self.report.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            self.report["loss_curves"] = self.rows
            training_rows = [
                row
                for name, rows in self.rows.items()
                if not name.startswith("probe")
                for row in rows
            ]
            if training_rows:
                numbered = [{**row, "step": float(i + 1)} for i, row in enumerate(training_rows)]
                measured = throughput(numbered)
                self.report["training_throughput"] = measured
                self.report["quoted_gpu_cost_per_billion_tokens"] = cost_per_billion(
                    measured["tokens_per_second"], self.spec.quoted_gpu_hourly
                )
            self.report["training_peak"] = self.peak()
            self.report["ended_at"] = utc_now()
            self.report["pod_elapsed_seconds"] = max(0, self.clock() - self.pod_created)
            self.report["quoted_accrued_cost_usd"] = (
                self.spec.limits.starting_cost_usd
                + self.report["pod_elapsed_seconds"] / 3600 * self.spec.limits.quoted_total_hourly
            )
            self.publish()
            if final_checkpoint is not None:
                self.export(*final_checkpoint)
            self.heartbeat(0 if trainer is None else trainer.step)
            del trainer
            self.release()
        return self.report


def smoke(root: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    spec = Specification(
        RunConfig(
            model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8),
            train=TrainConfig(steps=5, sequence_length=8, batch_size=2, accumulation_steps=2),
        ),
        Limits(60, 120, 300, 1, 1, 1, 1.2, 0.5, 60, 180, 500),
        (2, 4, 1),
        2,
        32,
        (0.0003, 0.0006, 0.0012),
        2,
        30,
        1,
        1,
        compile_probe_steps=3,
        compile_probe_seconds=120,
        compile_probe_warmup_updates=1,
    )
    return Worker(
        spec,
        root,
        [[256, 1, 2, 3] * 32],
        (ValidationWindow((256, 1, 2, 3), 3),),
        data_hash="authored-smoke",
        tokenizer_hash="authored-smoke",
        device="cpu",
        pod_created=time.time(),
        data_kind="authored-tiny-CPU-only",
    ).execute()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("specification", "readiness", "binding", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--binding-sha256", required=True)
    child = sub.add_parser("compile-probe", help="internal: the compile probe's child process")
    child.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compile-probe":
        raise SystemExit(run_compile_probe_child(args.request))
    if args.command == "smoke":
        report = smoke(args.output)
    else:
        spec = Specification.load(args.specification)
        if spec.config.train.precision != "bfloat16":
            raise ValueError("the A40 pilot specification must use bfloat16 autocast")
        receipt = json.loads(args.readiness.read_text())
        if receipt.get("specification_sha256") != file_sha256(args.specification):
            raise ValueError("specification differs from the operational receipt")
        if receipt.get("binding_sha256") != args.binding_sha256:
            raise ValueError("binding identity differs from the operational receipt")
        created = check_readiness(receipt, now=time.time(), spec=spec)
        if spec.limits.finalization_reserve_seconds < 600:
            raise ValueError("the A40 pilot needs at least a 600-second finalization reserve")
        armed_deadline = finite_number(receipt["self_destruct_deadline_unix"], "deadline")
        if armed_deadline - created < spec.limits.pod_seconds:
            # The in-pod timer fires earlier than the specification's wall; plan against it.
            try:
                spec = replace(
                    spec, limits=replace(spec.limits, pod_seconds=armed_deadline - created)
                )
            except ValueError as error:
                raise ValueError(
                    "the armed self-destruct deadline leaves too little pod lifetime for the "
                    "specification's probe, training and finalization budgets"
                ) from error
        if receipt.get("preflight_sha256") != file_sha256(args.preflight):
            raise ValueError("preflight report differs from bound receipt")
        preflight = json.loads(args.preflight.read_text())
        hardware = hardware_description(training_device("cuda:0"))
        check_preflight(preflight, hardware=hardware, spec=spec)
        if file_sha256(args.binding) != args.binding_sha256:
            raise ValueError("data binding differs from externally recorded SHA-256")
        binding = verify_binding(
            args.binding,
            sequence_length=spec.config.train.sequence_length,
            validation_documents=spec.config.train.validation_documents,
        )
        directory = args.binding.parent
        tokenizer = ByteBPE.load(directory / "tokenizer.json")
        if spec.config.model.vocab_size != tokenizer.vocab_size:
            raise ValueError("model vocabulary differs from bound tokenizer")
        documents = TokenShard(
            directory / "train.bin",
            tokenizer_hash=tokenizer.sha256,
            vocab_size=tokenizer.vocab_size,
            bos_id=tokenizer.bos_id,
        )
        heldout = TokenShard(
            directory / "validation.bin",
            tokenizer_hash=tokenizer.sha256,
            vocab_size=tokenizer.vocab_size,
            bos_id=tokenizer.bos_id,
        )
        if spec.config.train.validation_documents != binding["validation"]["documents"]:
            raise ValueError("the specification must validate on every bound held-out document")
        if not 0 < spec.config.train.warmup_steps < spec.screen_steps:
            raise ValueError("warmup must finish inside the learning-rate screen")
        windows = shard_windows(
            heldout,
            tokenizer,
            documents=spec.config.train.validation_documents,
            sequence_length=spec.config.train.sequence_length,
        )
        if len(windows) != spec.config.train.validation_documents:
            raise ValueError("validation shard holds fewer documents than specified")
        train_tokens = int(binding["train"]["tokens_including_bos"])
        arm_tokens = spec.screen_steps * spec.tokens_per_update
        winner_tokens = spec.config.train.steps * spec.tokens_per_update
        exposure = {
            "train_tokens_including_bos": train_tokens,
            "tokens_per_screen_arm": arm_tokens,
            "tokens_for_winner_at_full_run": winner_tokens,
            "epochs_for_winner_at_full_run": winner_tokens / train_tokens,
            "aggregate_training_tokens_maximum": (
                len(spec.learning_rates) * arm_tokens + winner_tokens - arm_tokens
            ),
            "repetition": (
                "none within one pass; a second pass would repeat the same seeded "
                "document permutation"
                if winner_tokens <= train_tokens
                else "the winner continuation wraps and repeats the seeded permutation"
            ),
        }
        worker = Worker(
            spec,
            args.output,
            documents,
            windows,
            data_hash=args.binding_sha256,
            tokenizer_hash=binding["tokenizer"]["identity"],
            device="cuda:0",
            pod_created=created,
        )
        worker.report["hardware"] = hardware
        worker.report["readiness_receipt"] = sanitize(receipt)
        worker.report["leakage_audit"] = binding.get("leakage_audit")
        worker.report["data_exposure"] = exposure
        worker.publish()
        worker.report["preflight_sha256"] = file_sha256(args.preflight)
        report = worker.execute()
    print(f"Pilot {report['status']}: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
