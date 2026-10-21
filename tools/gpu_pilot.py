"""Bounded pilot worker on an existing machine; never provisions compute.

The smoke command uses authored tiny data on CPU. The run command additionally
requires externally bound data and an operational receipt. A receipt documents
readiness; this worker cannot enforce a provider's resource lifetime.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig, finite_number
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
from scratchcoder.train.trainer import Trainer, synchronized_time, training_device
from scratchcoder.train.validation import ValidationWindow, shard_windows, validate
from tools.cpu_resume_check import assert_identical
from tools.pilot_data import verify_binding
from tools.pilot_preflight import hardware_description
from tools.runpod_watchdog import PodIdentity, Policy, parse_timestamp


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

    def __post_init__(self) -> None:
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


def check_readiness(receipt: dict[str, Any], *, now: float, spec: Specification) -> float:
    """Reject unresolved launch gates before model allocation or output creation."""
    required = (
        "owner_confirms_four_runpod_steps",
        "cloud_data_pre_staged",
        "ssh_transfer_tested",
        "external_watchdog_armed",
        "provider_hard_deadline_confirmed",
        "hardware_preflight_passed",
        "full_console_quote_confirmed",
        "staging_storage_cost_bound",
    )
    if any(receipt.get(name) is not True for name in required):
        raise ValueError("launch readiness gates are unresolved")
    if spec.limits.pod_seconds > 5400 or spec.limits.idle_seconds > 600:
        raise ValueError("specification exceeds the pilot wall or idle ceiling")
    identity = PodIdentity.from_dict(receipt.get("pod_identity"))
    policy = Policy(**receipt.get("watchdog_policy", {}))
    if (
        policy.max_wall_seconds > spec.limits.pod_seconds
        or policy.max_idle_seconds > spec.limits.idle_seconds
        or policy.max_cost_usd > spec.limits.cost_stop
        or policy.quoted_total_hourly < spec.limits.quoted_total_hourly
        or policy.starting_cost_usd < spec.limits.starting_cost_usd
    ):
        raise ValueError("watchdog policy is weaker than the bound specification")
    if (
        not isinstance(receipt.get("provider_deadline_evidence"), str)
        or not receipt["provider_deadline_evidence"].strip()
    ):
        raise ValueError("provider deadline evidence is required")
    created = finite_number(receipt.get("pod_created_at_unix"), "pod creation timestamp")
    if created != parse_timestamp(identity.created_at):
        raise ValueError("pod timestamp differs from captured provider identity")
    origin = policy.billing_started_at_unix
    if origin is None or not 0 < origin <= created:
        raise ValueError("conservative create-request billing origin is required")
    termination = finite_number(receipt.get("provider_termination_at_unix"), "provider deadline")
    if not now < termination <= origin + spec.limits.pod_seconds:
        raise ValueError("provider deadline is outside the approved pod lifetime")
    if created <= 0 or created > now:
        raise ValueError("pod creation timestamp is invalid")
    if spec.limits.stop_reason(pod_elapsed=now - origin):
        raise ValueError("pod already exceeds an operating limit")
    if now - origin >= spec.limits.probe_seconds:
        raise ValueError("initial batch-probe window already expired")
    return origin


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
    ) -> None:
        self.spec, self.root, self.documents, self.windows = spec, root, documents, windows
        self.data_hash, self.tokenizer_hash = data_hash, tokenizer_hash
        self.device = training_device(device)
        if not windows:
            raise ValueError("fixed held-out windows are required")
        if root.exists():
            raise FileExistsError("pilot output must be a new directory")
        self.clock, self.pod_created = clock, pod_created
        self.training_started: float | None = None
        self.last_activity = clock()
        self.stop_reason: str | None = None
        self.speed_reference = spec.expected_tokens_per_second
        self.phase = "initializing"
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
            "probes": [],
            "arms": [],
            "checkpoints": self.checkpoints,
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
            and now - self.pod_created >= self.spec.limits.probe_seconds
        ):
            self.stop_reason = "probe_time_limit"
        elif self.training_started is not None and (
            now - self.training_started >= self.spec.limits.training_seconds
        ):
            self.stop_reason = "training_time_limit"
        return self.stop_reason is not None

    def require_time(self) -> None:
        if self.stopped():
            raise PilotStop(str(self.stop_reason))

    def trainer(self, name: str, size: int, rate: float, *, probe: bool = False) -> Trainer:
        self.require_time()
        config = replace(
            self.spec.config,
            train=replace(
                self.spec.config.train,
                batch_size=size,
                accumulation_steps=1
                if probe
                else accumulation_steps(
                    size, self.spec.config.train.sequence_length, self.spec.tokens_per_update
                ),
                learning_rate=rate,
                steps=self.spec.probe_steps if probe else self.spec.config.train.steps,
                warmup_steps=min(self.spec.config.train.warmup_steps, self.spec.probe_steps)
                if probe
                else self.spec.config.train.warmup_steps,
            ),
        )
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

    def save(self, trainer: Trainer, name: str) -> Path:
        self.require_time()
        path = self.root / name / "last.pt"
        started = synchronized_time(self.device)
        trainer.checkpoint(path)
        seconds = synchronized_time(self.device) - started
        self.checkpoints.append(
            {
                "arm": name,
                "step": trainer.step,
                "path": str(path.relative_to(self.root)),
                "save_seconds": seconds,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        self.last_saved[name] = self.clock()
        self.publish()
        self.export(path, name, trainer.step)
        return path

    def export(self, checkpoint: Path, name: str, step: int) -> None:
        """Publish an immutable transfer bundle without duplicating checkpoint bytes."""
        self.export_count += 1
        directory = self.root / "exports" / f"{name}-step{step}-{self.export_count}"
        staging = directory.with_name("." + directory.name)
        staging.mkdir(parents=True, exist_ok=False)
        os.link(checkpoint, staging / "checkpoint.pt")
        shutil.copyfile(self.root / "updates.jsonl", staging / "training.jsonl")
        self.report["latest_transfer_bundle"] = str(directory.relative_to(self.root))
        atomic_json(staging / "record.json", self.report)
        staging.replace(directory)
        self.publish()
        for previous in directory.parent.glob(f"{name}-step*"):
            if previous != directory and previous.is_dir() and not previous.is_symlink():
                shutil.rmtree(previous)

    def on_step(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        row = {**metrics, "step": float(trainer.step)}
        if any(not math.isfinite(value) for value in row.values()):
            raise FloatingPointError("nonfinite pilot metric")
        self.rows.setdefault(self.phase, []).append(row)
        self.last_activity = self.clock()
        with (self.root / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"arm": self.phase, **row}, allow_nan=False) + "\n")
            stream.flush()
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
            and self.clock() - self.last_saved.get(self.phase, self.training_started)
            >= self.spec.checkpoint_seconds
        ):
            self.save(trainer, self.phase)

    def probe(self) -> int:
        for size in self.spec.microbatches:
            self.phase = f"probe-{size}"
            self.require_time()
            self.reset_peak()
            trainer: Trainer | None = None
            row: dict[str, Any] = {"microbatch": size}
            try:
                trainer = self.trainer(self.phase, size, self.spec.learning_rates[0], probe=True)
                trainer.train(should_stop=self.stopped, on_step=self.on_step)
                self.require_time()
                row.update(
                    status="completed", updates=trainer.step, **throughput(self.rows[self.phase])
                )
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda":
                    raise
                row.update(status="oom", updates=0 if trainer is None else trainer.step)
            finally:
                row.update(self.peak())
                self.report["probes"].append(row)
                self.publish()
                del trainer
                self.release()
        size = choose_microbatch(
            self.report["probes"], self.spec.microbatches, self.spec.probe_steps
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

    def resume_test(self, trainer: Trainer, name: str, checkpoint: Path) -> None:
        """Restore the actual last checkpoint, compare all state and held-out outputs."""
        self.require_time()
        before = validate(trainer.model, self.windows)
        expected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        trainer.resume(checkpoint)
        after = validate(trainer.model, self.windows)
        assert_identical(after, before)
        assert_identical(cpu_tree(trainer.model.state_dict()), expected["model"])
        assert_identical(cpu_tree(trainer.optimizer.state_dict()), expected["optimizer"])
        assert_identical(trainer.loader.state_dict(), expected["rank_states"][0]["loader"])
        assert_identical(torch.get_rng_state(), expected["rank_states"][0]["rng_cpu"])
        if self.device.type == "cuda":
            assert_identical(torch.cuda.get_rng_state_all(), expected["rank_states"][0]["rng_cuda"])
        if trainer.step != expected["step"]:
            raise AssertionError("restored checkpoint step differs")
        self.report["last_checkpoint_resume"] = {
            "tested": True,
            "arm": name,
            "step": trainer.step,
            "sha256": file_sha256(checkpoint),
            "scope": (
                "restored numerical state, loader, RNG and identical held-out outputs; "
                "no extra training update"
            ),
        }
        self.publish()

    def execute(self) -> dict[str, Any]:
        trainer: Trainer | None = None
        final_checkpoint: tuple[Path, str, int] | None = None
        try:
            size = self.probe()
            self.phase = "screen"
            self.training_started = self.clock()
            self.reset_peak()
            scores: list[tuple[float, float, str]] = []
            for i, rate in enumerate(self.spec.learning_rates):
                self.phase = f"lr-{i}"
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
                self.report["arms"].append(
                    {
                        "name": self.phase,
                        "learning_rate": rate,
                        "validation": heldout,
                        "step": trainer.step,
                    }
                )
                scores.append((heldout["loss"], rate, self.phase))
                del trainer
                trainer = None
                self.release()
            _, rate, name = min(scores)
            self.report["winner"] = name
            self.phase = name
            trainer = self.trainer(name + "-continued", size, rate)
            trainer.resume(self.root / name / "last.pt")
            self.rising, self.previous_loss = 0, None
            trainer.train(should_stop=self.stopped, on_step=self.on_step)
            # A training deadline permits final save/resume inside the remaining pod
            # budget; cost, idle and pod deadlines never permit additional work.
            if self.stop_reason == "training_time_limit":
                self.training_started = None
                self.stop_reason = None
            self.phase = "finalizing"
            path = self.save(trainer, name)
            self.resume_test(trainer, name, path)
            self.report["selected_step"] = trainer.step
            self.report["status"] = (
                "time_limited" if trainer.step < self.spec.config.train.steps else "completed"
            )
            final_checkpoint = (path, name, trainer.step)
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
    args = parser.parse_args(argv)
    if args.command == "smoke":
        report = smoke(args.output)
    else:
        spec = Specification.load(args.specification)
        receipt = json.loads(args.readiness.read_text())
        if receipt.get("specification_sha256") != file_sha256(args.specification):
            raise ValueError("specification differs from the operational receipt")
        if receipt.get("binding_sha256") != args.binding_sha256:
            raise ValueError("binding identity differs from the operational receipt")
        created = check_readiness(receipt, now=time.time(), spec=spec)
        if receipt.get("preflight_sha256") != file_sha256(args.preflight):
            raise ValueError("preflight report differs from bound receipt")
        preflight = json.loads(args.preflight.read_text())
        hardware = hardware_description(training_device("cuda:0"))
        if (
            preflight.get("status") != "completed"
            or preflight.get("hardware_precision_and_resume_checked") is not True
            or preflight.get("device") != "cuda:0"
            or preflight.get("hardware") != hardware
        ):
            raise ValueError("a matching A40 hardware precision/resume preflight is required")
        if file_sha256(args.binding) != args.binding_sha256:
            raise ValueError("data binding differs from externally recorded SHA-256")
        binding = verify_binding(args.binding)
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
        windows = shard_windows(
            heldout,
            tokenizer,
            documents=spec.config.train.validation_documents,
            sequence_length=spec.config.train.sequence_length,
        )
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
        worker.report["readiness_receipt"] = receipt
        worker.report["preflight_sha256"] = file_sha256(args.preflight)
        report = worker.execute()
    print(f"Pilot {report['status']}: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
