"""One training path for initial pretraining, continuation, and exact resume."""

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from scratchcoder.common.config import RunConfig
from scratchcoder.common.logging import EventLogger, debug_logger
from scratchcoder.common.records import RunRecord, validate_resume_details
from scratchcoder.data.loader import BatchSource
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import (
    AdamW,
    clip_gradients,
    cross_entropy,
    cross_entropy_and_z,
    learning_rate,
)
from scratchcoder.train.validation import ValidationWindow, validate


class TrainingStop(RuntimeError):
    """A deliberate, recorded stop raised from a step callback; not a training failure."""


def training_device(value: str) -> torch.device:
    """Resolve an explicit CPU/CUDA choice without falling back to another device."""
    if value == "cpu":
        return torch.device("cpu")
    if not isinstance(value, str) or re.fullmatch(r"cuda(?::[0-9]+)?", value) is None:
        raise ValueError("training device must be cpu, cuda or cuda:<nonnegative index>")
    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select cpu explicitly to use CPU")
    ordinal = value.partition(":")[2]
    index = int(ordinal) if ordinal else torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device cuda:{index} is not available")
    return torch.device("cuda", index)


def synchronized_time(device: torch.device) -> float:
    """Read wall time after pending work on the selected CUDA device completes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        loader: BatchSource,
        *,
        run_id: str,
        run_root: Path,
        data_hash: str,
        tokenizer_hash: str,
        device: str = "cpu",
        validation: dict[str, tuple[ValidationWindow, ...]] | None = None,
    ) -> None:
        self.config, self.loader = config, loader
        self.validation = dict(validation or {})
        if bool(self.validation) != bool(config.train.validation_interval):
            raise ValueError("validation windows and a positive interval must be supplied together")
        if any(not name or not windows for name, windows in self.validation.items()):
            raise ValueError("validation domains and windows must be nonempty")
        self.validation_hash = hashlib.sha256(
            json.dumps(
                {
                    name: [asdict(window) for window in windows]
                    for name, windows in self.validation.items()
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.device = training_device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        torch.manual_seed(config.train.seed)
        self.model = GPT(config.model).to(self.device)
        self.model.set_fused_attention(config.train.attention == "fused")
        # The compiled wrapper shares parameters with the eager module; checkpoints,
        # validation and generation keep using the eager module.
        self.forward: Callable[[Tensor], Tensor] = (
            torch.compile(self.model, dynamic=True) if config.train.compile else self.model
        )
        parent_run: str | None = None
        if config.parent_checkpoint:
            parent = torch.load(config.parent_checkpoint, map_location="cpu", weights_only=True)
            if parent["config"]["model"] != asdict(config.model):
                raise ValueError("parent checkpoint architecture differs")
            if parent["tokenizer_hash"] != tokenizer_hash:
                raise ValueError("parent checkpoint tokenizer differs")
            self.model.load_state_dict(parent["model"])
            parent_run = parent["run_id"]
        named = list(self.model.named_parameters())
        self.optimizer = AdamW(
            [parameter for _, parameter in named],
            learning_rate=config.train.learning_rate,
            betas=(config.train.beta1, config.train.beta2),
            eps=config.train.adam_epsilon,
            weight_decay=config.train.weight_decay,
            decay_mask=[
                parameter.ndim >= 2 and name != "embedding.weight" for name, parameter in named
            ],
            foreach=config.train.optimizer_foreach,
        )
        self.step = 0
        self.record = RunRecord.start(run_id, config, data_hash, tokenizer_hash, parent_run)
        self.record.hardware["training_device"] = str(self.device)
        self.run_root = run_root
        if self.rank == 0:
            self.record.save(run_root)

    def train(
        self,
        *,
        stop_after: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_step: Callable[["Trainer", dict[str, float]], None] | None = None,
    ) -> list[float]:
        if self.world_size != 1 and (should_stop is not None or on_step is not None):
            raise ValueError("step controls require single-process training")
        target = self.config.train.steps if stop_after is None else stop_after
        if not self.step <= target <= self.config.train.steps:
            raise ValueError("stop_after must be between current and configured steps")
        losses: list[float] = []
        options = self.config.train
        log_path = self.run_root / self.record.run_id / f"events-rank{self.rank}.jsonl"
        with EventLogger(log_path, self.record.run_id) as logger:
            try:
                if self.step == 0:
                    self._log_validation(logger)
                while self.step < target:
                    if should_stop is not None and should_stop():
                        logger.event("training_stopped", step=self.step, partial_update=False)
                        break
                    started = synchronized_time(self.device)
                    loader_before = self.loader.state_dict() if should_stop is not None else None
                    rng_before = torch.get_rng_state() if should_stop is not None else None
                    cuda_before = (
                        torch.cuda.get_rng_state_all()
                        if should_stop is not None and self.device.type == "cuda"
                        else []
                    )
                    self.optimizer.zero_grad()
                    loss_value = 0.0
                    z_value = 0.0
                    interrupted = False
                    for _ in range(options.accumulation_steps):
                        if should_stop is not None and should_stop():
                            interrupted = True
                            break
                        inputs, targets, _ = self.loader.next_batch()
                        x = torch.tensor(inputs, dtype=torch.long, device=self.device)
                        y = torch.tensor(targets, dtype=torch.long, device=self.device)
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=options.precision == "bfloat16",
                        ):
                            logits = self.forward(x)
                            if options.z_loss > 0:
                                nll, z_term = cross_entropy_and_z(logits, y)
                                objective = nll + options.z_loss * z_term
                            else:
                                nll = cross_entropy(logits, y)
                                objective = nll
                                z_term = None
                            loss = objective / options.accumulation_steps
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("nonfinite training loss")
                        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
                        # The logged loss is the next-token loss, comparable with validation;
                        # the z term is reported separately.
                        loss_value += float(nll.detach()) / options.accumulation_steps
                        if z_term is not None:
                            z_value += float(z_term.detach()) / options.accumulation_steps
                    if interrupted or should_stop is not None and should_stop():
                        # No optimizer update has occurred. Restore consumed inputs and
                        # RNG as well as discarding partial gradients, so the checkpoint
                        # resumes at a whole effective-batch boundary.
                        assert loader_before is not None and rng_before is not None
                        self.loader.load_state_dict(loader_before)
                        torch.set_rng_state(rng_before)
                        if self.device.type == "cuda":
                            torch.cuda.set_rng_state_all(cuda_before)
                        self.optimizer.zero_grad()
                        logger.event("training_stopped", step=self.step, partial_update=True)
                        break
                    if self.world_size > 1:
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                dist.all_reduce(parameter.grad)
                                parameter.grad.div_(self.world_size)
                        aggregate = torch.tensor(loss_value, device=self.device)
                        dist.all_reduce(aggregate)
                        loss_value = float(aggregate) / self.world_size
                    norm = clip_gradients(self.model.parameters(), options.gradient_clip)
                    self.optimizer.lr = learning_rate(
                        self.step,
                        total_steps=options.steps,
                        warmup_steps=options.warmup_steps,
                        peak=options.learning_rate,
                        minimum_ratio=options.min_lr_ratio,
                        schedule=options.schedule,
                        decay_fraction=options.decay_fraction,
                    )
                    self.optimizer.step()
                    elapsed = synchronized_time(self.device) - started
                    losses.append(loss_value)
                    self.step += 1
                    tokens = (
                        options.batch_size
                        * options.sequence_length
                        * options.accumulation_steps
                        * self.world_size
                    )
                    metrics = {
                        "loss": loss_value,
                        "learning_rate": self.optimizer.lr,
                        "gradient_norm": norm,
                        "tokens": tokens,
                        "step_seconds": elapsed,
                        "tokens_per_second": tokens / max(elapsed, 1e-9),
                    }
                    if options.z_loss > 0:
                        metrics["z_loss"] = z_value
                    logger.metrics(metrics, step=self.step)
                    if options.validation_interval and (
                        self.step % options.validation_interval == 0 or self.step == options.steps
                    ):
                        self._log_validation(logger)
                    if on_step is not None:
                        on_step(self, metrics)
                if self.step == options.steps:
                    self.record.finish(success=True)
            except TrainingStop as stop:
                logger.event(
                    "training_stopped", step=self.step, partial_update=False, reason=str(stop)
                )
                raise
            except Exception as error:
                logger.event("training_failed", error_type=type(error).__name__, step=self.step)
                diagnostic = debug_logger(log_path.with_name(f"debug-rank{self.rank}.jsonl"))
                try:
                    diagnostic.exception("Training aborted at step %s", self.step)
                finally:
                    for handler in diagnostic.handlers:
                        handler.close()
                self.record.finish(success=False)
                raise
            finally:
                if self.rank == 0:
                    self.record.save(self.run_root)
        return losses

    def _log_validation(self, logger: EventLogger) -> None:
        if self.rank == 0 and self.validation:
            metrics = {
                f"validation/{name}/{metric}": value
                for name, windows in self.validation.items()
                for metric, value in validate(self.model, windows).items()
            }
            logger.event(
                "validation", step=self.step, validation_hash=self.validation_hash, metrics=metrics
            )

    @torch.no_grad()
    def validation_loss(self, inputs: Tensor, targets: Tensor) -> float:
        return float(cross_entropy(self.model(inputs.to(self.device)), targets.to(self.device)))

    def checkpoint(self, path: Path) -> None:
        """All ranks participate; rank zero publishes a complete atomic checkpoint."""
        local = {
            "loader": self.loader.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }
        rank_states: list[Any] = [local]
        if self.world_size > 1:
            rank_states = [None] * self.world_size
            dist.all_gather_object(rank_states, local)
        if self.rank == 0:
            state = {
                "version": 1,
                "config": asdict(self.config),
                "run_id": self.record.run_id,
                "data_hash": self.record.data_manifest_hash,
                "tokenizer_hash": self.record.tokenizer_hash,
                "validation_hash": self.validation_hash,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "rank_states": rank_states,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)  # Make the rename itself durable, not only the bytes.
            finally:
                os.close(directory)
        if self.world_size > 1:
            dist.barrier()

    def finish(self, *, success: bool = True) -> None:
        """Close the run record for a run that ended deliberately before its final step."""
        if self.record.status == "running":
            self.record.finish(success=success)
            if self.rank == 0:
                self.record.save(self.run_root)

    def resume(self, path: Path, *, expected_sha256: str | None = None) -> None:
        """Restore exact training state; ``expected_sha256`` rejects altered bytes first."""
        checkpoint_hash: str | None = None
        try:
            with path.open("rb") as stream:
                checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if expected_sha256 is not None and checkpoint_hash != expected_sha256:
                    raise ValueError("checkpoint bytes differ from the expected identity")
                stream.seek(0)
                state = torch.load(stream, map_location="cpu", weights_only=True)
            self._restore_checkpoint(state, checkpoint_hash)
        except Exception as error:
            self.record.finish(success=False)
            if self.rank == 0:
                self.record.save(self.run_root)
            log_path = self.run_root / self.record.run_id / f"events-rank{self.rank}.jsonl"
            with EventLogger(log_path, self.record.run_id) as logger:
                logger.event(
                    "resume_failed",
                    error_type=type(error).__name__,
                    step=self.step,
                    checkpoint_sha256=checkpoint_hash,
                )
            diagnostic = debug_logger(log_path.with_name(f"debug-rank{self.rank}.jsonl"))
            try:
                diagnostic.exception("Checkpoint resume aborted")
            finally:
                for handler in diagnostic.handlers:
                    handler.close()
            raise

    def _restore_checkpoint(self, state: dict[str, Any], checkpoint_hash: str) -> None:
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 1
            or state.get("config") != asdict(self.config)
        ):
            raise ValueError("checkpoint configuration differs")
        if (
            state["data_hash"] != self.record.data_manifest_hash
            or state["tokenizer_hash"] != self.record.tokenizer_hash
        ):
            raise ValueError("checkpoint input identity differs")
        if state.get("validation_hash", hashlib.sha256(b"{}").hexdigest()) != self.validation_hash:
            raise ValueError("checkpoint validation identity differs")
        if len(state["rank_states"]) != self.world_size:
            raise ValueError("exact resume requires the same world size")
        details = {
            "sha256": checkpoint_hash,
            "source_run_id": state["run_id"],
            "step": state["step"],
        }
        validate_resume_details(details)
        if state["step"] > self.config.train.steps:
            raise ValueError("checkpoint step exceeds configured training")
        local = state["rank_states"][self.rank]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.loader.load_state_dict(local["loader"])
        torch.set_rng_state(local["rng_cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(local["rng_cuda"])
        self.step = state["step"]
        self.record.schema_version = 2
        self.record.resume_checkpoint = details
        self.record.status = "running"
        self.record.ended_at = None
        self.record.actual_cost_usd = None
        if self.rank == 0:
            self.record.save(self.run_root)
