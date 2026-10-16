"""One training path for initial pretraining, continuation, and exact resume."""

import hashlib
import json
import os
import time
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
from scratchcoder.optim.adamw import AdamW, clip_gradients, cross_entropy, learning_rate
from scratchcoder.train.validation import ValidationWindow, validate


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
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        torch.manual_seed(config.train.seed)
        self.model = GPT(config.model).to(self.device)
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
        )
        self.step = 0
        self.record = RunRecord.start(run_id, config, data_hash, tokenizer_hash, parent_run)
        self.run_root = run_root
        if self.rank == 0:
            self.record.save(run_root)

    def train(self, *, stop_after: int | None = None) -> list[float]:
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
                    started = time.perf_counter()
                    self.optimizer.zero_grad()
                    loss_value = 0.0
                    for _ in range(options.accumulation_steps):
                        inputs, targets, _ = self.loader.next_batch()
                        x = torch.tensor(inputs, dtype=torch.long, device=self.device)
                        y = torch.tensor(targets, dtype=torch.long, device=self.device)
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=options.precision == "bfloat16",
                        ):
                            loss = cross_entropy(self.model(x), y) / options.accumulation_steps
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("nonfinite training loss")
                        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's Tensor stub.
                        loss_value += float(loss.detach())
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
                    )
                    self.optimizer.step()
                    losses.append(loss_value)
                    self.step += 1
                    elapsed = time.perf_counter() - started
                    logger.metrics(
                        {
                            "loss": loss_value,
                            "learning_rate": self.optimizer.lr,
                            "gradient_norm": norm,
                            "tokens_per_second": options.batch_size
                            * options.sequence_length
                            * options.accumulation_steps
                            * self.world_size
                            / max(elapsed, 1e-9),
                        },
                        step=self.step,
                    )
                    if options.validation_interval and (
                        self.step % options.validation_interval == 0 or self.step == options.steps
                    ):
                        self._log_validation(logger)
                if self.step == options.steps:
                    self.record.finish(success=True)
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
        if self.world_size > 1:
            dist.barrier()

    def resume(self, path: Path) -> None:
        checkpoint_hash: str | None = None
        try:
            with path.open("rb") as stream:
                checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
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
