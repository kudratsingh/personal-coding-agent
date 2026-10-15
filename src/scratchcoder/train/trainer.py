"""One training path for initial pretraining, continuation, and exact resume."""

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
from scratchcoder.common.logging import EventLogger
from scratchcoder.common.records import RunRecord
from scratchcoder.data.loader import MixedLoader
from scratchcoder.model.gpt import GPT
from scratchcoder.optim.adamw import AdamW, clip_gradients, cross_entropy, learning_rate


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        loader: MixedLoader,
        *,
        run_id: str,
        run_root: Path,
        data_hash: str,
        tokenizer_hash: str,
        device: str = "cpu",
    ) -> None:
        self.config, self.loader = config, loader
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
                            enabled=self.device.type == "cuda",
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
                if self.step == options.steps:
                    self.record.finish(success=True)
            except Exception as error:
                logger.event("training_failed", error_type=type(error).__name__, step=self.step)
                self.record.finish(success=False)
                raise
            finally:
                if self.rank == 0:
                    self.record.save(self.run_root)
        return losses

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
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["version"] != 1 or state["config"] != asdict(self.config):
            raise ValueError("checkpoint configuration differs")
        if (
            state["data_hash"] != self.record.data_manifest_hash
            or state["tokenizer_hash"] != self.record.tokenizer_hash
        ):
            raise ValueError("checkpoint input identity differs")
        if len(state["rank_states"]) != self.world_size:
            raise ValueError("exact resume requires the same world size")
        local = state["rank_states"][self.rank]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.loader.load_state_dict(local["loader"])
        torch.set_rng_state(local["rng_cpu"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(local["rng_cuda"])
        self.step = state["step"]
