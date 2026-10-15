from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.train.trainer import Trainer


def config(batch_size: int) -> RunConfig:
    return RunConfig(
        model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(steps=2, sequence_length=4, batch_size=batch_size),
    )


def train_rank(rank: int, root: str) -> None:
    torch.set_num_threads(1)
    directory = Path(root)
    dist.init_process_group(
        "gloo", init_method=(directory / "init").as_uri(), rank=rank, world_size=2
    )
    try:
        options = config(1)
        loader = MixedLoader(
            {"web": [[1] * 16, [2] * 16]},
            {"web": 1.0},
            sequence_length=4,
            batch_size=1,
            seed=0,
            rank=rank,
            world_size=2,
        )
        trainer = Trainer(
            options,
            loader,
            run_id="distributed",
            run_root=directory,
            data_hash="fixture",
            tokenizer_hash="fixture",
        )
        trainer.train()
        trainer.checkpoint(directory / "distributed.pt")
    finally:
        dist.destroy_process_group()


def test_two_cpu_processes_match_effective_batch(tmp_path: Path) -> None:
    context = get_context("spawn")
    processes = [
        context.Process(target=train_rank, args=(rank, str(tmp_path))) for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    options = config(2)
    loader = CombinedRankBatches()
    trainer = Trainer(
        options,
        loader,
        run_id="serial",
        run_root=tmp_path,
        data_hash="fixture",
        tokenizer_hash="fixture",
    )
    trainer.train()
    saved = torch.load(tmp_path / "distributed.pt", weights_only=True)
    assert len(saved["rank_states"]) == 2
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, saved["model"][key], atol=1e-6, rtol=1e-5)


class CombinedRankBatches:
    """Serial reference consumes the exact union of two different rank batches."""

    def __init__(self) -> None:
        self.loaders = [
            MixedLoader(
                {"web": [[1] * 16, [2] * 16]},
                {"web": 1.0},
                sequence_length=4,
                batch_size=1,
                seed=0,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]

    def next_batch(self) -> tuple[list[list[int]], list[list[int]], list[str]]:
        first, second = [loader.next_batch() for loader in self.loaders]
        return first[0] + second[0], first[1] + second[1], first[2] + second[2]

    def state_dict(self) -> dict[str, Any]:
        return {"loaders": [loader.state_dict() for loader in self.loaders]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for loader, saved in zip(self.loaders, state["loaders"], strict=True):
            loader.load_state_dict(saved)


def test_gradient_accumulation_matches_effective_batch(tmp_path: Path) -> None:
    models = []
    for batch, accumulation in ((2, 1), (1, 2)):
        options = replace(
            config(batch), train=replace(config(batch).train, accumulation_steps=accumulation)
        )
        loader = MixedLoader(
            {"web": [[1] * 16]}, {"web": 1.0}, sequence_length=4, batch_size=batch, seed=0
        )
        trainer = Trainer(
            options,
            loader,
            run_id=f"batch-{batch}",
            run_root=tmp_path,
            data_hash="fixture",
            tokenizer_hash="fixture",
        )
        trainer.train()
        models.append(trainer.model)
    for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
