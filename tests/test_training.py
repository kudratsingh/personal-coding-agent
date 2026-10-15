from dataclasses import replace
from pathlib import Path

import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.train.trainer import Trainer


def tiny_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(steps=6, sequence_length=8, batch_size=2),
        data_mix={"web": 1.0},
    )


def make_trainer(config: RunConfig, root: Path, name: str) -> Trainer:
    loader = MixedLoader(
        {"web": [[256, 1, 2, 3] * 8]},
        {"web": 1.0},
        sequence_length=config.train.sequence_length,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )
    return Trainer(
        config,
        loader,
        run_id=name,
        run_root=root,
        data_hash="fixture-data",
        tokenizer_hash="fixture-tokenizer",
    )


def test_interrupted_resume_is_exact(tmp_path: Path) -> None:
    config = tiny_config()
    full = make_trainer(config, tmp_path, "full")
    full_losses = full.train()
    interrupted = make_trainer(config, tmp_path, "partial")
    assert interrupted.train(stop_after=3) == full_losses[:3]
    checkpoint = tmp_path / "partial.pt"
    interrupted.checkpoint(checkpoint)
    resumed = make_trainer(config, tmp_path, "resumed")
    resumed.resume(checkpoint)
    assert resumed.train() == full_losses[3:]
    for left, right in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_stage_two_loads_weights_but_resets_optimizer_and_schedule(tmp_path: Path) -> None:
    first = make_trainer(tiny_config(), tmp_path, "first")
    first.train()
    checkpoint = tmp_path / "first.pt"
    first.checkpoint(checkpoint)
    second = make_trainer(
        replace(tiny_config(), parent_checkpoint=str(checkpoint)), tmp_path, "second"
    )
    x = torch.tensor([[256, 1, 2, 3]])
    assert first.validation_loss(x, x) == second.validation_loss(x, x)
    assert second.step == 0
    assert set(second.optimizer.steps) == {0}
    assert second.record.parent_run_id == "first"
