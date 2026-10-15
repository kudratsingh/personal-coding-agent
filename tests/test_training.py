import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.engine.generation import generate
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


def test_one_batch_can_be_memorized(tmp_path: Path) -> None:
    config = replace(
        tiny_config(), train=replace(tiny_config().train, steps=100, learning_rate=0.02)
    )
    loader = MixedLoader({"web": [[1] * 32]}, {"web": 1.0}, sequence_length=8, batch_size=2, seed=0)
    trainer = Trainer(
        config,
        loader,
        run_id="memorize",
        run_root=tmp_path,
        data_hash="synthetic",
        tokenizer_hash="synthetic",
    )
    losses = trainer.train()
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 50


def test_tiny_model_learns_repeating_toy_corpus(tmp_path: Path) -> None:
    config = RunConfig(
        model=ModelConfig(vocab_size=272, width=32, layers=1, heads=4, context=32),
        train=TrainConfig(steps=120, sequence_length=12, batch_size=2, learning_rate=0.01),
    )
    loader = MixedLoader(
        {"web": [[97, 98, 99] * 20]}, {"web": 1.0}, sequence_length=12, batch_size=2, seed=0
    )
    trainer = Trainer(
        config,
        loader,
        run_id="toy",
        run_root=tmp_path,
        data_hash="synthetic",
        tokenizer_hash="bytes",
    )
    losses = trainer.train()
    assert losses[-1] < 0.1
    assert generate(trainer.model, torch.tensor([[97]]), max_new_tokens=8).tolist() == [
        [97, 98, 99] * 3
    ]


def test_nano_configuration_learns_toy_corpus(tmp_path: Path) -> None:
    original = RunConfig.load(Path("configs/nano.yaml"))
    config = replace(
        original,
        train=replace(
            original.train, steps=60, sequence_length=12, batch_size=2, learning_rate=0.003
        ),
    )
    loader = MixedLoader(
        {"web": [[97, 98, 99] * 20]},
        {"web": 1.0},
        sequence_length=12,
        batch_size=2,
        seed=config.train.seed,
    )
    trainer = Trainer(
        config,
        loader,
        run_id="nano-toy",
        run_root=tmp_path,
        data_hash="synthetic",
        tokenizer_hash="bytes",
    )
    losses = trainer.train()
    assert losses[-1] < 0.1
    assert generate(trainer.model, torch.tensor([[97]]), max_new_tokens=8).tolist() == [
        [97, 98, 99] * 3
    ]


def test_training_failure_records_traceback_and_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = make_trainer(tiny_config(), tmp_path, "failure")

    def fail() -> None:
        raise RuntimeError("Bearer synthetic-secret")

    monkeypatch.setattr(trainer.optimizer, "step", fail)
    with pytest.raises(RuntimeError):
        trainer.train()
    record = json.loads((tmp_path / "failure/record.json").read_text())
    trace = (tmp_path / "failure/debug-rank0.jsonl").read_text()
    assert record["status"] == "failed"
    assert "RuntimeError" in trace
    assert "synthetic-secret" not in trace
