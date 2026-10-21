import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.train.trainer import Trainer


def tiny_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(steps=6, sequence_length=8, batch_size=2),
    )


def make_trainer(config: RunConfig, root: Path, name: str) -> Trainer:
    return Trainer(
        config,
        MixedLoader(
            {"web": [[256, 1, 2, 3] * 8]},
            {"web": 1.0},
            sequence_length=config.train.sequence_length,
            batch_size=config.train.batch_size,
            seed=config.train.seed,
        ),
        run_id=name,
        run_root=root,
        data_hash="authored-data",
        tokenizer_hash="authored-tokenizer",
    )


def test_stopped_partial_update_restores_loader_rng_and_exact_continuation(tmp_path: Path) -> None:
    config = replace(tiny_config(), train=replace(tiny_config().train, accumulation_steps=3))
    reference = make_trainer(config, tmp_path, "reference")
    expected_losses = reference.train()
    interrupted = make_trainer(config, tmp_path, "interrupted")
    before_loader = interrupted.loader.state_dict()
    before_rng = torch.get_rng_state().clone()
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        # Outer check, first microbatch, then stop before the second microbatch.
        return calls >= 3

    assert interrupted.train(should_stop=stop) == []
    assert interrupted.step == 0 and set(interrupted.optimizer.steps) == {0}
    assert interrupted.loader.state_dict() == before_loader
    torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
    interrupted.checkpoint(tmp_path / "partial.pt")
    resumed = make_trainer(config, tmp_path, "resumed")
    resumed.resume(tmp_path / "partial.pt")
    assert resumed.train() == expected_losses
    for left, right in zip(reference.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    events = [
        json.loads(line)
        for line in (tmp_path / "interrupted/events-rank0.jsonl").read_text().splitlines()
    ]
    assert any(
        row["event"] == "training_stopped" and row["fields"]["partial_update"] for row in events
    )


def test_callback_observes_only_completed_updates_and_can_save_resume_state(tmp_path: Path) -> None:
    config = tiny_config()
    trainer = make_trainer(config, tmp_path, "controlled")
    observed: list[tuple[int, float]] = []

    def completed(instance: object, metrics: dict[str, float]) -> None:
        assert instance is trainer
        observed.append((trainer.step, metrics["tokens"]))
        if trainer.step == 2:
            trainer.checkpoint(tmp_path / "step2.pt")

    losses = trainer.train(should_stop=lambda: trainer.step >= 2, on_step=completed)
    assert len(losses) == 2 and observed == [(1, 16), (2, 16)]
    resumed = make_trainer(config, tmp_path, "resume-callback")
    resumed.resume(tmp_path / "step2.pt")
    assert resumed.step == 2
    assert resumed.train() == trainer.train()


def test_stop_before_first_update_consumes_nothing(tmp_path: Path) -> None:
    trainer = make_trainer(tiny_config(), tmp_path, "no-work")
    before = trainer.loader.state_dict()
    assert trainer.train(should_stop=lambda: True) == []
    assert trainer.step == 0 and trainer.loader.state_dict() == before


def test_rank_local_controls_rejected_for_distributed_training(tmp_path: Path) -> None:
    trainer = make_trainer(tiny_config(), tmp_path, "distributed-control")
    trainer.world_size = 2
    with pytest.raises(ValueError, match="single-process"):
        trainer.train(should_stop=lambda: False)
