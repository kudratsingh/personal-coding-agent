"""Actual two-process restarts, compared bit-for-bit with uninterrupted CPU training."""

import json
from dataclasses import asdict
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.loader import MixedLoader
from scratchcoder.data.shards import file_sha256
from scratchcoder.train.trainer import Trainer


def options(precision: str) -> RunConfig:
    return RunConfig(
        model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8),
        train=TrainConfig(
            seed=7,
            steps=6,
            sequence_length=4,
            batch_size=1,
            accumulation_steps=2,
            precision=precision,
        ),
        data_mix={"web": 0.4, "python": 0.6},
    )


def make_trainer(root: Path, phase: str, precision: str, rank: int, world: int) -> Trainer:
    config = options(precision)
    loader = MixedLoader(
        {
            "web": [[256, index, 10, 11, 12] for index in range(8)],
            "python": [[256, index + 20, 30, 31, 32, 33] for index in range(8)],
        },
        config.data_mix,
        sequence_length=config.train.sequence_length,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
        rank=rank,
        world_size=world,
    )
    return Trainer(
        config,
        loader,
        run_id=phase,
        run_root=root,
        data_hash="authored-two-source-fixture",
        tokenizer_hash="authored-fixture",
    )


def train_rank(rank: int, root: str, phase: str, precision: str) -> None:
    torch.set_num_threads(1)
    directory = Path(root)
    dist.init_process_group(
        "gloo",
        init_method=(directory / f"rendezvous-{phase}").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer = make_trainer(directory, phase, precision, rank, 2)
        if phase == "resumed":
            torch.manual_seed(999 + rank)  # Resume must replace this unrelated fresh-process RNG.
            trainer.resume(directory / "partial.pt")
            saved = torch.load(directory / "partial.pt", weights_only=True)
            assert torch.equal(torch.get_rng_state(), saved["rank_states"][rank]["rng_cpu"])
            assert trainer.loader.state_dict() == saved["rank_states"][rank]["loader"]
        else:
            torch.rand(rank + 1)  # Deliberately different RNG positions on each rank.
        losses = trainer.train(stop_after=3 if phase == "partial" else None)
        trainer.checkpoint(directory / f"{phase}.pt")
        torch.save(
            {"losses": losses, "next_random": torch.rand(5)},
            directory / f"{phase}-rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def launch(root: Path, phase: str, precision: str) -> None:
    context = get_context("spawn")
    processes = [
        context.Process(target=train_rank, args=(rank, str(root), phase, precision))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0, f"{phase} worker failed or exceeded its deadline"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_two_process_checkpoint_restart_matches_uninterrupted_training(
    tmp_path: Path, precision: str
) -> None:
    for phase in ("full", "partial", "resumed"):
        launch(tmp_path, phase, precision)
    full = torch.load(tmp_path / "full.pt", weights_only=True)
    partial = torch.load(tmp_path / "partial.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed.pt", weights_only=True)
    assert full["step"] == resumed["step"] == 6
    assert partial["step"] == 3
    assert full["config"] == resumed["config"] == asdict(options(precision))
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
    for field in ("m", "v"):
        for left, right in zip(full["optimizer"][field], resumed["optimizer"][field], strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert full["optimizer"]["steps"] == resumed["optimizer"]["steps"]
    assert len(full["rank_states"]) == len(resumed["rank_states"]) == 2
    assert full["rank_states"][0]["loader"] != full["rank_states"][1]["loader"]
    assert not torch.equal(full["rank_states"][0]["rng_cpu"], full["rank_states"][1]["rng_cpu"])
    for rank in range(2):
        left, right = full["rank_states"][rank], resumed["rank_states"][rank]
        assert left["loader"] == right["loader"]
        torch.testing.assert_close(left["rng_cpu"], right["rng_cpu"], atol=0, rtol=0)
        results: dict[str, Any] = {
            phase: torch.load(tmp_path / f"{phase}-rank{rank}.pt", weights_only=True)
            for phase in ("full", "partial", "resumed")
        }
        assert (
            results["full"]["losses"] == results["partial"]["losses"] + results["resumed"]["losses"]
        )
        torch.testing.assert_close(
            results["full"]["next_random"], results["resumed"]["next_random"], atol=0, rtol=0
        )
    record = json.loads((tmp_path / "resumed/record.json").read_text())
    assert record["status"] == "completed" and record["schema_version"] == 2
    assert record["resume_checkpoint"] == {
        "sha256": file_sha256(tmp_path / "partial.pt"),
        "source_run_id": "partial",
        "step": 3,
    }

    # This process has no distributed group: restoring two-rank state is not exact resume.
    single = make_trainer(tmp_path, "wrong-world", precision, 0, 1)
    with pytest.raises(ValueError, match="world size"):
        single.resume(tmp_path / "partial.pt")
    failed = json.loads((tmp_path / "wrong-world/record.json").read_text())
    assert failed["status"] == "failed" and failed["resume_checkpoint"] is None
