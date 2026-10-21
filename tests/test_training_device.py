import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.fixtures import prepare_fixture
from scratchcoder.data.loader import MixedLoader
from scratchcoder.train.command import train_from_files
from scratchcoder.train.trainer import Trainer, synchronized_time, training_device


def test_cpu_device_does_not_probe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected() -> bool:
        raise AssertionError("CPU selection must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    assert training_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("value", ["auto", "mps", "cpu:0", "cuda:-1", "cuda:1x", ""])
def test_unsupported_device_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="training device"):
        training_device(value)


@pytest.mark.parametrize("value, expected", [("cuda", 1), ("cuda:0", 0), ("cuda:1", 1)])
def test_cuda_device_resolves_visible_ordinal(
    value: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert training_device(value) == torch.device("cuda", expected)
    for invalid in ("cuda:2", "cuda:256", "cuda:18446744073709551616"):
        with pytest.raises(ValueError, match="not available"):
            training_device(invalid)


def test_unavailable_cuda_rejected_before_file_io_or_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        train_from_files(
            config_path=tmp_path / "absent.yaml",
            sources_path=tmp_path / "absent-sources.json",
            tokenizer_path=tmp_path / "absent-tokenizer.json",
            output_root=tmp_path / "runs",
            run_id="unavailable",
            device="cuda",
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("arguments, expected", [([], "cpu"), (["--device", "cuda:1"], "cuda:1")])
def test_cli_forwards_explicit_or_default_device(
    arguments: list[str], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("scratchcoder.train.command.train_from_files", capture)
    assert (
        main(
            [
                "train",
                "--config",
                "config.yaml",
                "--sources",
                "sources.json",
                "--tokenizer",
                "tokenizer.json",
                "--run-id",
                "test",
                *arguments,
            ]
        )
        == 0
    )
    assert captured["device"] == expected


def test_file_training_passes_resolved_device_to_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    tokenizer, _, _ = prepare_fixture(data, vocab_size=512)
    original = RunConfig.load(Path("configs/smoke.yaml"))
    config = replace(original, model=replace(original.model, vocab_size=tokenizer.vocab_size))
    config_path = tmp_path / "config.yaml"
    config.save(config_path)
    captured: list[str] = []

    class CaptureTrainer:
        def __init__(self, *args: object, device: str, **kwargs: object) -> None:
            captured.append(device)

        def train(self) -> None:
            pass

        def checkpoint(self, path: Path) -> None:
            pass

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr("scratchcoder.train.command.Trainer", CaptureTrainer)
    train_from_files(
        config_path=config_path,
        sources_path=data / "stage1-sources.json",
        tokenizer_path=data / "tokenizer.json",
        output_root=tmp_path / "runs",
        run_id="captured",
        device="cuda",
    )
    assert captured == ["cuda:1"]


@pytest.mark.parametrize("name", ["cpu", "cuda:1"])
def test_clock_waits_for_selected_cuda_device_before_each_read(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = torch.device(name)
    events: list[str] = []
    readings = iter([10.0, 12.5])

    def synchronize(selected: torch.device) -> None:
        assert selected == device
        events.append("sync")

    def clock() -> float:
        events.append("clock")
        return next(readings)

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr("scratchcoder.train.trainer.time.perf_counter", clock)
    start = synchronized_time(device)
    events.append("work")
    elapsed = synchronized_time(device) - start
    assert elapsed == 2.5
    assert events == (
        ["sync", "clock", "work", "sync", "clock"]
        if name.startswith("cuda")
        else ["clock", "work", "clock"]
    )


def test_step_timing_encloses_accumulation_and_optimizer_and_records_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RunConfig(
        model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=16),
        train=TrainConfig(steps=1, sequence_length=8, batch_size=2, accumulation_steps=2),
    )
    loader = MixedLoader(
        {"web": [[256, 1, 2, 3] * 8]}, {"web": 1.0}, sequence_length=8, batch_size=2, seed=0
    )
    trainer = Trainer(
        config,
        loader,
        run_id="timed",
        run_root=tmp_path,
        data_hash="authored",
        tokenizer_hash="authored",
    )
    events: list[str] = []
    readings = iter([10.0, 12.5])
    next_batch, update = loader.next_batch, trainer.optimizer.step

    def clock(device: torch.device) -> float:
        assert device == torch.device("cpu")
        events.append("clock")
        return next(readings)

    def batch() -> tuple[list[list[int]], list[list[int]], list[str]]:
        events.append("batch")
        return next_batch()

    def step() -> None:
        events.append("optimizer")
        update()

    monkeypatch.setattr("scratchcoder.train.trainer.synchronized_time", clock)
    monkeypatch.setattr(loader, "next_batch", batch)
    monkeypatch.setattr(trainer.optimizer, "step", step)
    trainer.train()
    assert events == ["clock", "batch", "batch", "optimizer", "clock"]
    record = json.loads((tmp_path / "timed/record.json").read_text())
    assert record["hardware"]["training_device"] == "cpu"
    rows = [
        json.loads(line)
        for line in (tmp_path / "timed/events-rank0.jsonl").read_text().splitlines()
    ]
    metrics = next(row["fields"]["metrics"] for row in rows if row["event"] == "metrics")
    assert metrics["tokens"] == 32
    assert metrics["step_seconds"] == 2.5
    assert metrics["tokens_per_second"] == 32 / 2.5
