import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.shards import file_sha256
from scratchcoder.train.pilot import Limits
from scratchcoder.train.trainer import Trainer
from scratchcoder.train.validation import ValidationWindow
from tools.gpu_pilot import Specification, Worker, check_readiness


def specification() -> Specification:
    return Specification(
        RunConfig(
            model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8),
            train=TrainConfig(steps=5, sequence_length=8, batch_size=2),
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


def worker(tmp_path: Path, spec: Specification | None = None, **kwargs: Any) -> Worker:
    return Worker(
        spec or specification(),
        tmp_path / "pilot",
        [[256, 1, 2, 3] * 32],
        (ValidationWindow((256, 1, 2, 3), 3),),
        data_hash="authored",
        tokenizer_hash="authored",
        device="cpu",
        pod_created=1000,
        clock=lambda: 1000,
        data_kind="authored-CPU-test",
        **kwargs,
    )


def test_complete_worker_selects_heldout_winner_and_restores_last_checkpoint(
    tmp_path: Path,
) -> None:
    pilot = worker(tmp_path)
    result = pilot.execute()
    assert result["status"] == "completed"
    assert result["chosen_microbatch"] == 4
    assert result["accumulation_steps"] == 1
    assert len(result["probes"]) == 3
    assert all(row["updates"] == 2 for row in result["probes"])
    assert len(result["arms"]) == 3
    winner = min(result["arms"], key=lambda row: (row["validation"]["loss"], row["learning_rate"]))
    assert result["winner"] == winner["name"]
    assert result["selected_step"] == 5
    assert result["last_checkpoint_resume"]["tested"] is True
    rows = [json.loads(line) for line in (pilot.root / "updates.jsonl").read_text().splitlines()]
    actual = [row for row in rows if not row["arm"].startswith("probe")]
    assert len(actual) == 3 * 2 + 3
    assert all(row["tokens"] == 32 for row in actual)
    assert result["training_throughput"]["tokens"] == 9 * 32
    bundle = pilot.root / result["latest_transfer_bundle"]
    assert file_sha256(bundle / "checkpoint.pt") == result["last_checkpoint_resume"]["sha256"]
    exported = json.loads((bundle / "record.json").read_text())
    assert exported["status"] == "completed"
    assert exported["latest_transfer_bundle"] == result["latest_transfer_bundle"]
    assert exported["last_checkpoint_resume"]["tested"] is True
    assert json.loads((pilot.root / "heartbeat.json").read_text())["status"] == "awaiting_transfer"
    # The CPU fixture produces no GPU performance evidence.
    assert result["training_peak"]["peak_allocated_bytes"] == 0
    assert result["actual_provider_cost"] is None


def test_probe_window_expires_without_any_real_training(tmp_path: Path) -> None:
    pilot = worker(tmp_path)
    pilot.clock = lambda: 1060
    result = pilot.execute()
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "probe_time_limit"
    assert not result["arms"]
    assert "chosen_microbatch" not in result
    assert not list(pilot.root.glob("*/last.pt"))


def test_real_training_deadline_preserves_whole_update_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = worker(
        tmp_path,
        replace(specification(), limits=replace(specification().limits, training_seconds=8)),
    )
    now = [1000.0]
    pilot.clock = lambda: now[0]
    original = pilot.on_step

    def advance(trainer: Trainer, metrics: dict[str, float]) -> None:
        now[0] += 1
        original(trainer, metrics)

    monkeypatch.setattr(pilot, "on_step", advance)
    result = pilot.execute()
    assert result["status"] == "time_limited"
    assert result["selected_step"] == 4  # six screen + two continuation updates
    state = torch.load(pilot.root / result["winner"] / "last.pt", weights_only=True)
    assert state["step"] == 4
    assert result["last_checkpoint_resume"]["step"] == 4


def test_real_oom_is_fatal_and_never_retried_at_smaller_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = worker(tmp_path)
    original = pilot.trainer

    def fail(name: str, size: int, rate: float, *, probe: bool = False) -> Trainer:
        if not probe:
            raise torch.cuda.OutOfMemoryError("authored real-update failure")
        return original(name, size, rate, probe=probe)

    monkeypatch.setattr(pilot, "trainer", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        pilot.execute()
    assert json.loads((pilot.root / "report.json").read_text())["status"] == "failed"
    assert not pilot.report["arms"]


def test_specification_roundtrip_and_invalid_effective_batch(tmp_path: Path) -> None:
    from dataclasses import asdict

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(asdict(specification())))
    assert Specification.load(path) == specification()
    with pytest.raises(ValueError, match="exact"):
        replace(specification(), tokens_per_update=33)


def test_all_launch_gates_and_deadline_evidence_are_required() -> None:
    receipt: dict[str, Any] = dict(
        owner_confirms_four_runpod_steps=True,
        cloud_data_pre_staged=True,
        ssh_transfer_tested=True,
        external_watchdog_armed=True,
        provider_hard_deadline_confirmed=True,
        hardware_preflight_passed=True,
        full_console_quote_confirmed=True,
        staging_storage_cost_bound=True,
        provider_deadline_evidence="authored-test-only",
        pod_created_at_unix=1000,
        provider_termination_at_unix=1300,
        pod_identity={
            "id": "testpod",
            "name": "test",
            "gpuId": "A40",
            "gpuCount": 1,
            "dataCenterId": "test",
            "createdAt": "1970-01-01T00:16:40+00:00",
        },
        watchdog_policy={
            "max_wall_seconds": 300,
            "max_cost_usd": 1,
            "max_idle_seconds": 180,
            "quoted_total_hourly": 1,
            "billing_started_at_unix": 1000,
        },
    )
    assert check_readiness(receipt, now=1001, spec=specification()) == 1000
    for name, value in receipt.items():
        if name in {"pod_identity", "watchdog_policy"}:
            continue
        bad = {**receipt, name: False if value is True else None}
        with pytest.raises(ValueError):
            check_readiness(bad, now=1001, spec=specification())
    with pytest.raises(ValueError, match="window"):
        check_readiness(receipt, now=1060, spec=specification())
    with pytest.raises(ValueError, match="timestamp"):
        check_readiness({**receipt, "pod_created_at_unix": 999}, now=1001, spec=specification())
    with pytest.raises(ValueError, match="deadline"):
        check_readiness(
            {**receipt, "provider_termination_at_unix": 1301}, now=1001, spec=specification()
        )
    with pytest.raises(ValueError, match="ceiling"):
        check_readiness(
            receipt,
            now=1001,
            spec=replace(specification(), limits=replace(specification().limits, pod_seconds=5401)),
        )
    with pytest.raises(ValueError, match="weaker"):
        check_readiness(
            {**receipt, "watchdog_policy": {**receipt["watchdog_policy"], "max_cost_usd": 5}},
            now=1001,
            spec=specification(),
        )


def test_low_speed_and_rising_loss_abort_at_recorded_update(tmp_path: Path) -> None:
    base = specification()
    pilot = worker(
        tmp_path,
        replace(
            base, expected_tokens_per_second=100, limits=replace(base.limits, speed_check_seconds=1)
        ),
    )
    pilot.training_started = 998
    pilot.phase = "lr-0"
    trainer = pilot.trainer("test", 4, 0.0006)
    trainer.step = 1
    from scratchcoder.train.pilot import PilotStop

    with pytest.raises(PilotStop, match="throughput"):
        pilot.on_step(trainer, {"loss": 1, "tokens": 32, "step_seconds": 10})
    assert len(pilot.rows["lr-0"]) == 1
