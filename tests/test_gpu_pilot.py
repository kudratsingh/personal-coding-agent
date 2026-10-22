import json
from collections.abc import Callable
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


def worker(
    tmp_path: Path,
    spec: Specification | None = None,
    *,
    pod_created: float = 1000,
    clock: Callable[[], float] = lambda: 1000,
    **kwargs: Any,
) -> Worker:
    return Worker(
        spec or specification(),
        tmp_path / "pilot",
        [[256, 1, 2, 3] * 32],
        (ValidationWindow((256, 1, 2, 3), 3),),
        data_hash="authored",
        tokenizer_hash="authored",
        device="cpu",
        pod_created=pod_created,
        clock=clock,
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


def test_probe_window_counts_from_first_probe_and_selects_among_completed(
    tmp_path: Path,
) -> None:
    # The pod existed for a long time before the worker started; that no longer
    # forfeits the probes. The window expires during the second probe.
    now = [5000.0]
    pilot = worker(tmp_path, clock=lambda: now[0], pod_created=4850)
    original = pilot.on_step

    def advance(trainer: Trainer, metrics: dict[str, float]) -> None:
        if pilot.phase == "probe-4" and trainer.step == 1:
            now[0] += 70  # second probe: the 60 s window expires after its first update
        original(trainer, metrics)

    pilot.on_step = advance  # type: ignore[method-assign]
    result = pilot.execute()
    assert result["probe_window_expired"] is True
    assert [row["status"] for row in result["probes"]] == ["completed", "incomplete", "skipped"]
    assert result["chosen_microbatch"] == 2
    assert result["accumulation_steps"] == 2
    assert result["status"] == "completed"
    assert len(result["arms"]) == 3


def test_probe_window_expiring_in_first_probe_stops_without_training(tmp_path: Path) -> None:
    now = [1000.0]
    pilot = worker(tmp_path, clock=lambda: now[0])
    original = pilot.on_step

    def advance(trainer: Trainer, metrics: dict[str, float]) -> None:
        now[0] += 100
        original(trainer, metrics)

    pilot.on_step = advance  # type: ignore[method-assign]
    result = pilot.execute()
    assert result["status"] == "stopped"
    assert result["stop_reason"] == "no measured microbatch fits"
    assert [row["status"] for row in result["probes"]] == ["incomplete", "skipped", "skipped"]
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
    winner = result["winner"]
    continued = torch.load(pilot.root / f"{winner}-continued" / "last.pt", weights_only=True)
    assert continued["step"] == 4
    # The winning arm's screen checkpoint (the fair-comparison artifact) is preserved.
    screen = torch.load(pilot.root / winner / "last.pt", weights_only=True)
    assert screen["step"] == 2
    assert result["continuation"]["resumed_from_step"] == 2
    assert result["last_checkpoint_resume"]["step"] == 4
    assert result["last_checkpoint_resume"]["arm"] == f"{winner}-continued"
    exports = sorted(path.name for path in (pilot.root / "exports").iterdir())
    assert any(name.startswith(f"{winner}-step2-") for name in exports)
    assert any(name.startswith(f"{winner}-continued-step4-") for name in exports)
    record = json.loads((pilot.root / f"{winner}-continued" / "record.json").read_text())
    assert record["status"] == "completed"
    assert record["resume_checkpoint"]["step"] == 2  # Provenance names the screen checkpoint.
    arm_record = json.loads((pilot.root / winner / "record.json").read_text())
    assert arm_record["status"] == "completed"


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


def receipt() -> dict[str, Any]:
    return dict(
        owner_confirms_four_runpod_steps=True,
        cloud_data_pre_staged=True,
        ssh_transfer_tested=True,
        external_watchdog_armed=True,
        pod_self_destruct_armed=True,
        hardware_preflight_passed=True,
        full_console_quote_confirmed=True,
        staging_storage_cost_bound=True,
        termination_guarantee="redundant_best_effort",
        owner_accepts_best_effort_termination=True,
        termination_evidence="authored-test-only",
        pod_created_at_unix=1000,
        self_destruct_deadline_unix=1290,
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


def test_all_launch_gates_and_termination_evidence_are_required() -> None:
    base = receipt()
    assert check_readiness(base, now=1001, spec=specification()) == 1000
    for name, value in base.items():
        if name in {"pod_identity", "watchdog_policy"}:
            continue
        bad = {**base, name: False if value is True else None}
        with pytest.raises(ValueError):
            check_readiness(bad, now=1001, spec=specification())
    with pytest.raises(ValueError, match="owner acceptance"):
        check_readiness(
            {**base, "owner_accepts_best_effort_termination": False},
            now=1001,
            spec=specification(),
        )
    with pytest.raises(ValueError, match="termination_guarantee"):
        check_readiness({**base, "termination_guarantee": "hard"}, now=1001, spec=specification())
    # A provider-enforced guarantee does not need the risk acceptance flag.
    provider = {**base, "termination_guarantee": "provider_enforced"}
    provider.pop("owner_accepts_best_effort_termination")
    assert check_readiness(provider, now=1001, spec=specification()) == 1000
    # A long-running pod is accepted while probes and finalization still fit.
    assert check_readiness(base, now=1200, spec=specification()) == 1000
    with pytest.raises(ValueError, match="too short"):
        check_readiness(base, now=1240, spec=specification())
    with pytest.raises(ValueError, match="timestamp"):
        check_readiness({**base, "pod_created_at_unix": 999}, now=1001, spec=specification())
    with pytest.raises(ValueError, match="deadline"):
        check_readiness(
            {**base, "self_destruct_deadline_unix": 1301}, now=1001, spec=specification()
        )
    with pytest.raises(ValueError, match="ceiling"):
        check_readiness(
            base,
            now=1001,
            spec=replace(specification(), limits=replace(specification().limits, pod_seconds=5401)),
        )
    with pytest.raises(ValueError, match="weaker"):
        check_readiness(
            {**base, "watchdog_policy": {**base["watchdog_policy"], "max_cost_usd": 5}},
            now=1001,
            spec=specification(),
        )


def test_finalization_reserve_ends_training_early_but_permits_final_checkpoint(
    tmp_path: Path,
) -> None:
    base = specification()
    spec = replace(
        base,
        limits=replace(base.limits, training_seconds=200, finalization_reserve_seconds=150),
    )
    now = [1000.0]
    pilot = worker(tmp_path, spec, clock=lambda: now[0])
    original = pilot.on_step

    def advance(trainer: Trainer, metrics: dict[str, float]) -> None:
        if not pilot.phase.startswith("probe"):
            now[0] += 20
        original(trainer, metrics)

    pilot.on_step = advance  # type: ignore[method-assign]
    result = pilot.execute()
    # Pod wall 300 s minus the 150 s reserve leaves 150 s, less than the 200 s
    # budget: six screen updates use 120 s, the continuation gets two more
    # updates (1140 s, 1160 s) before the reserve deadline at 1150 s stops it.
    assert result["status"] == "time_limited"
    assert result["training_stop_reason"] == "finalization_reserve"
    assert result["training_deadline_unix"] == 1150
    assert result["selected_step"] == 4
    assert result["last_checkpoint_resume"]["tested"] is True
    assert result["training_tokens_processed"] == 8 * 32


def test_tampered_checkpoint_is_rejected_before_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = worker(tmp_path)
    original = pilot.save

    def corrupt(trainer: Trainer, name: str) -> Path:
        path = original(trainer, name)
        if name == "lr-2":
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 0xFF
            path.write_bytes(bytes(raw))
        return path

    monkeypatch.setattr(pilot, "save", corrupt)
    with pytest.raises(ValueError, match="differ from the recorded save identity"):
        pilot.execute()
    assert json.loads((pilot.root / "report.json").read_text())["status"] == "failed"
    assert "winner" in pilot.report and "selected_step" not in pilot.report


def test_token_accounting_counts_only_real_updates(tmp_path: Path) -> None:
    pilot = worker(tmp_path)
    result = pilot.execute()
    assert result["training_tokens_processed"] == 9 * 32
    assert all(row["training_tokens_processed"] <= 9 * 32 for row in result["checkpoints"])
    heartbeat = json.loads((pilot.root / "heartbeat.json").read_text())
    assert heartbeat["training_tokens_processed"] == 9 * 32
    sidecars = list(pilot.root.glob("*/last.pt.sha256.json"))
    assert sidecars and all(
        json.loads(path.read_text())["sha256"] == file_sha256(path.with_name("last.pt"))
        for path in sidecars
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


def test_final_checkpoint_is_exported_once_and_previous_bundle_is_retained(
    tmp_path: Path,
) -> None:
    pilot = worker(tmp_path)
    result = pilot.execute()
    exports = sorted(path.name for path in (pilot.root / "exports").iterdir())
    final = [name for name in exports if name.startswith(f"{result['winner']}-continued-step5-")]
    assert len(final) == 1, exports  # No duplicate final bundle.
    assert result["latest_transfer_bundle"] == f"exports/{final[0]}"
    exported = json.loads(
        (pilot.root / result["latest_transfer_bundle"] / "record.json").read_text()
    )
    assert exported["status"] == "completed"
    # Every screen arm keeps its step-2 bundle; retention never removes the newest previous one.
    for arm in ("lr-0", "lr-1", "lr-2"):
        assert any(name.startswith(f"{arm}-step2-") for name in exports)


def test_cuda_library_allocation_failures_count_as_probe_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = worker(tmp_path)
    monkeypatch.setattr(pilot, "device", torch.device("cuda", 0))
    monkeypatch.setattr(pilot, "reset_peak", lambda: None)
    monkeypatch.setattr(
        pilot, "peak", lambda: {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2}
    )
    monkeypatch.setattr(pilot, "release", lambda: None)
    monkeypatch.setattr(pilot, "peak_budget_bytes", lambda: None)
    original = pilot.trainer

    def fail(name: str, size: int, rate: float, *, probe: bool = False) -> Trainer:
        if probe and size == 4:
            raise RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate")
        if probe and size == 1:
            raise RuntimeError("unrelated kernel failure")
        pilot.device = torch.device("cpu")  # The completing probe runs on this CPU box.
        try:
            return original(name, size, rate, probe=probe)
        finally:
            pilot.device = torch.device("cuda", 0)

    monkeypatch.setattr(pilot, "trainer", fail)
    with pytest.raises(RuntimeError, match="unrelated"):
        pilot.probe()
    statuses = {row["microbatch"]: row.get("status") for row in pilot.report["probes"]}
    assert statuses[4] == "oom" and pilot.report["probes"][1]["error_type"] == "RuntimeError"


def test_rising_guard_floor_is_enforced() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        replace(specification().limits, rising_steps=4)


def test_billing_origin_slightly_after_creation_is_clamped_to_creation() -> None:
    base = receipt()
    policy = {**base["watchdog_policy"], "billing_started_at_unix": 1030}
    assert (
        check_readiness({**base, "watchdog_policy": policy}, now=1031, spec=specification()) == 1000
    )
    policy["billing_started_at_unix"] = 1100
    with pytest.raises(ValueError, match="billing origin"):
        check_readiness({**base, "watchdog_policy": policy}, now=1101, spec=specification())
