import functools
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.data.shards import file_sha256
from scratchcoder.train.pilot import Limits
from scratchcoder.train.trainer import Trainer
from scratchcoder.train.validation import ValidationWindow
from tools.gpu_pilot import (
    COMPILE_SOFT_STOP_SECONDS,
    Specification,
    Worker,
    _completed_rows,
    _kill_group,
    check_preflight,
    check_readiness,
    document_source,
    run_compile_probe_child,
    trainer_config,
)


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


def compile_spec(**overrides: Any) -> Specification:
    """The test specification with the measurement-only compile probe enabled."""
    settings: dict[str, Any] = {
        "compile_probe_steps": 3,
        "compile_probe_seconds": 60,
        "compile_probe_warmup_updates": 1,
    }
    settings.update(overrides)
    return replace(specification(), **settings)


@functools.lru_cache(maxsize=1)
def _toolchain_available() -> bool:
    """Whether torch.compile can build even a trivial CPU function on this machine."""
    try:
        compiled = torch.compile(lambda x: x * 2 + 1)
        return bool(torch.equal(compiled(torch.ones(4)), torch.full((4,), 3.0)))
    except Exception:
        return False


SECRET = "rpa_ABCDEFGH12345678"

# A stand-in for the compile probe child: real compilation is slow, so these
# tests drive the parent's supervision with a tiny script that obeys the same
# request/rows/result protocol. One test below runs the real compiled child.
FAKE_CHILD = """
import json, os, subprocess, sys, time
from pathlib import Path
request = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text())
rows, result = Path(request["rows_path"]), Path(request["result_path"])
(rows.parent / "child.pid").write_text(str(os.getpid()))
mode = os.environ["FAKE_COMPILE_CHILD"]
SECRET = "rpa_ABCDEFGH12345678"
def row(step):
    seconds = 3.0 if step == 1 else 0.5
    return json.dumps({"loss": 5.0, "tokens": 64.0, "step_seconds": seconds, "step": float(step)})
def complete():
    with rows.open("a") as stream:
        stream.write("".join(row(step) + "\\n" for step in range(1, 4)))
    result.write_text(json.dumps({
        "status": "completed", "updates": 3, "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0, "torch_version": "fake", "compile": True,
    }))
if mode == "fail":
    sys.stderr.write("backend compiler failed: no working C++ compiler; key " + SECRET + "\\n")
    raise SystemExit(1)
if mode in ("result-oom", "result-failed"):
    error_type = "OutOfMemoryError" if mode == "result-oom" else "BackendCompilerFailed"
    result.write_text(json.dumps({
        "status": "oom" if mode == "result-oom" else "failed", "updates": 0,
        "error_type": error_type, "error": "boom with " + SECRET + " and Bearer abcdefghijkl",
        "peak_allocated_bytes": 7, "peak_reserved_bytes": 9, "torch_version": "fake",
        "compile": True,
    }))
    raise SystemExit(1)
if mode == "complete":
    complete()
    raise SystemExit(0)
if mode == "complete-then-abort":
    complete()
    os.abort()
if mode == "one-then-sleep":
    with rows.open("a") as stream:
        stream.write(row(1) + "\\n")
if mode == "complete-on-go":
    while not (rows.parent / "go").exists():
        time.sleep(0.01)
    complete()
    raise SystemExit(0)
if mode == "sleep":
    # Like an inductor compile worker: a grandchild in the same process group.
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (rows.parent / "grandchild.pid").write_text(str(grandchild.pid))
time.sleep(120)
"""


def fake_child_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, **kwargs: Any
) -> Worker:
    script = tmp_path / "fake_compile_child.py"
    script.write_text(FAKE_CHILD)
    monkeypatch.setenv("FAKE_COMPILE_CHILD", mode)
    spec = kwargs.pop("spec", compile_spec())
    return worker(
        tmp_path,
        spec,
        compile_probe_command=(sys.executable, str(script)),
        compile_poll_seconds=0.02,
        **kwargs,
    )


def _dead(pid: int, seconds: float = 10) -> bool:
    """True once the process is gone; an orphaned zombie is reaped by init shortly."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return True  # macOS: only a zombie awaiting its reaper remains.
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _wait_for(path: Path, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_compile_probe_is_recorded_separately_and_never_changes_the_arms(
    tmp_path: Path,
) -> None:
    eager = worker(tmp_path / "eager").execute()
    assert eager["compile_probe"] == {"status": "disabled"}
    # A real advancing clock, so the hard window would end a hung child.
    started = time.time()
    pilot = worker(
        tmp_path / "compiled",
        compile_spec(),
        clock=time.time,
        pod_created=started,
        compile_poll_seconds=0.05,
    )
    result = pilot.execute()
    # Compiler-independent guarantees first: the parent never compiles ...
    records = {
        path.parent.name: json.loads(path.read_text()) for path in pilot.root.glob("*/record.json")
    }
    assert {"probe-2", "probe-4", "probe-1", "lr-0", "lr-1", "lr-2"} <= set(records)
    assert any(name.endswith("-continued") for name in records)
    assert all(record["config"]["train"]["compile"] is False for record in records.values())
    # ... the child is asked for exactly the eager probe's configuration with compile on ...
    request = json.loads((pilot.root / "compile-probe" / "request.json").read_text())
    train = request["config"]["train"]
    assert (train["compile"], train["batch_size"], train["accumulation_steps"], train["steps"]) == (
        True,
        4,
        2,
        3,
    )
    assert {**train, "compile": False, "steps": 2, "warmup_steps": 2} == {
        **records["probe-4"]["config"]["train"],
        "warmup_steps": 2,
    }
    # ... and the eager probe table, the arms and the continuation are untouched.
    assert [r["status"] for r in result["probes"]] == [r["status"] for r in eager["probes"]]
    assert [a["validation"] for a in result["arms"]] == [a["validation"] for a in eager["arms"]]
    assert result["winner"] == eager["winner"]
    for arm in ("lr-0", "lr-1", "lr-2"):
        compiled_losses = [r["loss"] for r in result["loss_curves"][arm]]
        assert compiled_losses == [r["loss"] for r in eager["loss_curves"][arm]]
    assert result["training_tokens_processed"] == eager["training_tokens_processed"]
    assert result["training_throughput"]["tokens"] == 9 * 32
    assert result["status"] == "completed"
    assert result["last_checkpoint_resume"]["tested"] is True
    row = result["compile_probe"]
    if row["status"] != "completed" and not _toolchain_available():
        pytest.skip("torch.compile cannot build even a trivial CPU function here")
    assert row["status"] == "completed", row
    child = pilot.root / "compile-probe" / "run" / "probe-compiled-4" / "record.json"
    assert json.loads(child.read_text())["config"]["train"]["compile"] is True
    assert row["microbatch"] == result["chosen_microbatch"] == eager["chosen_microbatch"] == 4
    assert row["updates"] == row["steps_requested"] == 3
    assert row["warmup_updates_excluded"] == 1 and row["exit_code"] == 0
    assert row["compile"] is True and row["peak_allocated_bytes"] == 0  # CPU: no GPU evidence.
    assert "teardown_error" not in row and "kill_error" not in row
    assert row["speedup"] == pytest.approx(
        row["tokens_per_second"] / row["eager_tokens_per_second"]
    )
    assert row["speedup"] > 0 and row["first_update_seconds"] > 0
    assert row["tokens_per_probe_update"] == 4 * 8 * 2
    if row["break_even_tokens"] is not None:
        assert row["break_even_training_updates"] == pytest.approx(row["break_even_tokens"] / 32)
        assert row["break_even_tokens"] == pytest.approx(row["break_even_probe_updates"] * 64)
    assert len(result["loss_curves"]["probe-compiled-4"]) == 3
    assert row["elapsed_seconds"] < 60


def test_expired_eager_probe_window_never_stops_the_compile_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The soft eager window expires during the second probe, as in the probe-window
    # test above; the measurement-only compile probe must neither inherit it nor stop
    # the pilot.
    now = [5000.0]
    pilot = fake_child_worker(
        tmp_path, monkeypatch, "complete", clock=lambda: now[0], pod_created=4850
    )
    original = pilot.on_step

    def advance(trainer: Trainer, metrics: dict[str, float]) -> None:
        if pilot.phase == "probe-4" and trainer.step == 1:
            now[0] += 70
        original(trainer, metrics)

    pilot.on_step = advance  # type: ignore[method-assign]
    result = pilot.execute()
    assert result["probe_window_expired"] is True and result["chosen_microbatch"] == 2
    row = result["compile_probe"]
    assert row["status"] == "completed" and row["microbatch"] == 2 and row["updates"] == 3
    assert result["status"] == "completed" and len(result["arms"]) == 3
    assert "stop_reason" not in result


def test_compile_failure_is_recorded_and_the_pilot_trains_eagerly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = fake_child_worker(tmp_path, monkeypatch, "fail")
    result = pilot.execute()
    row = result["compile_probe"]
    assert row["status"] == "failed" and row["error_type"] == "ChildExit"
    assert row["error_source"] == "stderr tail" and "compiler" in row["error"]
    assert SECRET not in row["error"] and "[REDACTED]" in row["error"]
    assert SECRET not in (pilot.root / "report.json").read_text()
    assert row["updates"] == 0 and row["exit_code"] == 1
    assert result["status"] == "completed" and len(result["arms"]) == 3
    assert "stop_reason" not in result


@pytest.mark.parametrize(
    ("mode", "status", "error_type"),
    [
        ("result-oom", "oom", "OutOfMemoryError"),
        ("result-failed", "failed", "BackendCompilerFailed"),
    ],
)
def test_child_recorded_failures_map_status_and_redact_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, status: str, error_type: str
) -> None:
    pilot = fake_child_worker(tmp_path, monkeypatch, mode)
    result = pilot.execute()
    row = result["compile_probe"]
    assert (row["status"], row["error_type"], row["error_source"]) == (
        status,
        error_type,
        "child exception",
    )
    assert row["error"].startswith("boom with ")
    assert SECRET not in row["error"] and "abcdefghijkl" not in row["error"]
    report_text = (pilot.root / "report.json").read_text()
    assert SECRET not in report_text and "abcdefghijkl" not in report_text
    assert (row["peak_allocated_bytes"], row["peak_reserved_bytes"]) == (7, 9)
    assert result["status"] == "completed" and len(result["arms"]) == 3


def test_completed_child_that_crashes_in_teardown_keeps_its_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot = fake_child_worker(tmp_path, monkeypatch, "complete-then-abort")
    result = pilot.execute()
    row = result["compile_probe"]
    assert row["status"] == "completed" and row["updates"] == 3, row
    assert row["speedup"] > 0 and row["exit_code"] != 0
    assert row["teardown_error"]["exit_code"] == row["exit_code"]
    assert result["status"] == "completed" and len(result["arms"]) == 3


def test_compile_probe_timeout_kills_the_group_and_restarts_the_idle_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    pilot: Worker
    directory = tmp_path / "pilot" / "compile-probe"

    def sleep(_: float) -> None:
        time.sleep(0.02)
        if (directory / "grandchild.pid").exists():
            # One jump past both the 60 s window and the 180 s idle limit: the arms
            # only run if the idle clock restarts when the probe ends.
            now[0] += 200

    pilot = fake_child_worker(tmp_path, monkeypatch, "sleep", clock=lambda: now[0], sleep=sleep)
    result = pilot.execute()
    row = result["compile_probe"]
    assert row["status"] == "timeout" and row["updates"] == 0
    assert "did not finish within the window" in row["note"]
    assert row["elapsed_seconds"] >= 200 and row["exit_code"] is None
    assert _dead(int((directory / "child.pid").read_text()))
    assert _dead(int((directory / "grandchild.pid").read_text()))
    assert result["status"] == "completed" and len(result["arms"]) == 3
    assert "stop_reason" not in result


def test_compile_probe_window_keeps_every_finished_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    pilot: Worker

    def sleep(_: float) -> None:
        time.sleep(0.02)
        rows = pilot.root / "compile-probe" / "rows.jsonl"
        if rows.exists() and rows.read_text().count("\n") >= 1:
            now[0] += 61

    pilot = fake_child_worker(
        tmp_path, monkeypatch, "one-then-sleep", clock=lambda: now[0], sleep=sleep
    )
    result = pilot.execute()
    row = result["compile_probe"]
    assert row["status"] == "incomplete" and row["updates"] == 1
    assert row["first_update_seconds"] == 3.0 and "speedup" not in row
    assert len(result["loss_curves"]["probe-compiled-4"]) == 1
    assert _dead(int((pilot.root / "compile-probe" / "child.pid").read_text()))
    assert result["status"] == "completed" and len(result["arms"]) == 3


def test_wall_limit_during_the_compile_probe_stops_the_pilot_and_kills_the_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    pilot: Worker
    directory = tmp_path / "pilot" / "compile-probe"

    def sleep(_: float) -> None:
        time.sleep(0.02)
        if (directory / "grandchild.pid").exists():
            now[0] += 400  # Past the 300 s pod wall; the compile window is not the cause.

    pilot = fake_child_worker(tmp_path, monkeypatch, "sleep", clock=lambda: now[0], sleep=sleep)
    result = pilot.execute()
    row = result["compile_probe"]
    assert row["status"] == "stopped" and row["stop_reason"] == "pod_wall_limit"
    assert result["status"] == "stopped" and result["stop_reason"] == "pod_wall_limit"
    assert not result["arms"]
    assert _dead(int((directory / "child.pid").read_text()))
    assert _dead(int((directory / "grandchild.pid").read_text()))


def test_compile_probe_refreshes_the_idle_clock_and_heartbeat_while_the_child_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    seen: list[tuple[float, dict[str, Any]]] = []
    pilot: Worker
    directory = tmp_path / "pilot" / "compile-probe"

    def sleep(_: float) -> None:
        seen.append((now[0], json.loads((pilot.root / "heartbeat.json").read_text())))
        assert pilot.last_activity == now[0]  # Refreshed on every poll inside the window.
        now[0] += 40
        if len(seen) == 4:
            (directory / "go").write_text("go")
            _wait_for(directory / "result.json")
            time.sleep(0.3)  # The child exits right after writing its result.
        time.sleep(0.01)

    pilot = fake_child_worker(
        tmp_path,
        monkeypatch,
        "complete-on-go",
        spec=compile_spec(compile_probe_seconds=170),
        clock=lambda: now[0],
        sleep=sleep,
    )
    result = pilot.execute()
    assert len(seen) >= 4
    for when, beat in seen:
        assert beat["compile_probe_active"] is True and beat["phase"] == "probe-compiled-4"
        assert beat["last_gpu_activity_unix"] == when
    row = result["compile_probe"]
    assert row["status"] == "completed" and row["updates"] == 3, row
    assert row["first_update_seconds"] == 3.0 and row["compile"] is True
    assert result["status"] == "completed" and len(result["arms"]) == 3
    assert "stop_reason" not in result
    final = json.loads((pilot.root / "heartbeat.json").read_text())
    assert final["compile_probe_active"] is False


def _unreaped_exited_child() -> subprocess.Popen[bytes]:
    """A new-session child that has exited but was never reaped: a zombie group leader."""
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(process.pid)], capture_output=True, text=True
        ).stdout.strip()
        if state.startswith("Z"):
            return process
        time.sleep(0.02)
    raise AssertionError("child did not exit")


def test_kill_group_tolerates_an_exited_unreaped_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    # On macOS killpg on a group whose only member is a zombie leader raises EPERM.
    process = _unreaped_exited_child()
    assert _kill_group(process) is None and process.returncode == 0
    assert _kill_group(process) is None  # Idempotent after reaping.
    # The race itself: the child exits after the last poll and before the kill.
    raced = _unreaped_exited_child()
    monkeypatch.setattr(raced, "poll", lambda: None)
    assert _kill_group(raced) is None and raced.returncode == 0


def _child_request(tmp_path: Path, *, launched: float, window: float = 60) -> Path:
    """A compile-probe request for an eager CPU child: fast and deterministic."""
    request = {
        "schema_version": 1,
        "config": asdict(trainer_config(compile_spec(), 4, 0.0003, probe=True)),
        "documents": document_source([[256, 1, 2, 3] * 32]),
        "run_id": "probe-compiled-4",
        "run_root": str(tmp_path / "run"),
        "data_hash": "authored",
        "tokenizer_hash": "authored",
        "device": "cpu",
        "num_threads": torch.get_num_threads(),
        "launched_unix": launched,
        "window_seconds": window,
        "rows_path": str(tmp_path / "rows.jsonl"),
        "result_path": str(tmp_path / "result.json"),
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    return path


def test_child_soft_stops_at_an_update_boundary_before_the_window(tmp_path: Path) -> None:
    # The soft stop point (window minus the margin) has already passed, so the child
    # keeps its first update and stops before the second.
    path = _child_request(tmp_path, launched=time.time() - 60 + COMPILE_SOFT_STOP_SECONDS)
    assert run_compile_probe_child(path) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert (result["status"], result["updates"]) == ("incomplete", 1)
    assert (result["peak_allocated_bytes"], result["peak_reserved_bytes"]) == (0, 0)
    assert result["torch_version"] == torch.__version__ and result["compile"] is False
    assert len(_completed_rows(tmp_path / "rows.jsonl")) == 1


def test_child_records_a_sanitized_failure_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Trainer:
        raise RuntimeError("inductor exploded holding " + SECRET)

    monkeypatch.setattr("tools.gpu_pilot.Trainer", broken)
    path = _child_request(tmp_path, launched=time.time())
    assert run_compile_probe_child(path) == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert (result["status"], result["error_type"], result["updates"]) == (
        "failed",
        "RuntimeError",
        0,
    )
    assert SECRET not in result["error"] and "[REDACTED]" in result["error"]
    assert (result["peak_allocated_bytes"], result["peak_reserved_bytes"]) == (0, 0)


def test_completed_rows_ignore_a_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    good = [{"loss": 1.0, "tokens": 8.0, "step_seconds": 0.1, "step": float(i)} for i in (1, 2)]
    path.write_text("".join(json.dumps(row) + "\n" for row in good) + '{"loss": 1.0, "tok')
    assert _completed_rows(path) == good
    path.write_text(json.dumps({**good[0], "step": 2.0}) + "\n")
    assert _completed_rows(path) == []  # Steps must start at 1 and increase by one.
    assert _completed_rows(tmp_path / "missing.jsonl") == []


def test_compile_probe_specification_rules() -> None:
    base = specification()
    compiled_training = replace(base.config, train=replace(base.config.train, compile=True))
    with pytest.raises(ValueError, match="keep compile off"):
        compile_spec(config=compiled_training)
    with pytest.raises(ValueError, match="warmup"):
        compile_spec(compile_probe_warmup_updates=2)  # The eager probe has only 2 updates.
    with pytest.raises(ValueError, match="positive time window"):
        compile_spec(compile_probe_seconds=0)
    with pytest.raises(ValueError, match="must not reserve"):
        replace(base, compile_probe_seconds=60)
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(base, compile_probe_steps=-1)
    # The budgeted window is the only idle exception, so it may not exceed the idle limit.
    assert compile_spec(compile_probe_seconds=180).compile_probe_seconds == 180
    with pytest.raises(ValueError, match="must not exceed the idle limit"):
        compile_spec(compile_probe_seconds=181)
    with pytest.raises(ValueError, match="exceed the pod lifetime"):
        compile_spec(
            compile_probe_seconds=180,
            limits=replace(base.limits, finalization_reserve_seconds=100),
        )
    with pytest.raises(ValueError, match="measured only by the compile probe"):
        trainer_config(base, 2, 0.0003, probe=False, compiled=True)
    assert trainer_config(compile_spec(), 2, 0.0003, probe=True, compiled=True).train.compile
    assert not trainer_config(compile_spec(), 2, 0.0003, probe=True).train.compile


def test_readiness_reserves_the_compile_probe_window() -> None:
    assert check_readiness(receipt(), now=1200, spec=specification()) == 1000
    with pytest.raises(ValueError, match="too short"):
        check_readiness(receipt(), now=1200, spec=compile_spec())
    assert check_readiness(receipt(), now=1165, spec=compile_spec()) == 1000


def test_readiness_measures_remaining_lifetime_to_the_armed_deadline() -> None:
    # 120 s are needed. At 1175 the 1300 s wall leaves 125 s but the timer armed for
    # 1290 leaves only 115 s, and the worker plans against the armed deadline.
    with pytest.raises(ValueError, match="too short"):
        check_readiness(receipt(), now=1175, spec=compile_spec())
    at_the_wall = {**receipt(), "self_destruct_deadline_unix": 1300}
    assert check_readiness(at_the_wall, now=1175, spec=compile_spec()) == 1000


def test_preflight_gate_requires_the_on_device_fused_attention_check() -> None:
    hardware = {"device_type": "cuda", "name": "NVIDIA A40"}
    passed = {
        "status": "completed",
        "hardware_precision_and_resume_checked": True,
        "device": "cuda:0",
        "hardware": hardware,
    }
    base = specification()
    fused = replace(
        base, config=replace(base.config, train=replace(base.config.train, attention="fused"))
    )
    with pytest.raises(ValueError, match="fused attention"):
        check_preflight(passed, hardware=hardware, spec=fused)
    with pytest.raises(ValueError, match="fused attention"):
        check_preflight(
            {**passed, "fused_attention": {"status": "failed"}}, hardware=hardware, spec=fused
        )
    check_preflight(
        {**passed, "fused_attention": {"status": "passed"}}, hardware=hardware, spec=fused
    )
    check_preflight(
        passed, hardware=hardware, spec=base
    )  # Reference attention needs no kernel check.
    for bad in ({"device": "cpu"}, {"status": "failed"}, {"hardware": {"name": "other"}}):
        with pytest.raises(ValueError, match="matching A40"):
            check_preflight({**passed, **bad}, hardware=hardware, spec=base)
