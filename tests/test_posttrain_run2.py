"""Post-training run 2's worker: the specification and its gate, RL and the control through the
reward bridge, the stops that end the run, the liveness rules during a wait, and the uploads.

Every run uses the run-2 smoke's fixtures (a tiny model, withheld fixture tasks with the
Mac's facts, scripted programs) with the Mac's ``BridgeServer`` in a thread over the
directory transport, and an in-memory bucket. Nothing touches a network, no generated code
runs (the Mac scores with fakes; ``test_posttrain_run2_sandbox.py`` runs the sandbox), and the
reference BPE encoder is disabled so every prompt goes through the cached encoder.
"""

import copy
import json
import os
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.rl.bridge import BridgeRefused
from scratchcoder.rl.bridge_mac import LEASE_FORMAT
from scratchcoder.rl.fakes import FakeRewardSource
from scratchcoder.rl.rewards import RewardHostFailure, RewardTimeout
from scratchcoder.rl.withheld import is_withheld
from scratchcoder.tokenizer.bpe import ByteBPE
from tools import posttrain_approved, posttrain_ops
from tools.pod_self_destruct import SelfDestruct
from tools.posttrain_run import (
    BRIDGE_LOST_STOP,
    WAIT_MARGIN_STOP,
    Rates,
    RunLimits,
    Specification,
    launch_fit,
    planned_seconds,
)
from tools.posttrain_run2 import BRIDGE_BOXES, Run2Worker
from tools.posttrain_smoke import patched_benchmarks
from tools.posttrain_smoke2 import (
    HIDDEN_MARKERS,
    SMOKE2_POD_ID,
    Fixtures2,
    ReferenceDev,
    fixtures2,
    mac_server,
    mac_thread,
    make_run2_worker,
    scripted_programs,
)
from tools.rehearsal_run import SMOKE_CREDENTIALS, MemoryBucket

TEMPLATE = posttrain_approved.SPECIFICATION_PATH_2
PREFIX = posttrain_ops.run_prefix("2", SMOKE2_POD_ID)
STAGES = ["rl", "more-sft", "arc-post", "eval-rl", "eval-more-sft"]


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> Fixtures2:
    torch.set_num_threads(1)
    return fixtures2(tmp_path_factory.mktemp("posttrain-run2-fixtures"))


@dataclass
class Ran:
    report: dict[str, Any]
    worker: Run2Worker
    bucket: MemoryBucket
    beats: list[dict[str, Any]]
    error: BaseException | None = None


def reference_encoder(self: ByteBPE, text: str, *, allow_special: bool = False) -> list[int]:
    raise AssertionError("the reference BPE encoder ran on the worker's hot path")


def execute(
    found: Fixtures2,
    root: Path,
    *,
    mac: bool = True,
    reward_source: Any = None,
    mac_dev: Any = None,
    mac_wall: Callable[[], float] | None = None,
    pod_task_sha256: dict[str, str] | None = None,
    setup: Callable[[Run2Worker, Any], None] | None = None,
    raises: type[BaseException] | None = None,
    **options: Any,
) -> Ran:
    """Run a run-2 worker with the Mac in a thread; every heartbeat it writes is kept.
    ``setup`` sees the worker and the Mac's server before the run starts."""
    bucket = MemoryBucket()
    worker = make_run2_worker(found, root, bucket=bucket, **options)
    beats: list[dict[str, Any]] = []
    original = Run2Worker.heartbeat

    def keep(self: Run2Worker) -> None:
        original(self)
        beats.append(json.loads((self.root / "heartbeat.json").read_text()))

    error: BaseException | None = None
    with (
        patched_benchmarks(),
        scripted_programs(found.tokenizer),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr(ByteBPE, "encode", reference_encoder)
        patch.setattr(Run2Worker, "heartbeat", keep)
        if mac:
            server = mac_server(
                found,
                worker.plan,
                pod_root=worker.bridge_root,
                state=root.parent / f"{root.name}-mac",
                wall=mac_wall or worker.clock,
                reward_source=reward_source,
            )
            if pod_task_sha256 is not None:
                server.pod_task_sha256 = pod_task_sha256
            if mac_dev is not None:
                server.dev_execute = mac_dev
            if setup is not None:
                setup(worker, server)
            with mac_thread(server):
                error = run(worker, raises)
        else:
            error = run(worker, raises)
    return Ran(worker.report, worker, bucket, beats, error)


def run(worker: Run2Worker, raises: type[BaseException] | None) -> BaseException | None:
    if raises is None:
        worker.execute()
        return None
    with pytest.raises(raises) as caught:
        worker.execute()
    return caught.value


@pytest.fixture(scope="module")
def completed(found: Fixtures2, tmp_path_factory: pytest.TempPathFactory) -> Ran:
    return execute(found, tmp_path_factory.mktemp("posttrain-run2") / "run")


def statuses(report: dict[str, Any]) -> dict[str, str]:
    return {stage["name"]: stage["status"] for stage in report["stages"]}


def labels(report: dict[str, Any]) -> set[str]:
    return {row["label"] for row in report["artifacts"]}


class SlowClock:
    """Unix time that moves a thousand times slower than real time: a run's limits see no
    time pass, whatever the runner's speed, yet the clock never stands still."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.start, self.real = start, time.monotonic()

    def __call__(self) -> float:
        return self.start + (time.monotonic() - self.real) / 1_000


class JumpClock(SlowClock):
    """``SlowClock`` that a test can move forward at once (``offset``)."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = 0.0

    def __call__(self) -> float:
        return super().__call__() + self.offset


class FastClock:
    """A shared fake clock that moves only when the bridge client sleeps between looks at its
    inbox, by ``step`` seconds a look; ``on_sleep`` plays the Mac's part (a lease renewal)."""

    def __init__(self, *, step: float, on_sleep: Callable[[float], None] | None = None) -> None:
        self.now, self.step, self.on_sleep = 1_800_000_000.0, step, on_sleep
        self.sleeps = 0

    def wall(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now - 1_800_000_000.0

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += self.step
        if self.on_sleep is not None:
            self.on_sleep(self.now)


def lease_writer(root: Path, *, until: float | None = None) -> Callable[[float], None]:
    """The Mac's lease renewals, dated by the Pod's clock; none after ``until`` (a dead Mac)."""

    def renew(now: float) -> None:
        if until is not None and now > until:
            return
        root.mkdir(parents=True, exist_ok=True)
        body = {"format": LEASE_FORMAT, "server": "f" * 32, "renewed": now, "seconds": 60.0}
        (root / "server.lease").write_text(json.dumps(body))

    return renew


# The specification and its gate ------------------------------------------------------------


def gate_values() -> dict[str, Any]:
    pin = {"size_bytes": 10, "sha256": "a" * 64}
    return {
        "ceiling_usd": 5.80,
        "sft-weights": {"key": "posttrain-runs/1/podpost01/sft-weights/checkpoint.pt", **pin},
        "rl-tasks": dict(pin),
        "rl-dev": dict(pin),
        "rl-facts": dict(pin),
        "pool-test": dict(pin),
    }


def test_the_tracked_template_is_the_approved_run_two_with_its_gate_values_null() -> None:
    raw = json.loads(TEMPLATE.read_text())
    assert raw == posttrain_approved.run2_specification()
    with pytest.raises(ValueError, match="gate values are still null"):
        Specification.from_dict(raw)
    with pytest.raises(ValueError, match="gate values are still null"):
        posttrain_approved.check_run2(raw)
    spec = Specification.from_dict(raw, allow_gate_nulls=True)
    assert [stage.name for stage in spec.stages] == STAGES
    assert raw["limits"]["ceiling_usd"] is None
    for name in ("sft-weights", "rl-tasks", "rl-dev", "rl-facts", "pool-test"):
        assert raw["inputs"][name]["sha256"] is None, name
    assert raw["inputs"]["sft-weights"]["key"] is None
    # The frozen protocol.
    rl = spec.rl_config()
    assert (rl.updates, rl.tasks_per_update, rl.group_size) == (100, 32, 8)
    assert (rl.temperature, rl.top_p, rl.max_new_tokens) == (1.0, 1.0, 256)
    assert (rl.learning_rate, rl.warmup_updates, rl.weight_decay, rl.gradient_clip) == (
        3e-6,
        5,
        0.1,
        1.0,
    )
    assert (rl.beta1, rl.beta2, rl.shuffle_seed) == (0.9, 0.95, 0)
    assert rl.dev_checks == (0, 12, 25, 50, 75, 100) and rl.patience == 3
    assert (rl.leakage_rate, rl.collapse_distinct, rl.collapse_updates) == (0.01, 2.0, 5)
    assert (spec.rl["dev_tasks"], spec.rl["dev_samples"], spec.rl["attention"]) == (200, 8, "fused")
    control = spec.control_config(attention="fused")
    assert (control.learning_rate, control.warmup_updates, control.pass_checks) == (
        6e-5,
        10,
        (1, 2, 4, 8),
    )
    assert (control.batch_size, control.accumulation_steps, control.tolerance) == (16, 2, 0.05)
    files = spec.stage("eval-rl").files
    assert [entry.name for entry in files] == [
        "pool-dev-greedy",
        "pool-dev-n8",
        "pool-test-greedy",
        "pool-test-n8",
        "humaneval-greedy",
        "humaneval-plus-greedy",
        "mbpp-plus-greedy",
    ]
    assert {entry.protocol for entry in files} == {"chat-v1"}
    text = TEMPLATE.read_text()
    assert "private/" not in text and "D-3" not in text


def test_the_gate_fills_every_null_and_the_approved_check_refuses_any_other_change() -> None:
    filled = posttrain_approved.run2_specification(gate_values())
    posttrain_approved.check_run2(filled)
    spec = Specification.from_dict(filled)
    assert spec.ceiling.wall_seconds == 34_800 and spec.ceiling.self_destruct_seconds == 34_620
    for path, value in (
        (("rl", "learning_rate"), 1e-5),
        (("rl", "dev_checks"), [0, 50, 100]),
        (("control", "learning_rate"), 1e-4),
        (("bridge", "root"), "/tmp/bridge"),
        (("limits", "pod_idle_seconds"), 3_600),
        (("planning", "reward_wait_seconds_per_update"), 10),
        (("stages", 3, "files", 0, "protocol"), "base-pool-completion-v1"),
        (("inputs", "pool-dev", "sha256"), "0" * 64),
        (("inputs", "tokenizer", "key"), "elsewhere/tokenizer.json"),
    ):
        changed = copy.deepcopy(filled)
        target: Any = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="differs from the approved run 2"):
            posttrain_approved.check_run2(changed)
    with pytest.raises(ValueError, match="exactly"):
        posttrain_approved.fill_gate(posttrain_approved.run2_specification(), {"ceiling_usd": 5.8})
    with pytest.raises(ValueError, match="SHA-256"):
        posttrain_approved.run2_specification(
            {**gate_values(), "rl-dev": {"size_bytes": 10, "sha256": "A" * 64}}
        )
    with pytest.raises(ValueError, match="absolute"):
        posttrain_approved.run2_specification({**gate_values(), "ceiling_usd": 11.0})


def test_the_gate_command_takes_its_values_from_the_gates_own_files(
    found: Fixtures2, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scratchcoder.eval.export import write_export
    from scratchcoder.eval.tasks import save_tasks
    from tools.posttrain_smoke2 import _function_task

    weights = found.inputs["sft-weights"]
    report = {
        "kind": "posttrain-run-report",
        "run": "1",
        "pod_id": "podpost01",
        "status": "completed",
        "artifacts": [
            {
                "label": "sft-weights",
                "name": "checkpoint.pt",
                "key": "posttrain-runs/1/podpost01/sft-weights/checkpoint.pt",
                "sha256": "b" * 64,
                "bytes": 541_142_400,
            }
        ],
    }
    run1 = tmp_path / "report.json"
    run1.write_text(json.dumps(report))
    test = [_function_task(f"smoke2/test/{i}", "test", i, str(i)) for i in range(2_000)]
    save_tasks(tmp_path / "test.jsonl", test)
    pool_test = tmp_path / "test-export.jsonl"
    write_export(
        "pool", tasks_path=tmp_path / "test.jsonl", output=pool_test, record=tmp_path / "r"
    )
    output = tmp_path / "spec-2.json"
    argv = [
        "run2-gate",
        "--ceiling-usd",
        "5.80",
        "--run1-report",
        str(run1),
        "--rl-pod-export",
        str(found.root / "rl-pod-export"),
        "--pool-test-export",
        str(pool_test),
        "--output",
        str(output),
    ]
    assert posttrain_approved.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    filled = json.loads(output.read_text())
    assert filled["limits"]["ceiling_usd"] == 5.8
    assert filled["inputs"]["sft-weights"] == {
        "kind": "checkpoint",
        "key": "posttrain-runs/1/podpost01/sft-weights/checkpoint.pt",
        "size_bytes": 541_142_400,
        "sha256": "b" * 64,
    }
    record = json.loads((found.root / "rl-pod-export" / "rl-pod-export.json").read_text())
    for name in ("rl-tasks", "rl-dev", "rl-facts"):
        assert filled["inputs"][name]["sha256"] == record["files"][name]["sha256"]
    posttrain_approved.check_run2(filled)
    import hashlib

    assert printed["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert posttrain_approved.main(argv) == 1  # A launch specification is never rewritten.
    report["status"] = "stopped"
    run1.write_text(json.dumps(report))
    assert posttrain_approved.main([*argv[:-1], str(tmp_path / "other.json")]) == 1
    assert weights.exists()


def test_run_two_fits_its_ceiling_at_the_planning_rates() -> None:
    spec = Specification.from_dict(posttrain_approved.run2_specification(gate_values()))
    plan = planned_seconds(spec, Rates.planning(spec))
    stages = plan["stages"]
    assert stages["rl"] == pytest.approx(
        20
        + 100 * 256 * 256 / 10_000
        + 100 * 256 * 1_024 / 56_765
        + 100 * 48
        + 6 * (1_800 * 512 / 5_000 + 300 + 5)
    )
    assert stages["more-sft"] == pytest.approx(
        20 + 100 * 256 * 256 / 10_000 + 100 * 256 * 1_024 / 56_765 + 5 * (1_800 * 512 / 5_000 + 305)
    )
    assert stages["eval-rl"] == pytest.approx(7 * 20 + 18_792_960 / 5_000)
    assert stages["arc-post"] == pytest.approx(2 * (20 + 54))
    lifetime = spec.ceiling.self_destruct_seconds
    fit = launch_fit(spec, remaining_seconds=lifetime - spec.limits["startup_seconds"])
    assert fit["fits_at_planning"] and fit["planning_seconds_needed"] == pytest.approx(
        21_302, abs=1
    )


# A completed run --------------------------------------------------------------------------------


def test_every_stage_completes_and_every_artifact_reaches_r2(completed: Ran) -> None:
    report = completed.report
    assert report["status"] == "completed" and statuses(report) == dict.fromkeys(
        STAGES, "completed"
    )
    names = labels(report)
    assert {"rl", "rl-check-0000", "rl-check-0002", "rl-check-0003", "more-sft", "bridge"} <= names
    assert {f"arc-post-{arm}" for arm in ("rl", "more-sft")} <= names
    assert len([name for name in names if name.startswith("eval-")]) == 14
    assert any(name.startswith("more-sft-check-") for name in names)
    assert all(key.startswith(PREFIX) for key in completed.bucket.objects)
    for row in report["artifacts"]:
        assert completed.bucket.digests[row["key"]] == row["sha256"]
    weights = [row for row in report["artifacts"] if row["name"] == "checkpoint.pt"]
    assert all(
        set(r["name"] for r in report["artifacts"] if r["label"] == w["label"])
        == {
            "checkpoint.pt",
            "record.json",
        }
        for w in weights
    )  # Weights labels are weights only: the SSH rescue can carry them.
    beat = json.loads((completed.worker.root / "heartbeat.json").read_text())
    assert beat["status"] == "awaiting_transfer" and beat["r2_final"]["artifact_count"] == len(
        report["artifacts"]
    )


def test_rl_and_the_control_record_their_seconds_selection_and_mismatch(completed: Ran) -> None:
    report = completed.report
    rl = report["rl"]
    assert rl["stop_reason"] == "completed" and rl["updates"] == 3
    assert rl["busy_seconds"] == pytest.approx(
        rl["seconds"]["rollout_generation"] + rl["seconds"]["training_step"]
    )
    assert rl["compile_seconds"] == 0.0
    assert set(rl["excluded_seconds"]) == {
        "reward_wait",
        "dev_generation",
        "dev_wait",
        "checkpoint",
    }
    assert [row["check"] for row in rl["checks"]] == [0, 2, 3]
    control = report["more_sft"]
    assert control["budget_seconds"] == rl["busy_seconds"]
    assert control["mismatch_seconds"] == pytest.approx(
        control["busy_seconds"] - rl["busy_seconds"]
    )
    assert control["attention"] == rl["attention"] == "reference"
    assert control["start_check"]["label"] == "rl-check-0000"
    for arm in ("rl", "more-sft"):
        selection = report["selection"][arm]
        assert selection["rule"].startswith("the selectable check with the highest")
        chosen = selection["selected"]
        best = max(
            selection["checks"],
            key=lambda row: (row["greedy_pass_at_1"], row["mean_pass_n8"], row["check"]),
        )
        assert chosen["check"] == best["check"] and chosen["label"] in labels(report)
    assert report["rates"]["basis"]["mac"] == "measured"
    assert report["rates"]["rl_busy_seconds"] == rl["busy_seconds"]
    identical = report["selection"]["identical_endpoints"]
    assert identical["identical"] is False and identical["sha256"] is None
    assert identical["rl"] == report["selection"]["rl"]["selected"]["label"]


def test_the_waits_heartbeat_awaiting_rewards_and_a_fresh_lease_is_progress(completed: Ran) -> None:
    waiting = [beat for beat in completed.beats if beat["status"] == "awaiting_rewards"]
    assert {beat["awaiting"]["kind"] for beat in waiting} == {"reward", "dev"}
    for beat in waiting:
        assert set(beat["awaiting"]) == {
            "request_id",
            "kind",
            "index",
            "count",
            "started_unix",
            "age_seconds",
        }
    stamped = [beat["progress_unix"] for beat in completed.beats if "progress_unix" in beat]
    assert stamped and stamped == sorted(stamped)  # The Mac's lease, renewed while it scored.
    bridge = completed.report["bridge"]
    assert bridge["lease_last_renewed_unix"] == stamped[-1]
    assert [wait["kind"] for wait in bridge["waits"]][:2] == ["dev", "reward"]
    assert all(wait["outcome"] == "scored" for wait in bridge["waits"])
    assert bridge["requests_sent"] == len(bridge["waits"]) and bridge["lost"] is None


def test_a_dev_checks_generation_keeps_the_gpu_stamp_and_the_heartbeat_live(
    found: Fixtures2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dev check generates 1,800 programs between two waits (about 184 s at the planning
    speed): every batch stamps the GPU activity and refreshes the heartbeat, in both arms, so
    neither idle rule nor the heartbeat's staleness reads it as a hung worker. (The refresh
    interval is zeroed so the check does not depend on how fast this machine generates.)"""
    import tools.posttrain_run2 as run2

    monkeypatch.setattr(run2, "HEARTBEAT_REFRESH_SECONDS", 0.0)
    seen: list[tuple[str, bool, bool]] = []
    original = Run2Worker.dev_tick

    def record(self: Run2Worker, tokens: int) -> None:
        before = self.clock()
        original(self, tokens)
        seen.append((self.stage_name, self.last_activity >= before, self.last_heartbeat >= before))

    monkeypatch.setattr(Run2Worker, "dev_tick", record)
    ran = execute(found, tmp_path / "run")
    report = ran.report
    assert report["status"] == "completed"
    assert {stage for stage, _, _ in seen} == {"rl", "more-sft"}
    assert all(stamped and beat for _, stamped, beat in seen)
    checks = len(report["rl"]["checks"]) + len(report["more_sft"]["checks"])
    assert len(seen) >= 2 * checks  # at least the greedy batch and one sampled batch each


def test_the_bridge_directories_are_kept_and_no_hidden_test_reaches_the_pod(
    found: Fixtures2, completed: Ran
) -> None:
    worker = completed.worker
    archive = worker.root / "exports" / "bridge" / "bridge.tar"
    with tarfile.open(archive) as tar:
        members = tar.getnames()
    assert "server.lease" in members and "client.lock" in members
    assert any(name.startswith("done/") for name in members)
    # 6b's scan, with the run-2 fixtures' own markers: the Pod's inputs, the run's directory,
    # the bridge directories and every object in R2.
    pod_files = [*found.inputs.values(), *worker.root.rglob("*"), *worker.bridge_root.rglob("*")]
    texts = [
        path.read_bytes().decode("utf-8", "replace")
        for path in pod_files
        if path.is_file() and path.suffix not in {".pt"}
    ]
    texts += [data.decode("utf-8", "replace") for data in completed.bucket.objects.values()]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            extracted = tar.extractfile(member)
            if extracted is not None:
                texts.append(extracted.read().decode("utf-8", "replace"))
    leaked = [marker for marker in HIDDEN_MARKERS if any(marker in text for text in texts)]
    assert leaked == []
    full = (found.mac["train"].read_text(), found.mac["dev"].read_text())
    assert all(any(marker in text for text in full) for marker in HIDDEN_MARKERS)  # Live scan.
    for name in ("rl-tasks", "rl-dev"):
        from scratchcoder.eval.tasks import load_tasks

        assert all(is_withheld(task) for task in load_tasks(found.inputs[name]))


def test_the_r2_report_passes_the_macs_final_verification(completed: Ran, tmp_path: Path) -> None:
    beat = json.loads((completed.worker.root / "heartbeat.json").read_text())
    store = completed.bucket.store(SMOKE_CREDENTIALS)
    result = posttrain_ops.verify_final(
        store, beat["r2_final"], run="2", pod_id=SMOKE2_POD_ID, destination=tmp_path
    )
    assert result["status"] == "verified" and result["report_status"] == "completed"


# Stops that end the run --------------------------------------------------------------------


def test_a_reward_wait_timeout_uploads_the_latest_weights_and_finalizes_in_full(
    found: Fixtures2, tmp_path: Path
) -> None:
    """The Mac's own bound passes on update 2: its timeout reply stops RL with reward_wait,
    the weights trained so far go to R2, and the run ends without the control."""
    ran = execute(found, tmp_path / "run", reward_source=FakeRewardSource(timeout_at=2))
    report = ran.report
    assert (report["status"], report["stop_reason"]) == ("stopped", "reward_wait")
    assert report["rl"]["stop_reason"] == "reward_wait" and report["rl"]["updates"] == 1
    assert "rl-stop-weights" in labels(report)
    assert statuses(report) == {
        "rl": "stopped",
        "more-sft": "skipped",
        "arc-post": "skipped",
        "eval-rl": "skipped",
        "eval-more-sft": "skipped",
    }
    beat = json.loads((ran.worker.root / "heartbeat.json").read_text())
    assert beat["status"] == "stopped" and "r2_final" in beat  # The finalization ran in full.
    assert "bridge" in labels(report)
    assert [wait["outcome"] for wait in report["bridge"]["waits"]][-1] == "failed"


def test_a_pod_side_timeout_with_a_live_mac_stops_rl_cleanly(
    found: Fixtures2, tmp_path: Path
) -> None:
    """A Mac that renews its lease but never answers: the Pod's own dev timeout ends the
    wait, before the lost-bridge rule (the lease stays fresh)."""
    bridge = tmp_path / "bridge"
    clock = FastClock(step=100.0, on_sleep=lease_writer(bridge))
    ran = execute(
        found,
        tmp_path / "run",
        mac=False,
        clock=clock.wall,
        bridge_root=bridge,
        bridge_clock=clock.monotonic,
        bridge_sleep=clock.sleep,
    )
    timeouts = ran.worker.timeouts
    assert (ran.report["status"], ran.report["stop_reason"]) == ("stopped", "reward_wait")
    assert ran.report["bridge"]["lost"] is None
    assert "within" in ran.report["rl"]["stop_detail"]
    assert clock.sleeps * clock.step >= timeouts.dev_timeout_seconds
    assert "rl-stop-weights" in labels(ran.report)
    stamps = [
        beat["progress_unix"]
        for beat in ran.beats
        if beat["status"] == "awaiting_rewards" and "progress_unix" in beat
    ]
    assert len(stamps) > 10 and stamps == sorted(stamps) and stamps[-1] > stamps[0]
    withdrawn = list((bridge / "withdrawn").iterdir())
    assert withdrawn and not list((bridge / "outbox").iterdir())


def test_a_dead_mac_is_a_lost_bridge_and_the_pod_dies_within_the_idle_bound(
    found: Fixtures2, tmp_path: Path
) -> None:
    """The Mac renews its lease for a while, then dies: the worker gives the wait up 600 s
    after the last renewal (``bridge_lost``), uploads its weights and finalizes; had it hung
    instead, the in-Pod timer would delete the Pod 900 s after that renewal."""
    bridge = tmp_path / "bridge"
    start = 1_800_000_000.0
    clock = FastClock(step=50.0, on_sleep=lease_writer(bridge, until=start + 300))
    ran = execute(
        found,
        tmp_path / "run",
        mac=False,
        clock=clock.wall,
        bridge_root=bridge,
        bridge_clock=clock.monotonic,
        bridge_sleep=clock.sleep,
    )
    report = ran.report
    assert (report["status"], report["stop_reason"]) == ("stopped", BRIDGE_LOST_STOP)
    lost = report["bridge"]["lost"]
    last = lost["lease_last_renewed_unix"]
    assert last == start + 300 and lost["silent_seconds"] >= 600
    assert lost["at_unix"] - last < ran.worker.timeouts.dev_timeout_seconds
    assert "rl-stop-weights" in labels(report) and "r2_final" in json.loads(
        (ran.worker.root / "heartbeat.json").read_text()
    )
    stamped = [beat for beat in ran.beats if beat.get("progress_unix")]
    assert max(beat["progress_unix"] for beat in stamped) == last
    # The in-Pod timer on the last heartbeat written while waiting, armed when the GPU last
    # worked (the wait's start, 300 s before the last renewal): the lease's stamp, and only
    # it, carries the timer to 900 s after the last renewal.
    waiting = [beat for beat in ran.beats if beat["status"] == "awaiting_rewards"][-1]
    armed = waiting["last_gpu_activity_unix"]
    assert armed == start and waiting["progress_unix"] == last == armed + 300

    def timer(beat: dict[str, Any], name: str) -> tuple[SelfDestruct, list[float]]:
        snapshot = tmp_path / f"{name}.json"
        snapshot.write_text(json.dumps(beat))
        now = [armed]
        unit = SelfDestruct(
            pod_id=SMOKE2_POD_ID,
            token="",
            deadline_unix=armed + 34_000,
            receipt_path=str(tmp_path / f"{name}-self-destruct.json"),
            clock=lambda: now[0],
            monotonic=lambda: 0.0,
            sleep=lambda _: None,
            dry_run=True,
            idle_seconds=900.0,
            activity=(str(snapshot),),
        )
        return unit, now

    unit, now = timer(waiting, "with-lease")
    for at, due in ((armed + 900, None), (last + 899, None), (last + 900, "idle_limit")):
        now[0] = at
        assert unit.due() == due, at
    # Without the lease's stamp the same heartbeat dies 900 s after the GPU's last work.
    stripped = {key: value for key, value in waiting.items() if key != "progress_unix"}
    unit, now = timer(stripped, "without-lease")
    for at, due in ((armed + 899, None), (armed + 900, "idle_limit")):
        now[0] = at
        assert unit.due() == due, at


def test_the_cost_stop_keeps_one_full_wait_free(found: Fixtures2, tmp_path: Path) -> None:
    """With 1,550 s of spend left before the cost stop, dev check 0 (a dev wait of at most
    1,523.5 s here) may start, but no update may: the margin is one full wait, 1,572.5 s."""
    clock = SlowClock()
    probe = make_run2_worker(found, tmp_path / "probe", bucket=MemoryBucket(), clock=clock)
    bounds = probe.timeouts
    assert (bounds.dev_timeout_seconds, bounds.wait_margin_seconds) == (1_523.5, 1_572.5)
    window = 1_550.0
    limits = RunLimits(
        pod_seconds=18_000,
        quoted_total_hourly=0.6,
        cost_stop_usd=3.0,
        idle_seconds=600,
        starting_cost_usd=3.0 - window / 3600 * 0.6,
    )
    ran = execute(found, tmp_path / "run", clock=clock, limits=limits, pod_created=clock())
    report = ran.report
    assert (report["status"], report["stop_reason"]) == ("stopped", WAIT_MARGIN_STOP)
    assert report["rl"]["stop_reason"] == "external" and report["rl"]["updates"] == 0
    assert report["rl"]["stop_detail"].startswith("wait_margin: one full wait would pass the cost")
    margin = report["bridge"]["margin_stop"]
    assert (margin["limit"], margin["wait_seconds"]) == ("cost_limit", 1_572.5)
    assert [row["check"] for row in report["rl"]["checks"]] == [0]
    beat = json.loads((ran.worker.root / "heartbeat.json").read_text())
    assert "r2_final" in beat  # Not the cost stop itself: the finalization runs in full.


def test_a_wait_that_cannot_fit_never_starts(found: Fixtures2, tmp_path: Path) -> None:
    """With 1,500 s of spend left before the cost stop, less than a dev wait (1,523.5 s here),
    dev check 0 is refused before its request is written. (The projected-finish rule, against
    the wall, lets the stage start: the wait's own check is what keeps the cost stop.)"""
    clock = SlowClock()
    limits = RunLimits(
        pod_seconds=18_000,
        quoted_total_hourly=0.6,
        cost_stop_usd=3.0,
        idle_seconds=600,
        starting_cost_usd=3.0 - 1_500 / 3600 * 0.6,
    )
    ran = execute(found, tmp_path / "run", clock=clock, limits=limits, pod_created=clock())
    report = ran.report
    assert (report["status"], report["stop_reason"]) == ("stopped", WAIT_MARGIN_STOP)
    margin = report["bridge"]["margin_stop"]
    assert (margin["kind"], margin["index"], margin["limit"]) == ("dev", 0, "cost_limit")
    assert margin["wait_seconds"] == 1_523.5
    assert report["bridge"]["requests_sent"] == 0
    assert not list((ran.worker.bridge_root / "outbox").iterdir())
    assert report["rl"]["stop_reason"] == "reward_wait" and report["rl"]["checks"] == []


def test_a_limit_reached_during_a_wait_ends_the_run_with_that_limit(
    found: Fixtures2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense in depth behind the margin: with the margin disabled, a cost stop reached while
    the Pod waits ends the wait cleanly and the run with ``cost_limit``."""
    bridge = tmp_path / "bridge"
    clock = FastClock(step=50.0, on_sleep=lease_writer(bridge))
    monkeypatch.setattr(Run2Worker, "margin_reason", lambda self, wait_seconds: None)
    limits = RunLimits(
        pod_seconds=18_000,
        quoted_total_hourly=0.6,
        cost_stop_usd=3.0,
        idle_seconds=600,
        starting_cost_usd=3.0 - 120 / 3600 * 0.6,
    )
    ran = execute(
        found,
        tmp_path / "run",
        mac=False,
        clock=clock.wall,
        bridge_root=bridge,
        bridge_clock=clock.monotonic,
        bridge_sleep=clock.sleep,
        limits=limits,
        pod_created=clock.wall(),
    )
    report = ran.report
    assert (report["status"], report["stop_reason"]) == ("stopped", "cost_limit")
    assert report["rl"]["stop_reason"] == "reward_wait"
    assert "cost_limit stop was reached while waiting" in report["rl"]["stop_detail"]
    assert not list((bridge / "outbox").iterdir())  # The request was withdrawn.


def test_an_rl_early_stop_is_a_result_and_the_control_runs(
    found: Fixtures2, tmp_path: Path
) -> None:
    def collapse(raw: dict[str, Any]) -> None:
        raw["rl"]["collapse_distinct"] = 99.0
        raw["rl"]["collapse_updates"] = 1
        raw["bridge"] = posttrain_approved.bridge_section(
            {key: raw["bridge"][key] for key in posttrain_approved.RUN2_BRIDGE_SETTINGS}, raw["rl"]
        )

    ran = execute(found, tmp_path / "run", changes=collapse)
    report = ran.report
    assert report["status"] == "completed" and statuses(report) == dict.fromkeys(
        STAGES, "completed"
    )
    assert report["rl"]["stop_reason"] == "collapse" and report["rl"]["updates"] == 1
    assert [row["check"] for row in report["rl"]["checks"]] == [0, 1]  # The stop-time check.
    assert report["more_sft"]["stop_reason"] in {"budget", "passes_exhausted"}
    assert report["selection"]["rl"]["selected"]["check"] in {0, 1}


# Stops in the control ------------------------------------------------------------------------


def in_the_control(worker: Run2Worker) -> bool:
    return worker.stage_name == "more-sft"


class FailsInTheControl:
    """The Mac's dev verdicts: the reference rule during RL; ``error`` during the control."""

    def __init__(self, worker: Run2Worker, error: Exception) -> None:
        self.worker, self.error, self.reference = worker, error, ReferenceDev()

    def __call__(self, pairs: Any) -> list[bool]:
        if in_the_control(self.worker):
            raise self.error
        return self.reference(pairs)


def assert_the_control_ended_the_run(ran: Ran, reason: str) -> None:
    """RL's result stands, the control stopped with ``reason``, nothing after it ran, and the
    run finalized in full with the bridge directories kept."""
    report = ran.report
    assert (report["status"], report["stop_reason"]) == ("stopped", reason)
    assert statuses(report) == {
        "rl": "completed",
        "more-sft": "stopped",
        "arc-post": "skipped",
        "eval-rl": "skipped",
        "eval-more-sft": "skipped",
    }
    assert {"rl", "more-sft", "bridge"} <= labels(report)
    events = {row["name"] for row in report["artifacts"] if row["label"] == "more-sft"}
    assert {"control-result.json", "control-events.jsonl"} <= events
    beat = json.loads((ran.worker.root / "heartbeat.json").read_text())
    assert beat["status"] == "stopped" and "r2_final" in beat


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RewardTimeout("the Mac's dev bound passed"), "reward_wait"),
        (RewardHostFailure("the sandbox host failed through its retries"), "reward_host_failure"),
    ],
)
def test_a_channel_stop_in_the_control_ends_the_run_with_its_weights_in_r2(
    found: Fixtures2, tmp_path: Path, error: Exception, reason: str
) -> None:
    """The Mac answers the control's first own dev check with a timeout or a host failure."""
    ran = execute(
        found,
        tmp_path / "run",
        setup=lambda worker, server: setattr(
            server, "dev_execute", FailsInTheControl(worker, error)
        ),
    )
    assert_the_control_ended_the_run(ran, reason)
    control = ran.report["more_sft"]
    assert control["stop_reason"] == reason and control["checks"] == []
    assert "more-sft-stop-weights" in labels(ran.report)
    waits = ran.report["bridge"]["waits"]
    assert waits[-1]["kind"] == "dev" and waits[-1]["outcome"] == "failed"


def test_a_mac_that_dies_during_the_control_is_a_lost_bridge(
    found: Fixtures2, tmp_path: Path
) -> None:
    """The Mac stops renewing its lease while the control waits on a dev check: 700 s later
    the worker gives the wait up as ``bridge_lost``, uploads the control's weights and
    finalizes. (The Mac's lease is dated by the Pod's clock until it dies, and the Pod's clock
    jumps the 700 s at once, so the test does not depend on the runner's speed.)"""
    clock = JumpClock()
    died: list[float] = []

    def mac_wall() -> float:
        return died[0] if died else clock()

    class DiesInTheControl:
        def __init__(self, worker: Run2Worker) -> None:
            self.worker, self.reference = worker, ReferenceDev()

        def __call__(self, pairs: Any) -> list[bool]:
            if not in_the_control(self.worker):
                return self.reference(pairs)
            died.append(clock())  # no renewal is dated later than this
            clock.offset += 700.0
            deadline = time.monotonic() + 120  # wide: only a stalled runner takes this long
            while self.worker.report["bridge"]["lost"] is None:
                assert time.monotonic() < deadline, "the Pod never gave the wait up"
                time.sleep(0.01)
            return [False] * len(pairs)

    ran = execute(
        found,
        tmp_path / "run",
        clock=clock,
        mac_wall=mac_wall,
        setup=lambda worker, server: setattr(server, "dev_execute", DiesInTheControl(worker)),
    )
    assert_the_control_ended_the_run(ran, BRIDGE_LOST_STOP)
    lost = ran.report["bridge"]["lost"]
    assert lost["kind"] == "dev" and lost["silent_seconds"] >= 600
    assert lost["lease_last_renewed_unix"] <= died[0]
    assert ran.report["more_sft"]["stop_reason"] == "reward_wait"
    assert "more-sft-stop-weights" in labels(ran.report)


def test_no_control_update_starts_without_one_full_wait_free(
    found: Fixtures2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RL leaves 1,550 s of spend before the cost stop, less than one full wait (1,572.5 s
    here): the control stops before its first update (``wait_margin``) and finalizes in full."""
    clock = JumpClock()
    window = 10_000.0
    limits = RunLimits(
        pod_seconds=18_000,
        quoted_total_hourly=0.6,
        cost_stop_usd=3.0,
        idle_seconds=600,
        starting_cost_usd=3.0 - window / 3600 * 0.6,
    )
    original = Run2Worker.run_more_sft

    def after_a_long_rl(self: Run2Worker, stage: Any) -> list[dict[str, Any]]:
        clock.offset += window - 1_550.0
        self.last_activity = self.clock()
        return original(self, stage)

    monkeypatch.setattr(Run2Worker, "run_more_sft", after_a_long_rl)
    ran = execute(found, tmp_path / "run", clock=clock, limits=limits, pod_created=clock())
    assert_the_control_ended_the_run(ran, WAIT_MARGIN_STOP)
    control = ran.report["more_sft"]
    assert control["stop_reason"] == "external" and control["updates"] == 0
    assert control["stop_detail"].startswith("wait_margin: one full wait would pass the cost")
    margin = ran.report["bridge"]["margin_stop"]
    assert (margin["kind"], margin["limit"], margin["wait_seconds"]) == (
        "next update",
        "cost_limit",
        1_572.5,
    )


def test_a_refusal_during_the_control_fails_the_run_after_its_weights_and_events_reach_r2(
    found: Fixtures2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers: list[Any] = []
    original = Run2Worker.run_more_sft

    def refused(self: Run2Worker, stage: Any) -> list[dict[str, Any]]:
        servers[0].pod_task_sha256 = {**servers[0].pod_task_sha256, "dev": "d" * 64}
        return original(self, stage)

    monkeypatch.setattr(Run2Worker, "run_more_sft", refused)
    ran = execute(
        found,
        tmp_path / "run",
        setup=lambda worker, server: servers.append(server),
        raises=BridgeRefused,
    )
    report = ran.report
    assert report["status"] == "failed" and report["error_type"] == "BridgeRefused"
    assert report["more_sft"]["status"] == "failed"
    assert {"more-sft-latest-weights", "more-sft", "bridge"} <= labels(report)
    (events,) = [
        row
        for row in report["artifacts"]
        if row["label"] == "more-sft" and row["name"] == "control-events.jsonl"
    ]
    assert b"control_failed" in ran.bucket.objects[events["key"]]
    assert "salvage_errors" not in report


def test_a_file_that_vanishes_during_the_bridge_archive_is_left_out_not_the_label(
    found: Fixtures2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Mac may still push a late reply while the Pod archives its bridge directories: a
    file listed and then moved aside is named and left out; links and FIFOs are never read."""
    worker = make_run2_worker(found, tmp_path / "run", bucket=MemoryBucket())
    root = worker.bridge_root
    for box in BRIDGE_BOXES:
        (root / box).mkdir(parents=True)
    (root / "server.lease").write_text("lease")
    (root / "inbox" / "a.json").write_text("{}")
    (root / "done" / "b.json").write_text('{"x": 1}')
    (root / "outbox" / "link.json").symlink_to(root / "done" / "b.json")
    os.mkfifo(root / "outbox" / "fifo.json")
    listed = Path.iterdir

    def with_a_ghost(self: Path) -> Any:
        yield from listed(self)
        if self.name == "inbox":
            yield self / "gone.json"  # listed, then set aside before it is read

    monkeypatch.setattr(Path, "iterdir", with_a_ghost)
    archived = worker.archive_bridge(tmp_path / "bridge.tar")
    assert archived == {"files": 3, "bytes": 5 + 2 + 8, "vanished": ["inbox/gone.json"]}
    with tarfile.open(tmp_path / "bridge.tar") as tar:
        assert sorted(tar.getnames()) == ["done/b.json", "inbox/a.json", "server.lease"]
        member = tar.extractfile("done/b.json")
        assert member is not None and member.read() == b'{"x": 1}'
    monkeypatch.setattr("tools.posttrain_run2.MAX_BRIDGE_TAR_BYTES", 6)
    with pytest.raises(ValueError, match="bound"):
        worker.archive_bridge(tmp_path / "small.tar")


class FirstCheckOnly:
    """The Mac's dev verdicts: every program passes at the first dev check (RL's update 0),
    none after, so neither arm beats the start."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, pairs: Any) -> list[bool]:
        self.calls += 1
        return [self.calls == 1] * len(pairs)


def test_both_arms_selecting_the_shared_start_is_flagged_in_the_report(
    found: Fixtures2, tmp_path: Path
) -> None:
    """The control's start is RL's update-0 check: when neither arm beats it, both select
    that one file. The run completes, and the report says so before anything is frozen
    (the unseal record refuses one checkpoint under two names)."""
    ran = execute(found, tmp_path / "run", mac_dev=FirstCheckOnly())
    report = ran.report
    assert report["status"] == "completed"
    selection = report["selection"]
    assert selection["rl"]["selected"]["label"] == "rl-check-0000"
    assert selection["more-sft"]["selected"]["label"] == "rl-check-0000"
    identical = selection["identical_endpoints"]
    assert identical["identical"] is True
    assert (identical["rl"], identical["more-sft"]) == ("rl-check-0000", "rl-check-0000")
    assert identical["sha256"] == selection["rl"]["selected"]["sha256"]
    assert "owner rules" in identical["rule"]


def test_a_refusal_by_the_mac_fails_the_run_after_the_latest_weights_reach_r2(
    found: Fixtures2, tmp_path: Path
) -> None:
    ran = execute(
        found,
        tmp_path / "run",
        pod_task_sha256={"reward": "c" * 64, "dev": "d" * 64},
        raises=BridgeRefused,
    )
    report = ran.report
    assert report["status"] == "failed" and report["error_type"] == "BridgeRefused"
    assert "rl-latest-weights" in labels(report)
    beat = json.loads((ran.worker.root / "heartbeat.json").read_text())
    assert beat["status"] == "failed" and "r2_final" in beat


# Refusals before any stage ------------------------------------------------------------------


def test_the_worker_refuses_full_task_files_and_null_gate_values(
    found: Fixtures2, tmp_path: Path
) -> None:
    from tools.posttrain_smoke2 import smoke_specification2

    with pytest.raises(ValueError, match="withheld"):
        Run2Worker(
            Specification.from_dict(smoke_specification2(found)),
            tmp_path / "run",
            inputs={**found.inputs, "rl-tasks": found.mac["train"]},
            store=MemoryBucket().store(SMOKE_CREDENTIALS),
            pod_id=SMOKE2_POD_ID,
            pod_created=time.time(),
        )
    raw = smoke_specification2(found)
    raw["inputs"]["rl-facts"]["sha256"] = None
    with pytest.raises(ValueError, match="gate values are still null"):
        Specification.from_dict(raw)
    raw = smoke_specification2(found)
    raw["inputs"]["rl-facts"]["sha256"] = "0" * 64
    worker = make_run2_worker(
        found,
        tmp_path / "facts",
        bucket=MemoryBucket(),
        changes=lambda r: r.update(inputs=raw["inputs"]),
    )
    with patched_benchmarks(), pytest.raises(ValueError, match="facts file's SHA-256"):
        worker.check_planned_tasks()


def test_records_stay_within_their_bounds(completed: Ran) -> None:
    assert all(len(json.dumps(beat)) < 65_536 for beat in completed.beats)
    assert len((completed.worker.root / "report.json").read_bytes()) < 1_000_000
