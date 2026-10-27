"""Post-training safeguards outside the worker: the ceiling-derived lifetime, the in-Pod idle
self-destruct, the watchdog's post-training mode, the R2 prefix guard on the round trip and
the weights-only upload rate. Offline; no Runpod or R2 call is made."""

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from test_pod_self_destruct import Clock, FakeTransport, timer

from tools import pod_progress, posttrain_ops, rehearsal_ops, runpod_watchdog
from tools.pod_self_destruct import ABSOLUTE_DEADLINE_SECONDS, SelfDestruct, activity_unix
from tools.pod_self_destruct import main as self_destruct_main
from tools.rehearsal_run import MemoryBucket
from tools.runpod_watchdog import Heartbeat, Policy, read_heartbeat

SPEC = Path(__file__).resolve().parents[1] / "configs" / "rehearsal-a40-s-post-1.json"


# The ceiling-derived lifetime ---------------------------------------------------------------


def test_run_one_limits_derive_from_the_three_dollar_ceiling() -> None:
    limits = posttrain_ops.ceiling_limits(3.00, 0.60)
    assert limits.worker_cost_stop_usd == 3.00
    assert limits.watchdog_cost_stop_usd == 2.99
    assert limits.wall_seconds == 18_000
    assert limits.self_destruct_seconds == 17_820
    assert limits.deadline_unix(1_000.0) == 18_820.0
    # Never the pretraining rehearsal's fixed 60,000 s: another ceiling gives another wall.
    assert posttrain_ops.ceiling_limits(5.80, 0.60).wall_seconds == 34_800
    assert posttrain_ops.ceiling_limits(3.00, 0.59).wall_seconds == 18_305  # Rounded down.
    with pytest.raises(ValueError, match="absolute"):
        posttrain_ops.ceiling_limits(11.0, 0.60)  # 66,000 s: above the timer's 60,000 s cap.
    with pytest.raises(ValueError):
        posttrain_ops.ceiling_limits(0.0, 0.60)


def test_the_specification_yields_the_policy_and_the_timer_deadline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = tmp_path / "policy.json"
    arguments = ["limits", "--specification", str(SPEC), "--origin", "1000"]
    assert posttrain_ops.main([*arguments, "--policy-output", str(policy)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["self_destruct_deadline_unix"] == 18_820
    written = json.loads(policy.read_text())
    assert written == record["policy"]
    parsed = Policy(**written)
    assert (parsed.mode, parsed.max_wall_seconds, parsed.max_cost_usd) == (
        "posttrain",
        18_000,
        2.99,
    )
    assert (parsed.max_idle_seconds, parsed.max_transfer_seconds) == (600, 900)
    assert posttrain_ops.main([*arguments, "--field", "self_destruct_seconds"]) == 0
    assert capsys.readouterr().out.strip() == "17820"


def test_the_self_destruct_accepts_the_smaller_lifetime_and_keeps_its_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap passes the derived 17,820 s timer lifetime; 60,000 s stays the absolute
    cap. A deadline past the lifetime it is given is refused."""
    started = time.time()
    armed: list[dict[str, Any]] = []

    class Stop(Exception):
        pass

    def capture(self: SelfDestruct) -> str:
        armed.append({"deadline": self.deadline, "idle": self.idle_seconds, "files": self.activity})
        raise Stop

    monkeypatch.setattr(SelfDestruct, "run", capture)
    base = ["--pod-id", "podpost01", "--receipt", str(tmp_path / "r.json"), "--dry-run"]
    with pytest.raises(Stop):
        self_destruct_main(
            [
                *base,
                "--deadline-unix",
                str(started + 17_820),
                "--max-lifetime-seconds",
                "17820",
                "--idle-seconds",
                "900",
                "--activity",
                str(tmp_path / "heartbeat.json"),
            ]
        )
    assert armed == [
        {"deadline": started + 17_820, "idle": 900.0, "files": (str(tmp_path / "heartbeat.json"),)}
    ]
    for deadline, ceiling in (
        (started + 17_900, "17820"),
        (started + 1, str(ABSOLUTE_DEADLINE_SECONDS + 1)),
    ):
        with pytest.raises(SystemExit):
            self_destruct_main(
                [*base, "--deadline-unix", str(deadline), "--max-lifetime-seconds", ceiling]
            )
    with pytest.raises(SystemExit):  # An idle limit needs something to watch.
        self_destruct_main([*base, "--deadline-unix", str(started + 60), "--idle-seconds", "900"])


# The in-Pod idle self-destruct ------------------------------------------------------------


def write(path: Path, **values: Any) -> None:
    path.write_text(json.dumps(values))


def idle_timer(tmp_path: Path, clock: Clock, files: list[Path]) -> SelfDestruct:
    transport = FakeTransport([(204, {}, b""), (404, {}, b"")])
    return timer(
        tmp_path,
        transport,
        clock,
        idle_seconds=900.0,
        activity=tuple(str(path) for path in files),
    )


def test_the_pod_deletes_itself_after_900_seconds_without_activity(tmp_path: Path) -> None:
    clock = Clock()
    unit = idle_timer(tmp_path, clock, [tmp_path / "heartbeat.json"])
    unit.deadline = unit.deadline_monotonic = 10**12  # Only the idle rule can fire.
    assert unit.due() is None
    clock.sleep(899.0)
    assert unit.due() is None
    clock.sleep(1.0)
    assert unit.due() == "idle_limit"
    assert unit.terminate("idle_limit") == "terminated"
    receipt = json.loads(Path(unit.receipt_path).read_text())
    assert receipt["reason"] == "idle_limit_delete_confirmed_absent"
    assert receipt["idle_seconds"] == 900.0 and receipt["activity_files"]


def test_gpu_work_and_progress_stamps_hold_the_pod_but_nothing_else_does(tmp_path: Path) -> None:
    clock = Clock()
    heartbeat, setup = tmp_path / "heartbeat.json", tmp_path / "setup-progress.json"
    unit = idle_timer(tmp_path, clock, [heartbeat, setup])
    unit.deadline = unit.deadline_monotonic = 10**12
    clock.sleep(800.0)
    pod_progress.stamp(str(setup), "bootstrap:fetch", clock=lambda: clock.wall)  # Bootstrap step.
    clock.sleep(800.0)
    assert unit.due() is None
    write(heartbeat, last_gpu_activity_unix=clock.wall, updated_at_unix=clock.wall)
    clock.sleep(899.0)
    assert unit.due() is None
    # Run 2's hook: progress_unix (the Mac's reward files arriving) holds the Pod too.
    write(heartbeat, last_gpu_activity_unix=clock.wall - 899, progress_unix=clock.wall)
    clock.sleep(899.0)
    assert unit.due() is None
    # A heartbeat that is only rewritten (updated_at) is no activity, and neither is a stamp
    # from the future or a malformed file.
    write(heartbeat, last_gpu_activity_unix=clock.wall - 899, updated_at_unix=clock.wall)
    setup.write_text(json.dumps({"progress_unix": clock.wall + 3_600}))
    clock.sleep(1.0)
    assert unit.due() == "idle_limit"
    assert activity_unix(str(setup), now=clock.wall) is None
    heartbeat.write_text("{not json")
    assert activity_unix(str(heartbeat), now=clock.wall) is None
    assert activity_unix(str(tmp_path / "missing.json"), now=clock.wall) is None


def test_no_activity_file_can_crash_or_stall_the_timer(tmp_path: Path) -> None:
    """A huge integer, a deep document, a FIFO or a directory at an activity path reads as no
    activity; the timer's loop never raises or blocks, so its deadline still fires."""
    huge, deep, fifo, folder = (tmp_path / name for name in ("huge", "deep", "fifo", "dir"))
    huge.write_text('{"progress_unix": ' + "9" * 400 + "}")
    deep.write_text("[" * 60_000)
    os.mkfifo(fifo)
    folder.mkdir()
    for path in (huge, deep, fifo, folder):
        assert activity_unix(str(path), now=time.time()) is None
    clock = Clock()
    unit = idle_timer(tmp_path, clock, [huge, deep, fifo, folder])
    assert unit.due() is None  # Nothing raised, nothing blocked on the FIFO.
    clock.sleep(10.0)
    assert unit.due() == "wall_clock_deadline"


def test_an_activity_check_that_fails_never_skips_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    unit = idle_timer(tmp_path, clock, [tmp_path / "heartbeat.json"])

    def broken(self: SelfDestruct) -> float:
        raise OverflowError("a writer's stamp out of range")

    monkeypatch.setattr(SelfDestruct, "observe_activity", broken)
    assert unit.due() is None  # A failed read is no new activity, not a crash.
    clock.sleep(10.0)
    assert unit.run() == "terminated"
    assert json.loads(Path(unit.receipt_path).read_text())["reason"].startswith(
        "wall_clock_deadline"
    )


def test_a_receipt_write_failure_never_ends_the_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    unit = idle_timer(tmp_path, clock, [tmp_path / "heartbeat.json"])
    original = SelfDestruct.record
    failures = [0]

    def full_disk(self: SelfDestruct) -> None:
        if self.state == "armed":
            failures[0] += 1
            raise OSError(28, "No space left on device")
        original(self)

    monkeypatch.setattr(SelfDestruct, "record", full_disk)
    assert unit.run() == "terminated" and failures[0] >= 1


def test_an_idle_limit_needs_activity_files_and_stays_bounded(tmp_path: Path) -> None:
    clock = Clock()
    transport = FakeTransport([])
    for options in (
        {"idle_seconds": 900.0},
        {"idle_seconds": 30.0, "activity": ("a",)},
        {"idle_seconds": 7_200.0, "activity": ("a",)},
        {"activity": ("a",)},
    ):
        with pytest.raises(ValueError):
            timer(tmp_path, transport, clock, **options)


def test_progress_stamps_are_atomic_and_plain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "progress" / "setup.json"
    assert pod_progress.main(["--file", str(path), "--source", "runbook:receipt"]) == 0
    value = json.loads(path.read_text())
    assert value["source"] == "runbook:receipt" and math.isfinite(value["progress_unix"])
    with pytest.raises(SystemExit):
        pod_progress.main(["--file", str(path), "--source", "bad label; rm"])
    assert not list(path.parent.glob(".progress.*"))


# The watchdog's post-training mode ---------------------------------------------------------


def beat(path: Path, status: str, *, now: float, activity: float) -> None:
    write(
        path,
        schema_version=1,
        phase="eval-base",
        step=3,
        status=status,
        last_gpu_activity_unix=activity,
        updated_at_unix=now,
        pod_id="podpost01",
    )


def test_the_watchdog_reads_post_training_heartbeats_under_its_own_mode(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    now = 10_000.0
    beat(path, "awaiting_rewards", now=now, activity=now - 10)
    found = read_heartbeat(path, now=now, max_idle_seconds=600, mode="posttrain")
    assert found == Heartbeat(now - 10, None, "awaiting_rewards")
    # Pilot and rehearsal policies keep their own status sets unchanged.
    for mode in ("pilot", "rehearsal"):
        assert read_heartbeat(path, now=now, max_idle_seconds=600, mode=mode).error == (
            "heartbeat_malformed"
        )
    beat(path, "uploading", now=now, activity=now - 10)
    assert read_heartbeat(path, now=now, max_idle_seconds=600, mode="posttrain").error is None
    assert runpod_watchdog.MODE_WALL_SECONDS["posttrain"] == ABSOLUTE_DEADLINE_SECONDS
    with pytest.raises(ValueError, match="ceiling"):
        Policy(60_001, 2.99, 600, 0.6, mode="posttrain")


def test_the_watchdog_refuses_flags_other_than_its_policy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped $WALL or $WATCHDOG_COST can never start a weaker watchdog than the policy
    the readiness receipt checked."""
    origin = 1_800_000_000.0
    limits = posttrain_ops.ceiling_limits(3.00, 0.60)
    written = posttrain_ops.watchdog_policy(limits, origin=origin, idle_baseline=origin)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(written))
    flags = Policy(
        18_000,
        2.99,
        600,
        0.6,
        origin,
        max_transfer_seconds=900,
        idle_baseline_unix=origin + 300,  # Re-baselined on a restart: not pinned.
        mode="posttrain",
    )
    runpod_watchdog.check_policy_file(flags, path)
    for change in (
        {"max_wall_seconds": 60_000},
        {"max_cost_usd": 9.99},
        {"billing_started_at_unix": origin + 600},
        {"mode": "rehearsal"},
    ):
        weaker = Policy(**{**asdict(flags), **change})
        with pytest.raises(ValueError, match="differ from the policy file"):
            runpod_watchdog.check_policy_file(weaker, path)
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "id": "podpost01",
                "name": "posttrain-a40-s-1",
                "gpuId": "NVIDIA A40",
                "gpuCount": 1,
                "dataCenterId": "test",
                "createdAt": "2027-01-15T08:00:00+00:00",
            }
        )
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder-not-a-key")
    argv = [
        "--mode",
        "posttrain",
        "--identity",
        str(identity),
        "--heartbeat",
        str(tmp_path / "heartbeat.json"),
        "--receipt",
        str(tmp_path / "receipt.json"),
        "--max-wall-seconds",
        "60000",  # A wrong shell variable.
        "--max-cost-usd",
        "2.99",
        "--quoted-total-hourly",
        "0.60",
        "--billing-started-at-unix",
        str(origin),
        "--max-transfer-seconds",
        "900",
        "--policy",
        str(path),
    ]
    with pytest.raises(SystemExit):  # Refused before any API call is made.
        runpod_watchdog.main(argv)
    assert not (tmp_path / "receipt.json").exists()


def test_reward_waits_get_no_idle_extension_in_run_one(tmp_path: Path) -> None:
    """``awaiting_rewards`` is an in-progress status: the idle rule applies unchanged until run 2
    adds its Mac-stamped reward progress."""
    path = tmp_path / "heartbeat.json"
    now = 10_000.0
    beat(path, "awaiting_rewards", now=now, activity=now - 700)
    heartbeat = read_heartbeat(path, now=now, max_idle_seconds=600, mode="posttrain")
    policy = Policy(18_000, 2.99, 600, 0.6, max_transfer_seconds=900, mode="posttrain")
    assert not runpod_watchdog._transfer_in_progress(heartbeat, policy, now)
    assert "awaiting_rewards" not in runpod_watchdog.TRANSFER_STATUSES


# The R2 round trip under the run prefix, gated on the weights-only upload rate ---------------


def test_one_weights_label_per_drain_sets_the_upload_rate() -> None:
    assert posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND == pytest.approx(550e6 / 360 * 2)
    assert (
        posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND
        < rehearsal_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND
    )
    assert posttrain_ops.S_WEIGHTS_LABEL_BYTES >= 135_285_504 * 4  # float32 weights only.


def test_the_round_trip_stays_inside_the_run_prefix(tmp_path: Path) -> None:
    bucket = MemoryBucket()
    prefix = posttrain_ops.run_prefix("1", "podpost01")
    store = posttrain_ops.GuardedStore(bucket.store(), prefix)
    ticks = iter(range(100))
    record = rehearsal_ops.round_trip(
        store,
        run="1",
        pod_id="podpost01",
        receipt=tmp_path / "round-trip.json",
        size=4096,
        clock=lambda: float(next(ticks)),
        prefix=prefix,
        required_upload_bytes_per_second=posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND,
    )
    assert record["status"] == "passed" and record["key"].startswith(prefix + "preflight/")
    assert (
        record["required_upload_bytes_per_second"] == posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND
    )
    assert not bucket.objects and not store.refused
    with pytest.raises(posttrain_ops.PrefixViolation):
        rehearsal_ops.round_trip(
            store,
            run="1",
            pod_id="other",
            receipt=tmp_path / "r2.json",
            size=16,
            prefix="posttrain-runs/1/other/",
        )


def test_the_guard_needs_a_plain_prefix() -> None:
    bucket = MemoryBucket()
    for prefix in ("", "/abs/", "posttrain-runs/1", "posttrain-runs/../x/"):
        with pytest.raises(ValueError):
            posttrain_ops.GuardedStore(bucket.store(), prefix)
