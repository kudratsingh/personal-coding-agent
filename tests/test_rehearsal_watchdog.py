"""The rehearsal watchdog: its own PID file per Pod, and the ``uploading`` transfer window."""

import json
import os
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools import rehearsal_ops, runpod_watchdog
from tools.runpod_watchdog import (
    Outcome,
    PodIdentity,
    Policy,
    RunpodClient,
    build_parser,
    check_once,
    parse_timestamp,
    release_pidfile,
    write_pidfile,
)

IDENTITY = {
    "id": "podfixture01",
    "name": "rehearsal-a40-s-a",
    "gpuId": "NVIDIA A40",
    "gpuCount": 1,
    "dataCenterId": "CA-MTL-1",
    "createdAt": "2027-01-15T08:00:00+00:00",
}
CREATED = parse_timestamp(IDENTITY["createdAt"])
REHEARSAL = Policy(
    45_000,
    7.54,
    600,
    0.6,
    billing_started_at_unix=CREATED,
    max_transfer_seconds=rehearsal_ops.MAX_TRANSFER_SECONDS,
    mode="rehearsal",
)


class API:
    """The Runpod REST surface the watchdog reads: the Pod alive, then gone after DELETE."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append(method)
        if method == "DELETE":
            return 204, b""
        if "DELETE" in self.calls:
            return 404, b""
        pod = {
            "id": IDENTITY["id"],
            "name": IDENTITY["name"],
            "gpu": {"id": IDENTITY["gpuId"], "count": 1},
            "dataCenterId": IDENTITY["dataCenterId"],
            "createdAt": IDENTITY["createdAt"],
            "cost": 0.6,
        }
        return 200, json.dumps(pod).encode()


def guard(
    tmp_path: Path,
    status: str,
    *,
    idle: float,
    progress_age: float | None,
    policy: Policy = REHEARSAL,
    into_run: float = 30_000,
) -> Outcome:
    """One watchdog check ``idle`` seconds after the last GPU work, ``into_run`` s in."""
    activity = CREATED + into_run
    now = activity + idle
    beat: dict[str, Any] = {
        "schema_version": 1,
        "updated_at_unix": now - 5,  # The worker refreshes it every 5 s while uploading.
        "last_gpu_activity_unix": activity,
        "phase": "finalizing",
        "step": 15_259,
        "status": status,
    }
    if progress_age is not None:
        beat["transfer_progress_unix"] = now - progress_age  # Only the Mac stamps this.
    path = tmp_path / f"{status}-{idle}-{progress_age}-{policy.mode}.json"
    path.write_text(json.dumps(beat))
    return check_once(
        RunpodClient("authored-test-key", API()),
        PodIdentity.from_dict(IDENTITY),
        policy,
        path,
        now=lambda: now,
        sleep=lambda _: None,
        observed=True,
    )


def test_uploading_holds_the_pod_for_the_transfer_window_with_mac_seen_progress(
    tmp_path: Path,
) -> None:
    # 700 s after the last GPU work the idle limit has passed, but the worker is uploading
    # and the Mac saw a new R2 object 50 s ago: the watchdog waits.
    assert guard(tmp_path, "uploading", idle=700, progress_age=50).state == "watching"
    assert guard(tmp_path, "awaiting_transfer", idle=700, progress_age=50).state == "watching"
    # The transfer window is at most max_transfer_seconds after the last GPU work.
    late = guard(tmp_path, "uploading", idle=900, progress_age=10)
    assert (late.state, late.reason) == ("terminated", "idle_gpu_limit_delete_204_get_404")
    # Without Mac-observed progress, or with progress older than the idle limit, the
    # 600 s idle rule applies: a hung Pod cannot extend its own lease.
    for age in (None, 600):
        outcome = guard(tmp_path, "uploading", idle=700, progress_age=age)
        assert outcome.reason.startswith("idle_gpu_limit")
    # Training statuses never get the window.
    assert guard(tmp_path, "running", idle=700, progress_age=50).reason.startswith("idle_gpu")
    # Before the idle limit an uploading Pod is simply within limits.
    assert guard(tmp_path, "uploading", idle=300, progress_age=None).state == "watching"


def test_the_pilot_policy_never_accepts_an_uploading_heartbeat(tmp_path: Path) -> None:
    pilot = Policy(5_400, 3.0, 600, 0.6, billing_started_at_unix=CREATED)
    # Under a pilot policy the status stays malformed, exactly as before this mode existed.
    outcome = guard(tmp_path, "uploading", idle=10, progress_age=5, policy=pilot, into_run=1_000)
    assert outcome.reason.startswith("heartbeat_malformed")
    assert "uploading" not in runpod_watchdog.MODE_STATUSES["pilot"]


def old_pod_heartbeat(path: Path, **extra: Any) -> None:
    """What a reused controller directory still holds: the previous Pod's last mirror."""
    beat = {
        "schema_version": 1,
        "updated_at_unix": CREATED - 3_600,  # An hour before this Pod was created.
        "last_gpu_activity_unix": CREATED - 3_700,
        "phase": "finalizing",
        "step": 15_259,
        "status": "shutdown_ready",
        "mac_receipt_verified": True,
        "github_receipt_verified": True,
        "pod_id": "previous-pod",
        **extra,
    }
    path.write_text(json.dumps(beat))


def check(path: Path, now: float, api: API, *, policy: Policy = REHEARSAL) -> Outcome:
    return check_once(
        RunpodClient("authored-test-key", api),
        PodIdentity.from_dict(IDENTITY),
        policy,
        path,
        now=lambda: now,
        sleep=lambda _: None,
        observed=False,  # A new Pod's first polls.
    )


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({}, "heartbeat_names_another_pod"),
        ({"pod_id": IDENTITY["id"]}, "heartbeat_predates_pod_creation"),
        ({"pod_id": None}, "heartbeat_names_another_pod"),
        ({"status": "running", "pod_id": IDENTITY["id"]}, "heartbeat_predates_pod_creation"),
    ],
)
def test_a_reused_controller_directory_never_deletes_the_new_pod(
    tmp_path: Path, extra: dict[str, Any], reason: str
) -> None:
    """The previous Pod's mirrored heartbeat reads as absent, never as stale or ready."""
    path = tmp_path / "supervisor" / "heartbeat.json"
    path.parent.mkdir()
    old_pod_heartbeat(path, **extra)
    baseline = CREATED + 120  # The new watchdog's idle baseline (runbook step 5).
    policy = replace(REHEARSAL, idle_baseline_unix=baseline)
    api = API()
    outcome = check(path, CREATED + 140, api, policy=policy)
    assert (outcome.state, outcome.reason, outcome.detail) == ("watching", "within_limits", reason)
    assert api.calls == ["GET"]
    # Absent, not fresh: the pre-heartbeat idle rule from the baseline still ends it.
    late = check(path, baseline + 600, API(), policy=policy)
    assert late.reason == "idle_gpu_limit_delete_204_get_404" and late.detail == reason


def test_without_the_identity_rule_the_same_file_deletes_the_new_pod(tmp_path: Path) -> None:
    """The finding's reproduction: an hour-old heartbeat reads as stale and deletes."""
    path = tmp_path / "heartbeat.json"
    old_pod_heartbeat(path, pod_id=IDENTITY["id"])
    beat = runpod_watchdog.read_heartbeat(
        path, now=CREATED + 140, max_idle_seconds=600, mode="rehearsal"
    )
    assert beat.error == "heartbeat_stale"
    ignored = runpod_watchdog.read_heartbeat(
        path,
        now=CREATED + 140,
        max_idle_seconds=600,
        mode="rehearsal",
        pod_id="podfixture01",
        created_at_unix=CREATED,
    )
    assert (ignored.error, ignored.last_gpu_activity_unix, ignored.ignored) == (
        None,
        0.0,
        "heartbeat_predates_pod_creation",
    )


def test_this_pods_own_heartbeat_is_still_read(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    activity = CREATED + 2_000
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at_unix": activity + 5,
                "last_gpu_activity_unix": activity,
                "phase": "stage1",
                "step": 10,
                "status": "running",
                "pod_id": IDENTITY["id"],
            }
        )
    )
    watching = check(path, activity + 30, API())
    assert watching.state == "watching" and watching.detail is None
    idle = check(path, activity + 600, API())
    assert idle.reason == "idle_gpu_limit_delete_204_get_404" and idle.detail is None


def test_the_pilot_keeps_reading_heartbeats_without_the_creation_rule(tmp_path: Path) -> None:
    """The pilot path is unchanged: no creation-time rule; another Pod's ID is still ignored."""
    pilot = Policy(5_400, 3.0, 600, 0.6, billing_started_at_unix=CREATED)
    path = tmp_path / "heartbeat.json"
    old_pod_heartbeat(path)
    unnamed = json.loads(path.read_text())
    del unnamed["pod_id"]
    path.write_text(json.dumps(unnamed))
    stale = check(path, CREATED + 140, API(), policy=pilot)
    assert stale.reason == "heartbeat_stale_delete_204_get_404"
    old_pod_heartbeat(path)  # Naming another Pod: absent, then the creation-time baseline.
    outcome = check(path, CREATED + 140, API(), policy=pilot)
    assert outcome.state == "watching" and outcome.detail == "heartbeat_names_another_pod"


def test_the_runbook_window_is_the_d35_one_and_within_the_ceiling() -> None:
    assert REHEARSAL.max_transfer_seconds == 900 <= runpod_watchdog.MAX_TRANSFER_SECONDS
    assert rehearsal_ops.MAX_TRANSFER_SECONDS == rehearsal_ops.FINALIZATION_RESERVE_SECONDS
    with pytest.raises(ValueError, match="1800"):
        Policy(45_000, 7.54, 600, 0.6, max_transfer_seconds=1_801, mode="rehearsal")


def arguments(tmp_path: Path, *extra: str) -> list[str]:
    identity = tmp_path / "pod-identity.json"
    identity.write_text(
        json.dumps(
            {
                "id": "podfixture01",
                "name": "rehearsal-a40-s-a",
                "gpuId": "NVIDIA A40",
                "gpuCount": 1,
                "dataCenterId": "CA-MTL-1",
                "createdAt": "2027-01-15T08:00:00+00:00",
            }
        )
    )
    return [
        "--mode",
        "rehearsal",
        "--identity",
        str(identity),
        "--heartbeat",
        str(tmp_path / "supervisor" / "heartbeat.json"),
        "--receipt",
        str(tmp_path / "watchdog-receipt.json"),
        "--max-wall-seconds",
        "45000",
        "--max-cost-usd",
        "7.54",
        "--quoted-total-hourly",
        "0.60",
        *extra,
    ]


def test_pidfile_names_this_process_and_is_released_only_by_it(tmp_path: Path) -> None:
    path = tmp_path / "ctrl" / "watchdog.pid"
    write_pidfile(path)
    assert path.read_text() == f"{os.getpid()}\n"
    release_pidfile(path)
    assert not path.exists()
    path.write_text("1\n")  # Another run's watchdog, or a newer one, now owns the file.
    release_pidfile(path)
    assert path.read_text() == "1\n"
    release_pidfile(tmp_path / "missing.pid")


def test_main_holds_the_pidfile_while_watching_and_releases_it_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    path = tmp_path / "watchdog.pid"
    seen: list[str] = []

    def watch(*args: Any) -> int:
        seen.append(path.read_text().strip())
        return 0

    monkeypatch.setattr(runpod_watchdog, "_watch", watch)
    before = signal.getsignal(signal.SIGTERM)
    assert runpod_watchdog.main(arguments(tmp_path, "--pidfile", str(path))) == 0
    assert seen == [str(os.getpid())] and not path.exists()
    assert signal.getsignal(signal.SIGTERM) == before  # The handler is restored.


def test_sigterm_from_a_per_run_restart_releases_the_pidfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    path = tmp_path / "watchdog.pid"

    def watch(*args: Any) -> int:
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM must stop the watchdog")

    monkeypatch.setattr(runpod_watchdog, "_watch", watch)
    with pytest.raises(SystemExit) as error:
        runpod_watchdog.main(arguments(tmp_path, "--pidfile", str(path)))
    assert error.value.code == 128 + signal.SIGTERM and not path.exists()


def test_without_a_pidfile_the_watchdog_behaves_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    calls: list[int] = []

    def watch(*args: Any) -> int:
        calls.append(1)
        return 0

    monkeypatch.setattr(runpod_watchdog, "_watch", watch)
    assert runpod_watchdog.main(arguments(tmp_path)) == 0
    assert calls == [1] and not list(tmp_path.glob("*.pid"))
    assert build_parser().parse_args(arguments(tmp_path)).pidfile is None
