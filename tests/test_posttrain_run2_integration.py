"""Run 2's CPU smoke drives the real Mac-side supervisor and the watchdog under run 2's policy.

The worker runs every stage with the Mac's bridge server in a thread; each heartbeat it writes
is kept with ``report.json`` and the bucket at that moment, then replayed to a real
``PosttrainSupervisor`` for run 2 through fake SSH and a fake GitHub, with the Mac's own key
reading the bucket. While a heartbeat reads ``awaiting_rewards`` the watchdog, reading the
supervisor's mirror under the run-2 policy, keeps the idle GPU for the derived extension and
no longer; once the report is verified it ends the Pod on the receipts. The reward-wait stop
is verified and torn down the same way. Nothing touches a network.
"""

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from test_posttrain_integration import Moment, Pod, SmokeRunpod, mirrored
from test_rehearsal_integration import MAC_KEY, REPOSITORY, GitHub

from scratchcoder.rl.fakes import FakeRewardSource
from tools import pilot_supervisor, posttrain_ops
from tools.pilot_supervisor import PosttrainSupervisor, archive_bundle
from tools.posttrain_run import Specification
from tools.posttrain_run2 import Run2Worker
from tools.posttrain_smoke import patched_benchmarks
from tools.posttrain_smoke2 import (
    SMOKE2_POD_ID,
    Fixtures2,
    fixtures2,
    mac_server,
    mac_thread,
    make_run2_worker,
    scripted_programs,
)
from tools.rehearsal_run import MemoryBucket
from tools.runpod_watchdog import PodIdentity, Policy, RunpodClient, check_once

PREFIX = posttrain_ops.run_prefix("2", SMOKE2_POD_ID)


def record(found: Fixtures2, root: Path, **options: Any) -> tuple[Run2Worker, list[Moment]]:
    """Run the worker with the Mac in a thread, keeping every heartbeat with what SSH and R2
    showed at that moment."""
    torch.set_num_threads(1)
    bucket = MemoryBucket()
    reward_source = options.pop("reward_source", None)
    moments: list[Moment] = []
    original = Run2Worker.heartbeat

    def keep(self: Run2Worker) -> None:
        original(self)
        with bucket.lock:
            objects, digests = dict(bucket.objects), dict(bucket.digests)
        moments.append(
            Moment(
                json.loads((self.root / "heartbeat.json").read_text()),
                (self.root / "report.json").read_bytes(),
                objects,
                digests,
            )
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Run2Worker, "heartbeat", keep)
        worker = make_run2_worker(found, root, bucket=bucket, **options)
        server = mac_server(
            found,
            worker.plan,
            pod_root=worker.bridge_root,
            state=root.parent / f"{root.name}-mac",
            reward_source=reward_source,
        )
        with patched_benchmarks(), scripted_programs(found.tokenizer), mac_thread(server):
            worker.execute()
    return worker, moments


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> Fixtures2:
    return fixtures2(tmp_path_factory.mktemp("posttrain-run2-integration-inputs"))


@pytest.fixture(scope="module")
def smoked(found: Fixtures2, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    yield record(found, tmp_path_factory.mktemp("posttrain-run2-integration") / "run")


def supervise(
    root: Path, pod: Pod, monkeypatch: pytest.MonkeyPatch
) -> tuple[PosttrainSupervisor, GitHub, MemoryBucket]:
    github = GitHub()
    monkeypatch.setattr(pilot_supervisor, "_gh", github)
    bucket = MemoryBucket()
    now = [0.0]

    def archive(bundle: Path) -> dict[str, Any]:
        return archive_bundle(
            bundle,
            REPOSITORY,
            run_id=SMOKE2_POD_ID,
            files=("report.json", "summary.json"),
            root="posttrain-runs/2",
            message="Save post-training report and summary",
        )

    unit = PosttrainSupervisor(
        root,
        run="2",
        pod_id=SMOKE2_POD_ID,
        fetch=pod.fetch,
        fetch_report=pod.fetch_report,
        store=bucket.store(MAC_KEY),
        archive=archive,
        clock=lambda: now[0],
    )
    unit.now = now  # type: ignore[attr-defined]
    return unit, github, bucket


def replay(unit: PosttrainSupervisor, pod: Pod, bucket: MemoryBucket, moment: Moment) -> None:
    pod.moment = moment
    bucket.objects, bucket.digests = dict(moment.objects), dict(moment.digests)
    unit.now[0] += 5  # type: ignore[attr-defined]
    unit.refresh_heartbeat()
    unit.cycle()


def identity_at(created_unix: float) -> dict[str, Any]:
    return {
        "id": SMOKE2_POD_ID,
        "name": "posttrain-a40-s-2",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "test",
        "createdAt": datetime.fromtimestamp(created_unix, UTC).isoformat(),
    }


def watchdog(spec: Specification, mirror: Path, *, created: float, now: float) -> tuple[str, str]:
    """The run's watchdog, under the run-2 policy derived from the specification."""
    identity = identity_at(created)
    policy = Policy(
        **posttrain_ops.watchdog_policy(
            spec.ceiling, origin=created, idle_baseline=created, bridge=spec.bridge_timeouts
        )
    )
    outcome = check_once(
        RunpodClient("authored-test-key", SmokeRunpod(identity)),
        PodIdentity.from_dict(identity),
        policy,
        mirror,
        now=lambda: now,
        sleep=lambda _: None,
        observed=True,
    )
    return outcome.state, outcome.reason


def test_the_smoke_waits_on_the_mac_and_ends_in_one_prefix(smoked: Any) -> None:
    worker, moments = smoked
    statuses = [moment.heartbeat["status"] for moment in moments]
    assert statuses[0] == "running" and statuses[-1] == "awaiting_transfer"
    assert "awaiting_rewards" in statuses and "uploading" in statuses
    assert {moment.heartbeat["pod_id"] for moment in moments} == {SMOKE2_POD_ID}
    assert all(key.startswith(PREFIX) for key in moments[-1].objects)
    stages = {moment.heartbeat["stage"] for moment in moments}
    assert {"rl", "more-sft", "arc-post", "eval-rl", "eval-more-sft"} <= stages


def test_teardown_comes_only_after_the_mac_verifies_every_artifact(
    smoked: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, moments = smoked
    pod = Pod()
    unit, github, bucket = supervise(tmp_path / "ctrl", pod, monkeypatch)
    for moment in moments[:-1]:
        replay(unit, pod, bucket, moment)
        assert moment.heartbeat["status"] in {"running", "uploading", "awaiting_rewards"}
        assert not unit.shutdown_ready and unit.r2_verified is None and not github.files
    final = moments[-1]
    replay(unit, pod, bucket, final)
    assert unit.shutdown_ready and unit.r2_verified is not None
    assert unit.r2_verified["report_status"] == "completed"
    assert len(unit.r2_verified["artifacts"]) == final.heartbeat["r2_final"]["artifact_count"]
    prefix = f"posttrain-runs/2/{SMOKE2_POD_ID}/final"
    paths = {endpoint.rsplit("/contents/", 1)[1] for endpoint in github.files}
    assert paths == {f"{prefix}/report.json", f"{prefix}/summary.json"}
    local = mirrored(unit)
    assert local["status"] == "shutdown_ready" and local["github_receipt_verified"] is True
    created = time.time() - 600
    spec = Specification.from_dict(json.loads(final.report)["specification"])
    assert watchdog(spec, unit.root / "heartbeat.json", created=created, now=time.time()) == (
        "terminated",
        "shutdown_ready_verified_receipts_delete_204_get_404",
    )


def test_an_awaiting_heartbeat_through_the_mirror_keeps_the_pod_for_its_bound_only(
    smoked: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The supervisor mirrors an ``awaiting_rewards`` heartbeat as it is; the watchdog keeps the
    idle GPU for the request kind's Pod timeout plus the slack, and deletes it after that."""
    _, moments = smoked
    spec = Specification.from_dict(json.loads(moments[-1].report)["specification"])
    bounds = spec.bridge_timeouts
    assert bounds is not None
    for kind in ("reward", "dev"):
        moment = next(
            m
            for m in moments
            if m.heartbeat["status"] == "awaiting_rewards"
            and m.heartbeat["awaiting"]["kind"] == kind
        )
        pod = Pod()
        unit, _, bucket = supervise(tmp_path / f"ctrl-{kind}", pod, monkeypatch)
        last_gpu = float(moment.heartbeat["last_gpu_activity_unix"])
        extension = bounds.extension(kind)
        for idle, expected in ((extension - 1, "watching"), (extension, "terminated")):
            beat = {**moment.heartbeat, "updated_at_unix": last_gpu + idle - 1}
            replay(unit, pod, bucket, Moment(beat, moment.report, moment.objects, moment.digests))
            mirror = mirrored(unit)
            assert mirror["status"] == "awaiting_rewards" and mirror["awaiting"]["kind"] == kind
            state, reason = watchdog(
                spec, unit.root / "heartbeat.json", created=last_gpu - 100, now=last_gpu + idle
            )
            assert state == expected, (kind, idle, reason)
        assert reason.startswith("idle_gpu_limit")


def test_a_reward_wait_stop_is_verified_and_torn_down(
    found: Fixtures2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, moments = record(found, tmp_path / "run", reward_source=FakeRewardSource(timeout_at=2))
    final = moments[-1]
    assert final.heartbeat["status"] == "stopped" and "r2_final" in final.heartbeat
    report = json.loads(final.report)
    assert report["stop_reason"] == "reward_wait"
    pod = Pod()
    unit, _, bucket = supervise(tmp_path / "ctrl", pod, monkeypatch)
    replay(unit, pod, bucket, final)
    assert unit.shutdown_ready and unit.r2_verified is not None
    assert unit.r2_verified["report_status"] == "stopped"
    keys = {row["key"] for row in unit.r2_verified["artifacts"]}
    assert f"{PREFIX}rl-stop-weights/checkpoint.pt" in keys
    assert f"{PREFIX}bridge/bridge.tar" in keys
