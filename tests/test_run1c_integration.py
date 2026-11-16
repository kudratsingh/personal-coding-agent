"""Run 1c's CPU smoke drives the real Mac-side supervisor and watchdog, gate passing or not.

The worker runs run 1c's stages on the smoke's fixtures (a weights-only parent, one pass, the
final weights kept, the real format gate) into an in-memory bucket; each heartbeat is kept
with ``report.json`` and the bucket at that moment and replayed to a real
``PosttrainSupervisor`` through fake SSH and a fake GitHub, the Mac's own key reading the
bucket. The watchdog under the policy run 1c's cost limit derives (120% of the smoke's
estimate, below the $1.50 ceiling) then ends the Pod on the verified receipts.
"""

import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from test_posttrain_integration import Moment, Pod, Recorded, SmokeRunpod, mirrored, replay
from test_rehearsal_integration import MAC_KEY, REPOSITORY, GitHub

from tools import pilot_supervisor, posttrain_ops
from tools.pilot_supervisor import PosttrainSupervisor, archive_bundle
from tools.posttrain_run import FORMAT_GATE_STOP, PosttrainWorker
from tools.posttrain_smoke import (
    SMOKE_POD_ID_1C,
    Fixtures,
    failing_gate_1c,
    fixtures_1c,
    make_worker_1c,
    patched_benchmarks,
)
from tools.rehearsal_run import MemoryBucket
from tools.runpod_watchdog import PodIdentity, Policy, RunpodClient, check_once

PREFIX = posttrain_ops.run_prefix("1c", SMOKE_POD_ID_1C)


def record_1c(found: Fixtures, root: Path, **options: Any) -> Recorded:
    torch.set_num_threads(1)
    bucket = MemoryBucket()
    moments: list[Moment] = []
    original = PosttrainWorker.heartbeat

    def keep(self: PosttrainWorker) -> None:
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
        patch.setattr(PosttrainWorker, "heartbeat", keep)
        worker = make_worker_1c(found, root, bucket=bucket, **options)
        with patched_benchmarks():
            worker.execute()
    return Recorded(worker, bucket, moments)


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> Fixtures:
    return fixtures_1c(tmp_path_factory.mktemp("posttrain-1c-inputs"))


@pytest.fixture(scope="module")
def passed(found: Fixtures, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Recorded]:
    yield record_1c(found, tmp_path_factory.mktemp("posttrain-1c") / "run")


@pytest.fixture(scope="module")
def failed(found: Fixtures, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Recorded]:
    yield record_1c(
        found, tmp_path_factory.mktemp("posttrain-1c-fail") / "run", changes=failing_gate_1c
    )


def supervise(
    root: Path, pod: Pod, bucket: MemoryBucket, monkeypatch: pytest.MonkeyPatch
) -> tuple[PosttrainSupervisor, GitHub]:
    github = GitHub()
    monkeypatch.setattr(pilot_supervisor, "_gh", github)

    def archive(bundle: Path) -> dict[str, Any]:
        return archive_bundle(
            bundle,
            REPOSITORY,
            run_id=SMOKE_POD_ID_1C,
            files=("report.json", "summary.json"),
            root="posttrain-runs/1c",
            message="Save post-training report and summary",
        )

    now = [0.0]
    unit = PosttrainSupervisor(
        root,
        run="1c",
        pod_id=SMOKE_POD_ID_1C,
        fetch=pod.fetch,
        fetch_report=pod.fetch_report,
        store=bucket.store(MAC_KEY),
        archive=archive,
        clock=lambda: now[0],
    )
    unit.now = now  # type: ignore[attr-defined]
    return unit, github


def watchdog_ends_the_pod(unit: PosttrainSupervisor, recorded: Recorded) -> tuple[str, str]:
    """The watchdog under the policy the smoke's cost limit derives."""
    derived = recorded.worker.plan.ceiling
    assert isinstance(derived, posttrain_ops.EstimateLimits) and derived.limit_usd < 1.5
    policy = posttrain_ops.watchdog_policy(derived, origin=time.time() - 600, idle_baseline=0.0)
    created = datetime.fromtimestamp(time.time() - 600, UTC).isoformat()
    identity = {
        "id": SMOKE_POD_ID_1C,
        "name": "posttrain-a40-s-1c",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "test",
        "createdAt": created,
    }
    outcome = check_once(
        RunpodClient("authored-test-key", SmokeRunpod(identity)),
        PodIdentity.from_dict(identity),
        Policy(
            policy["max_wall_seconds"],
            policy["max_cost_usd"],
            600,
            0.6,
            max_transfer_seconds=900,
            mode="posttrain",
        ),
        unit.root / "heartbeat.json",
        now=time.time,
        sleep=lambda _: None,
        observed=True,
    )
    return outcome.state, outcome.reason


def test_a_passing_gate_runs_every_stage_into_run_1cs_prefix(passed: Recorded) -> None:
    final = json.loads(passed.moments[-1].report)
    assert final["status"] == "completed"
    assert [stage["name"] for stage in final["stages"]] == [
        "probe",
        "sft",
        "format-gate",
        "eval-sft",
    ]
    assert all(stage["status"] == "completed" for stage in final["stages"])
    assert final["sft"]["kept_weights"] == "final"
    assert final["format_gate"]["checkpoint_sha256"] == final["sft"]["selected"]["sha256"]
    assert all(key.startswith(PREFIX) for key in passed.bucket.objects)
    assert f"{PREFIX}environment/environment.json" in passed.bucket.objects
    assert f"{PREFIX}format-gate/gate.json" in passed.bucket.objects


@pytest.mark.parametrize("which", ["passed", "failed"])
def test_teardown_follows_the_macs_verification_gate_passing_or_not(
    which: str, request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: Recorded = request.getfixturevalue(which)
    pod, bucket = Pod(), MemoryBucket()
    unit, github = supervise(tmp_path / "ctrl", pod, bucket, monkeypatch)
    for moment in recorded.moments[:-1]:
        replay(unit, pod, bucket, moment)
        assert not unit.shutdown_ready and unit.r2_verified is None
    replay(unit, pod, bucket, recorded.moments[-1])
    status = "completed" if which == "passed" else "stopped"
    assert unit.shutdown_ready and unit.r2_verified is not None
    assert unit.r2_verified["report_status"] == status
    assert mirrored(unit)["status"] == "shutdown_ready"
    prefix = f"posttrain-runs/1c/{SMOKE_POD_ID_1C}/final"
    summary = json.loads(github.files[f"repos/{REPOSITORY}/contents/{prefix}/summary.json"])
    assert summary["format_gate"]["passed"] is (which == "passed")
    assert watchdog_ends_the_pod(unit, recorded) == (
        "terminated",
        "shutdown_ready_verified_receipts_delete_204_get_404",
    )


def test_a_failing_gate_stops_before_the_evaluation(failed: Recorded) -> None:
    final = json.loads(failed.moments[-1].report)
    assert final["status"] == "stopped" and final["stop_reason"] == FORMAT_GATE_STOP
    stages = {stage["name"]: stage["status"] for stage in final["stages"]}
    assert stages == {
        "probe": "completed",
        "sft": "completed",
        "format-gate": "stopped",
        "eval-sft": "skipped",
    }
    labels = {row["label"] for row in final["artifacts"]}
    assert labels == {"environment", "probe", "sft", "sft-weights", "format-gate"}
    keys = set(failed.bucket.objects)
    assert all(key.startswith(PREFIX) for key in keys)
    assert not any("eval-sft" in key for key in keys)
