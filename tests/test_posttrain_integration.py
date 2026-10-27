"""Run 1's CPU smoke drives the real Mac-side supervisor: heartbeats in, teardown out.

The worker runs every stage on the smoke's fixtures into an in-memory bucket. Each
heartbeat it writes is kept with ``report.json`` and the bucket at that moment, then
replayed to a real ``PosttrainSupervisor`` through fake SSH and a fake GitHub, with the
Mac's own key reading the bucket; the watchdog then ends the Pod on the verified receipts.
The stops end the same way: the probe stop, a label R2 refused (the weights pulled over the
verified SSH channel; any other label held for a rescue by hand) and the in-Pod idle
self-destruct reading the run's real heartbeat. Nothing touches a network.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import torch
from test_rehearsal_integration import MAC_KEY, REPOSITORY, GitHub, pod_program
from test_rehearsal_run import refuse_put

from scratchcoder.data.shards import file_sha256
from tools import pilot_supervisor, posttrain_ops
from tools.pilot_supervisor import (
    MILESTONE_ERROR_FILE,
    RECOVERED_ARTIFACTS,
    PosttrainSupervisor,
    archive_bundle,
)
from tools.pilot_transfer import pull_artifacts
from tools.pod_self_destruct import SelfDestruct
from tools.posttrain_run import PROBE_STOP, PosttrainWorker
from tools.posttrain_smoke import SMOKE_POD_ID, Fixtures, fixtures, make_worker, patched_benchmarks
from tools.rehearsal_run import MemoryBucket
from tools.runpod_watchdog import PodIdentity, Policy, RunpodClient, check_once

PREFIX = posttrain_ops.run_prefix("1", SMOKE_POD_ID)
SHORT_WINDOW = 120 + 420 + 1
"""A one-second final drain before the finalize margin and the pull reserve."""


@dataclass(frozen=True)
class Moment:
    heartbeat: dict[str, Any]
    report: bytes
    objects: dict[str, bytes]
    digests: dict[str, str]


@dataclass
class Recorded:
    worker: PosttrainWorker
    bucket: MemoryBucket
    moments: list[Moment]


def record(found: Fixtures, root: Path, **options: Any) -> Recorded:
    """Run the worker, keeping every heartbeat with what SSH and R2 showed at that moment."""
    torch.set_num_threads(1)
    bucket = options.pop("bucket", MemoryBucket())
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
        worker = make_worker(found, root, bucket=bucket, **options)
        with patched_benchmarks():
            worker.execute()
    return Recorded(worker, bucket, moments)


@pytest.fixture(scope="module")
def found(tmp_path_factory: pytest.TempPathFactory) -> Fixtures:
    return fixtures(tmp_path_factory.mktemp("posttrain-integration-inputs"))


@pytest.fixture(scope="module")
def smoked(found: Fixtures, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Recorded]:
    yield record(found, tmp_path_factory.mktemp("posttrain-integration") / "run")


class Pod:
    """Fake SSH: the worker's heartbeat and report.json at the moment being replayed."""

    def __init__(self) -> None:
        self.moment: Moment | None = None

    def fetch(self, name: str) -> dict[str, Any]:
        assert name == "heartbeat.json" and self.moment is not None
        return json.loads(json.dumps(self.moment.heartbeat))  # type: ignore[no-any-return]

    def fetch_report(self) -> bytes:
        assert self.moment is not None
        return self.moment.report


def supervise(
    root: Path, pod: Pod, bucket: MemoryBucket, monkeypatch: pytest.MonkeyPatch, **options: Any
) -> tuple[PosttrainSupervisor, GitHub, list[bool]]:
    github = GitHub()
    monkeypatch.setattr(pilot_supervisor, "_gh", github)
    ready_when_archived: list[bool] = []
    holder: list[PosttrainSupervisor] = []
    now = options.pop("now", [0.0])

    def archive(bundle: Path) -> dict[str, Any]:
        ready_when_archived.append(holder[0].shutdown_ready)
        return archive_bundle(
            bundle,
            REPOSITORY,
            run_id=SMOKE_POD_ID,
            files=("report.json", "summary.json"),
            root="posttrain-runs/1",
            message="Save post-training report and summary",
        )

    unit = PosttrainSupervisor(
        root,
        run="1",
        pod_id=SMOKE_POD_ID,
        fetch=options.pop("fetch", pod.fetch),
        fetch_report=options.pop("fetch_report", pod.fetch_report),
        store=bucket.store(MAC_KEY),
        archive=archive,
        clock=lambda: now[0],
        **options,
    )
    holder.append(unit)
    unit.now = now  # type: ignore[attr-defined]
    return unit, github, ready_when_archived


def replay(unit: PosttrainSupervisor, pod: Pod, bucket: MemoryBucket, moment: Moment) -> None:
    pod.moment = moment
    bucket.objects, bucket.digests = dict(moment.objects), dict(moment.digests)
    unit.now[0] += 5  # type: ignore[attr-defined]
    unit.refresh_heartbeat()
    unit.cycle()


def mirrored(unit: PosttrainSupervisor) -> dict[str, Any]:
    return json.loads((unit.root / "heartbeat.json").read_text())  # type: ignore[no-any-return]


class SmokeRunpod:
    """The Runpod API showing the smoke's Pod alive, then gone after a DELETE."""

    def __init__(self, identity: dict[str, Any]) -> None:
        self.identity = identity
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
            "id": self.identity["id"],
            "name": self.identity["name"],
            "gpu": {"id": self.identity["gpuId"], "count": 1},
            "dataCenterId": self.identity["dataCenterId"],
            "createdAt": self.identity["createdAt"],
            "cost": 0.59,
        }
        return 200, json.dumps(pod).encode()


def watchdog_ends_the_pod(unit: PosttrainSupervisor) -> tuple[str, str]:
    """The run's watchdog, under the run-1 policy, reading the supervisor's mirror."""
    created = datetime.fromtimestamp(time.time() - 600, UTC).isoformat()
    identity = {
        "id": SMOKE_POD_ID,
        "name": "posttrain-a40-s-1",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "test",
        "createdAt": created,
    }
    outcome = check_once(
        RunpodClient("authored-test-key", SmokeRunpod(identity)),
        PodIdentity.from_dict(identity),
        Policy(18_000, 2.99, 600, 0.6, max_transfer_seconds=900, mode="posttrain"),
        unit.root / "heartbeat.json",
        now=time.time,
        sleep=lambda _: None,
        observed=True,
    )
    return outcome.state, outcome.reason


def test_the_smoke_runs_every_stage_into_one_prefix(smoked: Recorded) -> None:
    statuses = [moment.heartbeat["status"] for moment in smoked.moments]
    assert statuses[0] == "running" and statuses[-1] == "awaiting_transfer"
    assert "uploading" in statuses
    assert {moment.heartbeat["pod_id"] for moment in smoked.moments} == {SMOKE_POD_ID}
    stages = [moment.heartbeat["stage"] for moment in smoked.moments]
    for stage in ("probe", "arc-base", "eval-base", "sft", "arc-sft", "eval-sft", "screen"):
        assert stage in stages
    assert all(key.startswith(PREFIX) for key in smoked.bucket.objects)


def test_teardown_comes_only_after_the_mac_verifies_every_artifact(
    smoked: Recorded, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod, bucket = Pod(), MemoryBucket()
    unit, github, ready_when_archived = supervise(tmp_path / "ctrl", pod, bucket, monkeypatch)
    for moment in smoked.moments[:-1]:
        replay(unit, pod, bucket, moment)
        status = moment.heartbeat["status"]
        assert status in {"running", "uploading"}, status
        assert not unit.shutdown_ready and unit.r2_verified is None and not github.files
    final = smoked.moments[-1]
    replay(unit, pod, bucket, final)
    assert unit.shutdown_ready and ready_when_archived == [False]
    assert unit.r2_verified is not None and unit.r2_verified["report_status"] == "completed"
    assert len(unit.r2_verified["artifacts"]) == final.heartbeat["r2_final"]["artifact_count"]
    local = mirrored(unit)
    assert local["status"] == "shutdown_ready" and local["github_receipt_verified"] is True
    assert local["transfer_progress_unix"] == unit.now[0]  # type: ignore[attr-defined]
    prefix = f"posttrain-runs/1/{SMOKE_POD_ID}/final"
    paths = {endpoint.rsplit("/contents/", 1)[1] for endpoint in github.files}
    assert paths == {f"{prefix}/report.json", f"{prefix}/summary.json"}
    summary = json.loads(github.files[f"repos/{REPOSITORY}/contents/{prefix}/summary.json"])
    assert summary["kind"] == "posttrain-run-summary" and summary["r2_final_verified"] is True
    assert watchdog_ends_the_pod(unit) == (
        "terminated",
        "shutdown_ready_verified_receipts_delete_204_get_404",
    )


@pytest.mark.parametrize("target", ["report", "weights", "completions"])
def test_a_mismatched_artifact_never_approves_teardown(
    smoked: Recorded, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    pod, bucket = Pod(), MemoryBucket()
    unit, github, _ = supervise(tmp_path / "ctrl", pod, bucket, monkeypatch)
    final = smoked.moments[-1]
    key = {
        "report": final.heartbeat["r2_final"]["report_key"],
        "weights": f"{PREFIX}sft-weights/checkpoint.pt",
        "completions": f"{PREFIX}eval-base-pool-test-n8/completions.jsonl",
    }[target]
    tampered = Moment(
        final.heartbeat, final.report, final.objects, {**final.digests, key: "0" * 64}
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="SHA-256"):
            replay(unit, pod, bucket, tampered)
        assert not unit.shutdown_ready and not github.files
    assert "mac_receipt_verified" not in mirrored(unit)


def test_a_probe_stop_is_verified_and_torn_down(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(raw: dict[str, Any]) -> None:
        raw["probe"]["minimum_tokens_per_second"] = {"4": 1e12}

    stopped = record(found, tmp_path / "run", changes=slow)
    final = stopped.moments[-1]
    assert final.heartbeat["status"] == "stopped" and "r2_final" in final.heartbeat
    report = json.loads(final.report)
    assert report["stop_reason"] == PROBE_STOP
    pod, bucket = Pod(), MemoryBucket()
    unit, github, _ = supervise(tmp_path / "ctrl", pod, bucket, monkeypatch)
    replay(unit, pod, bucket, final)
    assert unit.shutdown_ready and unit.r2_verified is not None
    assert unit.r2_verified["report_status"] == "stopped"
    assert [row["key"] for row in unit.r2_verified["artifacts"]] == [
        f"{PREFIX}probe/generation-benchmark.json",
        f"{PREFIX}probe/record.json",
    ]


def refused_run(found: Fixtures, root: Path, part: str) -> Recorded:
    """A run whose bucket refuses every PUT to keys containing ``part``."""
    bucket = MemoryBucket()
    bucket.fault = refuse_put(part)

    def short(raw: dict[str, Any]) -> None:
        raw["limits"]["max_transfer_seconds"] = SHORT_WINDOW

    return record(found, root, bucket=bucket, changes=short, upload_retry_seconds=0.05)


def test_unverified_weights_are_pulled_over_ssh_before_teardown(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = refused_run(found, tmp_path / "pod", "sft-weights/")
    worker = refused.worker
    beat = json.loads((worker.root / "heartbeat.json").read_text())
    finalization = json.loads((worker.root / "finalization.json").read_text())
    assert (beat["status"], beat["failure"]) == ("failed", "r2_milestones_unverified")
    (row,) = finalization["unverified_milestones"]
    assert row["label"] == "sft-weights" and {f["name"] for f in row["files"]} == {
        "checkpoint.pt",
        "record.json",
    }
    weights = (worker.root / "exports" / "sft-weights" / "checkpoint.pt").read_bytes()
    for name in ("key", "known_hosts"):
        (tmp_path / name).write_text("synthetic placeholder\n")
    activity = float(beat["last_gpu_activity_unix"])
    now = [activity + 480]
    pulls: list[str] = []

    def recover(name: str, destination: Path, timeout: float) -> Path:
        pulls.append(name)
        return pull_artifacts(
            host="root@203.0.113.7",
            port=40022,
            identity_file=tmp_path / "key",
            known_hosts=tmp_path / "known_hosts",
            remote_dir=f"{worker.root}/{name}",
            destination=destination,
            bundle_name=PurePosixPath(name).name,
            artifacts=RECOVERED_ARTIFACTS,
            timeout=timeout,
            run_ssh=pod_program,
        )

    unit, github, _ = supervise(
        tmp_path / "ctrl",
        Pod(),
        refused.bucket,
        monkeypatch,
        now=now,
        fetch=lambda name: json.loads((worker.root / name).read_text()),
        fetch_report=lambda: (worker.root / "report.json").read_bytes(),
        fetch_finalization=lambda: (worker.root / "finalization.json").read_bytes(),
        recover=recover,
    )
    unit.cycle()
    assert pulls == ["exports/sft-weights"] and unit.shutdown_ready
    pulled = unit.recovery_root / "sft-weights" / "checkpoint.pt"
    assert pulled.read_bytes() == weights and file_sha256(pulled) == row["checkpoint_sha256"]
    assert not (unit.root / MILESTONE_ERROR_FILE).exists()
    prefix = f"repos/{REPOSITORY}/contents/posttrain-runs/1/{SMOKE_POD_ID}/diagnostics"
    recovery = json.loads(github.files[f"{prefix}/summary.json"])["label_recovery"]
    assert recovery["failure"] == "r2_milestones_unverified"
    assert [item["label"] for item in recovery["pulled_over_ssh"]] == ["sft-weights"]


def test_an_unverified_label_that_ssh_cannot_carry_is_held_for_a_rescue(
    found: Fixtures, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = refused_run(found, tmp_path / "pod", "screen/")
    worker = refused.worker
    beat = json.loads((worker.root / "heartbeat.json").read_text())
    assert beat["failure"] == "r2_milestones_unverified"
    now = [float(beat["last_gpu_activity_unix"]) + 480]
    unit, github, _ = supervise(
        tmp_path / "ctrl",
        Pod(),
        refused.bucket,
        monkeypatch,
        now=now,
        fetch=lambda name: json.loads((worker.root / name).read_text()),
        fetch_report=lambda: (worker.root / "report.json").read_bytes(),
        fetch_finalization=lambda: (worker.root / "finalization.json").read_bytes(),
        recover=lambda *_: pytest.fail("no pull may start for this label"),
    )
    for _ in range(3):
        unit.cycle()
        assert not unit.shutdown_ready and not github.files
    error = json.loads((unit.root / MILESTONE_ERROR_FILE).read_text())
    (held,) = error["files"]
    assert held["label"] == "screen" and "exports/screen/attempts.jsonl" in held["pod_paths"]


def test_the_pod_deletes_itself_after_900_idle_seconds_on_its_own_heartbeat(
    smoked: Recorded, tmp_path: Path
) -> None:
    heartbeat = smoked.worker.root / "heartbeat.json"
    last = json.loads(heartbeat.read_text())["last_gpu_activity_unix"]
    clock = [last - 3_600]  # Armed by the bootstrap an hour before the run ended.
    unit = SelfDestruct(
        pod_id=SMOKE_POD_ID,
        token="",
        deadline_unix=last + 17_820,
        receipt_path=str(tmp_path / "self-destruct.json"),
        clock=lambda: clock[0],
        monotonic=lambda: clock[0],
        sleep=lambda seconds: None,
        dry_run=True,
        idle_seconds=900.0,
        activity=(str(heartbeat), str(tmp_path / "setup-progress.json")),
    )
    clock[0] = last + 899
    assert unit.due() is None  # The run's own last GPU work holds the Pod.
    clock[0] = last + 900
    assert unit.due() == "idle_limit"
    assert unit.terminate("idle_limit") == "dry_run_complete"
    assert json.loads((tmp_path / "self-destruct.json").read_text())["reason"] == (
        "idle_limit_no_delete_requested"
    )


def test_heartbeats_never_exceed_their_bound(smoked: Recorded) -> None:
    assert all(len(json.dumps(m.heartbeat)) < 65_536 for m in smoked.moments)
    assert all(len(m.report) < 1_000_000 for m in smoked.moments)
    assert time.time() > 0
