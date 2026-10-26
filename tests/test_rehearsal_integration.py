"""The worker's own records drive the Mac-side supervisor: smoke output in, teardown out.

The worker's CPU smoke trains both runs into one in-memory bucket. Every heartbeat it
writes is kept with the ``report.json`` and the bucket's objects at that moment, then
replayed to a real ``RehearsalSupervisor`` through fake SSH and a fake GitHub, with the
Mac's own key reading the same bucket. Nothing touches a network.
"""

import base64
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.tokenizer.bpe import ByteBPE
from tools import pilot_supervisor, rehearsal_run
from tools.pilot_supervisor import RehearsalSupervisor, archive_bundle
from tools.r2_store import Credentials
from tools.rehearsal_binding import RUNS
from tools.rehearsal_ops import run_prefix
from tools.rehearsal_run import MemoryBucket, Worker, smoke

MAC_KEY = Credentials(
    "https://smoke.r2.cloudflarestorage.com", "mac-key-id", "mac-secret-key", "smoke-bucket"
)
REPOSITORY = "owner/private-artifacts"


@dataclass(frozen=True)
class Moment:
    """What SSH and R2 would show right after one heartbeat."""

    heartbeat: dict[str, Any]
    report: bytes
    objects: dict[str, bytes]
    digests: dict[str, str]


@dataclass
class Smoke:
    bucket: MemoryBucket
    moments: dict[str, list[Moment]]
    summary: dict[str, Any]


@pytest.fixture(scope="module")
def smoked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Smoke]:
    """``rehearsal_run.smoke`` for both runs, every heartbeat kept with its moment."""
    bucket = MemoryBucket()
    moments: dict[str, list[Moment]] = {run: [] for run in RUNS}
    original = Worker.heartbeat

    def keep(self: Worker) -> None:
        original(self)
        with bucket.lock:
            objects, digests = dict(bucket.objects), dict(bucket.digests)
        moments[self.spec.run].append(
            Moment(
                json.loads((self.root / "heartbeat.json").read_text()),
                (self.root / "report.json").read_bytes(),
                objects,
                digests,
            )
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Worker, "heartbeat", keep)
        summary = smoke(tmp_path_factory.mktemp("smoke") / "out", bucket=bucket)
    yield Smoke(bucket, moments, summary)


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


class GitHub:
    """Fake ``gh api``: a private repository that stores and returns contents exactly."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def __call__(
        self,
        arguments: list[str],
        *,
        payload: dict[str, Any] | None = None,
        missing_ok: bool = False,
    ) -> dict[str, Any]:
        if arguments == [f"repos/{REPOSITORY}"]:
            return {"private": True, "full_name": REPOSITORY}
        endpoint = arguments[-1]
        if payload is not None:
            self.files[endpoint] = base64.b64decode(payload["content"])
            return {"commit": {"sha": "fixture-commit"}}
        if endpoint not in self.files:
            return {"missing": True}
        return {"encoding": "base64", "content": base64.b64encode(self.files[endpoint]).decode()}


def supervise(
    root: Path, run: str, pod: Pod, bucket: MemoryBucket, monkeypatch: pytest.MonkeyPatch
) -> tuple[RehearsalSupervisor, GitHub, list[bool]]:
    """A real rehearsal supervisor over fake SSH and GitHub and the Mac's key on ``bucket``."""
    github = GitHub()
    monkeypatch.setattr(pilot_supervisor, "_gh", github)
    ready_when_archived: list[bool] = []
    holder: list[RehearsalSupervisor] = []
    now = [0.0]

    def archive(bundle: Path) -> dict[str, Any]:
        # The archive runs after the R2 verification and before teardown is approved.
        ready_when_archived.append(holder[0].shutdown_ready)
        assert (bundle / "r2-verification.json").is_file()
        return archive_bundle(
            bundle,
            REPOSITORY,
            run_id=f"smoke-{run.lower()}",
            files=("report.json", "summary.json"),
            root=f"rehearsal-runs/{run}",
            message="Save rehearsal report and summary",
        )

    unit = RehearsalSupervisor(
        root,
        run=run,
        pod_id=f"smoke-{run.lower()}",
        fetch=pod.fetch,
        fetch_report=pod.fetch_report,
        store=bucket.store(MAC_KEY),
        archive=archive,
        clock=lambda: now[0],
    )
    holder.append(unit)
    unit.now = now  # type: ignore[attr-defined]
    return unit, github, ready_when_archived


def replay(unit: RehearsalSupervisor, pod: Pod, bucket: MemoryBucket, moment: Moment) -> None:
    pod.moment = moment
    bucket.objects, bucket.digests = dict(moment.objects), dict(moment.digests)
    unit.now[0] += 5  # type: ignore[attr-defined]  # One mirror cycle.
    unit.cycle()


def mirrored(unit: RehearsalSupervisor) -> dict[str, Any]:
    return json.loads((unit.root / "heartbeat.json").read_text())  # type: ignore[no-any-return]


def test_the_smoke_completes_both_runs_into_one_bucket(smoked: Smoke) -> None:
    assert smoked.summary["status"] == {"A": "completed", "B": "completed"}
    assert smoked.summary["same_multiset"] and smoked.summary["tickets_match"]
    for run in RUNS:
        statuses = [moment.heartbeat["status"] for moment in smoked.moments[run]]
        assert statuses[0] == "running" and statuses[-1] == "awaiting_transfer"
        assert "uploading" in statuses


@pytest.mark.parametrize("run", sorted(RUNS))
def test_supervisor_tears_down_only_after_verifying_the_smokes_r2_state(
    smoked: Smoke, run: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod, bucket = Pod(), MemoryBucket()
    unit, github, ready_when_archived = supervise(tmp_path / "ctrl", run, pod, bucket, monkeypatch)
    moments = smoked.moments[run]
    for index, moment in enumerate(moments[:-1]):
        replay(unit, pod, bucket, moment)
        status = moment.heartbeat["status"]
        # Every stage, run A's boundary between them, and the final upload: no teardown.
        assert status in {"running", "uploading"}, (index, status)
        assert not unit.shutdown_ready and unit.r2_verified is None and not github.files
        assert mirrored(unit)["status"] == status
    if run == "A":
        stages = [moment.heartbeat["stage"] for moment in moments]
        boundary = stages.index("stage2")
        phases = {moment.heartbeat["phase"] for moment in moments[:boundary]}
        assert "stage1-full-validation" in phases  # Stage 1 ended while still "running".
        # stage1-final reached R2 while the run went on: a milestone, never a run end.
        milestone = f"{run_prefix('A', 'smoke-a')}stage1-final/checkpoint.pt"
        assert any(milestone in moment.objects for moment in moments[:-1])
    final = moments[-1]
    assert final.heartbeat["status"] == "awaiting_transfer" and "r2_final" in final.heartbeat
    replay(unit, pod, bucket, final)
    # shutdown_ready only now, after R2 verification and the archive's read-back.
    assert unit.shutdown_ready and ready_when_archived == [False]
    assert unit.r2_verified is not None and unit.r2_verified["report_status"] == "completed"
    local = mirrored(unit)
    assert local["status"] == "shutdown_ready"
    assert local["mac_receipt_verified"] and local["github_receipt_verified"]
    # The uploading moment was the R2 baseline; report.json arriving is Mac-seen progress.
    assert local["transfer_progress_unix"] == unit.now[0]  # type: ignore[attr-defined]
    prefix = f"rehearsal-runs/{run}/smoke-{run.lower()}/final"
    paths = {endpoint.rsplit("/contents/", 1)[1] for endpoint in github.files}
    assert paths == {f"{prefix}/report.json", f"{prefix}/summary.json"}
    archived = github.files[f"repos/{REPOSITORY}/contents/{prefix}/report.json"]
    assert archived == final.report == final.objects[final.heartbeat["r2_final"]["report_key"]]
    summary = json.loads(github.files[f"repos/{REPOSITORY}/contents/{prefix}/summary.json"])
    assert summary["r2_final_verified"] is True
    assert [stage["name"] for stage in summary["stages"]] == [name for name, _ in RUNS[run]]


@pytest.mark.parametrize("target", ["checkpoint", "report", "stage1"])
def test_a_mismatched_r2_sha256_never_sets_shutdown_ready(
    smoked: Smoke, target: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod, bucket = Pod(), MemoryBucket()
    unit, github, _ = supervise(tmp_path / "ctrl", "A", pod, bucket, monkeypatch)
    final = smoked.moments["A"][-1]
    r2_final = final.heartbeat["r2_final"]
    key = {
        "checkpoint": r2_final["checkpoint_key"],
        "report": r2_final["report_key"],
        "stage1": f"{run_prefix('A', 'smoke-a')}stage1-final/checkpoint.pt",
    }[target]
    tampered = Moment(
        final.heartbeat, final.report, final.objects, {**final.digests, key: "0" * 64}
    )
    for _ in range(3):
        with pytest.raises(ValueError, match="SHA-256"):
            replay(unit, pod, bucket, tampered)
        assert not unit.shutdown_ready and unit.r2_verified is None and not github.files
    local = mirrored(unit)
    assert local["status"] == "awaiting_transfer"
    assert "mac_receipt_verified" not in local and "github_receipt_verified" not in local


def test_a_run_a_stopped_in_stage_two_closes_at_its_verified_stage_one_milestone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker publishes r2_final only for a verified stage-final milestone, which the
    supervisor's verification accepts for a stopped run; a stopped run without one gets
    none and goes to diagnostics."""
    tokenizer = ByteBPE()
    streams = rehearsal_run.author_streams(tmp_path / "data", tokenizer)
    specs = {run: rehearsal_run.smoke_specification(run, streams) for run in RUNS}
    initialization = tmp_path / "initialization.pt"
    digest = rehearsal_run.write_initialization(
        list(specs.values()), tokenizer.sha256, initialization
    )
    spec = specs["A"].with_limits(rising_steps=5)
    bucket = MemoryBucket()
    worker = Worker(
        spec,
        tmp_path / "run-A",
        rehearsal_run.authored_run_data(spec, streams, tokenizer),
        initialization=initialization,
        initialization_sha256=digest,
        store=bucket.store(),
        pod_id="smoke-a",
        pod_created=time.time(),
    )
    original = worker.on_step

    def rising(trainer: Any, metrics: dict[str, float]) -> None:
        if worker.stage_name == "stage2":
            metrics = {**metrics, "loss": 10.0 + trainer.step}
        original(trainer, metrics)

    worker.on_step = rising  # type: ignore[method-assign]
    report = worker.execute()
    assert report["status"] == "stopped" and report["stop_reason"] == "loss_rising"
    beat = json.loads((worker.root / "heartbeat.json").read_text())
    assert beat["status"] == "stopped"
    prefix = run_prefix("A", "smoke-a")
    assert beat["r2_final"]["checkpoint_key"] == f"{prefix}stage1-final/checkpoint.pt"
    pod = Pod()
    pod.moment = Moment(beat, (worker.root / "report.json").read_bytes(), {}, {})
    unit, github, ready_when_archived = supervise(tmp_path / "ctrl", "A", pod, bucket, monkeypatch)
    unit.cycle()
    assert unit.shutdown_ready and ready_when_archived == [False]
    assert unit.r2_verified is not None and unit.r2_verified["report_status"] == "stopped"
    assert [row["key"] for row in unit.r2_verified["stage_milestones"]] == [
        f"{prefix}stage1-final/checkpoint.pt"
    ]
