"""The supervisor's rehearsal mode with fake SSH, R2 and GitHub: teardown only after proof."""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools import pilot_supervisor
from tools.pilot_supervisor import RehearsalSupervisor, archive_bundle, fetch_status_bytes
from tools.pilot_transfer import child_environment
from tools.rehearsal_ops import run_prefix
from tools.runpod_watchdog import PodIdentity, Policy, RunpodClient, check_once

POD = "podfixture01"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeR2:
    """An in-memory bucket that answers HEAD with the stored SHA-256 metadata."""

    def __init__(self) -> None:
        self.objects_by_key: dict[str, tuple[bytes, str]] = {}
        self.calls: list[tuple[str, str]] = []

    def put(self, key: str, data: bytes, *, meta: str | None = None) -> None:
        self.objects_by_key[key] = (data, meta or sha(data))

    def head(self, key: str) -> dict[str, str] | None:
        self.calls.append(("HEAD", key))
        if key not in self.objects_by_key:
            return None
        data, meta = self.objects_by_key[key]
        return {"x-amz-meta-sha256": meta, "content-length": str(len(data))}

    def download(self, key: str, destination: Path, *, sha256: str, size: int) -> dict[str, Any]:
        self.calls.append(("GET", key))
        data, _ = self.objects_by_key[key]
        if len(data) != size or sha(data) != sha256:
            raise ValueError(f"{key}: size or SHA-256 differs from the pinned value")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return {"key": key, "bytes": size, "sha256": sha256, "attempts": 1, "seconds": 0.0}

    def objects(self) -> list[dict[str, Any]]:
        return [{"key": key, "bytes": len(data)} for key, (data, _) in self.objects_by_key.items()]


class Pod:
    """What SSH would show: the worker's heartbeat and report.json."""

    def __init__(self, heartbeat: dict[str, Any], report: bytes) -> None:
        self.heartbeat, self.report = heartbeat, report

    def fetch(self, name: str) -> dict[str, Any]:
        assert name == "heartbeat.json"
        return json.loads(json.dumps(self.heartbeat))  # type: ignore[no-any-return]

    def fetch_report(self) -> bytes:
        return self.report


class Archive:
    def __init__(self, supervisor: list[RehearsalSupervisor], status: str = "verified") -> None:
        self.supervisor, self.status = supervisor, status
        self.bundles: list[Path] = []

    def __call__(self, bundle: Path) -> dict[str, Any]:
        # The archive always runs before teardown is approved.
        assert not self.supervisor[0].shutdown_ready
        assert {path.name for path in bundle.iterdir()} >= {"report.json", "summary.json"}
        self.bundles.append(bundle)
        return {"status": self.status, "private": True}


def heartbeat(status: str = "running", **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at_unix": 1900,
        "last_gpu_activity_unix": 1899,
        "phase": "training",
        "step": 7630,
        "status": status,
        **extra,
    }


def stage(name: str, checkpoint: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "updates": 7630,
        "tokens": 1_000_000_000,
        "learning_rate": 6e-4,
        "warmup_updates": 153,
        "ticket": {"sha256": "1" * 64},
        "expected_ticket_match": True,
        "final_checkpoint_sha256": sha(checkpoint),
        "r2_labels": [f"{name}-final"],
        "validations": [{"global_update": 500, "web": 6.1, "python": 5.9}],
        "full_validation": {"web": 4.0, "python": 3.5},
    }


def finished(
    r2: FakeR2,
    *,
    run: str = "A",
    stages: tuple[str, ...] = ("stage1", "stage2"),
    status: str = "completed",
) -> tuple[bytes, dict[str, str]]:
    """Put a run's final state in R2; return the report bytes and the heartbeat's r2_final."""
    prefix = run_prefix(run, POD)
    rows = []
    for name in stages:
        checkpoint = f"checkpoint of {run} {name}".encode()
        label = f"{prefix}{name}-final/"
        r2.put(label + "checkpoint.pt", checkpoint)
        for other in ("checkpoint.pt.sha256", "record.json", "updates.jsonl", "ticket.json"):
            r2.put(label + other, f"{other} of {name}".encode())
        rows.append(stage(name, checkpoint))
    report = json.dumps(
        {"status": status, "stages": rows, "binding_sha256": "2" * 64, "cost": {"usd": 7.1}},
        sort_keys=True,
    ).encode()
    r2.put(prefix + "report.json", report)
    last = f"{prefix}{stages[-1]}-final/checkpoint.pt"
    final = {
        "report_key": prefix + "report.json",
        "report_sha256": sha(report),
        "checkpoint_key": last,
        "checkpoint_sha256": r2.objects_by_key[last][1],
    }
    return report, final


def supervisor(
    tmp_path: Path, pod: Pod, r2: FakeR2, *, run: str = "A", status: str = "verified"
) -> tuple[RehearsalSupervisor, Archive]:
    holder: list[RehearsalSupervisor] = []
    archive = Archive(holder, status)
    now = [1000.0]
    unit = RehearsalSupervisor(
        tmp_path / "controller",
        run=run,
        pod_id=POD,
        fetch=pod.fetch,
        fetch_report=pod.fetch_report,
        store=r2,
        archive=archive,
        clock=lambda: now[0],
        failure_grace_seconds=300,
    )
    holder.append(unit)
    unit.now = now  # type: ignore[attr-defined]
    return unit, archive


def mirrored(unit: RehearsalSupervisor) -> dict[str, Any]:
    return json.loads((unit.root / "heartbeat.json").read_text())  # type: ignore[no-any-return]


IDENTITY = PodIdentity.from_dict(
    {
        "id": POD,
        "name": "rehearsal-a40-s-a",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "test",
        "createdAt": "1970-01-01T00:16:40+00:00",
    }
)


class AlivePod:
    """The Runpod API showing this Pod alive; a DELETE would fail the test."""

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        assert method == "GET", "the watchdog must not delete this Pod"
        pod = {
            "id": POD,
            "name": "rehearsal-a40-s-a",
            "gpu": {"id": "NVIDIA A40", "count": 1},
            "dataCenterId": "test",
            "createdAt": "1970-01-01T00:16:40+00:00",
            "cost": 0.6,
        }
        return 200, json.dumps(pod).encode()


def test_run_a_stage_boundary_never_tears_down(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, final = finished(r2, stages=("stage1",), status="running")
    pod = Pod(heartbeat("running", phase="stage-boundary", r2_final=final), report)
    unit, archive = supervisor(tmp_path, pod, r2)
    for _ in range(3):
        unit.cycle()
    assert not unit.shutdown_ready and not archive.bundles and not r2.calls
    assert mirrored(unit)["status"] == "running"
    assert "github_receipt_verified" not in mirrored(unit)


def test_uploading_is_treated_like_running_and_r2_growth_is_the_only_progress(
    tmp_path: Path,
) -> None:
    r2 = FakeR2()
    report, final = finished(r2, stages=("stage1",), status="completed")
    prefix = run_prefix("A", POD)
    pod = Pod(heartbeat("uploading", phase="finalizing", last_gpu_activity_unix=1500), report)
    unit, archive = supervisor(tmp_path, pod, r2)
    unit.cycle()  # The first listing is only a baseline.
    assert not unit.shutdown_ready and not archive.bundles
    local = mirrored(unit)
    assert local["status"] == "uploading" and "transfer_progress_unix" not in local
    assert not [call for call in r2.calls if call[0] in {"HEAD", "GET"}]  # No verification yet.
    unit.now[0] = 1600.0  # type: ignore[attr-defined]
    r2.put(f"rehearsal-runs/B/{POD}/stage1-final/checkpoint.pt", b"another run's object")
    unit.cycle()
    assert "transfer_progress_unix" not in mirrored(unit)  # Another prefix is not progress.
    unit.now[0] = 1700.0  # type: ignore[attr-defined]
    r2.put(prefix + "stage2-final/checkpoint.pt", b"a new label arrives")
    unit.cycle()
    assert mirrored(unit)["transfer_progress_unix"] == 1700.0 and not unit.shutdown_ready
    unit.now[0] = 1800.0  # type: ignore[attr-defined]
    unit.cycle()  # Nothing new: the stamp stays where the Mac last saw progress.
    assert mirrored(unit)["transfer_progress_unix"] == 1700.0
    # The watchdog reads the mirror: 700 s after the last GPU work the idle limit has
    # passed, but the Mac saw progress 100 s ago, so it waits.
    policy = Policy(45_000, 7.54, 600, 0.6, max_transfer_seconds=900, mode="rehearsal")
    outcome = check_once(
        RunpodClient("authored-test-key", AlivePod()),
        IDENTITY,
        policy,
        unit.root / "heartbeat.json",
        now=lambda: 2200,
        observed=True,
    )
    assert outcome.state == "watching"

    def broken() -> list[dict[str, Any]]:
        raise RuntimeError("R2 list failed with HTTP 500")

    r2.objects = broken  # type: ignore[method-assign]
    unit.now[0] = 1900.0  # type: ignore[attr-defined]
    unit.cycle()  # A failed listing stamps nothing and is recorded.
    assert mirrored(unit)["transfer_progress_unix"] == 1700.0
    assert json.loads((unit.root / "r2-progress-error.json").read_text()) == {
        "error_type": "RuntimeError"
    }


def test_progress_is_never_stamped_while_training(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, _ = finished(r2, stages=("stage1",), status="running")
    unit, _ = supervisor(tmp_path, Pod(heartbeat("running"), report), r2)
    unit.cycle()
    r2.put(run_prefix("A", POD) + "stage1-step500/checkpoint.pt", b"a rolling label")
    unit.cycle()
    assert "transfer_progress_unix" not in mirrored(unit) and not r2.calls


def test_stage_one_final_never_counts_as_a_completed_run_a(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, final = finished(r2, stages=("stage1",), status="completed")
    unit, archive = supervisor(
        tmp_path, Pod(heartbeat("awaiting_transfer", r2_final=final), report), r2
    )
    with pytest.raises(ValueError, match="every stage"):
        unit.cycle()
    assert not unit.shutdown_ready and not archive.bundles
    assert mirrored(unit)["status"] == "awaiting_transfer"


@pytest.mark.parametrize(
    "tamper",
    ["checkpoint-meta", "report-meta", "ssh-report", "missing-ticket", "stage1-milestone"],
)
def test_mismatched_r2_state_never_sets_shutdown_ready(tmp_path: Path, tamper: str) -> None:
    r2 = FakeR2()
    report, final = finished(r2)
    prefix = run_prefix("A", POD)
    if tamper == "checkpoint-meta":
        data, _ = r2.objects_by_key[final["checkpoint_key"]]
        r2.put(final["checkpoint_key"], data, meta="9" * 64)
    elif tamper == "report-meta":
        r2.put(final["report_key"], report, meta="9" * 64)
    elif tamper == "ssh-report":
        report = report.replace(b'"usd": 7.1', b'"usd": 7.2')
    elif tamper == "missing-ticket":
        del r2.objects_by_key[prefix + "stage2-final/ticket.json"]
    else:
        r2.put(prefix + "stage1-final/checkpoint.pt", b"another stage 1 checkpoint")
    unit, archive = supervisor(
        tmp_path, Pod(heartbeat("awaiting_transfer", r2_final=final), report), r2
    )
    for _ in range(2):
        with pytest.raises(ValueError):
            unit.cycle()
    assert not unit.shutdown_ready and not archive.bundles and unit.r2_verified is None
    local = mirrored(unit)
    assert local["status"] == "awaiting_transfer"
    assert "mac_receipt_verified" not in local and "r2_final_verified" not in local


def test_happy_path_sets_shutdown_ready_only_after_the_archive(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, final = finished(r2)
    unit, archive = supervisor(
        tmp_path, Pod(heartbeat("awaiting_transfer", r2_final=final), report), r2
    )
    unit.cycle()
    assert unit.shutdown_ready and archive.bundles == [unit.root / "final"]
    assert (unit.root / "final" / "report.json").read_bytes() == report
    summary = json.loads((unit.root / "final" / "summary.json").read_text())
    assert summary["r2_final_verified"] is True and summary["run"] == "A"
    assert [row["name"] for row in summary["stages"]] == ["stage1", "stage2"]
    assert "updates" in summary["stages"][0] and "loss_curves" not in summary
    assert ("GET", final["report_key"]) in r2.calls
    assert not [key for method, key in r2.calls if method == "GET" and key.endswith(".pt")]
    local = mirrored(unit)
    assert local["status"] == "shutdown_ready"
    assert local["mac_receipt_verified"] and local["github_receipt_verified"]
    assert local["r2_final"] == final

    class API:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def request(
            self, method: str, url: str, headers: dict[str, str], timeout: float
        ) -> tuple[int, bytes]:
            self.calls.append(method)
            if len(self.calls) == 1:
                return 200, json.dumps(
                    {
                        "id": POD,
                        "name": "rehearsal-a40-s-a",
                        "gpu": {"id": "NVIDIA A40", "count": 1},
                        "dataCenterId": "test",
                        "createdAt": "1970-01-01T00:16:40+00:00",
                        "cost": 0.6,
                    }
                ).encode()
            return (204, b"") if method == "DELETE" else (404, b"")

    transport = API()
    identity = PodIdentity.from_dict(
        {
            "id": POD,
            "name": "rehearsal-a40-s-a",
            "gpuId": "NVIDIA A40",
            "gpuCount": 1,
            "dataCenterId": "test",
            "createdAt": "1970-01-01T00:16:40+00:00",
        }
    )
    outcome = check_once(
        RunpodClient("authored-test-key", transport),
        identity,
        Policy(45_000, 7.54, 600, 0.6, mode="rehearsal"),
        unit.root / "heartbeat.json",
        now=lambda: 1900,
    )
    assert outcome.state == "terminated" and transport.calls == ["GET", "DELETE", "GET"]


def test_archive_failure_keeps_the_r2_proof_and_retries_without_redownloading(
    tmp_path: Path,
) -> None:
    r2 = FakeR2()
    report, final = finished(r2, run="B", stages=("mixed",))
    unit, archive = supervisor(
        tmp_path,
        Pod(heartbeat("awaiting_transfer", r2_final=final), report),
        r2,
        run="B",
        status="failed",
    )
    with pytest.raises(ValueError, match="unverified"):
        unit.cycle()
    assert unit.r2_verified is not None and not unit.shutdown_ready
    local = unit.refresh_heartbeat()  # What the mirror thread writes next.
    assert local["r2_final_verified"] and local["mac_receipt_verified"]
    assert local["status"] == "awaiting_transfer" and "github_receipt_verified" not in local
    downloads = [call for call in r2.calls if call[0] == "GET"]
    archive.status = "verified"
    unit.cycle()
    assert unit.shutdown_ready and len(archive.bundles) == 2
    assert [call for call in r2.calls if call[0] == "GET"] == downloads


def test_awaiting_transfer_without_r2_final_waits(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, _ = finished(r2)
    unit, archive = supervisor(tmp_path, Pod(heartbeat("awaiting_transfer"), report), r2)
    unit.cycle()
    assert not unit.shutdown_ready and not archive.bundles and not r2.calls


def test_failed_worker_without_r2_final_archives_diagnostics_after_the_grace(
    tmp_path: Path,
) -> None:
    r2 = FakeR2()
    report, _ = finished(r2, stages=("stage1",), status="failed")
    unit, archive = supervisor(tmp_path, Pod(heartbeat("failed"), report), r2)
    unit.cycle()
    assert not unit.shutdown_ready
    unit.now[0] += 299  # type: ignore[attr-defined]
    unit.cycle()
    assert not unit.shutdown_ready
    unit.now[0] += 1  # type: ignore[attr-defined]
    unit.cycle()
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["r2_final_verified"] is False
    assert summary["r2_labels_present"] == ["report.json", "stage1-final"]
    assert (unit.root / "diagnostics" / "report.json").read_bytes() == report
    assert mirrored(unit)["status"] == "shutdown_ready"


def test_failed_worker_with_r2_final_is_verified_like_a_finished_one(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, final = finished(r2, run="B", stages=("mixed",), status="stopped")
    unit, archive = supervisor(
        tmp_path, Pod(heartbeat("stopped", r2_final=final), report), r2, run="B"
    )
    unit.cycle()
    assert unit.shutdown_ready and archive.bundles == [unit.root / "final"]
    assert unit.r2_verified is not None and unit.r2_verified["report_status"] == "stopped"


def test_pod_supplied_receipt_flags_are_discarded(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, _ = finished(r2)
    forged = heartbeat(
        "shutdown_ready",
        mac_receipt_verified=True,
        github_receipt_verified=True,
        r2_final_verified=True,
        transfer_progress_unix=1899,
    )
    unit, _ = supervisor(tmp_path, Pod(forged, report), r2)
    unit.cycle()
    local = mirrored(unit)
    for key in (
        "mac_receipt_verified",
        "github_receipt_verified",
        "r2_final_verified",
        "transfer_progress_unix",
    ):
        assert key not in local
    assert not unit.shutdown_ready


def test_rehearsal_mode_never_pulls_over_ssh_and_reuses_its_directory(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, _ = finished(r2)
    pod = Pod(heartbeat(), report)
    unit, _ = supervisor(tmp_path, pod, r2)
    with pytest.raises(RuntimeError, match="never copies checkpoints"):
        unit.pull("exports/lr-0-step4-1")
    again, _ = supervisor(tmp_path, pod, r2)  # A restart continues in the same directory.
    assert again.root == unit.root


def test_rehearsal_archive_paths_and_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "final"
    final.mkdir()
    (final / "report.json").write_text('{"status": "completed"}')
    (final / "summary.json").write_text('{"run": "A"}')
    (final / "updates.jsonl").write_text("{}\n")
    remote: dict[str, bytes] = {}

    def gh(
        arguments: list[str], *, payload: dict[str, Any] | None = None, missing_ok: bool = False
    ) -> dict[str, Any]:
        if arguments == ["repos/owner/private-artifacts"]:
            return {"private": True, "full_name": "owner/private-artifacts"}
        endpoint = arguments[-1]
        if payload is not None:
            assert payload["message"] == "Save rehearsal report and summary"
            remote[endpoint] = base64.b64decode(payload["content"])
            return {"commit": {"sha": "fixture-commit"}}
        if endpoint not in remote:
            return {"missing": True}
        return {"encoding": "base64", "content": base64.b64encode(remote[endpoint]).decode()}

    monkeypatch.setattr("tools.pilot_supervisor._gh", gh)
    receipt = archive_bundle(
        final,
        "owner/private-artifacts",
        run_id=POD,
        files=("report.json", "summary.json"),
        root="rehearsal-runs/A",
        message="Save rehearsal report and summary",
    )
    assert [row["path"] for row in receipt["files"]] == [
        f"rehearsal-runs/A/{POD}/final/report.json",
        f"rehearsal-runs/A/{POD}/final/summary.json",
    ]
    assert not any("updates.jsonl" in endpoint for endpoint in remote)
    (final / "report.json").write_bytes(b"x" * 1_000_001)
    with pytest.raises(ValueError, match="bounded"):
        archive_bundle(
            final,
            "owner/private-artifacts",
            run_id=POD,
            files=("report.json",),
            root="rehearsal-runs/A",
        )
    for root in ("../escape", "/absolute", "a//b"):
        with pytest.raises(ValueError, match="archive root"):
            archive_bundle(final, "owner/private-artifacts", run_id=POD, root=root)


def test_status_bounds_only_tighten_and_children_never_see_r2_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="tighten"):
        fetch_status_bytes(["ssh"], "/workspace/rehearsal-run", "heartbeat.json", max_bytes=10**6)
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "placeholder")
    monkeypatch.setenv("R2_POD_SECRET_ACCESS_KEY", "placeholder")
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    assert not [name for name in child_environment() if name.startswith("R2_")]
    assert "RUNPOD_API_KEY" not in child_environment()


def _cli(tmp_path: Path, policy: dict[str, Any], extra: list[str]) -> list[str]:
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "id": POD,
                "name": "rehearsal-a40-s-a",
                "gpuId": "NVIDIA A40",
                "gpuCount": 1,
                "dataCenterId": "test",
                "createdAt": "1970-01-01T00:16:40+00:00",
            }
        )
    )
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    for name in ("key", "known_hosts"):
        (tmp_path / name).write_text("fixture\n")
    return [
        "--identity",
        str(identity),
        "--policy",
        str(tmp_path / "policy.json"),
        "--identity-file",
        str(tmp_path / "key"),
        "--known-hosts",
        str(tmp_path / "known_hosts"),
        "--output",
        str(tmp_path / "supervisor"),
        "--host",
        "root@203.0.113.7",
        "--port",
        "40022",
        "--remote-root",
        "/workspace/rehearsal-run",
        "--repository",
        "owner/private-artifacts",
        *extra,
    ]


REHEARSAL_POLICY = {
    "mode": "rehearsal",
    "max_wall_seconds": 45_000,
    "max_cost_usd": 7.54,
    "max_idle_seconds": 600,
    "quoted_total_hourly": 0.6,
}


@pytest.mark.parametrize(
    ("policy", "extra", "message"),
    [
        (REHEARSAL_POLICY, ["--mode", "rehearsal"], "needs --run"),
        (
            {**REHEARSAL_POLICY, "mode": "pilot", "max_wall_seconds": 5400},
            ["--mode", "rehearsal", "--run", "A"],
            "mode",
        ),
        (
            REHEARSAL_POLICY,
            ["--mode", "rehearsal", "--run", "A", "--failure-grace-seconds", "900"],
            "grace",
        ),
        (REHEARSAL_POLICY, ["--mode", "rehearsal", "--run", "A"], "Mac R2 credentials"),
        (REHEARSAL_POLICY, ["--run", "A"], "only to --mode rehearsal"),
    ],
)
def test_cli_refuses_an_incomplete_rehearsal_setup_before_any_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    policy: dict[str, Any],
    extra: list[str],
    message: str,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    for name in ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)

    def no_ssh(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no SSH before the setup is complete")

    monkeypatch.setattr("tools.pilot_supervisor.subprocess.run", no_ssh)
    with pytest.raises(SystemExit) as error:
        pilot_supervisor.main(_cli(tmp_path, policy, extra))
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "supervisor").exists()
