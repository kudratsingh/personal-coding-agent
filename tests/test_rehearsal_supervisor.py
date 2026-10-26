"""The supervisor's rehearsal mode with fake SSH, R2 and GitHub: teardown only after proof."""

import base64
import hashlib
import json
import shutil
import subprocess
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from scratchcoder.common.records import atomic_json
from tools import pilot_supervisor, r2_store
from tools.pilot_supervisor import (
    MAC_CHECK_TIMEOUT_SECONDS,
    RehearsalSupervisor,
    archive_bundle,
    fetch_status_bytes,
    unverified_milestones,
)
from tools.pilot_transfer import child_environment
from tools.r2_store import Store
from tools.rehearsal_ops import run_prefix
from tools.runpod_watchdog import Outcome, PodIdentity, Policy, RunpodClient, check_once

POD = "podfixture01"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeR2:
    """An in-memory bucket that answers HEAD with the stored SHA-256 metadata."""

    def __init__(self) -> None:
        self.objects_by_key: dict[str, tuple[bytes, str]] = {}
        self.calls: list[tuple[str, str]] = []
        self.listings = 0

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
        self.listings += 1
        return [{"key": key, "bytes": len(data)} for key, (data, _) in self.objects_by_key.items()]


ADDRESSES = 4
"""The addresses an R2 endpoint is assumed to resolve to. urllib tries each one with the
whole socket timeout, so a request to an unreachable endpoint fails only after all four."""
STALL_SECONDS = ADDRESSES * MAC_CHECK_TIMEOUT_SECONDS
"""How long one request of the Mac's bounded store takes to fail against such an endpoint."""


class Stalled(FakeR2):
    """A bounded store against an R2 endpoint that never answers: every listing and HEAD
    ends in the socket timeout ``bounded_store`` sets. Given ``now``, a supervisor's fake
    clock, each request first takes ``STALL_SECONDS`` on it."""

    def __init__(self, now: list[float] | None = None) -> None:
        super().__init__()
        self.now = now

    def stall(self) -> None:
        if self.now is not None:
            self.now[0] += STALL_SECONDS

    def head(self, key: str) -> dict[str, str] | None:
        self.calls.append(("HEAD", key))
        self.stall()
        raise TimeoutError("The read operation timed out")

    def objects(self) -> list[dict[str, Any]]:
        self.listings += 1
        self.stall()
        raise TimeoutError("The read operation timed out")


class Pod:
    """What SSH would show: the worker's heartbeat, report.json and finalization.json."""

    def __init__(
        self, heartbeat: dict[str, Any], report: bytes, finalization: bytes | None = None
    ) -> None:
        self.heartbeat, self.report, self.finalization = heartbeat, report, finalization

    def fetch(self, name: str) -> dict[str, Any]:
        assert name == "heartbeat.json"
        return json.loads(json.dumps(self.heartbeat))  # type: ignore[no-any-return]

    def fetch_report(self) -> bytes:
        return self.report

    def fetch_finalization(self) -> bytes:
        if self.finalization is None:
            raise RuntimeError("bounded SSH status request failed")
        return self.finalization


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
    tmp_path: Path,
    pod: Pod,
    r2: FakeR2,
    *,
    run: str = "A",
    status: str = "verified",
    **channels: Any,
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
        **channels,
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


class Runpod:
    """The Runpod API showing this Pod alive until a DELETE, then gone."""

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
        return AlivePod().request("GET", url, headers, timeout)


REHEARSAL = Policy(45_000, 7.54, 600, 0.6, max_transfer_seconds=900, mode="rehearsal")
"""The rehearsal watchdog's policy: 600 s idle, a 900 s transfer window."""


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
    with pytest.raises(RuntimeError, match="never copies checkpoints over SSH, except"):
        unit.pull("exports/lr-0-step4-1")
    again, _ = supervisor(tmp_path, pod, r2)  # A restart continues in the same directory.
    assert again.root == unit.root
    # An unverified milestone's copy is kept per Pod, never mixed with another Pod's.
    assert again.recovery_root == unit.root / "recovered" / POD


# A milestone the worker could not verify in R2: pulled over SSH before any teardown ---------


class Recovery:
    """Fake SSH pull: publishes ``exports/<label>`` from the Pod's disk as the pilot's
    verified pull does (staging, the Pod's manifest digests, then an atomic rename)."""

    def __init__(
        self, holder: list[RehearsalSupervisor], disk: dict[str, bytes], fail: str | None = None
    ) -> None:
        self.holder, self.disk, self.fail = holder, disk, fail
        self.calls: list[tuple[str, float]] = []
        self.progress: list[object] = []

    def __call__(self, name: str, destination: Path, timeout: float) -> Path:
        unit = self.holder[0]
        assert not unit.shutdown_ready  # Teardown is never approved while a pull runs.
        self.calls.append((name, timeout))
        label = PurePosixPath(name).name
        assert name == f"exports/{label}" and destination == unit.recovery_root
        staging = destination / f".{label}.staging-fixture"
        staging.mkdir(parents=True)
        data = self.disk[label]
        (staging / "checkpoint.pt").write_bytes(data[: len(data) // 2])
        unit.now[0] += 1  # type: ignore[attr-defined]
        # The mirror thread runs meanwhile: bytes arriving on the Mac are transfer progress.
        self.progress.append(unit.refresh_heartbeat().get("transfer_progress_unix"))
        if self.fail is not None:
            shutil.rmtree(staging)  # As the real pull does: nothing half-copied remains.
            raise RuntimeError(self.fail)
        (staging / "checkpoint.pt").write_bytes(data)
        (staging / "record.json").write_bytes(b"{}")
        rows: list[dict[str, Any]] = [
            {"kind": "checkpoint", "name": "checkpoint.pt", "sha256": sha(data)},
            {"kind": "record", "name": "record.json", "sha256": sha(b"{}")},
        ]
        rows[0]["size_bytes"], rows[1]["size_bytes"] = len(data), 2
        (staging / "transfer-receipt.json").write_text(
            json.dumps({"status": "verified", "artifacts": rows})
        )
        staging.replace(destination / label)
        return destination / label


def unverified_run(
    r2: FakeR2,
    *,
    failure: str = "r2_milestones_unverified",
    in_r2: tuple[str, ...] = ("stage1",),
    status: str = "failed",
    flags: bool = True,
    size: int = 0,
) -> tuple[bytes, dict[str, bytes]]:
    """Run A's report after a final drain that left ``stage2-final`` only on the Pod.

    ``size`` pads each checkpoint to that many bytes.
    """
    prefix = run_prefix("A", POD)
    disk: dict[str, bytes] = {}
    rows = []
    for name in ("stage1", "stage2"):
        checkpoint = f"checkpoint of A {name}".encode().ljust(size, b".")
        disk[f"{name}-final"] = checkpoint
        if name in in_r2:
            r2.put(f"{prefix}{name}-final/checkpoint.pt", checkpoint)
        rows.append(stage(name, checkpoint))
    report: dict[str, Any] = {"status": status, "stages": rows, "training_status": "completed"}
    if failure:
        report["failure"] = failure
    if flags:
        verified = {f"{name}-final": name in in_r2 for name in ("stage1", "stage2")}
        report["r2"] = {"milestones_verified": verified}
    return json.dumps(report, sort_keys=True).encode(), disk


def export_row(label: str, checkpoint: bytes) -> dict[str, Any]:
    """One unverified milestone as ``rehearsal_run.Worker.export_listing`` writes it."""
    directory = f"/workspace/rehearsal-run/exports/{label}"
    files = [
        {
            "name": name,
            "path": f"{directory}/{name}",
            "sha256": sha(data),
            "size_bytes": len(data),
        }
        for name, data in (
            ("checkpoint.pt", checkpoint),
            ("checkpoint.pt.sha256", sha(checkpoint).encode()),
            ("record.json", b"{}"),
            ("updates.jsonl", b"{}\n"),
            ("ticket.json", b"{}"),
        )
    ]
    return {
        "label": label,
        "export": f"exports/{label}",
        "export_path": directory,
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_size_bytes": len(checkpoint),
        "files": files,
    }


def listing(disk: dict[str, bytes], *labels: str, legacy: bool = False) -> bytes:
    """The worker's finalization.json naming each unverified milestone's export.

    ``legacy`` writes an older row: only ``label``, ``path`` and ``sha256``.
    """
    rows = [
        {
            "label": label,
            "path": f"/workspace/rehearsal-run/exports/{label}",
            "sha256": sha(disk[label]),
        }
        if legacy
        else export_row(label, disk[label])
        for label in labels
    ]
    return json.dumps({"status": "failed", "unverified_milestones": rows}).encode()


def recovering(
    tmp_path: Path,
    r2: FakeR2,
    report: bytes,
    finalization: bytes | None,
    disk: dict[str, bytes],
    **beat: Any,
) -> tuple[RehearsalSupervisor, Archive, Recovery]:
    """A supervisor 400 s after the worker's last GPU work: 500 s of the window remain."""
    pod = Pod(heartbeat("failed", last_gpu_activity_unix=1500, **beat), report, finalization)
    holder: list[RehearsalSupervisor] = []
    recovery = Recovery(holder, disk)
    unit, archive = supervisor(
        tmp_path,
        pod,
        r2,
        fetch_finalization=pod.fetch_finalization,
        recover=recovery,
        transfer_window_seconds=900,
    )
    holder.append(unit)
    unit.now[0] = 1900.0  # type: ignore[attr-defined]
    return unit, archive, recovery


@pytest.mark.parametrize("legacy", [False, True])
def test_an_unverified_milestone_is_pulled_checked_and_archived_before_teardown(
    tmp_path: Path, legacy: bool
) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(r2)
    rows = listing(disk, "stage2-final", legacy=legacy)
    unit, archive, recovery = recovering(tmp_path, r2, report, rows, disk)
    unit.cycle()  # No grace timer: the pull starts at once, inside the transfer window.
    assert recovery.calls == [("exports/stage2-final", 500.0)]
    assert recovery.progress == [1901.0]  # The staged bytes stamped Mac-seen progress.
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
    pulled = unit.recovery_root / "stage2-final" / "checkpoint.pt"
    assert pulled.read_bytes() == disk["stage2-final"]
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    recovery_record = summary["milestone_recovery"]
    assert recovery_record["failure"] == "r2_milestones_unverified"
    assert [row["label"] for row in recovery_record["pulled_over_ssh"]] == ["stage2-final"]
    assert recovery_record["pulled_over_ssh"][0]["sha256"] == sha(disk["stage2-final"])
    assert recovery_record["pulled_over_ssh"][0]["path"] == str(pulled)
    # stage1-final reached R2: the Mac's own HEAD confirms it, nothing is pulled for it.
    assert [row["key"] for row in recovery_record["verified_in_r2_by_mac"]] == [
        f"{run_prefix('A', POD)}stage1-final/checkpoint.pt"
    ]
    assert (unit.root / "diagnostics" / "report.json").read_bytes() == report
    assert mirrored(unit)["status"] == "shutdown_ready"
    assert not (unit.root / "MILESTONES-NOT-SECURED.json").exists()
    unit.cycle()
    assert len(recovery.calls) == 1  # Done once.


def put_export(r2: FakeR2, label: str, checkpoint: bytes, *, skip: str = "") -> None:
    """Store every file ``export_row`` lists for ``label`` in R2, except ``skip``."""
    prefix = run_prefix("A", POD)
    for name, data in (
        ("checkpoint.pt", checkpoint),
        ("checkpoint.pt.sha256", sha(checkpoint).encode()),
        ("record.json", b"{}"),
        ("updates.jsonl", b"{}\n"),
        ("ticket.json", b"{}"),
    ):
        if name != skip:
            r2.put(f"{prefix}{label}/{name}", data)


def test_a_listed_milestone_the_mac_finds_whole_in_r2_is_not_pulled(tmp_path: Path) -> None:
    # The worker ran out of time to confirm stage2-final; every file did reach R2.
    r2 = FakeR2()
    report, disk = unverified_run(r2)
    put_export(r2, "stage2-final", disk["stage2-final"])
    unit, archive, recovery = recovering(tmp_path, r2, report, listing(disk, "stage2-final"), disk)
    unit.cycle()
    assert recovery.calls == []
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    secured = summary["milestone_recovery"]["verified_in_r2_by_mac"]
    assert [row["key"].split("/")[-2] for row in secured] == ["stage1-final", "stage2-final"]
    assert len(secured[1]["files"]) == 5  # Every listed file, not just the checkpoint.


@pytest.mark.parametrize("missing", ["checkpoint.pt", "ticket.json"])
def test_a_listed_milestone_missing_any_file_in_r2_is_pulled(tmp_path: Path, missing: str) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(r2)
    put_export(r2, "stage2-final", disk["stage2-final"], skip=missing)
    unit, _, recovery = recovering(tmp_path, r2, report, listing(disk, "stage2-final"), disk)
    unit.cycle()
    assert recovery.calls == [("exports/stage2-final", 500.0)]
    assert unit.shutdown_ready


def test_r2_finalization_checks_every_recorded_milestone_with_the_macs_key(
    tmp_path: Path,
) -> None:
    """An exception in finalization may leave no list: the Mac checks R2 itself."""
    r2 = FakeR2()
    report, disk = unverified_run(r2, failure="r2_finalization", flags=False)
    unit, archive, recovery = recovering(tmp_path, r2, report, b"{}", disk)
    unit.cycle()
    assert recovery.calls == [("exports/stage2-final", 500.0)]  # Missing in R2: pulled.
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    verified = summary["milestone_recovery"]["verified_in_r2_by_mac"]
    assert [row["key"] for row in verified] == [f"{run_prefix('A', POD)}stage1-final/checkpoint.pt"]
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]


def test_an_r2_final_for_the_last_milestone_does_not_skip_an_earlier_unverified_one(
    tmp_path: Path,
) -> None:
    """stage2-final is in R2 and named by r2_final; stage1-final reached only the Pod."""
    r2 = FakeR2()
    report, disk = unverified_run(r2, in_r2=("stage2",))
    prefix = run_prefix("A", POD)
    final = {
        "report_key": prefix + "report.json",
        "report_sha256": sha(report),
        "checkpoint_key": prefix + "stage2-final/checkpoint.pt",
        "checkpoint_sha256": sha(disk["stage2-final"]),
    }
    unit, archive, recovery = recovering(
        tmp_path, r2, report, listing(disk, "stage1-final"), disk, r2_final=final
    )
    unit.cycle()
    assert recovery.calls == [("exports/stage1-final", 500.0)]
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    assert [row["label"] for row in summary["milestone_recovery"]["pulled_over_ssh"]] == [
        "stage1-final"
    ]
    assert [row["key"] for row in summary["milestone_recovery"]["verified_in_r2_by_mac"]] == [
        final["checkpoint_key"]
    ]


def test_milestones_all_verified_in_r2_by_the_mac_need_no_ssh_pull(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(
        r2, failure="r2_finalization", in_r2=("stage1", "stage2"), flags=False
    )
    unit, archive, recovery = recovering(tmp_path, r2, report, b"{}", disk)
    unit.cycle()
    assert not recovery.calls and unit.shutdown_ready
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    assert len(summary["milestone_recovery"]["verified_in_r2_by_mac"]) == 2


@pytest.mark.parametrize("problem", ["pull", "tampered", "window", "no-channel"])
def test_a_failed_pull_never_approves_teardown_and_names_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], problem: str
) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(r2)
    unit, archive, recovery = recovering(tmp_path, r2, report, listing(disk, "stage2-final"), disk)
    expected = sha(disk["stage2-final"])
    if problem == "pull":
        recovery.fail = "SSH artifact stream failed: connection reset"
    elif problem == "tampered":
        recovery.disk = {**disk, "stage2-final": b"other bytes on the Pod's disk"}
    elif problem == "window":
        unit.now[0] = 2400.0  # type: ignore[attr-defined]  # 900 s after the last GPU work.
    else:
        unit.recover = None
    for _ in range(3):
        unit.cycle()
        unit.now[0] += 400  # type: ignore[attr-defined]  # Long past the failure grace.
    assert not unit.shutdown_ready and not archive.bundles
    assert mirrored(unit)["status"] == "failed"
    error = json.loads((unit.root / "MILESTONES-NOT-SECURED.json").read_text())
    assert error == json.loads((unit.root / "controller-error.json").read_text())
    assert error["error_type"] == "MilestonesNotSecured" and "NOT approved" in error["detail"]
    assert error["files"] == [
        {
            "label": "stage2-final",
            "pod_path": "exports/stage2-final/checkpoint.pt",
            "sha256": expected,
            "pulled": False,
        }
    ]
    printed = capsys.readouterr().err
    assert "MILESTONES NOT SECURED: exports/stage2-final/checkpoint.pt" in printed
    if problem in {"pull", "tampered"}:
        # Retried while the window lasts (1900 and 2300), never after it ends at 2400.
        assert [call[0] for call in recovery.calls] == ["exports/stage2-final"] * 2
        assert "transfer window is over" in error["detail"]
    else:
        assert not recovery.calls
    assert not (unit.recovery_root / "stage2-final").exists()


def test_the_watchdog_still_ends_a_pod_whose_milestones_are_not_secured(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(r2)
    unit, _, recovery = recovering(tmp_path, r2, report, listing(disk, "stage2-final"), disk)
    recovery.fail = "SSH artifact stream failed"
    unit.cycle()
    assert not unit.shutdown_ready

    def watchdog(now: float) -> Outcome:
        return check_once(
            RunpodClient("authored-test-key", Runpod()),
            IDENTITY,
            REHEARSAL,
            unit.root / "heartbeat.json",
            now=lambda: now,
            sleep=lambda _: None,
            observed=True,
        )

    # The pull's progress stamp lets the idle Pod wait, but only to the transfer window.
    assert mirrored(unit)["transfer_progress_unix"] == 1901.0
    assert watchdog(1500 + 899).state == "watching"
    assert watchdog(1500 + 900).reason == "idle_gpu_limit_delete_204_get_404"


def test_an_ordinary_failure_with_nothing_at_risk_keeps_the_grace(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(
        r2, failure="nonfinite_loss", in_r2=("stage1", "stage2"), status="failed"
    )
    unit, archive, recovery = recovering(tmp_path, r2, report, b'{"status": "failed"}', disk)
    unit.cycle()
    assert not unit.shutdown_ready  # The grace starts.
    unit.now[0] += 300  # type: ignore[attr-defined]
    unit.cycle()
    assert unit.shutdown_ready and not recovery.calls
    assert "milestone_recovery" not in json.loads(
        (unit.root / "diagnostics" / "summary.json").read_text()
    )


def test_a_stopped_run_with_a_flagged_milestone_is_secured_first(tmp_path: Path) -> None:
    """Any stopped or failed run: while a milestone is unverified, no grace timer."""
    r2 = FakeR2()
    report, disk = unverified_run(r2, failure="", status="stopped")
    unit, _, recovery = recovering(tmp_path, r2, report, None, disk)  # No finalization.json.
    unit.cycle()
    assert recovery.calls == [("exports/stage2-final", 500.0)] and unit.shutdown_ready


@pytest.mark.parametrize("failure", ["r2_milestones_unverified", "r2_finalization"])
def test_without_finalization_json_a_recovery_failure_never_approves(
    tmp_path: Path, failure: str
) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(r2, failure=failure, in_r2=("stage1", "stage2"))
    unit, archive, recovery = recovering(tmp_path, r2, report, None, disk)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="bounded SSH"):
            unit.cycle()
        unit.now[0] += 400  # type: ignore[attr-defined]
    assert not unit.shutdown_ready and not archive.bundles and not recovery.calls


def test_finalization_and_report_must_agree_on_a_milestones_sha256(tmp_path: Path) -> None:
    r2 = FakeR2()
    report, disk = unverified_run(r2)
    wrong = listing({**disk, "stage2-final": b"another checkpoint"}, "stage2-final")
    unit, archive, recovery = recovering(tmp_path, r2, report, wrong, disk)
    with pytest.raises(ValueError, match="two SHA-256s"):
        unit.cycle()
    assert not unit.shutdown_ready and not recovery.calls


def test_the_mirror_and_the_at_risk_checks_use_the_bounded_store(tmp_path: Path) -> None:
    """R2 never answers the Mac. The mirror's listing and the at-risk HEADs time out in the
    bounded store, each after the socket timeout on four addresses of the fake clock, are
    recorded and stamp nothing; the milestone R2 could not confirm is pulled too, and the
    pull still starts in time. The default store is never asked."""
    r2 = FakeR2()  # The default store: it holds stage1-final, for verification only.
    report, disk = unverified_run(r2)
    stalled = Stalled()
    beat = {"phase": "finalizing", "last_gpu_activity_unix": 1500}
    pod = Pod(heartbeat("uploading", **beat), report, listing(disk, "stage2-final"))
    holder: list[RehearsalSupervisor] = []
    recovery = Recovery(holder, disk)
    unit, archive = supervisor(
        tmp_path,
        pod,
        r2,
        quick_store=stalled,
        fetch_finalization=pod.fetch_finalization,
        recover=recovery,
        transfer_window_seconds=900,
    )
    holder.append(unit)
    stalled.now = unit.now  # type: ignore[attr-defined]
    assert supervisor(tmp_path / "default", pod, r2)[0].quick_store is r2  # The default.
    unit.now[0] = 1800.0  # type: ignore[attr-defined]
    for _ in range(2):
        unit.refresh_heartbeat()  # The mirror thread, every 5 s while the worker uploads.
    assert stalled.listings == 2 and r2.listings == 0
    assert json.loads((unit.root / "r2-progress-error.json").read_text()) == {
        "error_type": "TimeoutError"
    }
    assert "transfer_progress_unix" not in mirrored(unit) and unit.r2_seen is None
    pod.heartbeat = heartbeat("failed", failure="r2_milestones_unverified", **beat)
    unit.now[0] = 1500.0 + 480  # type: ignore[attr-defined]  # The worker's handover.
    unit.cycle()
    # The last stage first, then stage1-final, which the Mac's HEAD could not confirm.
    assert [name for name, _ in recovery.calls] == ["exports/stage2-final", "exports/stage1-final"]
    # Two stalled HEADs delay the first pull by 40 s: its first bytes still reach the Mac
    # (and stamp progress) before the 600 s idle limit, with 380 s of the window left.
    assert recovery.calls[0][1] == 900 - 480 - 2 * STALL_SECONDS >= 360
    assert recovery.progress[0] == 1500 + 480 + 2 * STALL_SECONDS + 1 < 1500 + 600
    # The listed stage2-final is checked first and times out at its first file; then the
    # recorded stage1-final. Both are pulled.
    assert stalled.calls == [
        ("HEAD", f"{run_prefix('A', POD)}stage2-final/checkpoint.pt"),
        ("HEAD", f"{run_prefix('A', POD)}stage1-final/checkpoint.pt"),
    ]
    assert stalled.listings == 3  # The two mirror listings, then the diagnostics one.
    assert not r2.calls and r2.listings == 0
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    assert summary["r2_listing_error"] == "TimeoutError"
    assert summary["milestone_recovery"]["verified_in_r2_by_mac"] == []


class Dropping(Recovery):
    """Drops the first pull of each milestone, as a reset SSH connection does, and records
    when each pull starts on the supervisor's fake clock."""

    def __init__(self, holder: list[RehearsalSupervisor], disk: dict[str, bytes]) -> None:
        super().__init__(holder, disk)
        self.starts: list[tuple[str, float]] = []

    def __call__(self, name: str, destination: Path, timeout: float) -> Path:
        retry = any(started == name for started, _ in self.starts)
        self.starts.append((name, self.holder[0].now[0]))  # type: ignore[attr-defined]
        self.fail = None if retry else "SSH artifact stream failed: connection reset"
        return super().__call__(name, destination, timeout)


def test_a_retried_pull_starts_within_a_cycle_and_no_milestone_is_checked_twice(
    tmp_path: Path,
) -> None:
    """R2 never answers the Mac, the worker lists both of run A's milestones, and each
    stalled HEAD takes the socket timeout on four addresses. The first pull of each one
    drops. Each retry starts in the next 5 s cycle: neither a milestone already pulled nor
    one whose R2 check already failed is checked again."""
    stalled = Stalled()
    report, disk = unverified_run(stalled, in_r2=())
    rows = listing(disk, "stage1-final", "stage2-final")
    pod = Pod(heartbeat("failed", last_gpu_activity_unix=1500), report, rows)
    holder: list[RehearsalSupervisor] = []
    recovery = Dropping(holder, disk)
    unit, archive = supervisor(
        tmp_path,
        pod,
        stalled,
        fetch_finalization=pod.fetch_finalization,
        recover=recovery,
        transfer_window_seconds=900,
    )
    holder.append(unit)
    stalled.now = unit.now  # type: ignore[attr-defined]
    handover = 1500.0 + 480
    unit.now[0] = handover  # type: ignore[attr-defined]
    for _ in range(3):
        unit.cycle()
        unit.now[0] += 5  # type: ignore[attr-defined]  # The wait between two cycles.
    prefix = run_prefix("A", POD)
    assert stalled.calls == [  # Once each, at the first cycle.
        ("HEAD", f"{prefix}stage1-final/checkpoint.pt"),
        ("HEAD", f"{prefix}stage2-final/checkpoint.pt"),
    ]
    # Each fake pull takes 1 s; a retry starts one 5 s wait after the dropped pull ended.
    first = handover + 2 * STALL_SECONDS
    assert recovery.starts == [
        ("exports/stage2-final", first),  # Dropped.
        ("exports/stage2-final", first + 1 + 5),
        ("exports/stage1-final", first + 1 + 5 + 1),  # Dropped.
        ("exports/stage1-final", first + 1 + 5 + 1 + 1 + 5),
    ]
    # The first pull's bytes reach the Mac before the 600 s idle limit, with 380 s left.
    assert recovery.calls[0] == ("exports/stage2-final", 1500 + 900 - first)
    assert 1500 + 900 - first >= 360 and recovery.progress[0] == first + 1 < 1500 + 600
    assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
    summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
    recovered = summary["milestone_recovery"]
    assert [row["label"] for row in recovered["pulled_over_ssh"]] == [
        "stage2-final",
        "stage1-final",
    ]
    assert recovered["verified_in_r2_by_mac"] == []


# The SSH pull inside the watchdog's transfer window, on one fake clock -------------------

T0 = 10_000.0
"""The worker's last GPU work (``last_gpu_activity_unix``): the end of its final save."""
HANDOVER = T0 + 480
"""The worker's final heartbeat, ``failed`` / ``r2_milestones_unverified``, by contract."""


class FinalDrain(Pod):
    """The worker after its last GPU work: ``uploading`` and refreshed until the handover,
    then its final heartbeat, written once. ``now`` is the shared fake clock."""

    def __init__(self, report: bytes, finalization: bytes) -> None:
        super().__init__(heartbeat("uploading"), report, finalization)
        self.now = [0.0]

    def fetch(self, name: str) -> dict[str, Any]:
        common = {"phase": "finalizing", "pod_id": POD, "last_gpu_activity_unix": T0}
        if self.now[0] < HANDOVER:
            self.heartbeat = heartbeat("uploading", updated_at_unix=self.now[0], **common)
        else:
            self.heartbeat = heartbeat(
                "failed", updated_at_unix=HANDOVER, failure="r2_milestones_unverified", **common
            )
        return super().fetch(name)


@pytest.mark.parametrize("pulled", [True, False])
def test_the_ssh_pull_holds_the_pod_only_while_its_bytes_reach_the_mac(
    tmp_path: Path, pulled: bool
) -> None:
    """R2 never answers the Mac. The worker hands over at T0 + 480 s; the Mac's HEADs of
    stage2-final and stage1-final each stall for the socket timeout on four addresses, so
    the pull of stage2-final starts at T0 + 520 s. Its staged bytes grow from T0 + 530 s
    to T0 + 800 s and it publishes at T0 + 855 s; stage1-final, small here, follows. The
    real mirror (``refresh_heartbeat``) and watchdog (``check_once``) share one fake
    clock."""
    quick = Stalled()  # The Mac's bounded store: stage1-final is in R2, but unreachable.
    report, disk = unverified_run(quick, size=64 * 1024)
    store = FakeR2()  # The default store: verification and downloads only.
    pod = FinalDrain(report, listing(disk, "stage2-final"))
    holder: list[RehearsalSupervisor] = []
    api = Runpod()
    seen: dict[float, tuple[str, str]] = {}
    checks = (599, 610, 700, 850) if pulled else (599, 600)
    starts: dict[str, float] = {}
    stamps: list[object] = []

    def watchdog(after: float) -> tuple[str, str]:
        outcome = check_once(
            RunpodClient("authored-test-key", api),
            IDENTITY,
            REHEARSAL,
            holder[0].root / "heartbeat.json",
            now=lambda: T0 + after,
            sleep=lambda _: None,
            observed=True,
        )
        seen[after] = (outcome.state, outcome.reason)
        return seen[after]

    def publish(label: str, staging: Path) -> Path:
        data = disk[label]
        rows: list[dict[str, Any]] = [
            {"kind": "checkpoint", "name": "checkpoint.pt", "sha256": sha(data)},
            {"kind": "record", "name": "record.json", "sha256": sha(b"{}")},
        ]
        rows[0]["size_bytes"], rows[1]["size_bytes"] = len(data), 2
        (staging / "transfer-receipt.json").write_text(
            json.dumps({"status": "verified", "artifacts": rows})
        )
        staging.replace(staging.parent / label)  # Verified and published.
        return staging.parent / label

    def pull(name: str, destination: Path, timeout: float) -> Path:
        unit = holder[0]
        label = PurePosixPath(name).name
        start = starts[label] = unit.now[0] - T0  # type: ignore[attr-defined]
        # Every pull must end with the transfer window, 900 s after the last GPU work.
        assert (name, destination, timeout) == (f"exports/{label}", unit.recovery_root, 900 - start)
        data = disk[label]
        staging = destination / f".{label}.staging-timeline"
        if label == "stage1-final":
            # R2 did not confirm it for the Mac either; the fixture's copy lands at once.
            staging.mkdir()
            (staging / "checkpoint.pt").write_bytes(data)
            (staging / "record.json").write_bytes(b"{}")
            unit.now[0] += 1  # type: ignore[attr-defined]
            return publish(label, staging)
        begin = 5 * (int(start) // 5 + 1)  # The mirror's first tick: no bytes yet.
        for after in sorted({*range(begin, 860, 5), *checks}):
            if "DELETE" in api.calls:
                raise RuntimeError("SSH artifact stream failed: Connection closed by remote host")
            unit.now[0] = T0 + after  # type: ignore[attr-defined]
            if after % 5 == 0:  # The mirror thread's tick.
                if pulled and begin < after <= 800:
                    staging.mkdir(exist_ok=True)
                    arrived = len(data) * (after - begin) // (800 - begin)  # All by T0 + 800.
                    (staging / "checkpoint.pt").write_bytes(data[:arrived])
                    if after == 800:
                        (staging / "record.json").write_bytes(b"{}")
                stamps.append(unit.refresh_heartbeat().get("transfer_progress_unix"))
            if after in checks:
                watchdog(after)
            assert not unit.shutdown_ready and mirrored(unit)["status"] == "failed"
        if not pulled:
            raise TimeoutError("artifact transfer exceeded its total timeout")
        return publish(label, staging)  # At T0 + 855.

    unit, archive = supervisor(
        tmp_path,
        pod,
        store,
        quick_store=quick,
        fetch_finalization=pod.fetch_finalization,
        recover=pull,
        transfer_window_seconds=900,
    )
    holder.append(unit)
    pod.now = quick.now = unit.now  # type: ignore[attr-defined]
    for after in range(5, 480, 5):  # The final drain: the supervisor cycles every 5 s.
        unit.now[0] = T0 + after  # type: ignore[attr-defined]
        unit.cycle()
        if after == 300:
            assert watchdog(after)[0] == "watching"
    local = mirrored(unit)
    assert local["status"] == "uploading" and "transfer_progress_unix" not in local
    assert quick.listings == 95  # One per mirror refresh, each a timeout.
    unit.now[0] = HANDOVER  # type: ignore[attr-defined]
    unit.cycle()  # Reads the final heartbeat, checks R2 and pulls stage2-final over SSH.
    assert quick.calls == [
        ("HEAD", f"{run_prefix('A', POD)}stage2-final/checkpoint.pt"),
        ("HEAD", f"{run_prefix('A', POD)}stage1-final/checkpoint.pt"),
    ]
    # Two stalled HEADs delay the pull by 40 s; it still has 380 s of the window.
    assert starts["stage2-final"] == 480 + 2 * STALL_SECONDS
    assert 900 - starts["stage2-final"] >= 360
    assert not store.calls and store.listings == 0
    if pulled:
        # The first bytes reach the Mac, and stamp progress, before the 600 s idle limit.
        assert min(stamp for stamp in stamps if isinstance(stamp, float)) < T0 + 600
        assert seen == {
            300: ("watching", "within_limits"),
            **{after: ("watching", "within_limits") for after in checks},
        }
        assert "DELETE" not in api.calls
        # Teardown is approved once the pulls are verified, and the watchdog acts on it.
        assert unit.shutdown_ready and archive.bundles == [unit.root / "diagnostics"]
        summary = json.loads((unit.root / "diagnostics" / "summary.json").read_text())
        assert [
            (row["label"], row["sha256"])
            for row in summary["milestone_recovery"]["pulled_over_ssh"]
        ] == [(label, sha(disk[label])) for label in ("stage2-final", "stage1-final")]
        local = mirrored(unit)
        assert local["status"] == "shutdown_ready" and local["transfer_progress_unix"] == T0 + 800
        ending = watchdog(860)
        assert ending == ("terminated", "shutdown_ready_verified_receipts_delete_204_get_404")
    else:
        # No byte reached the Mac: nothing stamps progress, and the idle limit ends the Pod.
        assert seen[599] == ("watching", "within_limits")
        assert seen[600] == ("terminated", "idle_gpu_limit_delete_204_get_404")
        assert "transfer_progress_unix" not in mirrored(unit) and not any(stamps)
        assert not unit.shutdown_ready and not archive.bundles and list(starts) == ["stage2-final"]
        assert (unit.root / "MILESTONES-NOT-SECURED.json").exists()


def test_unverified_milestone_listings_are_parsed_strictly() -> None:
    worker = export_row("stage2-final", b"stage 2 weights")
    digest = worker["checkpoint_sha256"]
    keyed = {key: value for key, value in worker.items() if key != "label"}
    for listed in (
        [worker],  # The worker's own row (rehearsal_run.Worker.export_listing).
        {"stage2-final": keyed},
        [{**worker, "sha256": digest}],  # Every copy of the digest agrees.
        [{"label": "stage2-final", "checkpoint_sha256": digest}],
        # Older rows: sha256 and a path.
        [{"label": "stage2-final", "sha256": digest}],
        [{"path": "exports/stage2-final", "sha256": digest}],
        [{"path": "/workspace/rehearsal-run/exports/stage2-final/checkpoint.pt", "sha256": digest}],
        [{"export_path": "exports/stage2-final/", "label": "stage2-final", "sha256": digest}],
        {"stage2-final": {"path": "exports/stage2-final", "sha256": digest}},
    ):
        found = unverified_milestones({"unverified_milestones": listed}, run="A")
        assert found == {"stage2-final": digest}, listed
    assert unverified_milestones({"status": "failed"}, run="A") == {}
    other = "b" * 64
    files = worker["files"]
    for bad, message in (
        ([{"label": "stage2-step15000", "sha256": digest}], "not a stage-final"),  # Rolling.
        ([{"label": "mixed-final", "sha256": digest}], "not a stage-final"),  # Run B's.
        ([{"label": "stage1-final", "path": "exports/stage2-final", "sha256": digest}], "single"),
        ([{**worker, "export": "exports/stage1-final"}], "single"),
        ([{"path": "elsewhere/stage2-final", "sha256": digest}], "export path"),
        ([{"label": "stage2-final"}], "no checkpoint SHA-256"),
        ([{"label": "stage2-final", "sha256": "A" * 64}], "no checkpoint SHA-256"),
        ([{**worker, "checkpoint_sha256": "A" * 64}], "two checkpoint SHA-256s"),
        ([{**worker, "checkpoint_sha256": None}], "two checkpoint SHA-256s"),
        ([{key: value for key, value in worker.items() if key != "checkpoint_sha256"}], "no"),
        ([{**worker, "sha256": other}], "two checkpoint SHA-256s"),
        ([{**worker, "files": [{**files[0], "sha256": other}, *files[1:]]}], "two checkpoint"),
        ([{**worker, "files": [{**files[0], "sha256": None}, *files[1:]]}], "two checkpoint"),
        ([{**worker, "files": [*files, {**files[0], "sha256": other}]}], "two checkpoint"),
        ([{**worker, "files": {"checkpoint.pt": digest}}], "files malformed"),
        ([{**worker, "files": ["checkpoint.pt"]}], "files malformed"),
        ([worker, {**worker, "checkpoint_sha256": other, "files": []}], "listed with two"),
        ([{"sha256": digest}], "single"),
        (["stage2-final"], "must be an object"),
        ("stage2-final", "a list or an object"),
    ):
        with pytest.raises(ValueError, match=message):
            unverified_milestones({"unverified_milestones": bad}, run="A")


# A heartbeat that cannot be this Pod's is never mirrored ---------------------------------


@pytest.mark.parametrize(
    ("extra", "created", "reason"),
    [
        ({"pod_id": "previous-pod"}, None, "heartbeat_names_another_pod"),
        ({"pod_id": POD}, 1950.0, "heartbeat_predates_pod_creation"),
        ({}, 1950.0, "heartbeat_predates_pod_creation"),
    ],
)
def test_a_heartbeat_for_another_pod_is_never_mirrored_or_acted_on(
    tmp_path: Path, extra: dict[str, Any], created: float | None, reason: str
) -> None:
    r2 = FakeR2()
    report, final = finished(r2)
    pod = Pod(heartbeat("awaiting_transfer", r2_final=final, **extra), report)
    unit, archive = supervisor(tmp_path, pod, r2, pod_created_unix=created)
    old = {"status": "shutdown_ready", "pod_id": "previous-pod", "updated_at_unix": 10}
    atomic_json(unit.root / "heartbeat.json", old)  # The reused directory's old mirror.
    with pytest.raises(ValueError, match=reason):
        unit.cycle()
    with pytest.raises(ValueError, match=reason):
        unit.refresh_heartbeat()
    assert json.loads((unit.root / "heartbeat.json").read_text()) == old  # Not overwritten.
    assert not unit.shutdown_ready and not archive.bundles and not r2.calls
    # The same heartbeat naming this Pod, after its creation, is read as usual.
    pod.heartbeat = heartbeat("awaiting_transfer", r2_final=final, pod_id=POD)
    unit.pod_created_unix = 1000.0
    unit.cycle()
    assert unit.shutdown_ready and mirrored(unit)["pod_id"] == POD


def test_finalization_json_is_a_bounded_status_file(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, Any]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        requests.append(json.loads(kwargs["input"]))
        assert not [name for name in kwargs["env"] if name.startswith("R2_")]
        return subprocess.CompletedProcess(command, 0, b'{"unverified_milestones": []}', b"")

    monkeypatch.setattr("tools.pilot_supervisor.subprocess.run", run)
    data = fetch_status_bytes(["ssh"], "/workspace/rehearsal-run", "finalization.json")
    assert json.loads(data) == {"unverified_milestones": []}
    assert requests == [
        {
            "root": "/workspace/rehearsal-run",
            "action": "stream",
            "artifact": "finalization.json",
            "max_bytes": 1_000_000,
        }
    ]
    with pytest.raises(ValueError, match="unsupported"):
        fetch_status_bytes(["ssh"], "/workspace/rehearsal-run", "exports")


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


def test_main_bounds_the_checks_a_deadline_depends_on_with_the_macs_own_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", "rehearsal")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "placeholder-mac-key-id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "placeholder-mac-secret")
    monkeypatch.setenv("R2_POD_ACCESS_KEY_ID", "placeholder-pod-key-id")
    built: dict[str, Any] = {}

    class Built(Exception):
        pass

    def capture(*args: Any, **kwargs: Any) -> Any:
        built.update(kwargs)
        raise Built  # Stop before the first SSH request.

    def no_ssh(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no SSH in this test")

    monkeypatch.setattr("tools.pilot_supervisor.subprocess.run", no_ssh)
    monkeypatch.setattr(pilot_supervisor, "RehearsalSupervisor", capture)
    policy = {**REHEARSAL_POLICY, "max_transfer_seconds": 900}
    with pytest.raises(Built):
        pilot_supervisor.main(_cli(tmp_path, policy, ["--mode", "rehearsal", "--run", "A"]))
    store, quick = built["store"], built["quick_store"]
    assert isinstance(store, Store) and isinstance(quick, Store) and store is not quick
    assert quick.credentials == store.credentials  # The same Mac key, never the Pod's.
    assert store.credentials.access_key_id == "placeholder-mac-key-id"
    # verify_final and its downloads keep the default timeouts and attempts.
    assert store.attempts == 4 and store.transport is r2_store._urllib_transport
    # The listing and HEAD checks: 5 s per socket operation, so a dead endpoint costs the
    # Mac seconds before the pull can start stamping progress.
    assert pilot_supervisor.MAC_CHECK_TIMEOUT_SECONDS == 5.0
    for call in (quick.transport, quick.opener):
        assert isinstance(call, partial) and call.keywords == {"timeout": 5.0}


def test_listed_files_are_plain_names_with_the_checkpoint_among_them() -> None:
    row = export_row("stage2-final", b"weights")
    files = pilot_supervisor.listed_files({"unverified_milestones": [row]}, run="A")
    assert [name for name, _ in files["stage2-final"]][0] == "checkpoint.pt"
    legacy = {"label": "stage2-final", "path": "exports/stage2-final", "sha256": sha(b"w")}
    assert pilot_supervisor.listed_files({"unverified_milestones": [legacy]}, run="A") == {
        "stage2-final": [("checkpoint.pt", sha(b"w"))]
    }
    for change in (
        {"name": "../escape"},
        {"name": ".hidden"},
        {"sha256": "X" * 64},
        {"sha256": None},
    ):
        bad = {**row, "files": [row["files"][0], {**row["files"][1], **change}]}
        with pytest.raises(ValueError, match="malformed"):
            pilot_supervisor.listed_files({"unverified_milestones": [bad]}, run="A")
    without = {**row, "files": row["files"][1:]}  # No checkpoint row: its digest is then unchecked.
    with pytest.raises(ValueError, match="malformed"):
        pilot_supervisor.listed_files({"unverified_milestones": [without]}, run="A")
    twice = {**row, "files": [*row["files"], row["files"][1]]}
    with pytest.raises(ValueError, match="malformed"):
        pilot_supervisor.listed_files({"unverified_milestones": [twice]}, run="A")
