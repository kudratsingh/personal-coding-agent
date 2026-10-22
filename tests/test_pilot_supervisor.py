import base64
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools.pilot_supervisor import Supervisor, archive_bundle, verify_received
from tools.runpod_watchdog import PodIdentity, Policy, RunpodClient, check_once


def bundle(root: Path, *, status: str = "completed") -> Path:
    root.mkdir()
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "version": 1,
            "data_hash": "fixture-data",
            "tokenizer_hash": "fixture-tokenizer",
            "step": 4,
            "model": {"weight": torch.ones(2)},
            "optimizer": {"m": [torch.zeros(2)]},
        },
        checkpoint,
    )
    atomic_json(
        root / "record.json",
        {
            "status": status,
            "binding_sha256": "fixture-data",
            "tokenizer_identity": "fixture-tokenizer",
            "checkpoints": [{"sha256": file_sha256(checkpoint), "step": 4}],
        },
    )
    (root / "training.jsonl").write_text('{"step":4,"loss":1}\n')
    atomic_json(
        root / "transfer-receipt.json",
        {
            "status": "verified",
            "artifacts": [
                {
                    "name": name,
                    "sha256": file_sha256(root / name),
                    "size_bytes": (root / name).stat().st_size,
                }
                for name in ("checkpoint.pt", "record.json", "training.jsonl")
            ],
        },
    )
    return root


def heartbeat() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at_unix": 1900,
        "last_gpu_activity_unix": 1899,
        "phase": "finalizing",
        "step": 4,
        "status": "awaiting_transfer",
    }


def test_remote_heartbeat_and_receipts_connect_to_external_delete(tmp_path: Path) -> None:
    copied = bundle(tmp_path / "received")
    report = {"status": "completed", "latest_transfer_bundle": "exports/lr-0-step4-1"}
    supervisor = Supervisor(
        tmp_path / "controller",
        fetch=lambda name: heartbeat() if name == "heartbeat.json" else report,
        pull=lambda _: copied,
        archive=lambda _: {"status": "verified", "private": True},
    )
    supervisor.cycle()
    local = json.loads((supervisor.root / "heartbeat.json").read_text())
    assert local["last_gpu_activity_unix"] == 1899
    assert local["status"] == "shutdown_ready"
    assert local["mac_receipt_verified"] and local["github_receipt_verified"]

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
                        "id": "fixture",
                        "name": "pilot",
                        "gpu": {"id": "A40", "count": 1},
                        "dataCenterId": "test",
                        "createdAt": "1970-01-01T00:16:40+00:00",
                        "cost": 0.5,
                    }
                ).encode()
            return (204, b"") if method == "DELETE" else (404, b"")

    transport = API()
    identity = PodIdentity.from_dict(
        {
            "id": "fixture",
            "name": "pilot",
            "gpuId": "A40",
            "gpuCount": 1,
            "dataCenterId": "test",
            "createdAt": "1970-01-01T00:16:40+00:00",
        }
    )
    outcome = check_once(
        RunpodClient("authored-test-key", transport),
        identity,
        Policy(5400, 5, 600, 0.5),
        supervisor.root / "heartbeat.json",
        now=lambda: 1900,
    )
    assert outcome.state == "terminated"
    assert transport.calls == ["GET", "DELETE", "GET"]


def test_unverified_archive_never_sets_shutdown_flags(tmp_path: Path) -> None:
    copied = bundle(tmp_path / "received")
    supervisor = Supervisor(
        tmp_path / "controller",
        fetch=lambda name: (
            heartbeat()
            if name == "heartbeat.json"
            else {"status": "completed", "latest_transfer_bundle": "exports/lr-0-step4-1"}
        ),
        pull=lambda _: copied,
        archive=lambda _: {"status": "failed", "private": True},
    )
    with pytest.raises(ValueError, match="unverified"):
        supervisor.cycle()
    local = json.loads((supervisor.root / "heartbeat.json").read_text())
    assert local["status"] == "awaiting_transfer"
    # The verified Mac copy is recorded even though the GitHub archive failed.
    assert "exports/lr-0-step4-1" in supervisor.received and not supervisor.archived
    supervisor.refresh_heartbeat()
    local = json.loads((supervisor.root / "heartbeat.json").read_text())
    assert local.get("mac_receipt_verified") is True
    assert not local.get("github_receipt_verified") and local["status"] == "awaiting_transfer"
    pulls: list[str] = []

    def pull(name: str) -> Path:
        pulls.append(name)
        return copied

    supervisor.pull = pull
    supervisor.archive = lambda _: {"status": "verified", "private": True}
    supervisor.cycle()
    assert not pulls  # Retry the already verified Mac copy instead of copying over it.
    assert supervisor.shutdown_ready
    supervisor.refresh_heartbeat()
    assert (
        json.loads((supervisor.root / "heartbeat.json").read_text())["status"] == "shutdown_ready"
    )


def test_intermediate_export_does_not_end_live_training(tmp_path: Path) -> None:
    copied = bundle(tmp_path / "received", status="running")
    report = {"status": "running", "latest_transfer_bundle": "exports/lr-0-step4-1"}
    archives: list[Path] = []

    def archive(path: Path) -> dict[str, Any]:
        archives.append(path)
        return {"status": "verified", "private": True}

    supervisor = Supervisor(
        tmp_path / "controller",
        fetch=lambda name: (
            {**heartbeat(), "status": "running"} if name == "heartbeat.json" else report
        ),
        pull=lambda _: copied,
        archive=archive,
    )
    supervisor.cycle()
    supervisor.cycle()
    assert archives == [copied]
    assert json.loads((supervisor.root / "heartbeat.json").read_text())["status"] == "running"


def test_failed_remote_fetch_does_not_refresh_activity(tmp_path: Path) -> None:
    def unreachable(name: str) -> dict[str, Any]:
        raise TimeoutError("authored timeout")

    supervisor = Supervisor(
        tmp_path / "controller", fetch=unreachable, pull=lambda _: tmp_path, archive=lambda _: {}
    )
    with pytest.raises(TimeoutError):
        supervisor.cycle()
    assert not (supervisor.root / "heartbeat.json").exists()


def test_received_bytes_and_checkpoint_identity_must_match(tmp_path: Path) -> None:
    copied = bundle(tmp_path / "received")
    assert verify_received(copied)["status"] == "completed"
    (copied / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        verify_received(copied)


def test_archive_is_private_record_log_only_and_readback_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = bundle(tmp_path / "received")
    remote: dict[str, bytes] = {}
    uploaded: list[str] = []

    def gh(
        arguments: list[str], *, payload: dict[str, Any] | None = None, missing_ok: bool = False
    ) -> dict[str, Any]:
        if arguments == ["repos/owner/private-artifacts"]:
            return {"private": True, "full_name": "owner/private-artifacts"}
        endpoint = arguments[-1]
        if payload is not None:
            if endpoint in remote:
                assert payload["sha"] == "existing-sha"
            uploaded.append(endpoint)
            remote[endpoint] = base64.b64decode(payload["content"])
            return {"commit": {"sha": "fixture-commit"}}
        if endpoint not in remote:
            assert missing_ok
            return {"missing": True}
        return {
            "encoding": "base64",
            "content": base64.b64encode(remote[endpoint]).decode(),
            "sha": "existing-sha",
        }

    monkeypatch.setattr("tools.pilot_supervisor._gh", gh)
    receipt = archive_bundle(copied, "owner/private-artifacts", run_id="fixture-pod")
    assert receipt["status"] == "verified" and receipt["checkpoint_uploaded"] is False
    assert len(uploaded) == 2 and all(not path.endswith(".pt") for path in uploaded)
    # Retry/update uses current remote SHA, including a partially completed prior upload.
    (copied / "github-receipt.json").unlink()
    archive_bundle(copied, "owner/private-artifacts", run_id="fixture-pod")
    assert len(uploaded) == 4


def test_public_repository_rejected_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tools.pilot_supervisor._gh", lambda *args, **kwargs: {"private": False})
    with pytest.raises(ValueError, match="private"):
        archive_bundle(tmp_path, "owner/public", run_id="fixture-pod")


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_terminal_failure_preserved_even_without_checkpoint(
    tmp_path: Path, with_checkpoint: bool
) -> None:
    copied = bundle(tmp_path / "received", status="running") if with_checkpoint else tmp_path
    report: dict[str, Any] = {
        "status": "failed",
        "error_type": "FloatingPointError",
        "loss_curves": {"lr-0": [{"step": 1, "loss": 2.0}]},
    }
    if with_checkpoint:
        report["latest_transfer_bundle"] = "exports/lr-0-step4-1"
    supervisor = Supervisor(
        tmp_path / "controller",
        fetch=lambda name: (
            {**heartbeat(), "status": "failed"} if name == "heartbeat.json" else report
        ),
        pull=lambda _: copied,
        archive=lambda _: {"status": "verified", "private": True},
    )
    supervisor.cycle()
    assert supervisor.shutdown_ready and supervisor.latest is not None
    saved = json.loads((supervisor.latest / "record.json").read_text())
    assert saved["status"] == "failed" and saved["error_type"] == "FloatingPointError"
    assert saved["checkpoint_present_on_mac"] is with_checkpoint
    assert json.loads((supervisor.latest / "training.jsonl").read_text())["loss"] == 2.0


def test_transfer_progress_is_stamped_only_from_growing_mac_side_staging(tmp_path: Path) -> None:
    now = [100.0]
    supervisor = Supervisor(
        tmp_path / "controller",
        fetch=lambda name: (
            {**heartbeat(), "transfer_progress_unix": 1.0}
            if name == "heartbeat.json"
            else {"status": "running"}
        ),
        pull=lambda _: tmp_path,
        archive=lambda _: {},
        clock=lambda: now[0],
    )
    first = supervisor.refresh_heartbeat()
    assert "transfer_progress_unix" not in first  # Remote-supplied stamps are discarded.
    staging = supervisor.root / "received" / ".bundle.staging-abc"
    staging.mkdir(parents=True)
    (staging / "checkpoint.pt").write_bytes(b"x" * 10)
    now[0] = 110.0
    second = supervisor.refresh_heartbeat()
    assert second["transfer_progress_unix"] == 110.0
    now[0] = 120.0
    third = supervisor.refresh_heartbeat()
    assert third["transfer_progress_unix"] == 110.0  # No growth, no new stamp.
    (staging / "checkpoint.pt").write_bytes(b"x" * 20)
    now[0] = 130.0
    assert supervisor.refresh_heartbeat()["transfer_progress_unix"] == 130.0
    local = json.loads((supervisor.root / "heartbeat.json").read_text())
    assert local["transfer_progress_unix"] == 130.0
    assert local["last_gpu_activity_unix"] == 1899  # GPU activity is never fabricated.


def test_continuation_export_names_are_accepted_and_unsafe_names_rejected(tmp_path: Path) -> None:
    from tools.pilot_supervisor import _BUNDLE

    assert _BUNDLE.fullmatch("exports/lr-1-continued-step500-7")
    assert _BUNDLE.fullmatch("exports/lr-0-step100-2")
    assert not _BUNDLE.fullmatch("exports/../lr-0-step100-2")
    assert not _BUNDLE.fullmatch("exports/probe-8-step20-1")


def test_gh_failure_with_non_json_output_is_a_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from tools.pilot_supervisor import _gh

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert "RUNPOD_API_KEY" not in kwargs["env"]
        return subprocess.CompletedProcess(args[0], 1, b"gh: not logged in", b"")

    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    monkeypatch.setattr("tools.pilot_supervisor.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="exit 1"):
        _gh(["repos/owner/private"])
