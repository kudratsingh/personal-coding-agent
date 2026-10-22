import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools.pilot_transfer import _REMOTE_SCRIPT, Artifact, pull_artifacts


def credentials(tmp_path: Path) -> tuple[Path, Path]:
    identity = tmp_path / "pilot-key"
    known_hosts = tmp_path / "known_hosts"
    identity.write_text("synthetic private key placeholder")
    known_hosts.write_text("synthetic host key placeholder")
    return identity, known_hosts


def fake_ssh(
    remote: dict[str, bytes], seen: list[tuple[list[str], dict[str, Any]]]
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append((argv, kwargs))
        request = json.loads(kwargs["input"])
        if request["action"] == "manifest":
            manifest = {
                name: {
                    "sha256": hashlib.sha256(remote[name]).hexdigest(),
                    "size_bytes": len(remote[name]),
                }
                for name in request["artifacts"]
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(manifest).encode(), b"")
        kwargs["stdout"].write(remote[request["artifact"]])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    return run


def artifacts() -> list[Artifact]:
    return [
        Artifact("checkpoint", "checkpoint.pt"),
        Artifact("record", "record.json"),
        Artifact("log", "events-rank0.jsonl"),
        Artifact("log", "debug-rank0.jsonl"),
    ]


def test_pull_verifies_selected_files_and_atomically_publishes_bundle(tmp_path: Path) -> None:
    identity, known_hosts = credentials(tmp_path)
    payloads = {
        "checkpoint.pt": b"synthetic checkpoint bytes",
        "record.json": b'{"status":"completed"}',
        "events-rank0.jsonl": b'{"event":"step"}\n',
        "debug-rank0.jsonl": b"",
    }
    calls: list[tuple[list[str], dict[str, Any]]] = []
    result = pull_artifacts(
        host="root@203.0.113.8",
        port=2222,
        identity_file=identity,
        known_hosts=known_hosts,
        remote_dir="/workspace/pilot",
        destination=tmp_path / "received",
        bundle_name="run-001",
        artifacts=artifacts(),
        run_ssh=fake_ssh(payloads, calls),
    )

    assert result == tmp_path / "received/run-001"
    for name, payload in payloads.items():
        assert (result / name).read_bytes() == payload
    receipt = json.loads((result / "transfer-receipt.json").read_text())
    assert receipt["status"] == "verified"
    assert {row["name"] for row in receipt["artifacts"]} == set(payloads)
    assert receipt["source_checkpoint"] == {
        "path": "/workspace/pilot/checkpoint.pt",
        "sha256": hashlib.sha256(payloads["checkpoint.pt"]).hexdigest(),
        "size_bytes": len(payloads["checkpoint.pt"]),
    }
    assert len(calls) == len(payloads) + 1
    command = calls[0][0]
    assert command[0] == "ssh"
    assert "-T" in command
    assert "-i" in command and str(identity) in command
    assert "-p" in command and command[command.index("-p") + 1] == "2222"
    assert "StrictHostKeyChecking=yes" in command
    assert "BatchMode=yes" in command
    assert "IdentitiesOnly=yes" in command
    assert "UserKnownHostsFile=" + str(known_hosts) in command
    assert "/workspace/pilot" not in command[-1]
    assert json.loads(calls[0][1]["input"])["root"] == "/workspace/pilot"
    assert not list((tmp_path / "received").glob("*.staging-*"))


def test_bad_download_is_never_promoted_and_staging_is_removed(tmp_path: Path) -> None:
    identity, known_hosts = credentials(tmp_path)
    payloads = {"checkpoint.pt": b"expected"}

    def corrupt(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        request = json.loads(kwargs["input"])
        if request["action"] == "manifest":
            manifest = {
                name: {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
                for name, payload in payloads.items()
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(manifest).encode(), b"")
        kwargs["stdout"].write(b"truncated")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    parent = tmp_path / "received"
    with pytest.raises(ValueError, match="verification"):
        pull_artifacts(
            host="pod.example",
            port=22,
            identity_file=identity,
            known_hosts=known_hosts,
            remote_dir="/workspace/pilot",
            destination=parent,
            bundle_name="run-002",
            artifacts=[Artifact("checkpoint", "checkpoint.pt")],
            run_ssh=corrupt,
        )

    assert not (parent / "run-002").exists()
    assert not list(parent.glob(".run-002.staging-*"))


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("-oProxyCommand=bad", 22),
        ("user@-host", 22),
        ("node.example", True),
        ("node.example", 65536),
    ],
)
def test_host_option_injection_and_invalid_ports_reject_before_ssh(
    tmp_path: Path, host: str, port: int
) -> None:
    identity, known_hosts = credentials(tmp_path)
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        nonlocal called
        called = True
        raise AssertionError("SSH must not run for invalid options")

    with pytest.raises(ValueError):
        pull_artifacts(
            host=host,
            port=port,
            identity_file=identity,
            known_hosts=known_hosts,
            remote_dir="/workspace/pilot",
            destination=tmp_path / "received",
            bundle_name="run-003",
            artifacts=[Artifact("record", "record.json")],
            run_ssh=forbidden,
        )
    assert not called


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("checkpoint", "../checkpoint.pt"),
        ("record", ".env"),
        ("log", "corpus.jsonl"),
        ("log", "data-000.jsonl"),
        ("log", "events-data-000.jsonl"),
        ("record", "nested/record.json"),
    ],
)
def test_only_safe_explicit_artifact_basenames_are_accepted(
    tmp_path: Path, kind: str, path: str
) -> None:
    identity, known_hosts = credentials(tmp_path)
    with pytest.raises(ValueError):
        pull_artifacts(
            host="node.example",
            port=22,
            identity_file=identity,
            known_hosts=known_hosts,
            remote_dir="/workspace/pilot",
            destination=tmp_path / "received",
            bundle_name="run-004",
            artifacts=[Artifact(kind, path)],
            run_ssh=lambda *_args, **_kwargs: pytest.fail("SSH must not run"),
        )


def test_remote_program_opens_paths_without_following_symlinks() -> None:
    assert "O_NOFOLLOW" in _REMOTE_SCRIPT
    assert "dir_fd=fd" in _REMOTE_SCRIPT
    assert "dir_fd=root_fd" in _REMOTE_SCRIPT
    assert "os.path.realpath" not in _REMOTE_SCRIPT


def test_remote_manifest_hashes_files_and_rejects_symlinks(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.pt").write_bytes(b"fixture weights")
    request = {
        "action": "manifest",
        "root": str(tmp_path),
        "artifacts": ["checkpoint.pt"],
    }
    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_SCRIPT],
        input=json.dumps(request).encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    manifest = json.loads(result.stdout)
    assert manifest["checkpoint.pt"] == {
        "sha256": hashlib.sha256(b"fixture weights").hexdigest(),
        "size_bytes": len(b"fixture weights"),
    }

    (tmp_path / "checkpoint.pt").unlink()
    (tmp_path / "checkpoint.pt").symlink_to(tmp_path / "outside.pt")
    (tmp_path / "outside.pt").write_bytes(b"must not follow")
    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_SCRIPT],
        input=json.dumps(request).encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout == b""


def test_existing_bundle_is_never_overwritten(tmp_path: Path) -> None:
    identity, known_hosts = credentials(tmp_path)
    parent = tmp_path / "received"
    bundle = parent / "run-005"
    bundle.mkdir(parents=True)
    (bundle / "marker").write_text("keep")

    with pytest.raises(FileExistsError):
        pull_artifacts(
            host="node.example",
            port=22,
            identity_file=identity,
            known_hosts=known_hosts,
            remote_dir="/workspace/pilot",
            destination=parent,
            bundle_name="run-005",
            artifacts=[Artifact("record", "record.json")],
            run_ssh=lambda *_args, **_kwargs: pytest.fail("SSH must not run"),
        )
    assert (bundle / "marker").read_text() == "keep"


def test_ssh_command_has_keepalives_and_rejects_the_runpod_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.pilot_transfer import _ssh_command, _validate_host, child_environment

    identity, known_hosts = credentials(tmp_path)
    command = _ssh_command("root@203.0.113.8", 22, identity, known_hosts)
    assert "ConnectTimeout=10" in command and "ServerAliveInterval=15" in command
    with pytest.raises(ValueError, match="proxy"):
        _validate_host("abc123-6441@ssh.runpod.io")
    monkeypatch.setenv("RUNPOD_API_KEY", "placeholder")
    assert "RUNPOD_API_KEY" not in child_environment()
