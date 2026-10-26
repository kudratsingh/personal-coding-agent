"""Pull explicitly selected pilot artifacts over pinned-host-key SSH."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

_REMOTE_SCRIPT = r"""
import hashlib, json, os, stat, sys

def open_root(path):
    if not isinstance(path, str) or not path.startswith('/') or '\x00' in path:
        raise ValueError('remote root must be an absolute path')
    parts = path.split('/')[1:]
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError('remote root has an unsafe component')
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open('/', flags)
    try:
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise

def open_artifact(root_fd, name):
    if not isinstance(name, str) or name in ('', '.', '..') or '/' in name or '\x00' in name:
        raise ValueError('artifact must be a basename')
    fd = os.open(name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=root_fd)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ValueError('artifact is not a regular file')
    return fd, info.st_size

request = json.load(sys.stdin)
if not isinstance(request, dict):
    raise ValueError('invalid request')
root_fd = open_root(request.get('root'))
try:
    action = request.get('action')
    if action == 'manifest':
        rows = request.get('artifacts')
        if not isinstance(rows, list):
            raise ValueError('invalid artifact list')
        result = {}
        for name in rows:
            fd, size = open_artifact(root_fd, name)
            digest = hashlib.sha256()
            try:
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            finally:
                os.close(fd)
            result[name] = {'sha256': digest.hexdigest(), 'size_bytes': size}
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(',', ':')))
    elif action == 'stream':
        fd, size = open_artifact(root_fd, request.get('artifact'))
        try:
            maximum = request.get('max_bytes')
            if maximum is not None and (type(maximum) is not int or maximum < size):
                raise ValueError('status file exceeds requested bound')
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
        finally:
            os.close(fd)
    else:
        raise ValueError('unsupported action')
finally:
    os.close(root_fd)
"""

_SAFE_HOST = re.compile(r"[A-Za-z0-9._-]+\Z")
_PROXY_SUFFIX = "ssh.runpod.io"  # The Runpod SSH proxy carries no file transfers.
_SAFE_USER = re.compile(r"[A-Za-z0-9._-]+\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SAFE_LOG = re.compile(
    r"(?:events-rank[0-9]+|debug-rank[0-9]+|console|training|metrics)\.(?:jsonl|log|txt)\Z"
)
_FORBIDDEN_NAME_PARTS = (
    "corpus",
    "shard",
    "dataset",
    "tokenizer",
    "credential",
    "secret",
    "privatekey",
)


@dataclass(frozen=True)
class Artifact:
    """A single explicitly selected pilot artifact in the remote run directory."""

    kind: str
    path: str

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name


RunSSH = Callable[..., subprocess.CompletedProcess[bytes]]


def _validate_host(host: str) -> str:
    if not isinstance(host, str) or not host or host.startswith("-") or host.count("@") > 1:
        raise ValueError("SSH host must be a user@host or host value without a leading dash")
    if "@" in host:
        user, hostname = host.split("@")
        if not _SAFE_USER.fullmatch(user) or not _SAFE_HOST.fullmatch(hostname):
            raise ValueError(
                "SSH user and host must contain only letters, digits, dot, underscore, or dash"
            )
    elif not _SAFE_HOST.fullmatch(host):
        raise ValueError("SSH host must contain only letters, digits, dot, underscore, or dash")
    if host.split("@")[-1].startswith("-"):
        raise ValueError("SSH hostname must not start with a dash")
    if host.split("@")[-1].lower().endswith(_PROXY_SUFFIX):
        raise ValueError("the Runpod SSH proxy cannot carry transfers; use the direct public IP")
    return host


def _validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("SSH port must be an integer from 1 to 65535")
    return port


def _validate_local_key(path: Path, name: str) -> Path:
    resolved = path.expanduser().absolute()
    try:
        info = resolved.lstat()
    except OSError as exc:
        raise ValueError(f"{name} file is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular, non-symlink file")
    return resolved


def _validate_remote_root(root: str) -> str:
    if not isinstance(root, str) or not root.startswith("/"):
        raise ValueError("remote directory must be an absolute POSIX path")
    path = PurePosixPath(root)
    if any(part in (".", "..") for part in path.parts) or "\x00" in root:
        raise ValueError("remote directory contains an unsafe path component")
    return root


def _validate_artifacts(artifacts: Sequence[Artifact]) -> tuple[Artifact, ...]:
    result = tuple(artifacts)
    if not result:
        raise ValueError("select at least one checkpoint, log, or record")
    names: set[str] = set()
    kinds: set[str] = set()
    for artifact in result:
        if artifact.kind not in {"checkpoint", "log", "record"}:
            raise ValueError("artifact kind must be checkpoint, log, or record")
        path = PurePosixPath(artifact.path)
        if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in (".", ".."):
            raise ValueError("artifact path must be a single basename")
        name = path.name
        if not _SAFE_NAME.fullmatch(name) or name.startswith("."):
            raise ValueError("artifact filename contains unsafe characters")
        if any(part in name.lower() for part in _FORBIDDEN_NAME_PARTS) or name.lower() in {
            "id_rsa",
            "id_ed25519",
            "credentials",
            "known_hosts",
        }:
            raise ValueError("sensitive or raw-data files cannot be transferred")
        if artifact.kind == "checkpoint" and not (
            name.startswith("checkpoint") and name.endswith(".pt")
        ):
            raise ValueError("checkpoint selection must name a checkpoint .pt file")
        if artifact.kind == "record" and name != "record.json":
            raise ValueError("record selection must be record.json")
        if artifact.kind == "log" and not _SAFE_LOG.fullmatch(name):
            raise ValueError(
                "log selection must be an events, debug, console, training, or metrics log"
            )
        if name in names:
            raise ValueError("artifact filenames must be unique")
        if artifact.kind in kinds and artifact.kind != "log":
            raise ValueError(f"only one {artifact.kind} artifact may be selected")
        names.add(name)
        kinds.add(artifact.kind)
    return result


def _ssh_command(
    host: str,
    port: int,
    identity_file: Path,
    known_hosts: Path,
) -> list[str]:
    host = _validate_host(host)
    port = _validate_port(port)
    identity = _validate_local_key(identity_file, "identity")
    hosts = _validate_local_key(known_hosts, "known-hosts")
    # The only remote command is fixed code; all remote paths and filenames are
    # sent as JSON on stdin, never interpolated into the remote shell command.
    import shlex

    remote = "python3 -c " + shlex.quote(_REMOTE_SCRIPT)
    return [
        "ssh",
        "-T",
        "-F",
        "/dev/null",
        "-p",
        str(port),
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        host,
        remote,
    ]


def child_environment() -> dict[str, str]:
    """Environment for ssh/gh children without the Runpod API key or any R2 credential."""
    return {
        key: value
        for key, value in os.environ.items()
        if key != "RUNPOD_API_KEY" and not key.startswith("R2_")
    }


def _request(
    root: str, action: str, *, artifacts: Sequence[Artifact] = (), artifact: str = ""
) -> bytes:
    value: dict[str, object] = {"root": root, "action": action}
    if action == "manifest":
        value["artifacts"] = [item.name for item in artifacts]
    else:
        value["artifact"] = artifact
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _remote_manifest(raw: bytes, artifacts: Sequence[Artifact]) -> dict[str, dict[str, int | str]]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("remote returned an invalid artifact manifest") from exc
    if not isinstance(value, dict) or set(value) != {item.name for item in artifacts}:
        raise ValueError("remote manifest does not match the selected artifacts")
    result: dict[str, dict[str, int | str]] = {}
    for name, row in value.items():
        if not isinstance(row, dict) or set(row) != {"sha256", "size_bytes"}:
            raise ValueError("remote manifest entry has an invalid shape")
        digest, size = row["sha256"], row["size_bytes"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError("remote manifest entry has invalid size or SHA-256")
        result[name] = {"sha256": digest, "size_bytes": size}
    return result


def _safe_bundle_name(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or value.startswith("."):
        raise ValueError("bundle name must be a simple filename-safe component")
    return value


def pull_artifacts(
    *,
    host: str,
    port: int,
    identity_file: Path,
    known_hosts: Path,
    remote_dir: str,
    destination: Path,
    bundle_name: str,
    artifacts: Sequence[Artifact],
    timeout: float = 900,
    run_ssh: RunSSH | None = None,
) -> Path:
    """Verify a remote manifest, stream selected files, and atomically publish a bundle."""
    host = _validate_host(host)
    port = _validate_port(port)
    root = _validate_remote_root(remote_dir)
    selected = _validate_artifacts(artifacts)
    bundle = _safe_bundle_name(bundle_name)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number of seconds")
    ssh = _ssh_command(host, port, identity_file, known_hosts)
    runner = subprocess.run if run_ssh is None else run_ssh
    deadline = time.monotonic() + float(timeout)

    def remaining_time() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("artifact transfer exceeded its total timeout")
        return remaining

    parent = destination.expanduser().absolute()
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("destination must be a real directory")
    final = parent / bundle
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"destination bundle already exists: {final}")

    manifest_result = runner(
        ssh,
        input=_request(root, "manifest", artifacts=selected),
        capture_output=True,
        timeout=remaining_time(),
        check=False,
        env=child_environment(),
    )
    if manifest_result.returncode != 0:
        tail = (manifest_result.stderr or b"")[-300:].decode("utf-8", errors="replace")
        raise RuntimeError(f"SSH manifest request failed: {tail.strip()}")
    manifest = _remote_manifest(manifest_result.stdout, selected)
    required_bytes = sum(int(row["size_bytes"]) for row in manifest.values())
    if required_bytes > shutil.disk_usage(parent).free:
        raise OSError("insufficient local free space for selected artifacts")

    staging = parent / f".{bundle}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        verified: list[dict[str, object]] = []
        for artifact in selected:
            expected = manifest[artifact.name]
            staged_path = staging / artifact.name
            digest = hashlib.sha256()
            size = 0
            with staged_path.open("xb") as output:
                result = runner(
                    ssh,
                    input=_request(root, "stream", artifact=artifact.name),
                    stdout=output,
                    stderr=subprocess.PIPE,
                    timeout=remaining_time(),
                    check=False,
                    env=child_environment(),
                )
                if result.returncode != 0:
                    tail = (result.stderr or b"")[-300:].decode("utf-8", errors="replace")
                    raise RuntimeError(f"SSH artifact stream failed: {tail.strip()}")
                output.flush()
                os.fsync(output.fileno())
            with staged_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if size != expected["size_bytes"] or actual_hash != expected["sha256"]:
                raise ValueError(
                    f"downloaded artifact failed size/SHA-256 verification: {artifact.name}"
                )
            verified.append(
                {
                    "kind": artifact.kind,
                    "name": artifact.name,
                    "sha256": actual_hash,
                    "size_bytes": size,
                }
            )

        receipt = {
            "schema_version": 1,
            "status": "verified",
            "host": host,
            "remote_dir": root,
            "transferred_at": datetime.now(UTC).isoformat(),
            "artifacts": verified,
            "source_checkpoint": next(
                (
                    {
                        "path": f"{root}/{row['name']}",
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in verified
                    if row["kind"] == "checkpoint"
                ),
                None,
            ),
        }
        receipt_path = staging / "transfer-receipt.json"
        with receipt_path.open("x", encoding="utf-8") as receipt_file:
            json.dump(receipt, receipt_file, sort_keys=True, separators=(",", ":"))
            receipt_file.write("\n")
            receipt_file.flush()
            os.fsync(receipt_file.fileno())
        staging_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.replace(staging, final)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="RunPod SSH host, optionally user@host")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument(
        "--remote-dir", required=True, help="absolute path to the prepared run directory"
    )
    parser.add_argument("--destination", type=Path, required=True, help="local parent directory")
    parser.add_argument("--bundle-name", required=True, help="new local directory name")
    parser.add_argument("--checkpoint", help="explicit checkpoint basename, such as checkpoint.pt")
    parser.add_argument("--record", action="store_true", help="pull record.json")
    parser.add_argument(
        "--log", action="append", default=[], help="explicit events/debug/console log basename"
    )
    parser.add_argument("--timeout", type=float, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    artifacts: list[Artifact] = []
    if args.checkpoint:
        artifacts.append(Artifact("checkpoint", args.checkpoint))
    if args.record:
        artifacts.append(Artifact("record", "record.json"))
    artifacts.extend(Artifact("log", name) for name in args.log)
    try:
        result = pull_artifacts(
            host=args.host,
            port=args.port,
            identity_file=args.identity_file,
            known_hosts=args.known_hosts,
            remote_dir=args.remote_dir,
            destination=args.destination,
            bundle_name=args.bundle_name,
            artifacts=artifacts,
            timeout=args.timeout,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"pilot-transfer: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
