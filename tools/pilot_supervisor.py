"""Mac-side heartbeat, verified transfer and private record/log archival.

An independent thread checks Runpod while SSH transfers run. This manages only
an explicitly identified existing Pod and cannot guarantee API availability.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools.pilot_transfer import Artifact, _ssh_command, child_environment, pull_artifacts
from tools.runpod_watchdog import (
    API_KEY_ENV,
    Outcome,
    PodIdentity,
    Policy,
    RunpodClient,
    UrllibTransport,
    atomic_receipt,
    check_once,
    load_observed,
    save_observed,
)

_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_BUNDLE = re.compile(r"exports/lr-[0-9]+(?:-continued)?-step[0-9]+-[0-9]+\Z")


def fetch_status(ssh: list[str], root: str, name: str) -> dict[str, Any]:
    if name not in {"heartbeat.json", "report.json"}:
        raise ValueError("unsupported status file")
    limit = 65_536 if name == "heartbeat.json" else 4_194_304
    request = json.dumps({"root": root, "action": "stream", "artifact": name, "max_bytes": limit})
    result = subprocess.run(
        ssh,
        input=request.encode(),
        capture_output=True,
        timeout=15,
        check=False,
        env=child_environment(),
    )
    if result.returncode or len(result.stdout) > limit:
        raise RuntimeError("bounded SSH status request failed")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("remote status must be an object")
    return value


def _gh(
    arguments: list[str], *, payload: dict[str, Any] | None = None, missing_ok: bool = False
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pilot-github-") as directory:
        if payload is not None:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(payload))
            arguments = [*arguments, "--input", str(path)]
        result = subprocess.run(
            ["gh", "api", *arguments],
            capture_output=True,
            timeout=30,
            check=False,
            env=child_environment(),
        )
        try:
            value = json.loads(result.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError):
            value = None
        if result.returncode:
            if missing_ok and isinstance(value, dict) and str(value.get("status")) == "404":
                return {"missing": True}
            # gh diagnostics can contain request data; never echo them here.
            raise RuntimeError(f"private GitHub request failed (exit {result.returncode})")
        if not isinstance(value, dict):
            raise ValueError("GitHub returned a malformed object")
        return value


def archive_bundle(bundle: Path, repository: str, *, run_id: str) -> dict[str, Any]:
    """Upload only record/log, then read both back and compare exact bytes."""
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("invalid repository name")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is None:
        raise ValueError("run ID must be a safe path component")
    repo = _gh([f"repos/{repository}"])
    if repo.get("private") is not True or repo.get("full_name") != repository:
        raise ValueError("artifact repository must be verified private")
    prefix = f"pilot-runs/{run_id}/{bundle.name}"
    rows = []
    for name in ("record.json", "training.jsonl"):
        path = bundle / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("only bounded record/log files may be archived")
        content = path.read_bytes()
        endpoint = f"repos/{repository}/contents/{prefix}/{name}"
        payload: dict[str, Any] = {
            "message": "Save pilot record and log",
            "content": base64.b64encode(content).decode(),
        }
        previous = _gh([endpoint], missing_ok=True)
        if previous.get("missing") is not True:
            payload["sha"] = previous["sha"]
        response = _gh(["--method", "PUT", endpoint], payload=payload)
        restored = _gh([endpoint])
        encoded = restored.get("content")
        if not isinstance(encoded, str) or restored.get("encoding") != "base64":
            raise ValueError("GitHub readback omitted file contents")
        if base64.b64decode("".join(encoded.split()), validate=True) != content:
            raise ValueError("private GitHub readback differs")
        rows.append(
            {
                "path": f"{prefix}/{name}",
                "sha256": file_sha256(path),
                "size_bytes": len(content),
                "commit": response["commit"]["sha"],
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "repository": repository,
        "private": True,
        "files": rows,
        "checkpoint_uploaded": False,
    }
    atomic_json(bundle / "github-receipt.json", receipt)
    return receipt


def verify_received(bundle: Path) -> dict[str, Any]:
    receipt = json.loads((bundle / "transfer-receipt.json").read_text())
    report = json.loads((bundle / "record.json").read_text())
    if not isinstance(report, dict):
        raise ValueError("run report must be an object")
    if receipt.get("status") != "verified":
        raise ValueError("Mac transfer is unverified")
    files = {row["name"]: row for row in receipt["artifacts"]}
    if set(files) != {"checkpoint.pt", "record.json", "training.jsonl"}:
        raise ValueError("transfer receipt does not cover the complete bundle")
    for name, row in files.items():
        path = bundle / name
        if (
            path.is_symlink()
            or path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError("local transfer bytes differ from receipt")
    checkpoint = torch.load(bundle / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint.get("data_hash") != report["binding_sha256"]:
        raise ValueError("copied checkpoint identity differs from run record")
    if checkpoint.get("tokenizer_hash") != report["tokenizer_identity"]:
        raise ValueError("copied checkpoint tokenizer differs from run record")
    candidates = [
        row for row in report["checkpoints"] if row["sha256"] == files["checkpoint.pt"]["sha256"]
    ]
    if not candidates or checkpoint.get("step") != candidates[-1]["step"]:
        raise ValueError("copied checkpoint step differs from run record")
    if not isinstance(checkpoint.get("model"), dict) or not isinstance(
        checkpoint.get("optimizer"), dict
    ):
        raise ValueError("copied checkpoint lacks model/optimizer state")
    return report


class Supervisor:
    def __init__(
        self,
        root: Path,
        *,
        fetch: Callable[[str], dict[str, Any]],
        pull: Callable[[str], Path],
        archive: Callable[[Path], dict[str, Any]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root, self.fetch, self.pull, self.archive = root, fetch, pull, archive
        self.received: dict[str, Path] = {}
        self.archived: set[str] = set()
        self.pending: dict[str, Path] = {}
        self.latest: Path | None = None
        self.checkpoint_bundle: Path | None = None
        self.shutdown_ready = False
        self.clock = clock
        self.staged_bytes = -1
        self.transfer_progress_unix: float | None = None
        root.mkdir(parents=True, exist_ok=False)

    def staging_bytes(self) -> int:
        """Bytes currently staged by an in-progress pull, measured on the Mac."""
        received = self.root / "received"
        if not received.is_dir():
            return 0
        total = 0
        for staging in received.glob(".*.staging-*"):
            if staging.is_dir() and not staging.is_symlink():
                for item in staging.iterdir():
                    if item.is_file() and not item.is_symlink():
                        total += item.stat().st_size
        return total

    def refresh_heartbeat(self) -> dict[str, Any]:
        heartbeat = self.fetch("heartbeat.json")
        heartbeat.pop("mac_receipt_verified", None)
        heartbeat.pop("github_receipt_verified", None)
        heartbeat.pop("transfer_progress_unix", None)  # Only Mac-observed progress counts.
        staged = self.staging_bytes()
        if staged > self.staged_bytes:
            # Growing staging bytes are evidence the Mac is receiving the artifact.
            if self.staged_bytes >= 0 or staged > 0:
                self.transfer_progress_unix = self.clock()
            self.staged_bytes = staged
        if self.transfer_progress_unix is not None:
            heartbeat["transfer_progress_unix"] = self.transfer_progress_unix
        if self.checkpoint_bundle is not None and not self.shutdown_ready:
            heartbeat["mac_receipt_verified"] = True  # Durable copy exists even if GitHub lags.
        if self.shutdown_ready:
            heartbeat.update(
                status="shutdown_ready", mac_receipt_verified=True, github_receipt_verified=True
            )
        atomic_json(self.root / "heartbeat.json", heartbeat)
        return heartbeat

    def cycle(self) -> None:
        heartbeat = self.refresh_heartbeat()
        report = self.fetch("report.json")
        atomic_json(self.root / "remote-report.json", report)
        failed = report.get("status") in {"stopped", "failed"}
        name = report.get("latest_transfer_bundle")
        if name is None:
            if failed:
                self.diagnostics(report)
                self.shutdown_ready = True
                self.refresh_heartbeat()
            return
        if not isinstance(name, str) or not _BUNDLE.fullmatch(name):
            raise ValueError("remote export path is unsafe")
        if name not in self.received:
            bundle = self.pending.get(name)
            if bundle is None:
                bundle = self.pull(name)
                self.pending[name] = bundle
            verified = verify_received(bundle)
            # The Mac copy is durable from here on, whatever GitHub does next.
            self.checkpoint_bundle = bundle
            self.received[name], self.latest = bundle, bundle
            self.pending.pop(name, None)
        else:
            bundle = self.received[name]
            verified = json.loads((bundle / "record.json").read_text())
        if not failed and name not in self.archived:
            self.archive_verified(bundle)  # Retried every cycle until it succeeds.
            self.archived.add(name)
        # Only a final export can complete a successful/time-limited run. A failed
        # run may use its last verified periodic export, while preserving failure.
        terminal = report.get("status") in {"completed", "time_limited", "stopped", "failed"}
        final_export = verified.get("status") in {"completed", "time_limited"}
        if terminal and (final_export or report.get("status") in {"stopped", "failed"}):
            if failed:
                self.diagnostics(report)
            self.shutdown_ready = True
            heartbeat.update(
                status="shutdown_ready", mac_receipt_verified=True, github_receipt_verified=True
            )
            atomic_json(self.root / "heartbeat.json", heartbeat)

    def archive_verified(self, bundle: Path) -> None:
        archived = self.archive(bundle)
        if archived.get("status") != "verified" or archived.get("private") is not True:
            raise ValueError("private GitHub archive is unverified")

    def diagnostics(self, report: dict[str, Any]) -> None:
        """Preserve terminal state even before the first checkpoint exists."""
        digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]
        directory = self.root / f"terminal-{digest}"
        directory.mkdir(exist_ok=True)
        diagnostic = {
            **report,
            "diagnostic_log_source": (
                "reconstructed from the authenticated terminal loss-curves record"
            ),
            "checkpoint_present_on_mac": self.checkpoint_bundle is not None,
            "checkpoint_bundle_on_mac": None
            if self.checkpoint_bundle is None
            else str(self.checkpoint_bundle),
        }
        atomic_json(directory / "record.json", diagnostic)
        with (directory / "training.jsonl").open("w", encoding="utf-8") as stream:
            for arm, rows in report.get("loss_curves", {}).items():
                for row in rows:
                    stream.write(json.dumps({"arm": arm, **row}, allow_nan=False) + "\n")
        atomic_json(
            directory / "diagnostic-receipt.json",
            {
                "status": "verified",
                "checkpoint_present": self.checkpoint_bundle is not None,
                "files": {
                    name: {
                        "sha256": file_sha256(directory / name),
                        "size_bytes": (directory / name).stat().st_size,
                    }
                    for name in ("record.json", "training.jsonl")
                },
            },
        )
        self.archive_verified(directory)
        self.latest = directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("identity", "policy", "identity-file", "known-hosts", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("host", "remote-root", "repository"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--transfer-timeout",
        type=float,
        default=1800.0,
        help="total seconds allowed for one bundle pull (1.5 GiB needs ~7 Mbit/s at 1800 s)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="API guard interval; the REST API is rate limited per minute",
    )
    parser.add_argument("--cycle-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if any(
        not (0 < value < 3600)
        for value in (args.transfer_timeout, args.poll_seconds, args.cycle_seconds)
    ):
        parser.error("timeouts and intervals must be positive and below one hour")
    identity = PodIdentity.from_dict(json.loads(args.identity.read_text()))
    policy = Policy(**json.loads(args.policy.read_text()))
    token = os.environ.get(API_KEY_ENV)
    if not token:
        parser.error(f"set {API_KEY_ENV} in the local environment")
    if not PurePosixPath(args.remote_root).is_absolute():
        parser.error("remote root must be absolute")
    ssh = _ssh_command(args.host, args.port, args.identity_file, args.known_hosts)
    supervisor = Supervisor(
        args.output,
        fetch=lambda name: fetch_status(ssh, args.remote_root, name),
        pull=lambda name: pull_artifacts(
            host=args.host,
            port=args.port,
            identity_file=args.identity_file,
            known_hosts=args.known_hosts,
            remote_dir=f"{args.remote_root}/{name}",
            destination=args.output / "received",
            bundle_name=PurePosixPath(name).name,
            artifacts=(
                Artifact("checkpoint", "checkpoint.pt"),
                Artifact("record", "record.json"),
                Artifact("log", "training.jsonl"),
            ),
            timeout=args.transfer_timeout,
        ),
        archive=lambda bundle: archive_bundle(bundle, args.repository, run_id=identity.pod_id),
    )
    client = RunpodClient(token, UrllibTransport())
    done = threading.Event()
    outcomes: list[Outcome] = []

    state_path = args.output / "watchdog-receipt.json.observed.json"
    observed = load_observed(state_path, identity)

    def guard() -> None:
        nonlocal observed
        while not done.is_set():
            try:
                outcome = check_once(
                    client, identity, policy, args.output / "heartbeat.json", observed=observed
                )
                if outcome.state in {"watching", "terminated"} and not observed:
                    observed = True
                    save_observed(state_path, identity, outcome.checked_at_unix)
            except Exception as error:
                outcome = Outcome(
                    "unknown", "watchdog_error", identity.pod_id, time.time(), type(error).__name__
                )
            atomic_receipt(args.output / "watchdog-receipt.json", outcome)
            if outcome.state in {"terminated", "absent", "identity_mismatch"}:
                outcomes.append(outcome)
                done.set()
                return
            done.wait(args.poll_seconds)

    watcher = threading.Thread(target=guard, name="independent-runpod-guard", daemon=False)

    def mirror() -> None:
        while not done.is_set():
            try:
                supervisor.refresh_heartbeat()
            except Exception as error:
                atomic_json(
                    args.output / "heartbeat-error.json", {"error_type": type(error).__name__}
                )
            done.wait(args.cycle_seconds)

    heartbeat_watcher = threading.Thread(target=mirror, name="ssh-heartbeat-mirror", daemon=False)
    watcher.start()
    heartbeat_watcher.start()
    interrupted = False
    try:
        while not done.is_set():
            try:
                supervisor.cycle()
            except Exception as error:
                atomic_json(
                    args.output / "controller-error.json",
                    {"error_type": type(error).__name__, "detail": str(error)[:500]},
                )
                # Preserve the last authenticated heartbeat; never fabricate GPU
                # activity or receipt flags when SSH/transfer/archival fails.
            done.wait(args.cycle_seconds)
    except KeyboardInterrupt:
        interrupted = True  # The API guard keeps the Pod bounded; stop supervising cleanly.
        done.set()
    finally:
        watcher.join()
        heartbeat_watcher.join()
    archive_failed = False
    if supervisor.latest is not None:
        final = args.output / (supervisor.latest.name + "-final")
        final.mkdir(exist_ok=True)
        path = final / "record.json"
        record = json.loads((supervisor.latest / "record.json").read_text())
        receipt = args.output / "watchdog-receipt.json"
        record["shutdown_receipt"] = json.loads(receipt.read_text()) if receipt.exists() else None
        receipt_path = supervisor.latest / "transfer-receipt.json"
        if not receipt_path.exists():
            receipt_path = supervisor.latest / "diagnostic-receipt.json"
        record["durable_copy_receipt"] = (
            json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        )
        record["resource_audit"] = (
            "Pod only; network volumes and account billing require separate audit"
        )
        record["supervisor_interrupted"] = interrupted
        atomic_json(path, record)  # Always written locally, before any GitHub call.
        (final / "training.jsonl").write_bytes((supervisor.latest / "training.jsonl").read_bytes())
        try:
            archive_bundle(final, args.repository, run_id=identity.pod_id)
        except Exception as error:
            archive_failed = True
            atomic_json(
                args.output / "final-archive-error.json",
                {"error_type": type(error).__name__, "detail": str(error)[:500]},
            )
    if interrupted or archive_failed:
        return 2
    return 0 if outcomes and outcomes[-1].state in {"terminated", "absent"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
