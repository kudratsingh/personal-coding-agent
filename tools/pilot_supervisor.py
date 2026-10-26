"""Mac-side heartbeat, verified transfer and private record/log archival.

An independent thread checks Runpod while SSH transfers run. This manages only
an explicitly identified existing Pod and cannot guarantee API availability.

``--mode rehearsal`` supervises one of the two rehearsal Pods (D-35) instead. It never
copies a checkpoint over SSH: the worker uploads its checkpoints to R2 itself, and once
its heartbeat says ``awaiting_transfer`` with ``r2_final``, the supervisor checks the final
milestone and ``report.json`` in R2 with the Mac's own key, archives the small records
privately, and only then reports ``shutdown_ready``. A ``running`` heartbeat, including
run A's boundary between its two stages, never leads to teardown.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import torch

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools.pilot_transfer import Artifact, _ssh_command, child_environment, pull_artifacts
from tools.r2_store import Store
from tools.rehearsal_binding import RUNS, r2_credentials
from tools.rehearsal_ops import (
    FAILURE_STATUSES,
    IDLE_SECONDS,
    MAC_KEY_PREFIX,
    REPORT_MAX_BYTES,
    SUCCESS_STATUSES,
    ObjectStore,
    check_r2_final,
    run_prefix,
    run_summary,
    verify_final,
)
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
_ARCHIVE_ROOT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*\Z")
_ARCHIVE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def fetch_status(
    ssh: list[str], root: str, name: str, *, max_bytes: int | None = None
) -> dict[str, Any]:
    value = json.loads(fetch_status_bytes(ssh, root, name, max_bytes=max_bytes))
    if not isinstance(value, dict):
        raise ValueError("remote status must be an object")
    return value


def fetch_status_bytes(
    ssh: list[str], root: str, name: str, *, max_bytes: int | None = None
) -> bytes:
    """One bounded status file's exact bytes over the fixed-code SSH channel."""
    if name not in {"heartbeat.json", "report.json"}:
        raise ValueError("unsupported status file")
    limit = 65_536 if name == "heartbeat.json" else 4_194_304
    if max_bytes is not None:
        if type(max_bytes) is not int or not 0 < max_bytes <= limit:
            raise ValueError("a status bound can only tighten the default")
        limit = max_bytes
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
    return result.stdout


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


def archive_bundle(
    bundle: Path,
    repository: str,
    *,
    run_id: str,
    files: Sequence[str] = ("record.json", "training.jsonl"),
    root: str = "pilot-runs",
    message: str = "Save pilot record and log",
) -> dict[str, Any]:
    """Upload only small records, then read each back and compare exact bytes.

    The pilot archives a bundle's record and log under ``pilot-runs/``; a rehearsal
    archives its report and summary under ``rehearsal-runs/<run>``. Every file stays
    at most 1,000,000 bytes; checkpoints and loss curves never go to GitHub.
    """
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("invalid repository name")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is None:
        raise ValueError("run ID must be a safe path component")
    if not _ARCHIVE_ROOT.fullmatch(root):
        raise ValueError("archive root must be plain path components")
    if (
        not files
        or len(set(files)) != len(files)
        or not all(_ARCHIVE_NAME.fullmatch(name) for name in files)
    ):
        raise ValueError("archive files must be distinct plain file names")
    repo = _gh([f"repos/{repository}"])
    if repo.get("private") is not True or repo.get("full_name") != repository:
        raise ValueError("artifact repository must be verified private")
    prefix = f"{root}/{run_id}/{bundle.name}"
    rows = []
    for name in files:
        path = bundle / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
            raise ValueError("only bounded record/log files may be archived")
        content = path.read_bytes()
        endpoint = f"repos/{repository}/contents/{prefix}/{name}"
        payload: dict[str, Any] = {
            "message": message,
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
        reuse_output: bool = False,
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
        # A rehearsal supervisor restarted after a crash continues in its directory, where
        # the run's watchdog reads the mirrored heartbeat; the pilot's must be new.
        root.mkdir(parents=True, exist_ok=reuse_output)

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


MAC_ONLY_HEARTBEAT_KEYS = (
    "mac_receipt_verified",
    "github_receipt_verified",
    "transfer_progress_unix",
    "r2_final_verified",
)
READY_STATUSES = frozenset({"awaiting_transfer", *SUCCESS_STATUSES})


class SupervisedStore(ObjectStore, Protocol):
    def objects(self) -> list[dict[str, Any]]: ...


def _no_ssh_pull(name: str) -> Path:
    raise RuntimeError("rehearsal mode never copies checkpoints over SSH")


class RehearsalSupervisor(Supervisor):
    """Supervise one rehearsal run: R2 verification instead of SSH checkpoint pulls.

    Teardown is approved (``shutdown_ready``) in exactly two cases:

    - the heartbeat is ``awaiting_transfer`` with ``r2_final``, the SSH copy of
      ``report.json`` hashes to ``r2_final``'s SHA-256, ``verify_final`` passes with the
      Mac's own R2 key, and the report and a summary are archived privately with
      read-back; or
    - the worker reported ``stopped`` or ``failed`` with no ``r2_final`` for the failure
      grace period, and its diagnostics are archived.

    A ``running`` heartbeat never approves teardown, whatever stage the run is in. A
    mismatching or missing R2 object never approves it either; the watchdog's idle,
    wall and cost limits still bound the Pod.
    """

    def __init__(
        self,
        root: Path,
        *,
        run: str,
        pod_id: str,
        fetch: Callable[[str], dict[str, Any]],
        fetch_report: Callable[[], bytes],
        store: SupervisedStore,
        archive: Callable[[Path], dict[str, Any]],
        clock: Callable[[], float] = time.time,
        failure_grace_seconds: float = 300.0,
    ) -> None:
        run_prefix(run, pod_id)
        if not 0 < failure_grace_seconds <= IDLE_SECONDS:
            raise ValueError("the failure grace must be positive and within the idle limit")
        super().__init__(
            root, fetch=fetch, pull=_no_ssh_pull, archive=archive, clock=clock, reuse_output=True
        )
        self.run, self.pod_id, self.store = run, pod_id, store
        self.fetch_report = fetch_report
        self.failure_grace_seconds = failure_grace_seconds
        self.r2_verified: dict[str, Any] | None = None
        self.failure_seen_unix: float | None = None
        self.summary: dict[str, Any] | None = None

    def refresh_heartbeat(self) -> dict[str, Any]:
        heartbeat = self.fetch("heartbeat.json")
        for key in MAC_ONLY_HEARTBEAT_KEYS:
            heartbeat.pop(key, None)  # Only the Mac's own checks may set these.
        if self.r2_verified is not None:
            # The durable copy is verified in R2 even if the GitHub archive still lags.
            heartbeat.update(r2_final_verified=True, mac_receipt_verified=True)
        if self.shutdown_ready:
            heartbeat.update(
                status="shutdown_ready", mac_receipt_verified=True, github_receipt_verified=True
            )
        atomic_json(self.root / "heartbeat.json", heartbeat)
        return heartbeat

    def cycle(self) -> None:
        heartbeat = self.refresh_heartbeat()
        raw = self.fetch_report()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError("remote report must be an object")
        atomic_json(self.root / "remote-report.json", report)
        if self.shutdown_ready:
            return
        status = heartbeat.get("status")
        r2_final = heartbeat.get("r2_final")
        if status in READY_STATUSES:
            self.failure_seen_unix = None
            if r2_final is not None:
                self.finish(r2_final, raw, statuses=SUCCESS_STATUSES)
            return  # Until the worker publishes r2_final there is nothing to verify.
        if status in FAILURE_STATUSES:
            if r2_final is not None:
                self.finish(r2_final, raw, statuses=(str(status),))
                return
            now = self.clock()
            if self.failure_seen_unix is None:
                self.failure_seen_unix = now
            if now - self.failure_seen_unix >= self.failure_grace_seconds:
                self.diagnose(raw, report)
            return
        # "running" (every stage, and run A's boundary between them) or unknown: wait.
        self.failure_seen_unix = None

    def finish(self, r2_final: object, raw: bytes, *, statuses: Sequence[str]) -> None:
        """Verify the final state in R2, archive the report and summary, then approve."""
        final = self.root / "final"
        if self.r2_verified is None:
            checked, _ = check_r2_final(r2_final, run=self.run, pod_id=self.pod_id)
            if hashlib.sha256(raw).hexdigest() != checked["report_sha256"]:
                raise ValueError("the SSH copy of report.json differs from the final R2 report")
            verification = verify_final(
                self.store,
                checked,
                run=self.run,
                pod_id=self.pod_id,
                destination=final,
                statuses=statuses,
            )
            if (final / "report.json").read_bytes() != raw:
                raise ValueError("the R2 report.json differs from the SSH copy")
            summary = run_summary(
                json.loads(raw),
                run=self.run,
                pod_id=self.pod_id,
                verification=verification,
                source="r2-final-report",
            )
            atomic_json(final / "r2-verification.json", verification)
            atomic_json(final / "summary.json", summary)
            self.r2_verified, self.summary, self.latest = verification, summary, final
        self.archive_verified(final)  # Retried every cycle until it succeeds.
        self.shutdown_ready = True
        self.refresh_heartbeat()

    def diagnose(self, raw: bytes, report: dict[str, Any]) -> None:
        """Archive a failed run's own report and what reached R2, then approve teardown."""
        directory = self.root / "diagnostics"
        directory.mkdir(exist_ok=True)
        (directory / "report.json").write_bytes(raw)  # Bounded by the SSH fetch.
        summary = run_summary(
            report,
            run=self.run,
            pod_id=self.pod_id,
            verification=None,
            source="ssh-terminal-report",
        )
        prefix = run_prefix(self.run, self.pod_id)
        try:
            summary["r2_labels_present"] = sorted(
                {
                    str(item["key"])[len(prefix) :].split("/", 1)[0]
                    for item in self.store.objects()
                    if str(item["key"]).startswith(prefix)
                }
            )
        except Exception as error:  # recorded; diagnostics are archived regardless
            summary["r2_listing_error"] = type(error).__name__
        atomic_json(directory / "summary.json", summary)
        self.summary, self.latest = summary, directory
        self.archive_verified(directory)
        self.shutdown_ready = True
        self.refresh_heartbeat()


def build_parser() -> argparse.ArgumentParser:
    """The supervisor command line; the runbooks' commands are tested against it."""
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
    parser.add_argument(
        "--mode",
        choices=("pilot", "rehearsal"),
        default="pilot",
        help="rehearsal: verify the final state in R2 instead of pulling checkpoints",
    )
    parser.add_argument("--run", choices=sorted(RUNS), help="the rehearsal run (A or B)")
    parser.add_argument(
        "--failure-grace-seconds",
        type=float,
        default=300.0,
        help="rehearsal: wait this long for r2_final after a stopped/failed heartbeat",
    )
    return parser


def _supervise(
    supervisor: Supervisor,
    *,
    client: RunpodClient,
    identity: PodIdentity,
    policy: Policy,
    output: Path,
    poll_seconds: float,
    cycle_seconds: float,
) -> tuple[list[Outcome], bool]:
    """Run the API guard, the heartbeat mirror and the supervisor cycle until the Pod ends."""
    done = threading.Event()
    outcomes: list[Outcome] = []

    state_path = output / "watchdog-receipt.json.observed.json"
    observed = load_observed(state_path, identity)

    def guard() -> None:
        nonlocal observed
        while not done.is_set():
            try:
                outcome = check_once(
                    client, identity, policy, output / "heartbeat.json", observed=observed
                )
                if outcome.state in {"watching", "terminated"} and not observed:
                    observed = True
                    save_observed(state_path, identity, outcome.checked_at_unix)
            except Exception as error:
                outcome = Outcome(
                    "unknown", "watchdog_error", identity.pod_id, time.time(), type(error).__name__
                )
            atomic_receipt(output / "watchdog-receipt.json", outcome)
            if outcome.state in {"terminated", "absent", "identity_mismatch"}:
                outcomes.append(outcome)
                done.set()
                return
            done.wait(poll_seconds)

    watcher = threading.Thread(target=guard, name="independent-runpod-guard", daemon=False)

    def mirror() -> None:
        while not done.is_set():
            try:
                supervisor.refresh_heartbeat()
            except Exception as error:
                atomic_json(output / "heartbeat-error.json", {"error_type": type(error).__name__})
            done.wait(cycle_seconds)

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
                    output / "controller-error.json",
                    {"error_type": type(error).__name__, "detail": str(error)[:500]},
                )
                # Preserve the last authenticated heartbeat; never fabricate GPU
                # activity or receipt flags when SSH/transfer/archival fails.
            done.wait(cycle_seconds)
    except KeyboardInterrupt:
        interrupted = True  # The API guard keeps the Pod bounded; stop supervising cleanly.
        done.set()
    finally:
        watcher.join()
        heartbeat_watcher.join()
    return outcomes, interrupted


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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
    if args.mode == "rehearsal":
        return _rehearsal_main(args, parser, identity, policy, token, ssh)
    if args.run is not None:
        parser.error("--run applies only to --mode rehearsal")
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
    outcomes, interrupted = _supervise(
        supervisor,
        client=client,
        identity=identity,
        policy=policy,
        output=args.output,
        poll_seconds=args.poll_seconds,
        cycle_seconds=args.cycle_seconds,
    )
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


def _rehearsal_main(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    identity: PodIdentity,
    policy: Policy,
    token: str,
    ssh: list[str],
) -> int:
    """Supervise one rehearsal Pod; nothing is pulled over SSH except bounded status files."""
    if args.run is None:
        parser.error("--mode rehearsal needs --run A or --run B")
    if policy.mode != "rehearsal":
        parser.error('a rehearsal supervisor needs a policy with "mode": "rehearsal"')
    if not 0 < args.failure_grace_seconds <= IDLE_SECONDS:
        parser.error(f"--failure-grace-seconds must be within the {IDLE_SECONDS}-second idle limit")
    try:
        # The Mac's own key: it checks what the Pod's key wrote.
        store = Store(r2_credentials(os.environ, key_prefix=MAC_KEY_PREFIX))
    except ValueError as error:
        parser.error(f"Mac R2 credentials: {error}")
    root = f"rehearsal-runs/{args.run}"

    def archive(bundle: Path) -> dict[str, Any]:
        return archive_bundle(
            bundle,
            args.repository,
            run_id=identity.pod_id,
            files=("report.json", "summary.json"),
            root=root,
            message="Save rehearsal report and summary",
        )

    supervisor = RehearsalSupervisor(
        args.output,
        run=args.run,
        pod_id=identity.pod_id,
        fetch=lambda name: fetch_status(ssh, args.remote_root, name),
        fetch_report=lambda: fetch_status_bytes(
            ssh, args.remote_root, "report.json", max_bytes=REPORT_MAX_BYTES
        ),
        store=store,
        archive=archive,
        failure_grace_seconds=args.failure_grace_seconds,
    )
    # Mirror one heartbeat before the API guard starts: without it the guard's first check
    # would read no heartbeat and apply the policy's (long past) pre-heartbeat idle baseline.
    for _ in range(6):
        try:
            supervisor.refresh_heartbeat()
            break
        except Exception as error:
            atomic_json(args.output / "heartbeat-error.json", {"error_type": type(error).__name__})
            time.sleep(args.cycle_seconds)
    else:
        print(
            "rehearsal supervisor: no heartbeat over SSH; not started (the run's watchdog "
            "still guards the Pod)",
            file=sys.stderr,
        )
        return 2
    client = RunpodClient(token, UrllibTransport())
    outcomes, interrupted = _supervise(
        supervisor,
        client=client,
        identity=identity,
        policy=policy,
        output=args.output,
        poll_seconds=args.poll_seconds,
        cycle_seconds=args.cycle_seconds,
    )
    closing = args.output / "closing"
    closing.mkdir(exist_ok=True)
    receipt = args.output / "watchdog-receipt.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "rehearsal-supervisor-closing",
        "run": args.run,
        "pod_id": identity.pod_id,
        "shutdown_ready": supervisor.shutdown_ready,
        "shutdown_receipt": json.loads(receipt.read_text()) if receipt.exists() else None,
        "summary": supervisor.summary,
        "resource_audit": "Pod only; R2 objects, network volumes and billing need separate audit",
        "supervisor_interrupted": interrupted,
    }
    atomic_json(closing / "summary.json", record)  # Always written locally first.
    archive_failed = False
    try:
        archive_bundle(
            closing,
            args.repository,
            run_id=identity.pod_id,
            files=("summary.json",),
            root=root,
            message="Save rehearsal closing record",
        )
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
