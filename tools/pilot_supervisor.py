"""Mac-side heartbeat, verified transfer and private record/log archival.

An independent thread checks Runpod while SSH transfers run. This manages only
an explicitly identified existing Pod and cannot guarantee API availability.

``--mode rehearsal`` supervises one of the two rehearsal Pods (D-35) instead. The worker
uploads its checkpoints to R2 itself, and once its heartbeat says ``awaiting_transfer``
with ``r2_final``, the supervisor checks the final milestone and ``report.json`` in R2 with
the Mac's own key, archives the small records privately, and only then reports
``shutdown_ready``. A ``running`` or ``uploading`` heartbeat, including run A's boundary
between its two stages, never leads to teardown. While the worker uploads, new objects
under its R2 prefix (listed with the Mac's key, through a store bounded to 5 s per socket
operation) are the Mac-observed progress that lets the watchdog wait out the transfer
window. The one checkpoint copy over SSH is the fallback for a stage-final milestone the
worker could not verify in R2: the supervisor pulls it with the pilot's verified pull,
checks its SHA-256, archives the records and only then approves teardown; if that fails
it never approves, and the watchdog's limits end the Pod.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import torch

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools import posttrain_ops
from tools.pilot_transfer import Artifact, _ssh_command, child_environment, pull_artifacts
from tools.r2_store import SHA256_HEX, Store, bounded_store
from tools.rehearsal_binding import RUNS, r2_credentials
from tools.rehearsal_ops import (
    FAILURE_STATUSES,
    IDLE_SECONDS,
    IN_PROGRESS_STATUSES,
    MAC_KEY_PREFIX,
    REPORT_MAX_BYTES,
    SUCCESS_STATUSES,
    UPLOADING_STATUS,
    ObjectStore,
    check_r2_final,
    head_sha256,
    run_prefix,
    run_stages,
    run_summary,
    verify_final,
)
from tools.rehearsal_ops import MAX_TRANSFER_SECONDS as REHEARSAL_TRANSFER_SECONDS
from tools.runpod_watchdog import (
    API_KEY_ENV,
    Outcome,
    PodIdentity,
    Policy,
    RunpodClient,
    UrllibTransport,
    atomic_receipt,
    check_once,
    foreign_heartbeat,
    load_observed,
    parse_timestamp,
    save_observed,
)
from tools.runpod_watchdog import MAX_TRANSFER_SECONDS as WATCHDOG_TRANSFER_CEILING_SECONDS

_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_BUNDLE = re.compile(r"exports/lr-[0-9]+(?:-continued)?-step[0-9]+-[0-9]+\Z")
_ARCHIVE_ROOT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*\Z")
_ARCHIVE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
STATUS_FILE_LIMITS = {
    "heartbeat.json": 65_536,
    "report.json": 4_194_304,
    "finalization.json": 1_000_000,  # The rehearsal worker's drain and check record.
}
"""The status files the fixed-code SSH channel may stream, each with its byte bound."""


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
    if name not in STATUS_FILE_LIMITS:
        raise ValueError("unsupported status file")
    limit = STATUS_FILE_LIMITS[name]
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
PROGRESS_STATUSES = frozenset({UPLOADING_STATUS, "awaiting_transfer"})
"""Heartbeat statuses during which new R2 objects count as transfer progress."""
DEFAULT_FAILURE_GRACE_SECONDS = 60.0
"""The worker writes ``stopped`` or ``failed`` only after its own drain and report upload
(it reads ``uploading`` until then), so the grace only covers a status change in flight;
the idle limit counts from the last GPU work, before that drain."""
RECOVERY_FAILURES = frozenset({"r2_milestones_unverified", "r2_finalization"})
"""Worker failures after which the Pod may hold the only copy of a stage-final milestone.
They never lead to teardown on the grace timer: every milestone must first be verified in
R2 with the Mac's key or pulled over SSH and checked."""
MILESTONE_ERROR_FILE = "MILESTONES-NOT-SECURED.json"
"""Written to the supervisor directory when an unverified milestone could not be pulled."""
RECOVERED_ARTIFACTS = (Artifact("checkpoint", "checkpoint.pt"), Artifact("record", "record.json"))
"""What the SSH fallback pulls from ``exports/<label>/``: the weights and the label's record."""
EXPORT_PATH_KEYS = ("path", "export_path", "local_path", "export")
"""Keys under which ``finalization.json`` may give an unverified milestone's export path."""
LISTED_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
"""A file name the worker lists for an unverified milestone: one plain path component."""
MAC_CHECK_TIMEOUT_SECONDS = 5.0
"""The socket timeout of the Mac's own R2 listings and HEAD checks (``quick_store``). A HEAD
is one request, but urllib tries every address the endpoint resolves to with the whole
timeout, and name resolution has no timeout at all. The worker hands over 480 s after its
last GPU work and the pull must stamp progress before the 600 s idle limit, so an
unreachable R2 delays the pull by up to addresses x 5 s for each milestone checked (each
check stops at its first failure and is not repeated while a pull can run): about 40 s for
run A's two milestones with four addresses, plus name resolution."""
CHECKPOINT_DIGEST_KEYS = ("checkpoint_sha256", "sha256")
"""Keys under which an unverified milestone gives its checkpoint's SHA-256: the worker's
``checkpoint_sha256`` (``rehearsal_run.Worker.export_listing``), else an older row's ``sha256``."""

Recover = Callable[[str, Path, float], Path]
"""Pull ``exports/<label>`` (relative to the worker's root) into a destination directory
within a timeout in seconds, verified against the Pod's own manifest; return the bundle."""


class SupervisedStore(ObjectStore, Protocol):
    def objects(self) -> list[dict[str, Any]]: ...


def _no_ssh_pull(name: str) -> Path:
    raise RuntimeError(
        "rehearsal mode never copies checkpoints over SSH, except an unverified milestone"
    )


def _milestone_stage(label: object, *, run: str) -> str:
    """The stage whose milestone ``label`` is (``<stage>-final`` of this run), or raise."""
    stages = [stage for stage in run_stages(run) if label == f"{stage}-final"]
    if len(stages) != 1:
        raise ValueError(f"{label!r} is not a stage-final milestone of run {run}")
    return stages[0]


def _export_label(path: object) -> str:
    """The label of a worker export path: ``[<root>/]exports/<label>[/checkpoint.pt]``."""
    if not isinstance(path, str):
        raise ValueError("an export path must be a string")
    parts = PurePosixPath(path).parts
    if parts and parts[-1] == "checkpoint.pt":
        parts = parts[:-1]
    if len(parts) < 2 or parts[-2] != "exports":
        raise ValueError(f"{path!r} is not a worker export path")
    return parts[-1]


def _listed_digest(row: dict[Any, Any], label: str) -> str:
    """The checkpoint SHA-256 of one unverified milestone; every copy it gives must agree."""
    given = [row[key] for key in CHECKPOINT_DIGEST_KEYS if key in row]
    if not given:
        raise ValueError(f"unverified milestone {label} carries no checkpoint SHA-256")
    files = row.get("files", [])
    if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
        raise ValueError(f"unverified milestone {label} lists its files malformed")
    given += [item.get("sha256") for item in files if item.get("name") == "checkpoint.pt"]
    digest = given[0]
    if any(value != digest for value in given):
        raise ValueError(f"unverified milestone {label} gives two checkpoint SHA-256s")
    if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
        raise ValueError(f"unverified milestone {label} carries no checkpoint SHA-256")
    return digest


def unverified_milestones(finalization: object, *, run: str) -> dict[str, str]:
    """Each milestone ``finalization.json`` lists as unverified in R2: label to SHA-256.

    The worker lists them under ``unverified_milestones``, as a list of objects or a map
    from label to object, in ``rehearsal_run.Worker.export_listing``'s shape: the
    ``label``, its ``export`` (``exports/<label>``) and ``export_path``, the checkpoint's
    ``checkpoint_sha256`` and size, and every file under ``files``. An older row gives
    ``sha256`` and ``path`` instead. Every identifier given must name the same stage-final
    milestone of this run, and every checkpoint SHA-256 given (``checkpoint_sha256``,
    ``sha256`` and the ``files`` row named ``checkpoint.pt``) must be the same; the SSH
    pull always reads ``exports/<label>``.
    """
    return {label: digest for label, digest, _ in _listed_rows(finalization, run=run)}


def listed_files(finalization: object, *, run: str) -> dict[str, list[tuple[str, str]]]:
    """Each unverified milestone's files as the worker lists them: (name, SHA-256).

    An older row without ``files`` gives its checkpoint alone. The checkpoint is always
    among them, with the milestone's checkpoint SHA-256.
    """
    found: dict[str, list[tuple[str, str]]] = {}
    for label, digest, row in _listed_rows(finalization, run=run):
        files = [(str(item.get("name")), item.get("sha256")) for item in row.get("files", [])]
        files = files or [("checkpoint.pt", digest)]
        for name, sha256 in files:
            if not LISTED_FILE.fullmatch(name) or not isinstance(sha256, str):
                raise ValueError(f"unverified milestone {label} lists a malformed file")
            if not SHA256_HEX.fullmatch(sha256):
                raise ValueError(f"unverified milestone {label} lists a malformed file")
        if ("checkpoint.pt", digest) not in files or len({name for name, _ in files}) != len(files):
            raise ValueError(f"unverified milestone {label} lists its files malformed")
        found[label] = [(name, str(sha256)) for name, sha256 in files]
    return found


def _listed_rows(finalization: object, *, run: str) -> list[tuple[str, str, dict[Any, Any]]]:
    """(label, checkpoint SHA-256, row) for each listed milestone, every check applied."""
    if not isinstance(finalization, dict):
        raise ValueError("finalization.json must be an object")
    raw = finalization.get("unverified_milestones")
    if raw is None:
        return []
    rows: list[object]
    if isinstance(raw, dict):
        rows = [
            {"label": label, **row} if isinstance(row, dict) else row for label, row in raw.items()
        ]
    elif isinstance(raw, list):
        rows = list(raw)
    else:
        raise ValueError("unverified_milestones must be a list or an object")
    found: dict[str, str] = {}
    listed: list[tuple[str, str, dict[Any, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each unverified milestone must be an object")
        names = {row["label"]} if "label" in row else set()
        names |= {_export_label(row[key]) for key in EXPORT_PATH_KEYS if key in row}
        if len(names) != 1:
            raise ValueError(f"an unverified milestone names no single label: {sorted(names)}")
        (label,) = names
        _milestone_stage(label, run=run)
        digest = _listed_digest(row, label)
        if found.get(label, digest) != digest:
            raise ValueError(f"unverified milestone {label} is listed with two SHA-256s")
        if label not in found:
            listed.append((label, digest, row))
        found[label] = digest
    return listed


def recorded_milestones(report: dict[str, Any], *, run: str) -> dict[str, str]:
    """Every stage-final milestone the worker's report records: label to checkpoint SHA-256."""
    found: dict[str, str] = {}
    for entry in report.get("stages") or []:
        if not isinstance(entry, dict) or entry.get("final_checkpoint_sha256") is None:
            continue
        label = f"{entry.get('name')}-final"
        _milestone_stage(label, run=run)
        digest = entry["final_checkpoint_sha256"]
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            raise ValueError(f"the report's {label} SHA-256 is malformed")
        found[label] = digest
    return found


def verify_recovered(bundle: Path, *, label: str, sha256: str) -> dict[str, Any]:
    """Check a pulled milestone against the expected SHA-256 and its transfer receipt."""
    receipt = json.loads((bundle / "transfer-receipt.json").read_text())
    if not isinstance(receipt, dict) or receipt.get("status") != "verified":
        raise ValueError(f"the SSH pull of {label} is unverified")
    rows = {row.get("name"): row for row in receipt.get("artifacts", []) if isinstance(row, dict)}
    checkpoint = bundle / "checkpoint.pt"
    if set(rows) != {item.name for item in RECOVERED_ARTIFACTS} or checkpoint.is_symlink():
        raise ValueError(f"the SSH pull of {label} does not hold exactly the milestone files")
    size = checkpoint.stat().st_size
    if rows["checkpoint.pt"].get("sha256") != sha256 or file_sha256(checkpoint) != sha256:
        raise ValueError(f"the pulled {label} checkpoint differs from its recorded SHA-256")
    if rows["checkpoint.pt"].get("size_bytes") != size:
        raise ValueError(f"the pulled {label} checkpoint differs from its transfer receipt")
    return {
        "label": label,
        "remote_export": f"exports/{label}",
        "path": str(checkpoint),
        "sha256": sha256,
        "size_bytes": size,
    }


class RehearsalSupervisor(Supervisor):
    """Supervise one rehearsal run: R2 verification, and SSH only for unverified milestones.

    Teardown is approved (``shutdown_ready``) in exactly three cases:

    - the heartbeat is ``awaiting_transfer`` (or ``stopped``/``failed`` with no milestone
      at risk) with ``r2_final``, the SSH copy of ``report.json`` hashes to ``r2_final``'s
      SHA-256, ``verify_final`` passes with the Mac's own R2 key, and the report and a
      summary are archived privately with read-back;
    - the worker reported ``stopped`` or ``failed`` with no ``r2_final``, no milestone is
      at risk, the failure grace period passed, and its diagnostics are archived; or
    - a milestone is at risk (the worker failed ``r2_milestones_unverified`` or
      ``r2_finalization``, or listed or flagged a milestone it could not verify in R2),
      and every one of them is now verified in R2 with the Mac's key or was pulled over
      SSH with the pilot's verified pull, its SHA-256 checked, and the records archived.

    A ``running`` or ``uploading`` heartbeat never approves teardown, whatever stage the
    run is in. A mismatching or missing R2 object never approves it either, and neither
    does a failed SSH pull: that writes ``MILESTONES-NOT-SECURED.json`` and
    ``controller-error.json`` naming the files, and the watchdog's idle, transfer, wall
    and cost limits still bound the Pod. A pull is bounded by the transfer window: it
    must end ``transfer_window_seconds`` after the heartbeat's last GPU work, when the
    watchdog stops waiting.

    While the heartbeat reads ``uploading`` or ``awaiting_transfer``, the mirror lists
    the run's R2 prefix with the Mac's key. The first listing is a baseline; any object
    (key and size) not seen before stamps ``transfer_progress_unix``, the Mac-observed
    progress the watchdog requires before it lets an idle GPU wait out the policy's
    ``max_transfer_seconds``. Bytes arriving from an SSH pull stamp it too. The Pod can
    never stamp it itself.

    Every R2 call a deadline depends on goes through ``quick_store`` (``_rehearsal_main``
    builds it with ``r2_store.bounded_store``: ``MAC_CHECK_TIMEOUT_SECONDS``, 5 s, per
    socket operation): the mirror's listing, the at-risk HEAD checks and the diagnostics
    listing. A HEAD, like each page of a listing, is a single request that is never
    retried. An R2 endpoint that stalls then holds the mirror or the start of a pull for up
    to 5 s per address it resolves to on each request, instead of 300 s, plus name
    resolution, which no socket timeout bounds; its error is recorded and stamps nothing.
    While a pull can run, a milestone is checked at most once: one already pulled, or one
    whose check failed, goes straight to the pull on a retry. One whose check timed out or
    could not connect is checked once more after the transfer window has closed, only so
    the record says what R2 holds. ``verify_final`` and its downloads use ``store``.

    A heartbeat naming another Pod, or updated before this Pod's creation, is ignored
    (``runpod_watchdog.foreign_heartbeat``): it is never mirrored and never acted on.
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
        failure_grace_seconds: float = DEFAULT_FAILURE_GRACE_SECONDS,
        fetch_finalization: Callable[[], bytes] | None = None,
        recover: Recover | None = None,
        transfer_window_seconds: float = REHEARSAL_TRANSFER_SECONDS,
        pod_created_unix: float | None = None,
        quick_store: SupervisedStore | None = None,
    ) -> None:
        self.layout_prefix(run, pod_id)
        if not 0 < failure_grace_seconds <= IDLE_SECONDS:
            raise ValueError("the failure grace must be positive and within the idle limit")
        if not 0 <= transfer_window_seconds <= WATCHDOG_TRANSFER_CEILING_SECONDS:
            raise ValueError("the transfer window must be within the watchdog's ceiling")
        super().__init__(
            root, fetch=fetch, pull=_no_ssh_pull, archive=archive, clock=clock, reuse_output=True
        )
        self.run, self.pod_id, self.store = run, pod_id, store
        self.quick_store = store if quick_store is None else quick_store
        self.prefix = self.layout_prefix(run, pod_id)
        self.fetch_report = fetch_report
        self.fetch_finalization = fetch_finalization
        self.recover = recover
        self.transfer_window_seconds = transfer_window_seconds
        self.pod_created_unix = pod_created_unix
        self.failure_grace_seconds = failure_grace_seconds
        self.r2_verified: dict[str, Any] | None = None
        self.failure_seen_unix: float | None = None
        self.summary: dict[str, Any] | None = None
        self.r2_seen: set[tuple[str, int]] | None = None
        self.progress_lock = threading.Lock()
        # Milestones at risk: pulled over SSH and checked, or verified in R2 by the Mac.
        # Keyed by Pod, so a reused controller directory never mixes two Pods' copies.
        self.recovery_root = root / "recovered" / pod_id
        self.recovered: dict[str, dict[str, Any]] = {}
        self.r2_secured: dict[str, dict[str, Any]] = {}
        # Milestones whose R2 check by the Mac failed: R2 answered missing or different
        # (never checked again), or did not answer (checked once more after the window), so
        # a retried pull starts at once instead of after the HEADs.
        self.r2_unconfirmed: set[str] = set()
        self.r2_unreachable: set[str] = set()
        self.r2_rechecked: set[str] = set()
        self.window_closed = False
        """Set once no SSH pull can start: the watchdog's transfer window is over."""
        self.recovering = False
        self.recovery_staged = 0
        self.milestone_error: dict[str, Any] | None = None

    @staticmethod
    def layout_prefix(run: str, pod_id: str) -> str:
        """The run's R2 prefix: ``rehearsal-runs/<run>/<pod id>/`` (a post-training run has
        its own layout, ``PosttrainSupervisor``)."""
        return run_prefix(run, pod_id)

    def observe_r2_progress(self) -> None:
        """Stamp ``transfer_progress_unix`` when the run's prefix holds a new object.

        The listing uses the bounded ``quick_store``, so a stalled R2 cannot hold the
        mirror thread for long. A listing error, a timeout included, is recorded and
        stamps nothing: without fresh progress the watchdog's idle limit applies again.
        """
        try:
            listed = {
                (str(item["key"]), int(item["bytes"]))
                for item in self.quick_store.objects()
                if str(item["key"]).startswith(self.prefix)
            }
        except Exception as error:  # recorded; the mirror keeps running
            atomic_json(self.root / "r2-progress-error.json", {"error_type": type(error).__name__})
            return
        with self.progress_lock:
            if self.r2_seen is not None and listed - self.r2_seen:
                self.transfer_progress_unix = self.clock()
            self.r2_seen = listed if self.r2_seen is None else self.r2_seen | listed

    def observe_recovery_progress(self) -> None:
        """Stamp ``transfer_progress_unix`` while an SSH pull's staged bytes grow on the Mac."""
        total = 0
        if self.recovery_root.is_dir():
            for staging in self.recovery_root.glob(".*.staging-*"):
                if staging.is_dir() and not staging.is_symlink():
                    total += sum(
                        item.stat().st_size
                        for item in staging.iterdir()
                        if item.is_file() and not item.is_symlink()
                    )
        with self.progress_lock:
            if total > self.recovery_staged:
                self.transfer_progress_unix = self.clock()
            self.recovery_staged = total

    def refresh_heartbeat(self, *, observe: bool = True) -> dict[str, Any]:
        """Mirror the Pod's heartbeat, with the Mac's own progress stamps and flags.

        Only the mirror thread lists R2 (``observe``); ``cycle`` mirrors without listing,
        so a stalled listing never delays its reading of the worker's final heartbeat.
        """
        heartbeat = self.fetch("heartbeat.json")
        foreign = foreign_heartbeat(
            heartbeat, pod_id=self.pod_id, created_at_unix=self.pod_created_unix
        )
        if foreign is not None:
            # Another Pod's heartbeat is no evidence about this one: never mirror it.
            raise ValueError(f"the fetched heartbeat is ignored: {foreign}")
        for key in MAC_ONLY_HEARTBEAT_KEYS:
            heartbeat.pop(key, None)  # Only the Mac's own checks may set these.
        if observe and heartbeat.get("status") in PROGRESS_STATUSES and not self.shutdown_ready:
            self.observe_r2_progress()
        if self.recovering:
            self.observe_recovery_progress()
        if self.transfer_progress_unix is not None:
            heartbeat["transfer_progress_unix"] = self.transfer_progress_unix
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
        heartbeat = self.refresh_heartbeat(observe=False)
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
            # Never on the grace timer while the Pod may hold a milestone's only copy, even
            # when the worker published r2_final for its last verified milestone.
            at_risk = self.milestones_at_risk(report)
            if at_risk:
                self.secure(raw, report, heartbeat, at_risk)
                return
            if r2_final is not None:
                self.finish(r2_final, raw, statuses=(str(status),))
                return
            if at_risk is not None:  # Every milestone verified in R2 with the Mac's key.
                self.secure(raw, report, heartbeat, at_risk)
                return
            now = self.clock()
            if self.failure_seen_unix is None:
                self.failure_seen_unix = now
            if now - self.failure_seen_unix >= self.failure_grace_seconds:
                self.diagnose(raw, report)
            return
        if status in IN_PROGRESS_STATUSES:
            # "running" (every stage, and run A's boundary between them) or "uploading"
            # (the worker's last drain to R2): never teardown, never verification yet.
            self.failure_seen_unix = None
            return
        # An unknown status: wait; the watchdog treats a malformed heartbeat on its own.
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

    def read_finalization(self) -> dict[str, Any]:
        """The worker's ``finalization.json`` over the bounded SSH channel, kept locally."""
        if self.fetch_finalization is None:
            raise RuntimeError("no finalization.json channel is configured")
        value = json.loads(self.fetch_finalization())
        if not isinstance(value, dict):
            raise ValueError("finalization.json must be an object")
        atomic_json(self.root / "remote-finalization.json", value)
        return value

    def milestones_at_risk(self, report: dict[str, Any]) -> dict[str, str] | None:
        """Milestones a stopped or failed run may hold only on the Pod (label to SHA-256).

        ``None`` means none is at risk and the ordinary grace applies. Each milestone
        ``finalization.json`` lists is first checked in R2 with the Mac's key, every file
        it lists by HEAD through the bounded ``quick_store``, stopping at the first that is
        missing, differs or times out: the worker may only have run out of time, and a
        milestone the Mac finds whole needs no pull. After ``r2_milestones_unverified`` or
        ``r2_finalization`` the answer is never ``None``: ``finalization.json`` must be
        readable (an error is raised and retried each cycle), and every milestone the
        report records that the worker did not list is checked in R2 the same way, by its
        checkpoint; any that fails or times out is at risk too. After any other failure,
        the listed milestones the Mac could not find and those the report flags as
        unverified are. A milestone already pulled, or one whose R2 check failed, stays at
        risk with no new HEAD while a pull can still run, so a pull retried after a dropped
        one starts within a cycle and no recheck spends the SSH reserve. A check that timed
        out or could not connect (``r2_unreachable``, unlike R2's own answer that the object
        is missing or different, ``r2_unconfirmed``) is made once more after the transfer
        window has closed (``window_closed``), only to keep the record accurate: a milestone
        R2 holds whole is then secured, and teardown approved.
        """
        recovery = report.get("failure") in RECOVERY_FAILURES
        try:
            finalization = self.read_finalization()
        except Exception:
            if recovery:
                raise  # The unverified list is needed; never approve without it.
            finalization = {}
        listed = unverified_milestones(finalization, run=self.run)
        recorded = recorded_milestones(report, run=self.run)
        for label, digest in listed.items():
            if recorded.get(label, digest) != digest:
                raise ValueError(f"finalization.json and report.json give {label} two SHA-256s")
        files = listed_files(finalization, run=self.run)
        settled = self.recovered.keys() | self.r2_unconfirmed  # Never checked in R2 again.
        if not self.window_closed:
            settled |= self.r2_unreachable  # While a pull can run, it gets the whole reserve.
        else:
            self.r2_unreachable -= self.r2_rechecked  # Each is checked once more, at most.
            settled |= self.r2_rechecked
            self.r2_rechecked |= self.r2_unreachable
        at_risk: dict[str, str] = {}
        for label, digest in listed.items():
            if label in self.r2_secured:
                continue
            if label in settled or not self.secured_in_r2(label, files[label]):
                at_risk[label] = digest
        r2 = report.get("r2")
        flags = r2.get("milestones_verified") if isinstance(r2, dict) else None
        if isinstance(flags, dict):
            for label, verified in flags.items():
                if verified is True or label in at_risk or label in self.r2_secured:
                    continue
                if label not in recorded:
                    raise ValueError(f"the report flags {label} unverified with no SHA-256")
                at_risk[label] = recorded[label]
        if not recovery:
            return at_risk or None
        for label, digest in recorded.items():
            if label in at_risk or label in self.r2_secured:
                continue
            if label in settled:
                at_risk[label] = digest
                continue
            try:
                row = head_sha256(
                    self.quick_store, f"{self.prefix}{label}/checkpoint.pt", expected=digest
                )
            except Exception as error:  # missing, mismatched or unreachable: secure it over SSH
                self.r2_check_failed(label, error)
                at_risk[label] = digest
            else:
                self.r2_secured[label] = row
        return at_risk

    def r2_check_failed(self, label: str, error: Exception) -> None:
        """Record a failed R2 check of ``label``: for good when R2 answered (``head_sha256``
        raises ``ValueError`` for a missing object, other bytes or no SHA-256), else as
        unreachable (a timeout, a refused connection, an HTTP error), checked once more
        after the transfer window has closed."""
        if isinstance(error, ValueError):
            self.r2_unconfirmed.add(label)
        else:
            self.r2_unreachable.add(label)

    def secured_in_r2(self, label: str, files: Sequence[tuple[str, str]]) -> bool:
        """True, and recorded in ``r2_secured``, when every listed file of ``label`` is in R2
        with its SHA-256; checked by HEAD with the bounded store, stopping at the first
        that is missing, differs or cannot be reached (recorded by ``r2_check_failed``)."""
        rows = []
        for name, digest in files:
            try:
                rows.append(
                    head_sha256(self.quick_store, f"{self.prefix}{label}/{name}", expected=digest)
                )
            except Exception as error:  # missing, mismatched or unreachable: secure it over SSH
                self.r2_check_failed(label, error)
                return False
        checkpoint = rows[[name for name, _ in files].index("checkpoint.pt")]
        self.r2_secured[label] = {**checkpoint, "files": rows}
        return True

    def secure(
        self,
        raw: bytes,
        report: dict[str, Any],
        heartbeat: dict[str, Any],
        at_risk: dict[str, str],
    ) -> None:
        """Pull every milestone at risk over SSH, check it, archive, and only then approve.

        The last stage's milestone goes first. Every pull ends with the transfer window,
        measured from the heartbeat's last GPU work as the watchdog measures it. A failure
        writes the prominent error, is retried next cycle while the window lasts, and
        never approves teardown.
        """
        stages = run_stages(self.run)
        order = sorted(
            at_risk, key=lambda label: -stages.index(_milestone_stage(label, run=self.run))
        )
        pending = [label for label in order if label not in self.recovered]
        activity = heartbeat.get("last_gpu_activity_unix")
        for label in pending:
            if self.recover is None:
                self.fail_milestones(at_risk, "no SSH pull is configured")
                return
            if (
                not isinstance(activity, (int, float))
                or isinstance(activity, bool)
                or not 0 < activity < math.inf
            ):
                self.fail_milestones(at_risk, "the heartbeat carries no last GPU activity")
                return
            left = float(activity) + self.transfer_window_seconds - self.clock()
            if left <= 0:
                # No pull can start now: the next cycle checks unreachable milestones in R2
                # once more, so the record says what R2 holds.
                self.window_closed = True
                self.fail_milestones(at_risk, "the watchdog's transfer window is over")
                return
            self.recovering = True
            try:
                bundle = self.recovery_root / label
                if not bundle.exists():
                    self.recovery_root.mkdir(parents=True, exist_ok=True)
                    bundle = self.recover(f"exports/{label}", self.recovery_root, left)
                try:
                    row = verify_recovered(bundle, label=label, sha256=at_risk[label])
                except Exception:
                    # Kept for inspection under another name; a retry pulls a fresh copy.
                    bundle.replace(bundle.with_name(f"{label}.rejected-{uuid.uuid4().hex[:8]}"))
                    raise
            except Exception as error:
                self.fail_milestones(at_risk, f"{label}: {type(error).__name__}: {error}")
                return
            finally:
                self.recovering = False
            self.recovered[label] = row
        self.diagnose(
            raw,
            report,
            recovery={
                "failure": report.get("failure"),
                "pulled_over_ssh": [self.recovered[label] for label in order],
                "verified_in_r2_by_mac": [
                    self.r2_secured[label] for label in sorted(self.r2_secured)
                ],
            },
        )

    def final_r2_check(self) -> dict[str, bool]:
        """Once supervision has ended, the Pod perhaps gone: check in R2 once more each
        milestone still at risk for which R2 never answered that it is missing or different:
        one whose check timed out or could not connect, or one never checked (the report
        flagged it, or no cycle reached it).

        The watchdog may end the Pod before any cycle runs after the transfer window, so
        this needs no SSH: the listed files and recorded SHA-256s come from the local copies
        of ``finalization.json`` and ``report.json``. A milestone R2 holds whole is recorded
        in ``r2_secured``, and ``MILESTONES-NOT-SECURED.json`` gains
        ``verified_in_r2_after_close`` (and ``resolved_after_close`` when nothing at risk
        is left), so the record never calls a checkpoint R2 holds unsecured. Returns label
        to whether R2 holds it.
        """
        at_risk_now = {row["label"] for row in (self.milestone_error or {}).get("files", [])}
        pending = sorted(
            (self.r2_unreachable | at_risk_now)
            - self.r2_unconfirmed
            - self.r2_rechecked
            - self.r2_secured.keys()
            - self.recovered.keys()
        )
        if not pending:
            return {}

        def local(name: str) -> dict[str, Any]:
            try:
                value = json.loads((self.root / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
            return value if isinstance(value, dict) else {}

        try:
            files = listed_files(local("remote-finalization.json"), run=self.run)
            recorded = recorded_milestones(local("remote-report.json"), run=self.run)
        except ValueError:
            return {}
        checked: dict[str, bool] = {}
        for label in pending:
            self.r2_rechecked.add(label)
            if label in files:
                checked[label] = self.secured_in_r2(label, files[label])
            elif label in recorded:
                key = f"{self.prefix}{label}/checkpoint.pt"
                try:
                    self.r2_secured[label] = head_sha256(
                        self.quick_store, key, expected=recorded[label]
                    )
                except Exception:  # Still not confirmed: the record keeps it at risk.
                    checked[label] = False
                else:
                    checked[label] = True
        if self.milestone_error is not None and any(checked.values()):
            at_risk = {row["label"] for row in self.milestone_error.get("files", [])}
            secured = self.r2_secured.keys() | self.recovered.keys()
            record = {
                **self.milestone_error,
                "verified_in_r2_after_close": [
                    {"label": label, **self.r2_secured[label]}
                    for label in sorted(checked)
                    if checked[label]
                ],
                "resolved_after_close": at_risk <= secured,
            }
            atomic_json(self.root / MILESTONE_ERROR_FILE, record)
            self.milestone_error = record
        return checked

    def fail_milestones(self, at_risk: dict[str, str], detail: str) -> None:
        """Record, prominently, that milestones are not secured; never approve teardown."""
        files = [
            {
                "label": label,
                "pod_path": f"exports/{label}/checkpoint.pt",
                "sha256": at_risk[label],
                "pulled": label in self.recovered,
            }
            for label in sorted(at_risk)
        ]
        record = {
            "error_type": "MilestonesNotSecured",
            "detail": (
                "these stage-final milestones are neither verified in R2 nor pulled to the "
                "Mac; teardown is NOT approved and the watchdog's limits will end the Pod: "
                + detail
            )[:2000],
            "files": files,
            "local_destination": str(self.recovery_root),
            "recorded_unix": self.clock(),
        }
        atomic_json(self.root / MILESTONE_ERROR_FILE, record)
        atomic_json(self.root / "controller-error.json", record)
        if self.milestone_error is None or self.milestone_error["detail"] != record["detail"]:
            names = ", ".join(
                f"exports/{label}/checkpoint.pt"
                for label in sorted(at_risk)
                if label not in self.recovered
            )
            print(
                f"\n*** run {self.run} Pod {self.pod_id}: MILESTONES NOT SECURED: {names}\n"
                f"*** {detail}\n*** see {self.root / MILESTONE_ERROR_FILE}\n",
                file=sys.stderr,
                flush=True,
            )
        self.milestone_error = record

    def diagnose(
        self, raw: bytes, report: dict[str, Any], *, recovery: dict[str, Any] | None = None
    ) -> None:
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
            # Bounded: a stalled listing must not hold teardown after a verified pull.
            summary["r2_labels_present"] = sorted(
                {
                    str(item["key"])[len(prefix) :].split("/", 1)[0]
                    for item in self.quick_store.objects()
                    if str(item["key"]).startswith(prefix)
                }
            )
        except Exception as error:  # recorded; diagnostics are archived regardless
            summary["r2_listing_error"] = type(error).__name__
        if recovery is not None:
            summary["milestone_recovery"] = recovery
        atomic_json(directory / "summary.json", summary)
        self.summary, self.latest = summary, directory
        self.archive_verified(directory)
        self.shutdown_ready = True
        self.milestone_error = None
        self.refresh_heartbeat()


WEIGHTS_LABEL_FILES = frozenset(item.name for item in RECOVERED_ARTIFACTS)
"""A post-training label the verified SSH pull can carry: weights (``checkpoint.pt``) and its
``record.json``, nothing else (``pilot_transfer`` allows no other file names)."""


def posttrain_unverified(finalization: object) -> dict[str, list[tuple[str, str]]]:
    """Each label ``finalization.json`` lists as unverified: its files as (name, SHA-256)."""
    if not isinstance(finalization, dict):
        raise ValueError("finalization.json must be an object")
    rows = finalization.get("unverified_milestones") or []
    if not isinstance(rows, list):
        raise ValueError("unverified_milestones must be a list")
    found: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("files"), list):
            raise ValueError("each unverified label must list its files")
        label = row.get("label")
        if not isinstance(label, str) or not LISTED_FILE.fullmatch(label) or label in found:
            raise ValueError(f"an unverified label has an unsafe or repeated name: {label!r}")
        files = []
        for item in row["files"]:
            name, sha256 = (
                (item.get("name"), item.get("sha256")) if isinstance(item, dict) else (None, None)
            )
            if not isinstance(name, str) or not LISTED_FILE.fullmatch(name):
                raise ValueError(f"unverified label {label} lists a malformed file")
            if not isinstance(sha256, str) or not SHA256_HEX.fullmatch(sha256):
                raise ValueError(f"unverified label {label} lists a malformed file")
            files.append((name, sha256))
        if not files or len({name for name, _ in files}) != len(files):
            raise ValueError(f"unverified label {label} lists its files malformed")
        found[label] = files
    return found


class PosttrainSupervisor(RehearsalSupervisor):
    """Supervise one post-training run (``tools.posttrain_run``): its R2 layout and final check.

    The machinery is the rehearsal supervisor's: the mirrored heartbeat with the Mac's own
    progress stamps, the watchdog guard, the failure grace and the archive. Teardown is
    approved when:

    - the heartbeat carries ``r2_final`` (``awaiting_transfer``, or ``stopped``/``failed``
      with every label verified), the SSH copy of ``report.json`` hashes to it, and
      ``posttrain_ops.verify_final`` finds the report and every artifact it lists in R2 with
      the Mac's own key; the report and a summary are then archived;
    - the run stopped or failed with no ``r2_final`` and no label at risk, after the grace;
    - labels ``finalization.json`` lists as unverified are each found whole in R2 by the
      Mac's HEAD checks, or, for the weights label (``checkpoint.pt`` and ``record.json``),
      pulled over SSH with the pilot's verified pull and checked. Any other unverified label
      is recorded in ``MILESTONES-NOT-SECURED.json`` for a rescue by hand, and teardown is not
      approved: the watchdog's limits end the Pod.
    """

    def __init__(self, root: Path, **options: Any) -> None:
        super().__init__(root, **options)
        self.at_risk_files: dict[str, list[tuple[str, str]]] = {}
        """Each label at risk and its files, from the last ``finalization.json`` read."""

    @staticmethod
    def layout_prefix(run: str, pod_id: str) -> str:
        return posttrain_ops.run_prefix(run, pod_id)

    def finish(self, r2_final: object, raw: bytes, *, statuses: Sequence[str]) -> None:
        final = self.root / "final"
        if self.r2_verified is None:
            checked = posttrain_ops.check_r2_final(r2_final, run=self.run, pod_id=self.pod_id)
            if hashlib.sha256(raw).hexdigest() != checked["report_sha256"]:
                raise ValueError("the SSH copy of report.json differs from the final R2 report")
            verification = posttrain_ops.verify_final(
                self.store,
                checked,
                run=self.run,
                pod_id=self.pod_id,
                destination=final,
                statuses=statuses,
            )
            if (final / "report.json").read_bytes() != raw:
                raise ValueError("the R2 report.json differs from the SSH copy")
            summary = posttrain_ops.run_summary(
                json.loads(raw),
                run=self.run,
                pod_id=self.pod_id,
                verification=verification,
                source="r2-final-report",
            )
            atomic_json(final / "r2-verification.json", verification)
            atomic_json(final / "summary.json", summary)
            self.r2_verified, self.summary, self.latest = verification, summary, final
        self.archive_verified(final)
        self.shutdown_ready = True
        self.refresh_heartbeat()

    def milestones_at_risk(self, report: dict[str, Any]) -> dict[str, str] | None:
        """Labels a stopped or failed run may hold only on the Pod: label to its first file's
        SHA-256 (the weights' for the weights label); ``None`` when none is at risk.

        Every label ``finalization.json`` lists is checked in R2 with the Mac's key, file by
        file, through the bounded ``quick_store``; one found whole is secured. After an R2
        failure (``r2_milestones_unverified``, ``r2_finalization``) the list must be readable.
        """
        recovery = report.get("failure") in RECOVERY_FAILURES
        try:
            finalization = self.read_finalization()
        except Exception:
            if recovery:
                raise
            finalization = {}
        listed = posttrain_unverified(finalization)
        self.at_risk_files = listed
        settled = self.recovered.keys() | self.r2_unconfirmed
        if not self.window_closed:
            settled |= self.r2_unreachable
        at_risk: dict[str, str] = {}
        for label, files in listed.items():
            if label in self.r2_secured:
                continue
            if label in settled or not self.secured_in_posttrain(label, files):
                weights = dict(files).get("checkpoint.pt")
                at_risk[label] = weights if weights is not None else files[0][1]
        if not at_risk and not recovery:
            return None
        return at_risk

    def secured_in_posttrain(self, label: str, files: Sequence[tuple[str, str]]) -> bool:
        rows = []
        for name, digest in files:
            try:
                rows.append(
                    head_sha256(self.quick_store, f"{self.prefix}{label}/{name}", expected=digest)
                )
            except Exception as error:  # missing, mismatched or unreachable
                self.r2_check_failed(label, error)
                return False
        self.r2_secured[label] = {"label": label, "files": rows}
        return True

    def secure(
        self,
        raw: bytes,
        report: dict[str, Any],
        heartbeat: dict[str, Any],
        at_risk: dict[str, str],
    ) -> None:
        """Pull the weights label over SSH; any other label at risk needs a rescue by hand."""
        files = self.at_risk_files
        unpullable = [
            label
            for label in at_risk
            if {name for name, _ in files.get(label, [])} != WEIGHTS_LABEL_FILES
        ]
        if unpullable:
            self.fail_milestones(
                at_risk,
                f"{', '.join(sorted(unpullable))}: not carried by the verified SSH pull; copy "
                "each listed file from the Pod's exports by hand and check its SHA-256",
            )
            return
        activity = heartbeat.get("last_gpu_activity_unix")
        for label in sorted(at_risk):
            if label in self.recovered:
                continue
            if self.recover is None:
                self.fail_milestones(at_risk, "no SSH pull is configured")
                return
            if (
                not isinstance(activity, (int, float))
                or isinstance(activity, bool)
                or not 0 < activity < math.inf
            ):
                self.fail_milestones(at_risk, "the heartbeat carries no last GPU activity")
                return
            left = float(activity) + self.transfer_window_seconds - self.clock()
            if left <= 0:
                self.window_closed = True
                self.fail_milestones(at_risk, "the watchdog's transfer window is over")
                return
            self.recovering = True
            try:
                bundle = self.recovery_root / label
                if not bundle.exists():
                    self.recovery_root.mkdir(parents=True, exist_ok=True)
                    bundle = self.recover(f"exports/{label}", self.recovery_root, left)
                row = verify_recovered(bundle, label=label, sha256=at_risk[label])
            except Exception as error:
                self.fail_milestones(at_risk, f"{label}: {type(error).__name__}: {error}")
                return
            finally:
                self.recovering = False
            self.recovered[label] = row
        self.diagnose(
            raw,
            report,
            recovery={
                "failure": report.get("failure"),
                "pulled_over_ssh": [self.recovered[label] for label in sorted(at_risk)],
                "verified_in_r2_by_mac": [
                    self.r2_secured[label] for label in sorted(self.r2_secured)
                ],
            },
        )

    def fail_milestones(self, at_risk: dict[str, str], detail: str) -> None:
        listed = self.at_risk_files
        record = {
            "error_type": "MilestonesNotSecured",
            "detail": (
                "these labels are neither verified in R2 nor pulled to the Mac; teardown is "
                "NOT approved and the watchdog's limits will end the Pod: " + detail
            )[:2000],
            "files": [
                {
                    "label": label,
                    "pod_paths": [f"exports/{label}/{name}" for name, _ in listed.get(label, [])],
                    "sha256": dict(listed.get(label, [])),
                    "pulled": label in self.recovered,
                }
                for label in sorted(at_risk)
            ],
            "local_destination": str(self.recovery_root),
            "recorded_unix": self.clock(),
        }
        atomic_json(self.root / MILESTONE_ERROR_FILE, record)
        atomic_json(self.root / "controller-error.json", record)
        if self.milestone_error is None or self.milestone_error["detail"] != record["detail"]:
            print(
                f"\n*** post-training run {self.run} Pod {self.pod_id}: LABELS NOT SECURED: "
                f"{', '.join(sorted(at_risk))}\n*** {detail}\n*** see "
                f"{self.root / MILESTONE_ERROR_FILE}\n",
                file=sys.stderr,
                flush=True,
            )
        self.milestone_error = record

    def final_r2_check(self) -> dict[str, bool]:
        """After supervision: check each unreachable label in R2 once more, for the record."""
        pending = sorted(
            self.r2_unreachable - self.r2_unconfirmed - self.r2_rechecked - self.r2_secured.keys()
        )
        listed = self.at_risk_files
        checked: dict[str, bool] = {}
        for label in pending:
            self.r2_rechecked.add(label)
            if label in listed:
                checked[label] = self.secured_in_posttrain(label, listed[label])
        return checked

    def diagnose(
        self, raw: bytes, report: dict[str, Any], *, recovery: dict[str, Any] | None = None
    ) -> None:
        directory = self.root / "diagnostics"
        directory.mkdir(exist_ok=True)
        (directory / "report.json").write_bytes(raw)
        summary = posttrain_ops.run_summary(
            report,
            run=self.run,
            pod_id=self.pod_id,
            verification=None,
            source="ssh-terminal-report",
        )
        try:
            summary["r2_labels_present"] = sorted(
                {
                    str(item["key"])[len(self.prefix) :].split("/", 1)[0]
                    for item in self.quick_store.objects()
                    if str(item["key"]).startswith(self.prefix)
                }
            )
        except Exception as error:  # recorded; diagnostics are archived regardless
            summary["r2_listing_error"] = type(error).__name__
        if recovery is not None:
            summary["label_recovery"] = recovery
        atomic_json(directory / "summary.json", summary)
        self.summary, self.latest = summary, directory
        self.archive_verified(directory)
        self.shutdown_ready = True
        self.milestone_error = None
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
        choices=("pilot", "rehearsal", "posttrain"),
        default="pilot",
        help="rehearsal or posttrain: verify the final state in R2 instead of pulling checkpoints",
    )
    parser.add_argument(
        "--run",
        choices=[*sorted(RUNS), *posttrain_ops.RUNS],
        help="the rehearsal run (A or B) or the post-training run (1)",
    )
    parser.add_argument(
        "--failure-grace-seconds",
        type=float,
        default=DEFAULT_FAILURE_GRACE_SECONDS,
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
    if args.mode in ("rehearsal", "posttrain"):
        return _rehearsal_main(args, parser, identity, policy, token, ssh)
    if args.run is not None:
        parser.error("--run applies only to --mode rehearsal or posttrain")
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
    """Supervise one rehearsal or post-training Pod; nothing is pulled over SSH except bounded
    status files and, to rescue them, unverified milestones."""
    posttrain = args.mode == "posttrain"
    runs = posttrain_ops.RUNS if posttrain else tuple(sorted(RUNS))
    if args.run not in runs:
        parser.error(f"--mode {args.mode} needs --run {' or --run '.join(runs)}")
    if policy.mode != args.mode:
        parser.error(f'a {args.mode} supervisor needs a policy with "mode": "{args.mode}"')
    if not 0 < args.failure_grace_seconds <= IDLE_SECONDS:
        parser.error(f"--failure-grace-seconds must be within the {IDLE_SECONDS}-second idle limit")
    try:
        # The Mac's own key: it checks what the Pod's key wrote.
        credentials = r2_credentials(os.environ, key_prefix=MAC_KEY_PREFIX)
    except ValueError as error:
        parser.error(f"Mac R2 credentials: {error}")
    store = Store(credentials)  # verify_final and its downloads.
    # The same key, bounded to 5 s per socket operation, for the checks the watchdog's
    # transfer window depends on: the mirror's listing and the at-risk HEADs.
    quick_store = bounded_store(credentials, timeout=MAC_CHECK_TIMEOUT_SECONDS)
    root = f"{posttrain_ops.RUN_ROOT if posttrain else 'rehearsal-runs'}/{args.run}"
    kind = "post-training" if posttrain else "rehearsal"

    def archive(bundle: Path) -> dict[str, Any]:
        return archive_bundle(
            bundle,
            args.repository,
            run_id=identity.pod_id,
            files=("report.json", "summary.json"),
            root=root,
            message=f"Save {kind} report and summary",
        )

    def recover(name: str, destination: Path, timeout: float) -> Path:
        # The pilot's verified pull: the Pod's own manifest, then a size and SHA-256 check
        # of every streamed file before the bundle is published on the Mac.
        return pull_artifacts(
            host=args.host,
            port=args.port,
            identity_file=args.identity_file,
            known_hosts=args.known_hosts,
            remote_dir=f"{args.remote_root}/{name}",
            destination=destination,
            bundle_name=PurePosixPath(name).name,
            artifacts=RECOVERED_ARTIFACTS,
            timeout=min(timeout, args.transfer_timeout),
        )

    supervisor = (PosttrainSupervisor if posttrain else RehearsalSupervisor)(
        args.output,
        run=args.run,
        pod_id=identity.pod_id,
        fetch=lambda name: fetch_status(ssh, args.remote_root, name),
        fetch_report=lambda: fetch_status_bytes(
            ssh, args.remote_root, "report.json", max_bytes=REPORT_MAX_BYTES
        ),
        store=store,
        quick_store=quick_store,
        archive=archive,
        failure_grace_seconds=args.failure_grace_seconds,
        fetch_finalization=lambda: fetch_status_bytes(ssh, args.remote_root, "finalization.json"),
        recover=recover,
        transfer_window_seconds=policy.max_transfer_seconds,
        pod_created_unix=parse_timestamp(identity.created_at),
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
            f"{kind} supervisor: no heartbeat over SSH; not started (the run's watchdog "
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
    # The Pod may be gone: milestones whose R2 check only timed out are checked once more.
    # Only the record depends on it, so it can never keep the closing record from being written.
    after_close: dict[str, Any]
    try:
        after_close = dict(supervisor.final_r2_check())
    except Exception as error:
        after_close = {"error_type": type(error).__name__, "detail": str(error)[:500]}
    closing = args.output / "closing"
    closing.mkdir(exist_ok=True)
    receipt = args.output / "watchdog-receipt.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "posttrain-supervisor-closing" if posttrain else "rehearsal-supervisor-closing",
        "run": args.run,
        "pod_id": identity.pod_id,
        "shutdown_ready": supervisor.shutdown_ready,
        "shutdown_receipt": json.loads(receipt.read_text()) if receipt.exists() else None,
        "summary": supervisor.summary,
        "milestones_not_secured": supervisor.milestone_error,
        "milestones_checked_in_r2_after_close": after_close,
        "resource_audit": "Pod only; R2 objects, network volumes and billing need separate audit",
        "supervisor_interrupted": interrupted,
    }
    atomic_json(closing / "summary.json", record)  # Always written locally first.
    if supervisor.milestone_error is not None and supervisor.milestone_error.get(
        "resolved_after_close"
    ):
        print(
            f"{kind} supervisor: run {args.run}: every milestone at risk was verified in R2 "
            f"by the Mac after supervision ended; see {args.output / MILESTONE_ERROR_FILE}",
            file=sys.stderr,
        )
    elif supervisor.milestone_error is not None:
        print(
            f"{kind} supervisor: run {args.run} ended with milestones NOT secured; see "
            f"{args.output / MILESTONE_ERROR_FILE}",
            file=sys.stderr,
        )
    archive_failed = False
    try:
        archive_bundle(
            closing,
            args.repository,
            run_id=identity.pod_id,
            files=("summary.json",),
            root=root,
            message=f"Save {kind} closing record",
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
