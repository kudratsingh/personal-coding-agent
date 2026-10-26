"""Operations for the two rehearsal Pods (D-35): limits, the R2 run layout and R2 checks.

``pod-env`` (Mac)
    Write the Pod's R2 environment file to a pipe, never to a terminal, so the operator
    can deliver it over SSH stdin into a 0600 file on the Pod. Only ``R2_ENDPOINT``,
    ``R2_BUCKET``, ``R2_POD_ACCESS_KEY_ID`` and ``R2_POD_SECRET_ACCESS_KEY`` are written;
    the Mac's own key never leaves the Mac.
``fetch-object`` (Pod)
    Download one object pinned to its size and SHA-256, such as the shared initialization.
``round-trip`` (Pod)
    PUT, HEAD, GET and DELETE one random 128 MiB probe object under the run's R2 prefix
    with the Pod's key, and write a receipt with the measured upload and download
    throughput; the readiness receipt requires it and the upload rate the window needs.
``verify-final`` (Mac)
    Check a finished run's final milestone and ``report.json`` in R2 with the Mac's own key,
    the same check the supervisor applies before it lets a Pod be deleted.

Credentials come only from the environment and are never printed or written to a receipt.
Nothing here contacts Runpod or creates any resource.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.data.shards import file_sha256
from tools.r2_store import SHA256_HEX, Store
from tools.rehearsal_binding import RUNS, r2_credentials

# The owner-approved run parameters (D-35). The worker's specification carries its own
# copy; these are the ceilings the Mac-side tools check every artifact against.
POD_LIFETIME_SECONDS = 45_000
SELF_DESTRUCT_MARGIN_SECONDS = 180
"""The runbook arms the in-Pod timer this long before the 45,000-second wall."""
APPROVED_COST_CAP_USD = 10.0
ESTIMATED_TOTAL_COST_USD = 6.2917
COST_MULTIPLIER = 1.2
COST_STOP_USD = min(APPROVED_COST_CAP_USD, ESTIMATED_TOTAL_COST_USD * COST_MULTIPLIER)
"""The worker's own cost stop: $7.55 (120% of the estimate, inside the $10 cap)."""
WATCHDOG_COST_STOP_USD = 7.54
"""The Mac watchdog's cost stop, one cent inside the worker's so it is never the weaker."""
IDLE_SECONDS = 600
MAX_TRANSFER_SECONDS = 900
"""How long after its last GPU work an idle worker may keep the Pod (watchdog policy).

The worker's last drain to R2 leaves the GPU idle. 900 s equals the finalization reserve, the
time every run already plans for its final save, uploads and checks; it is half of the
watchdog's 1,800 s ceiling and costs at most $0.15 more per Pod at $0.60/h. Past the 600 s idle
limit the watchdog grants the window only while the Mac sees progress: new objects in R2, or
an SSH pull's bytes growing. The window is split three ways: the worker's final drain
(``DRAIN_SECONDS``), its bounded checks and final heartbeat (``TRANSFER_MARGIN_SECONDS``),
then the Mac supervisor's SSH pull of any milestone still unverified
(``PULL_RESERVE_SECONDS``)."""
TRANSFER_MARGIN_SECONDS = 120
"""After the drain, the worker's HEAD checks, the ``report.json`` upload and its final
heartbeat (its ``FINALIZE_MARGIN_SECONDS``). Every R2 call in it uses a bounded store, and
none starts late, so the final heartbeat is written by 480 s after the last GPU work."""
PULL_RESERVE_SECONDS = 420
"""The end of the window, kept for the Mac's SSH pull of an unverified milestone. The worker
hands over 480 s after its last GPU work, before the 600 s idle limit, so the pull's growing
bytes stamp progress in time; 1.65 GB in the roughly 370 s left after the supervisor's
bounded R2 check needs about 4.5 MB/s. The pilot's pulls bound one 1.62 GB pull at 150 s or
less (about 10.7 MB/s or faster)."""
DRAIN_SECONDS = MAX_TRANSFER_SECONDS - TRANSFER_MARGIN_SECONDS - PULL_RESERVE_SECONDS
"""The worker's final drain to R2: at most 360 s after its last GPU work."""
S_LABEL_CHECKPOINT_BYTES = 1_650_000_000
"""One S label's checkpoint, weights and AdamW state: the measured 1,623,502,465 bytes
(``pilot_feasibility.MEASURED_UNTRAINED_CHECKPOINT_BYTES``) rounded up to 1.65 GB."""
REQUIRED_UPLOAD_BYTES_PER_SECOND = S_LABEL_CHECKPOINT_BYTES / DRAIN_SECONDS * 2
"""The Pod's measured R2 upload rate must reach this, about 9.17 MB/s: the stage-final
milestone inside the 360 s drain, with a factor-2 safety margin. One label is the worst case
at this rate: no rolling save starts within 900 s of a stage end (``ROLLING_SKIP_SECONDS``)
and one uploads in at most 180 s, so none is still in flight when the final save is made.
The readiness receipt's ``r2_round_trip_tested`` gate requires it."""
ROUND_TRIP_PROBE_BYTES = 128 * 1024 * 1024
"""The round trip's probe: large enough (128 MiB) that its PUT and GET measure throughput
rather than request latency. The readiness receipt refuses a smaller probe."""
SPEED_CHECK_SECONDS = 900
REFERENCE_TOKENS_PER_SECOND = 56_479.0
"""The pilot's measured eager 35,974 tok/s times its measured 1.57x compile speedup."""
MINIMUM_SPEED_FRACTION = 0.85
FINALIZATION_RESERVE_SECONDS = 900
QUOTED_GPU_HOURLY = 0.59
QUOTED_TOTAL_HOURLY = 0.60
CONTAINER_DISK_GB = 40
CHECKPOINT_SECONDS = 1_800
VALIDATION_INTERVAL_UPDATES = 500
VALIDATION_TARGETS_PER_DOMAIN = 1_000_000
TOKENS_PER_UPDATE = 16 * 8 * 1024

READINESS_KIND = "rehearsal-readiness"
"""The ``kind`` of the receipt ``pilot_readiness_receipt rehearsal`` writes and the worker reads."""
OWNER_GATES = (
    "owner_confirms_four_runpod_steps",
    "cloud_data_pre_staged",
    "external_watchdog_armed",
    "hardware_preflight_passed",
    "full_console_quote_confirmed",
    "staging_storage_cost_bound",
    "r2_token_bucket_scoped",
)
"""The owner's statements. A rehearsal moves checkpoints through R2, not SSH, so the
pilot's ``ssh_transfer_tested`` is replaced by the evidence gates below."""
EVIDENCE_GATES = ("r2_round_trip_tested", "r2_shards_staged_and_hash_checked")
"""Set only from the Pod's receipts: the R2 round trip and the fetched, re-hashed data."""
READINESS_GATES = (*OWNER_GATES, *EVIDENCE_GATES, "pod_self_destruct_armed")
"""Every gate the receipt sets and the worker requires to be an explicit ``true``."""
REVISION_KEYS = frozenset({"git_commit", "clean", "archive_sha256"})
"""``revision.json``: the Mac's commit, its clean tree and the shipped tarball's SHA-256. The
runbook writes it next to the tarball and ships it to the Pod; the readiness receipt records it
as ``revision`` and the worker requires its own copy to be equal."""
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

RUN_ROOT = "rehearsal-runs"
REPORT_MAX_BYTES = 1_000_000
"""The worker's report.json stays below this, which is also the GitHub archive limit."""
LABEL_FILES = ("checkpoint.pt", "checkpoint.pt.sha256", "record.json", "updates.jsonl")
FINAL_LABEL_FILES = (*LABEL_FILES, "ticket.json")
R2_FINAL_KEYS = frozenset({"report_key", "report_sha256", "checkpoint_key", "checkpoint_sha256"})
SUCCESS_STATUSES = ("completed", "time_limited")
FAILURE_STATUSES = ("stopped", "failed")
UPLOADING_STATUS = "uploading"
"""The worker's heartbeat while it drains to R2, checks R2 and uploads ``report.json``."""
IN_PROGRESS_STATUSES = ("running", UPLOADING_STATUS)
"""Heartbeat statuses that never approve teardown, whatever stage the run is in."""
POD_KEY_PREFIX = "R2_POD_"
MAC_KEY_PREFIX = "R2_"
POD_ENVIRONMENT = ("R2_ENDPOINT", "R2_BUCKET", "R2_POD_ACCESS_KEY_ID", "R2_POD_SECRET_ACCESS_KEY")
_SAFE_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class ObjectStore(Protocol):
    """The read side of ``tools.r2_store.Store`` that verification needs."""

    def head(self, key: str) -> dict[str, str] | None: ...

    def download(
        self, key: str, destination: Path, *, sha256: str, size: int
    ) -> dict[str, Any]: ...


class ProbeStore(ObjectStore, Protocol):
    """Everything the R2 round trip touches."""

    def put_known(self, key: str, path: Path, sha256: str, size: int) -> int: ...

    def delete(self, key: str) -> None: ...


def run_stages(run: str) -> tuple[str, ...]:
    """The run's stage names in training order: A is stage1 then stage2, B is mixed."""
    if run not in RUNS:
        raise ValueError(f"unknown run {run!r}; the runs are {sorted(RUNS)}")
    return tuple(name for name, _ in RUNS[run])


def run_prefix(run: str, pod_id: str) -> str:
    """``rehearsal-runs/<run>/<pod id>/``, where the run's Pod writes every artifact."""
    run_stages(run)
    if not isinstance(pod_id, str) or not _SAFE_POD_ID.fullmatch(pod_id):
        raise ValueError("the Pod ID must be a plain identifier")
    return f"{RUN_ROOT}/{run}/{pod_id}/"


def _hex(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be 64 lower-case hex digits")
    return value


def check_revision(value: object) -> dict[str, Any]:
    """The code revision record: exactly a 40-hex commit, ``clean: true`` and a tarball hash.

    A dirty tree or any other shape is refused: the run must name the exact code it trains.
    """
    if not isinstance(value, dict) or set(value) != REVISION_KEYS:
        raise ValueError(f"the revision record must have exactly the keys {sorted(REVISION_KEYS)}")
    commit = value["git_commit"]
    if not isinstance(commit, str) or not _GIT_COMMIT.fullmatch(commit):
        raise ValueError("the revision's git_commit must be 40 lower-case hex digits")
    if value["clean"] is not True:
        raise ValueError("the revision was taken from a dirty tree; commit or stash first")
    _hex(value["archive_sha256"], "the revision's archive_sha256")
    return {"git_commit": commit, "clean": True, "archive_sha256": value["archive_sha256"]}


def check_r2_final(value: object, *, run: str, pod_id: str) -> tuple[dict[str, str], str]:
    """Validate a heartbeat's ``r2_final``; return it and the stage its checkpoint closes.

    Both keys must sit under this run's own prefix, and the checkpoint must be a
    stage-final milestone of this run, never a rolling label.
    """
    if not isinstance(value, dict) or set(value) != R2_FINAL_KEYS:
        raise ValueError(f"r2_final must have exactly the keys {sorted(R2_FINAL_KEYS)}")
    final = {key: value[key] for key in sorted(R2_FINAL_KEYS)}
    if not all(isinstance(item, str) for item in final.values()):
        raise ValueError("r2_final values must be strings")
    _hex(final["report_sha256"], "r2_final report_sha256")
    _hex(final["checkpoint_sha256"], "r2_final checkpoint_sha256")
    prefix = run_prefix(run, pod_id)
    if final["report_key"] != prefix + "report.json":
        raise ValueError(f"the final report key must be {prefix}report.json")
    stages = [
        stage
        for stage in run_stages(run)
        if final["checkpoint_key"] == f"{prefix}{stage}-final/checkpoint.pt"
    ]
    if len(stages) != 1:
        raise ValueError("the final checkpoint key must be a stage-final milestone of this run")
    return final, stages[0]


def head_sha256(store: ObjectStore, key: str, *, expected: str | None) -> dict[str, Any]:
    """HEAD one object: it must exist and carry SHA-256 metadata (equal to ``expected``)."""
    headers = store.head(key)
    if headers is None:
        raise ValueError(f"{key} is missing in R2")
    stored = headers.get("x-amz-meta-sha256")
    if stored is None or not SHA256_HEX.fullmatch(stored):
        raise ValueError(f"{key} carries no SHA-256 metadata in R2")
    if expected is not None and stored != expected:
        raise ValueError(f"{key}: the SHA-256 stored in R2 differs from the expected value")
    size = headers.get("content-length", "")
    if not size.isdigit():
        raise ValueError(f"{key}: R2 reported no object size")
    return {"key": key, "sha256": stored, "bytes": int(size)}


def verify_final(
    store: ObjectStore,
    r2_final: object,
    *,
    run: str,
    pod_id: str,
    destination: Path,
    statuses: Sequence[str] = SUCCESS_STATUSES,
) -> dict[str, Any]:
    """Verify a run's final milestone and report in R2; keep the pinned report locally.

    HEAD must show the expected SHA-256 for the final checkpoint and ``report.json``
    and the final label's other files must exist. ``report.json`` is then downloaded
    pinned to its SHA-256 and must close this run. The worker lists every planned stage
    in order, a stage it never started as ``pending`` with no final checkpoint. The
    stages with a final checkpoint must come first; the last of them is the heartbeat's
    checkpoint, and every earlier one's milestone is in R2 with the hash the report
    records. A completed run must have closed every stage, so run A's stage-1 end is
    never completion; a time-limited or stopped run A may close at ``stage1-final``.
    """
    final, stage = check_r2_final(r2_final, run=run, pod_id=pod_id)
    prefix = run_prefix(run, pod_id)
    report_head = head_sha256(store, final["report_key"], expected=final["report_sha256"])
    if not 0 < report_head["bytes"] <= REPORT_MAX_BYTES:
        raise ValueError(f"report.json in R2 must hold 1 to {REPORT_MAX_BYTES:,} bytes")
    checkpoint_head = head_sha256(
        store, final["checkpoint_key"], expected=final["checkpoint_sha256"]
    )
    if checkpoint_head["bytes"] <= 0:
        raise ValueError("the final checkpoint in R2 is empty")
    label = f"{prefix}{stage}-final/"
    label_files = [checkpoint_head] + [
        head_sha256(store, label + name, expected=None)
        for name in FINAL_LABEL_FILES
        if name != "checkpoint.pt"
    ]
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    store.download(
        final["report_key"], report_path, sha256=final["report_sha256"], size=report_head["bytes"]
    )
    report = json.loads(report_path.read_bytes())
    if not isinstance(report, dict):
        raise ValueError("the R2 report must be a JSON object")
    status = report.get("status")
    if status not in statuses:
        raise ValueError(f"the R2 report status {status!r} is not one of {list(statuses)}")
    entries = report.get("stages")
    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError("the R2 report must list its stages")
    names = [entry.get("name") for entry in entries]
    expected = list(run_stages(run))
    if names != expected[: len(names)]:
        raise ValueError(f"the report's stages {names} differ from run {run}'s {expected}")
    closed = 0
    while closed < len(entries) and entries[closed].get("final_checkpoint_sha256") is not None:
        closed += 1
    if any(entry.get("final_checkpoint_sha256") is not None for entry in entries[closed:]):
        raise ValueError("a stage after an unfinished one reports a final checkpoint")
    if status == "completed" and (names != expected or closed != len(expected)):
        raise ValueError(f"a completed run {run} must report every stage {expected} closed")
    if closed == 0 or names[closed - 1] != stage:
        raise ValueError("the final checkpoint does not close the report's last finished stage")
    if entries[closed - 1].get("final_checkpoint_sha256") != final["checkpoint_sha256"]:
        raise ValueError("the report's final checkpoint SHA-256 differs from the heartbeat's")
    milestones = [
        head_sha256(
            store,
            f"{prefix}{entry['name']}-final/checkpoint.pt",
            expected=_hex(entry.get("final_checkpoint_sha256"), f"{entry['name']} checkpoint"),
        )
        for entry in entries[: closed - 1]
    ]
    milestones.append(checkpoint_head)
    return {
        "schema_version": 1,
        "kind": "rehearsal-r2-final-verification",
        "status": "verified",
        "run": run,
        "pod_id": pod_id,
        "prefix": prefix,
        "report": report_head,
        "checkpoint": checkpoint_head,
        "final_label_files": label_files,
        "stage_milestones": milestones,
        "report_status": status,
        "verified_at": utc_now(),
    }


def run_summary(
    report: Mapping[str, Any],
    *,
    run: str,
    pod_id: str,
    verification: Mapping[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    """A small record of a run for the private archive; the loss curve stays in R2."""
    stages = []
    for entry in report.get("stages") or []:
        if not isinstance(entry, dict):
            continue
        validations = entry.get("validations")
        stages.append(
            {
                "name": entry.get("name"),
                "updates": entry.get("updates"),
                "tokens": entry.get("tokens"),
                "learning_rate": entry.get("learning_rate"),
                "warmup_updates": entry.get("warmup_updates"),
                "expected_ticket_match": entry.get("expected_ticket_match"),
                "final_checkpoint_sha256": entry.get("final_checkpoint_sha256"),
                "validations": len(validations) if isinstance(validations, list) else None,
                "last_validation": validations[-1]
                if isinstance(validations, list) and validations
                else None,
                "full_validation": entry.get("full_validation"),
            }
        )
    return {
        "schema_version": 1,
        "kind": "rehearsal-run-summary",
        "run": run,
        "pod_id": pod_id,
        "r2_prefix": run_prefix(run, pod_id),
        "source": source,
        "status": report.get("status"),
        "binding_sha256": report.get("binding_sha256"),
        "initial_weights_sha256": report.get("initial_weights_sha256"),
        "stages": stages,
        "speed_check": report.get("speed_check"),
        "cost": report.get("cost"),
        "timings": report.get("timings"),
        "limits": report.get("limits"),
        "r2_final_verified": verification is not None,
        "r2_verification": None if verification is None else dict(verification),
        "created_at": utc_now(),
    }


def pod_environment(environ: Mapping[str, str]) -> str:
    """The Pod's R2 environment file: the shared endpoint and bucket and the Pod's key.

    The Pods get their own bucket-scoped key; the Mac's key is refused if it is the same.
    """
    r2_credentials(environ, key_prefix=POD_KEY_PREFIX)  # Presence and endpoint/bucket shape.
    if environ.get("R2_ACCESS_KEY_ID") == environ["R2_POD_ACCESS_KEY_ID"]:
        raise ValueError("the Pods need their own bucket-scoped R2 key, not the Mac's")
    lines = []
    for name in POD_ENVIRONMENT:
        value = environ[name]
        if any(character in value for character in "\n\r\x00"):
            raise ValueError(f"{name} contains a line break")
        lines.append(f"{name}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def fetch_object(
    store: ObjectStore, *, key: str, sha256: str, size: int, output: Path, receipt: Path
) -> dict[str, Any]:
    """Download ``key`` to ``output`` pinned to ``size`` and ``sha256``; write a receipt.

    A present file with the pinned bytes is kept; any other present file stops here.
    """
    _hex(sha256, "the pinned SHA-256")
    if type(size) is not int or size <= 0:
        raise ValueError("the pinned size must be a positive integer")
    if output.is_symlink():
        raise ValueError(f"{output} must not be a symlink")
    started = time.monotonic()
    if output.exists():
        if output.stat().st_size != size or file_sha256(output) != sha256:
            raise ValueError(f"{output} is present with other bytes; remove it to refetch")
        row: dict[str, Any] = {"method": "present", "bytes_transferred": 0}
    else:
        got = store.download(key, output, sha256=sha256, size=size)
        row = {
            "method": "downloaded",
            "bytes_transferred": got["bytes"],
            "attempts": got["attempts"],
        }
    record = {
        "schema_version": 1,
        "kind": "r2-pinned-object",
        "created_at": utc_now(),
        "key": key,
        "path": str(output),
        "bytes": size,
        "sha256": sha256,
        "seconds": time.monotonic() - started,
        **row,
    }
    atomic_json(receipt, record)
    return record


def round_trip(
    store: ProbeStore,
    *,
    run: str,
    pod_id: str,
    receipt: Path,
    size: int = ROUND_TRIP_PROBE_BYTES,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """PUT, HEAD, GET and DELETE one random object under the run's prefix; write a receipt.

    The PUT and the GET are timed, and the receipt records ``upload_bytes_per_second`` and
    ``download_bytes_per_second`` (retries included, so the rates err low). The probe is
    deleted again on any failure after it was stored. The receipt says ``passed`` only when
    every step succeeded and the object is gone afterwards; whether the upload rate is
    enough for the transfer window is the readiness receipt's decision.
    """
    if type(size) is not int or size <= 0:
        raise ValueError("the probe size must be a positive integer")
    prefix = run_prefix(run, pod_id)
    key = f"{prefix}preflight/round-trip-{uuid.uuid4().hex}.bin"
    steps: dict[str, float] = {}
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "r2-round-trip",
        "created_at": utc_now(),
        "run": run,
        "pod_id": pod_id,
        "prefix": prefix,
        "key": key,
        "bytes": size,
        "steps": steps,
        "required_upload_bytes_per_second": REQUIRED_UPLOAD_BYTES_PER_SECOND,
    }
    stored = False
    with tempfile.TemporaryDirectory(prefix="r2-round-trip-") as directory:
        source = Path(directory) / "probe.bin"
        hasher = hashlib.sha256()
        with source.open("wb") as stream:
            left = size
            while left:
                block = os.urandom(min(left, 8 * 1024 * 1024))
                hasher.update(block)
                stream.write(block)
                left -= len(block)
        digest = hasher.hexdigest()
        record["sha256"] = digest
        try:
            began = clock()
            store.put_known(key, source, digest, size)
            stored = True
            steps["put_seconds"] = clock() - began
            began = clock()
            if head_sha256(store, key, expected=digest)["bytes"] != size:
                raise ValueError("HEAD reported another size than was written")
            steps["head_seconds"] = clock() - began
            began = clock()
            store.download(key, Path(directory) / "back.bin", sha256=digest, size=size)
            steps["get_seconds"] = clock() - began
            if steps["put_seconds"] <= 0 or steps["get_seconds"] <= 0:
                raise ValueError("the clock did not advance over the PUT or the GET")
            record["upload_bytes_per_second"] = size / steps["put_seconds"]
            record["download_bytes_per_second"] = size / steps["get_seconds"]
            began = clock()
            store.delete(key)
            if store.head(key) is not None:
                raise ValueError("the probe object is still present after DELETE")
            stored = False
            steps["delete_seconds"] = clock() - began
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {str(error)[:300]}")
            if stored:
                try:
                    store.delete(key)
                    record["probe_deleted"] = True
                except Exception as cleanup:  # recorded; the original error is raised
                    record["probe_delete_error"] = type(cleanup).__name__
            atomic_json(receipt, record)
            raise
    record["status"] = "passed"
    atomic_json(receipt, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pod-env", help="write the Pod's R2 environment file to a pipe (Mac)")
    fetch = commands.add_parser("fetch-object", help="download one pinned object (Pod)")
    fetch.add_argument("--key", required=True)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--size", type=int, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--receipt", type=Path, required=True)
    probe = commands.add_parser("round-trip", help="test PUT/HEAD/GET/DELETE in R2 (Pod)")
    probe.add_argument("--run", choices=sorted(RUNS), required=True)
    probe.add_argument("--pod-id", required=True)
    probe.add_argument("--receipt", type=Path, required=True)
    for command in (fetch, probe):
        command.add_argument(
            "--key-prefix",
            default=POD_KEY_PREFIX,
            help="prefix of the key-pair variables (the Pod's: R2_POD_)",
        )
    final = commands.add_parser("verify-final", help="check a run's final state in R2 (Mac)")
    final.add_argument("--run", choices=sorted(RUNS), required=True)
    final.add_argument("--pod-id", required=True)
    final.add_argument("--report-sha256", required=True)
    final.add_argument("--checkpoint-sha256", required=True)
    final.add_argument("--stage", help="the stage whose final milestone closes the run")
    final.add_argument("--output", type=Path, required=True, help="directory for report.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pod-env":
            if sys.stdout.isatty():
                parser.error("pipe this into ssh; credentials are never written to a terminal")
            sys.stdout.write(pod_environment(os.environ))
            sys.stdout.flush()
            return 0
        if args.command == "verify-final":
            prefix = run_prefix(args.run, args.pod_id)
            stage = args.stage or run_stages(args.run)[-1]
            r2_final = {
                "report_key": prefix + "report.json",
                "report_sha256": args.report_sha256,
                "checkpoint_key": f"{prefix}{stage}-final/checkpoint.pt",
                "checkpoint_sha256": args.checkpoint_sha256,
            }
            store = Store(r2_credentials(key_prefix=MAC_KEY_PREFIX))
            result = verify_final(
                store,
                r2_final,
                run=args.run,
                pod_id=args.pod_id,
                destination=args.output,
                statuses=(*SUCCESS_STATUSES, *FAILURE_STATUSES),
            )
            atomic_json(args.output / "r2-verification.json", result)
            print(json.dumps(result, indent=1, sort_keys=True))
            return 0
        store = Store(r2_credentials(key_prefix=args.key_prefix))
        if args.command == "fetch-object":
            record = fetch_object(
                store,
                key=args.key,
                sha256=args.sha256,
                size=args.size,
                output=args.output,
                receipt=args.receipt,
            )
        else:
            record = round_trip(store, run=args.run, pod_id=args.pod_id, receipt=args.receipt)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"rehearsal-ops: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
        return 1
    print(json.dumps({key: record[key] for key in sorted(record) if key != "steps"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
