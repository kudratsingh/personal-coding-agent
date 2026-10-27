"""Operations for the post-training Pods: the ceiling-derived lifetime, the R2 run layout,
the R2 prefix guard and the Mac's final verification. Nothing here contacts Runpod.

The owner fixes each post-training run by a dollar ceiling (run 1: $3.00). Every time and
cost limit is derived from that ceiling and the quoted rate in one place,
``ceiling_limits``, never from the pretraining rehearsal's fixed 60,000 s:

- the worker's cost stop is the ceiling;
- the Mac watchdog's cost stop is one cent inside it;
- the Pod wall is the ceiling's worth of seconds at the quoted rate, rounded down;
- the in-Pod timer fires ``self_destruct_margin_seconds`` (180 s) before that wall.

For run 1 at $0.60/h: $3.00, $2.99, 18,000 s and 17,820 s. The worker, the readiness
receipt, the runbook's watchdog policy and the bootstrap's timer limit all take them from
here (``limits`` prints them), and the receipt and the worker refuse anything weaker.

``limits`` (Mac)
    Print a specification's derived limits; with ``--origin`` also the timer deadline, and
    with ``--policy-output`` write the watchdog policy file.
``round-trip`` (Pod)
    The D-35 R2 round trip under ``posttrain-runs/<run>/<pod id>/``, gated on the upload rate
    one weights-only label needs (``REQUIRED_UPLOAD_BYTES_PER_SECOND``).
``verify-final`` (Mac)
    Check a finished run's ``report.json`` and every artifact it lists in R2 with the Mac's
    own key, the check the supervisor applies before it lets a Pod be deleted.

Credentials come only from the environment and are never printed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

from scratchcoder.common.records import atomic_json, utc_now
from tools import rehearsal_ops
from tools.r2_store import SHA256_HEX, Store
from tools.rehearsal_binding import r2_credentials
from tools.rehearsal_ops import ObjectStore, head_sha256

RUN_ROOT = "posttrain-runs"
RUNS = ("1",)
"""The post-training runs this code drives. Run 2 (RL, the more-SFT control and their
evaluation, with the Mac in the loop) is added with its own stages and specification."""
ABSOLUTE_LIFETIME_SECONDS = 60_000
"""The in-Pod timer's absolute cap (``pod_self_destruct.ABSOLUTE_DEADLINE_SECONDS``)."""
SELF_DESTRUCT_MARGIN_SECONDS = 180
WATCHDOG_MARGIN_USD = 0.01
QUOTED_GPU_HOURLY = rehearsal_ops.QUOTED_GPU_HOURLY
QUOTED_TOTAL_HOURLY = rehearsal_ops.QUOTED_TOTAL_HOURLY
MAC_IDLE_SECONDS = rehearsal_ops.IDLE_SECONDS
"""The Mac watchdog's idle limit, as in D-35 (600 s)."""
POD_IDLE_SECONDS = 900
"""The in-Pod idle self-destruct: the Pod deletes itself after this long with no GPU work and
no progress, independent of the Mac (``pod_self_destruct.py --idle-seconds``)."""
MAX_TRANSFER_SECONDS = rehearsal_ops.MAX_TRANSFER_SECONDS
DRAIN_SECONDS = rehearsal_ops.DRAIN_SECONDS
PULL_RESERVE_SECONDS = rehearsal_ops.PULL_RESERVE_SECONDS
S_WEIGHTS_LABEL_BYTES = 550_000_000
"""One weights-only S label: the 541,142,016 bytes of float32 weights, the save's framing and
the label's record, rounded up to 0.55 GB. SFT, RL and more-SFT each start a fresh optimizer,
so no post-training label carries optimizer state (D-35's labels were 1.65 GB)."""
REQUIRED_UPLOAD_BYTES_PER_SECOND = S_WEIGHTS_LABEL_BYTES / DRAIN_SECONDS * 2
"""About 3.06 MB/s: one weights-only label inside the 360 s final drain with a factor-2
margin, re-derived as D-35 derived 9.17 MB/s for 1.65 GB. One label is the worst case for
run 1: its only weights label (the selected SFT checkpoint) is uploaded an hour before the
final drain, and every other artifact is a small JSON or JSON Lines file."""
ROUND_TRIP_PROBE_BYTES = rehearsal_ops.ROUND_TRIP_PROBE_BYTES
UPLOADING_STATUS = rehearsal_ops.UPLOADING_STATUS
AWAITING_REWARDS_STATUS = "awaiting_rewards"
"""Run 2's heartbeat while the worker waits for the Mac's RL rewards (the hook for its
Mac-stamped reward progress); run 1 never writes it."""
IN_PROGRESS_STATUSES = ("running", UPLOADING_STATUS, AWAITING_REWARDS_STATUS)
SUCCESS_STATUSES = rehearsal_ops.SUCCESS_STATUSES
FAILURE_STATUSES = rehearsal_ops.FAILURE_STATUSES
READINESS_KIND = "posttrain-readiness"
REPORT_KIND = "posttrain-run-report"
OWNER_GATES = rehearsal_ops.OWNER_GATES
EVIDENCE_GATES = (
    "r2_round_trip_tested",
    "r2_inputs_staged_and_hash_checked",
    "mac_parent_copies_verified",
)
"""Set only from receipts: the Pod's R2 round trip, the Pod's fetch and re-verification of
every bound input, and the Mac's own pinned copies of the parent checkpoints."""
READINESS_GATES = (*OWNER_GATES, *EVIDENCE_GATES, "pod_self_destruct_armed")
REPORT_MAX_BYTES = rehearsal_ops.REPORT_MAX_BYTES
R2_FINAL_KEYS = frozenset({"report_key", "report_sha256", "artifact_count"})
LIMIT_KEYS = frozenset(
    {
        "ceiling_usd",
        "quoted_gpu_hourly",
        "quoted_total_hourly",
        "self_destruct_margin_seconds",
        "watchdog_margin_usd",
        "mac_idle_seconds",
        "pod_idle_seconds",
        "max_transfer_seconds",
        "finalization_reserve_seconds",
        "startup_seconds",
    }
)
_SAFE_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


@dataclass(frozen=True)
class CeilingLimits:
    """Every time and cost limit of one run, derived from its ceiling and quoted rate."""

    ceiling_usd: float
    quoted_total_hourly: float
    wall_seconds: int
    worker_cost_stop_usd: float
    watchdog_cost_stop_usd: float
    self_destruct_seconds: int
    self_destruct_margin_seconds: int

    def deadline_unix(self, origin: float) -> float:
        """The in-Pod timer's deadline for a Pod whose billing began at ``origin``."""
        return float(origin) + self.self_destruct_seconds


def ceiling_limits(
    ceiling_usd: float,
    quoted_total_hourly: float,
    *,
    self_destruct_margin_seconds: float = SELF_DESTRUCT_MARGIN_SECONDS,
    watchdog_margin_usd: float = WATCHDOG_MARGIN_USD,
) -> CeilingLimits:
    """The run's limits from its dollar ceiling: the one place they are derived.

    The wall is the ceiling's worth of whole seconds at the quoted rate, rounded down, so
    the wall's quoted cost never exceeds the ceiling. It must fit the in-Pod timer's
    absolute 60,000 s cap.
    """
    ceiling = _positive(ceiling_usd, "the ceiling")
    hourly = _positive(quoted_total_hourly, "the quoted total hourly rate")
    margin = _positive(self_destruct_margin_seconds, "the self-destruct margin")
    cents = _positive(watchdog_margin_usd, "the watchdog margin")
    if margin != int(margin):
        raise ValueError("the self-destruct margin must be whole seconds")
    wall = math.floor(ceiling * 3600 / hourly + 1e-6)
    if wall > ABSOLUTE_LIFETIME_SECONDS:
        raise ValueError(
            f"a ${ceiling:.2f} ceiling at ${hourly:.2f}/h is {wall:,} s, above the in-Pod "
            f"timer's absolute {ABSOLUTE_LIFETIME_SECONDS:,} s cap"
        )
    if margin >= wall or cents >= ceiling:
        raise ValueError("the margins must be smaller than the wall and the ceiling")
    return CeilingLimits(
        ceiling_usd=ceiling,
        quoted_total_hourly=hourly,
        wall_seconds=wall,
        worker_cost_stop_usd=ceiling,
        watchdog_cost_stop_usd=round(ceiling - cents, 2),
        self_destruct_seconds=int(wall - margin),
        self_destruct_margin_seconds=int(margin),
    )


def read_limits(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A specification's ``limits`` object, exactly ``LIMIT_KEYS``, each a positive number."""
    if not isinstance(raw, Mapping) or set(raw) != LIMIT_KEYS:
        raise ValueError(f"limits must have exactly the keys {sorted(LIMIT_KEYS)}")
    for key in LIMIT_KEYS:
        _positive(raw[key], f"limits.{key}")
    return dict(raw)


def specification_limits(specification: Mapping[str, Any]) -> CeilingLimits:
    """``ceiling_limits`` for a specification file's ``limits`` object."""
    limits = read_limits(specification.get("limits", {}))
    return ceiling_limits(
        limits["ceiling_usd"],
        limits["quoted_total_hourly"],
        self_destruct_margin_seconds=limits["self_destruct_margin_seconds"],
        watchdog_margin_usd=limits["watchdog_margin_usd"],
    )


def watchdog_policy(
    limits: CeilingLimits,
    *,
    origin: float,
    idle_baseline: float,
    idle_seconds: float = MAC_IDLE_SECONDS,
    transfer_seconds: float = MAX_TRANSFER_SECONDS,
) -> dict[str, Any]:
    """The Mac watchdog's policy file for a run: wall and cost stop from ``ceiling_limits``."""
    return {
        "mode": "posttrain",
        "max_wall_seconds": limits.wall_seconds,
        "max_cost_usd": limits.watchdog_cost_stop_usd,
        "max_idle_seconds": idle_seconds,
        "quoted_total_hourly": limits.quoted_total_hourly,
        "max_transfer_seconds": transfer_seconds,
        "billing_started_at_unix": origin,
        "idle_baseline_unix": idle_baseline,
    }


def run_prefix(run: str, pod_id: str) -> str:
    """``posttrain-runs/<run>/<pod id>/``, where the run's Pod writes every artifact."""
    if run not in RUNS:
        raise ValueError(f"unknown post-training run {run!r}; the runs are {list(RUNS)}")
    if not isinstance(pod_id, str) or not _SAFE_POD_ID.fullmatch(pod_id):
        raise ValueError("the Pod ID must be a plain identifier")
    return f"{RUN_ROOT}/{run}/{pod_id}/"


class PrefixViolation(PermissionError):
    """A write or delete outside the run's own R2 prefix was refused before any request."""


class GuardedStore(Store):
    """A store that refuses every write and delete outside one prefix: the R2 prefix guard.

    Every request a ``Store`` makes goes through ``_request`` except ``download``'s signed
    GET, so the guard sits there: any method but GET and HEAD whose key does not lie under
    ``prefix`` (plain components, no ``.`` or ``..``) raises ``PrefixViolation`` before a
    socket is opened. Bucket listings (GET with no key) and reads anywhere stay allowed:
    the worker fetches its inputs from other prefixes.
    """

    READ_METHODS = frozenset({"GET", "HEAD"})

    def __init__(self, inner: Store, prefix: str) -> None:
        if not isinstance(prefix, str) or not prefix.endswith("/") or prefix.startswith("/"):
            raise ValueError("the guarded prefix must be relative and end in '/'")
        if any(part in ("", ".", "..") for part in prefix[:-1].split("/")):
            raise ValueError("the guarded prefix must be plain path components")
        super().__init__(
            inner.credentials,
            transport=inner.transport,
            opener=inner.opener,
            clock=inner.clock,
            attempts=inner.attempts,
            sleep=inner.sleep,
        )
        self.prefix = prefix
        self.refused: list[dict[str, str]] = []

    def allowed(self, method: str, key: str) -> bool:
        if method in self.READ_METHODS:
            return True
        parts = key.split("/")
        return key.startswith(self.prefix) and all(part not in ("", ".", "..") for part in parts)

    def _request(
        self,
        method: str,
        key: str = "",
        *,
        query: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO[bytes] | bytes | None = None,
        payload_sha256: str | None = None,
    ) -> tuple[int, Mapping[str, str], bytes]:
        if not self.allowed(method, key):
            self.refused.append({"method": method, "key": key})
            raise PrefixViolation(
                f"R2 {method} {key!r} refused: outside the run prefix {self.prefix}"
            )
        if payload_sha256 is None:
            return super()._request(method, key, query=query, headers=headers, body=body)
        return super()._request(
            method, key, query=query, headers=headers, body=body, payload_sha256=payload_sha256
        )


def check_r2_final(value: object, *, run: str, pod_id: str) -> dict[str, Any]:
    """A heartbeat's ``r2_final``: the report key under this run's prefix and its SHA-256."""
    if not isinstance(value, dict) or set(value) != R2_FINAL_KEYS:
        raise ValueError(f"r2_final must have exactly the keys {sorted(R2_FINAL_KEYS)}")
    final = dict(value)
    if final["report_key"] != run_prefix(run, pod_id) + "report.json":
        raise ValueError("the final report key must be this run's report.json")
    if not isinstance(final["report_sha256"], str) or not SHA256_HEX.fullmatch(
        final["report_sha256"]
    ):
        raise ValueError("r2_final report_sha256 must be 64 lower-case hex digits")
    count = final["artifact_count"]
    if type(count) is not int or count < 1:
        raise ValueError("r2_final artifact_count must be a positive integer")
    return final


def report_artifacts(report: Mapping[str, Any], *, prefix: str) -> list[dict[str, Any]]:
    """The artifacts a worker report lists, each under ``prefix`` with a size and SHA-256."""
    rows = report.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("the report lists no artifacts")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"label", "name", "key", "sha256", "bytes"}:
            raise ValueError("an artifact row has exactly label, name, key, sha256 and bytes")
        label, name = row["label"], row["name"]
        if not all(isinstance(item, str) and _LABEL.fullmatch(item) for item in (label, name)):
            raise ValueError("artifact labels and names must be plain components")
        if row["key"] != f"{prefix}{label}/{name}" or row["key"] in seen:
            raise ValueError(f"artifact {label}/{name} has another key or is listed twice")
        if not isinstance(row["sha256"], str) or not SHA256_HEX.fullmatch(row["sha256"]):
            raise ValueError(f"artifact {label}/{name} has no SHA-256")
        if type(row["bytes"]) is not int or row["bytes"] < 1:
            raise ValueError(f"artifact {label}/{name} has no size")
        seen.add(row["key"])
    return [dict(row) for row in rows]


def verify_final(
    store: ObjectStore,
    r2_final: object,
    *,
    run: str,
    pod_id: str,
    destination: Path,
    statuses: Sequence[str] = SUCCESS_STATUSES,
) -> dict[str, Any]:
    """Verify a run's ``report.json`` and every artifact it lists in R2; keep the report.

    HEAD shows the report's SHA-256, and the report is downloaded pinned to it. It must be
    this run's and Pod's post-training report with one of ``statuses``, list exactly
    ``artifact_count`` artifacts under this run's prefix, and each one's HEAD must show its
    recorded SHA-256 and size. Every completed stage's outputs must be among them.
    """
    final = check_r2_final(r2_final, run=run, pod_id=pod_id)
    prefix = run_prefix(run, pod_id)
    report_head = head_sha256(store, final["report_key"], expected=final["report_sha256"])
    if not 0 < report_head["bytes"] <= REPORT_MAX_BYTES:
        raise ValueError(f"report.json in R2 must hold 1 to {REPORT_MAX_BYTES:,} bytes")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    store.download(
        final["report_key"], path, sha256=final["report_sha256"], size=report_head["bytes"]
    )
    report = json.loads(path.read_bytes())
    if not isinstance(report, dict) or report.get("kind") != REPORT_KIND:
        raise ValueError("the R2 report is not a post-training run report")
    if report.get("run") != run or report.get("pod_id") != pod_id:
        raise ValueError("the R2 report belongs to another run or Pod")
    status = report.get("status")
    if status not in statuses:
        raise ValueError(f"the R2 report status {status!r} is not one of {list(statuses)}")
    artifacts = report_artifacts(report, prefix=prefix)
    if len(artifacts) != final["artifact_count"]:
        raise ValueError("the report lists another number of artifacts than the heartbeat")
    listed = {(row["label"], row["name"]) for row in artifacts}
    stages = report.get("stages")
    if not isinstance(stages, list):
        raise ValueError("the R2 report must list its stages")
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("the R2 report's stages must be objects")
        if stage.get("status") != "completed":
            continue
        for output in stage.get("outputs", []):
            if (output.get("label"), output.get("name")) not in listed:
                raise ValueError(f"stage {stage.get('name')} output {output} is not in R2")
    heads = []
    for row in artifacts:
        head = head_sha256(store, row["key"], expected=row["sha256"])
        if head["bytes"] != row["bytes"]:
            raise ValueError(f"{row['key']}: the size in R2 differs from the report's")
        heads.append(head)
    return {
        "schema_version": 1,
        "kind": "posttrain-r2-final-verification",
        "status": "verified",
        "run": run,
        "pod_id": pod_id,
        "prefix": prefix,
        "report": report_head,
        "report_status": status,
        "artifacts": heads,
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
    """A small record of a run for the private archive; the artifacts stay in R2."""
    stages = []
    for entry in report.get("stages") or []:
        if isinstance(entry, dict):
            stages.append(
                {
                    key: entry.get(key)
                    for key in ("name", "kind", "status", "seconds", "outputs", "stop_reason")
                }
            )
    return {
        "schema_version": 1,
        "kind": "posttrain-run-summary",
        "run": run,
        "pod_id": pod_id,
        "r2_prefix": run_prefix(run, pod_id),
        "source": source,
        "status": report.get("status"),
        "stop_reason": report.get("stop_reason"),
        "stages": stages,
        "probe": report.get("probe"),
        "sft": report.get("sft"),
        "cost": report.get("cost"),
        "timings": report.get("timings"),
        "limits": report.get("limits"),
        "r2_final_verified": verification is not None,
        "r2_verification": None if verification is None else dict(verification),
        "created_at": utc_now(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    limits = commands.add_parser("limits", help="print a specification's derived limits (Mac)")
    limits.add_argument("--specification", type=Path, required=True)
    limits.add_argument("--origin", type=float, help="the billing origin (unix seconds)")
    limits.add_argument("--idle-baseline", type=float, help="the watchdog's idle baseline")
    limits.add_argument("--policy-output", type=Path, help="write the watchdog policy here")
    limits.add_argument("--field", help="print only this value (for shell variables)")
    probe = commands.add_parser("round-trip", help="test PUT/HEAD/GET/DELETE in R2 (Pod)")
    probe.add_argument("--run", choices=RUNS, required=True)
    probe.add_argument("--pod-id", required=True)
    probe.add_argument("--receipt", type=Path, required=True)
    probe.add_argument("--key-prefix", default=rehearsal_ops.POD_KEY_PREFIX)
    final = commands.add_parser("verify-final", help="check a run's final state in R2 (Mac)")
    final.add_argument("--run", choices=RUNS, required=True)
    final.add_argument("--pod-id", required=True)
    final.add_argument("--report-sha256", required=True)
    final.add_argument("--artifact-count", type=int, required=True)
    final.add_argument("--output", type=Path, required=True, help="directory for report.json")
    return parser


def limits_record(args: argparse.Namespace) -> dict[str, Any]:
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    if not isinstance(specification, dict):
        raise ValueError("the specification must be a JSON object")
    derived = specification_limits(specification)
    record: dict[str, Any] = {"run": specification.get("run"), **asdict(derived)}
    if args.origin is not None:
        record["self_destruct_deadline_unix"] = int(derived.deadline_unix(args.origin))
        baseline = args.origin if args.idle_baseline is None else args.idle_baseline
        record["policy"] = watchdog_policy(derived, origin=args.origin, idle_baseline=baseline)
        if args.policy_output is not None:
            atomic_json(args.policy_output, record["policy"])
    elif args.policy_output is not None or args.idle_baseline is not None:
        raise ValueError("--policy-output and --idle-baseline need --origin")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "limits":
            record = limits_record(args)
            if args.field is not None:
                if args.field not in record or isinstance(record[args.field], dict):
                    raise ValueError(f"no plain field {args.field!r} among {sorted(record)}")
                print(record[args.field])
            else:
                print(json.dumps(record, indent=1, sort_keys=True))
            return 0
        if args.command == "verify-final":
            prefix = run_prefix(args.run, args.pod_id)
            r2_final = {
                "report_key": prefix + "report.json",
                "report_sha256": args.report_sha256,
                "artifact_count": args.artifact_count,
            }
            store = Store(r2_credentials(key_prefix=rehearsal_ops.MAC_KEY_PREFIX))
            result = verify_final(
                store,
                r2_final,
                run=args.run,
                pod_id=args.pod_id,
                destination=args.output,
                statuses=(*SUCCESS_STATUSES, *FAILURE_STATUSES),
            )
            atomic_json(args.output / "r2-verification.json", result)
            print(json.dumps({key: result[key] for key in ("status", "report_status")}))
            return 0
        prefix = run_prefix(args.run, args.pod_id)
        store = GuardedStore(Store(r2_credentials(key_prefix=args.key_prefix)), prefix)
        record = rehearsal_ops.round_trip(
            store,
            run=args.run,
            pod_id=args.pod_id,
            receipt=args.receipt,
            prefix=prefix,
            required_upload_bytes_per_second=REQUIRED_UPLOAD_BYTES_PER_SECOND,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"posttrain-ops: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr)
        return 1
    print(json.dumps({key: record[key] for key in sorted(record) if key != "steps"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
