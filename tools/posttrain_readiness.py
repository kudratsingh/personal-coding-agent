"""The post-training readiness receipt: ``python -m tools.pilot_readiness_receipt posttrain``.

Built on the Mac from the Pod's own evidence and the owner's confirmations, in exactly the
form ``tools.posttrain_run run`` accepts (its ``check_readiness`` is applied before anything
is written). Besides the pretraining rehearsal's owner gates and termination checks it requires:

- the specification to be the approved run 1 or run 2 (``posttrain_approved.check_run1``,
  ``check_run2``; for run 2 with every gate value filled) and the binding to name exactly its
  inputs at every pinned value;
- a watchdog policy at least as strict as the ceiling-derived limits (mode ``posttrain``,
  wall and cost stop from ``posttrain_ops.ceiling_limits``, the 900 s transfer window), with
  run 2's ``awaiting_rewards`` idle extensions exactly the derived bounds (run 1: none);
- a live in-Pod timer, not a dry run, armed no later than the billing origin plus the
  derived timer lifetime (17,820 s), with the 900 s idle rule watching the worker's
  heartbeat (``self_destruct_idle``);
- the Pod's R2 round trip under ``posttrain-runs/<run>/<pod id>/`` at the upload rate one
  weights-only label needs (``r2_round_trip_tested``);
- the Pod's fetch receipt and re-verification of every bound input
  (``r2_inputs_staged_and_hash_checked``);
- the Mac's own pinned copies of the parent checkpoints (``mac_parent_copies_verified``; run
  2's parent is run 1's selected SFT weights);
- a completed A40 preflight with the fused-attention and compile checks, a clean code
  revision and a captured Pod price no higher than the GPU quote;
- the lifetime left to hold every stage at the planning rates (the worker's launch rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools import posttrain_binding, posttrain_ops, rehearsal_ops
from tools.gpu_pilot import TERMINATION_GUARANTEES
from tools.pilot_readiness_receipt import (
    pod_cost_within_quote,
    round_trip_passed,
    self_destruct_live,
)
from tools.posttrain_approved import PARENTS, check_run1, check_run2
from tools.posttrain_run import Specification, check_policy, check_readiness, launch_fit
from tools.rehearsal_run import lifetime_end
from tools.runpod_watchdog import PodIdentity, Policy, parse_timestamp

MAC_COPIES = {"1": PARENTS, "2": ("sft-weights",)}
"""The inputs the Mac must hold its own pinned copy of before the Pod exists: run 1's three
parent checkpoints, run 2's SFT weights (run 1's selected checkpoint)."""
APPROVED_CHECKS = {"1": check_run1, "2": check_run2}
DEFAULT_HEARTBEAT = "/workspace/posttrain-run/heartbeat.json"
"""The worker's heartbeat on the Pod (its ``--output``), which the in-Pod timer must watch."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def check_verification(record: dict[str, Any], binding: dict[str, Any]) -> None:
    """``posttrain_binding verify``'s output on the Pod covers every input at its pin."""
    if record.get("binding_sha256") != posttrain_binding.binding_sha256(binding):
        raise ValueError("the input verification is for another binding")
    expected = {name: entry["sha256"] for name, entry in binding["inputs"].items()}
    if record.get("inputs") != expected:
        raise ValueError("the input verification does not cover exactly the bound inputs")


def build_receipt(
    *,
    specification: Path,
    binding: Path,
    binding_sha256: str,
    preflight: Path,
    identity: Path,
    policy: Path,
    self_destruct_receipt: Path,
    confirmations: Path,
    data_fetch: Path,
    data_verification: Path,
    round_trip: Path,
    mac_copy: Path,
    revision: Path,
    connection: Path,
    billing_origin_unix: float,
    heartbeat: str = DEFAULT_HEARTBEAT,
    now: float | None = None,
) -> dict[str, Any]:
    created_at = time.time() if now is None else now
    spec = Specification.load(specification)
    APPROVED_CHECKS[spec.run](spec.raw)  # Run 2 also refuses any gate value still null.
    bound = posttrain_binding.load_binding(binding, expected_sha256=binding_sha256)
    posttrain_binding.require_specification(bound, spec.raw)
    pod_identity = PodIdentity.from_dict(_load(identity))
    policy_value = _load(policy)
    if policy_value.get("billing_started_at_unix") not in (None, billing_origin_unix):
        raise ValueError("the policy file's billing origin differs from --billing-origin-unix")
    watchdog = Policy(**{**policy_value, "billing_started_at_unix": billing_origin_unix})
    check_policy(watchdog, spec)
    timer = _load(self_destruct_receipt)
    deadline = self_destruct_live(timer, pod_id=pod_identity.pod_id, now=created_at)
    if timer.get("dry_run") is not False:
        # A dry-run timer reads "armed" too, but it never sends the delete.
        raise ValueError("the in-Pod timer is a dry run: it would never delete the Pod")
    idle = {
        "idle_seconds": timer.get("idle_seconds"),
        "activity_files": timer.get("activity_files"),
    }
    cost = pod_cost_within_quote(
        _load(connection),
        pod_id=pod_identity.pod_id,
        quoted=float(spec.limits["quoted_gpu_hourly"]),
    )
    code = rehearsal_ops.check_revision(_load(revision))
    owner = _load(confirmations)
    guarantee = owner.get("termination_guarantee")
    if guarantee not in TERMINATION_GUARANTEES:
        raise ValueError("confirmations must state termination_guarantee")
    preflight_report = _load(preflight)
    for name in ("fused_attention", "compile_equivalence"):
        check = preflight_report.get(name)
        if not isinstance(check, dict) or check.get("status") != "passed":
            raise ValueError(f"the A40 preflight's {name} check has not passed")
    if (
        preflight_report.get("status") != "completed"
        or preflight_report.get("hardware_precision_and_resume_checked") is not True
        or preflight_report.get("device") != "cuda:0"
    ):
        raise ValueError("the A40 preflight has not completed on cuda:0")
    posttrain_binding.check_fetch_receipt(_load(data_fetch), bound)
    check_verification(_load(data_verification), bound)
    posttrain_binding.check_fetch_receipt(_load(mac_copy), bound, names=MAC_COPIES[spec.run])
    prefix = posttrain_ops.run_prefix(spec.run, pod_identity.pod_id)
    probed = round_trip_passed(
        _load(round_trip),
        run=spec.run,
        pod_id=pod_identity.pod_id,
        prefix=prefix,
        required=posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND,
    )
    origin = min(billing_origin_unix, parse_timestamp(pod_identity.created_at))
    derived = spec.ceiling
    end = lifetime_end(
        watchdog,
        origin=origin,
        deadline=deadline,
        pod_seconds=derived.wall_seconds,
        cost_stop_usd=derived.worker_cost_stop_usd,
        quoted_total_hourly=derived.quoted_total_hourly,
    )
    fit = launch_fit(spec, remaining_seconds=end - created_at)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": posttrain_ops.READINESS_KIND,
        "run": spec.run,
        "created_at_unix": created_at,
        "specification_sha256": file_sha256(specification),
        "binding_sha256": binding_sha256,
        "preflight_sha256": file_sha256(preflight),
        "preflight_status": preflight_report.get("status"),
        "pod_identity": _load(identity),
        "pod_created_at_unix": parse_timestamp(pod_identity.created_at),
        "watchdog_policy": {**policy_value, "billing_started_at_unix": billing_origin_unix},
        "pod_self_destruct_armed": True,
        "self_destruct_deadline_unix": deadline,
        "self_destruct_receipt_sha256": file_sha256(self_destruct_receipt),
        "self_destruct_idle": idle,
        "self_destruct_dry_run": False,
        "termination_guarantee": guarantee,
        "termination_evidence": (
            f"in-pod self-destruct receipt {file_sha256(self_destruct_receipt)[:16]} armed at "
            f"{timer.get('armed_at_unix')} for deadline {deadline} with a "
            f"{idle['idle_seconds']} s idle rule; Mac watchdog policy bound above; no "
            "provider-enforced Pod lifetime exists"
        ),
        "owner_accepts_best_effort_termination": owner.get("owner_accepts_best_effort_termination"),
        "r2_run_prefix": prefix,
        "r2_round_trip_tested": probed,
        "r2_inputs_staged_and_hash_checked": True,
        "mac_parent_copies_verified": True,
        "revision": code,
        "pod_cost_usd_per_hour": cost,
        "evidence_sha256": {
            name: file_sha256(path)
            for name, path in (
                ("data_fetch", data_fetch),
                ("data_verification", data_verification),
                ("round_trip", round_trip),
                ("mac_copy", mac_copy),
                ("connection", connection),
                ("revision", revision),
            )
        },
        "limits": {
            **asdict(derived),
            "pod_idle_seconds": spec.limits["pod_idle_seconds"],
            "mac_idle_seconds": spec.limits["mac_idle_seconds"],
            "max_transfer_seconds": spec.max_transfer_seconds,
            "required_upload_bytes_per_second": posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND,
        },
        "time_fit": {**fit, "lifetime_end_unix": end, "checked_at_unix": created_at},
    }
    bridge = spec.bridge_timeouts
    if bridge is not None:
        receipt["bridge_timeouts"] = asdict(bridge)  # Run 2's derived wait bounds.
    for gate in posttrain_ops.OWNER_GATES:
        receipt[gate] = owner.get(gate)
    # Exactly the worker's own acceptance check, so a written receipt is one it accepts.
    check_readiness(
        receipt,
        now=created_at,
        spec=spec,
        specification_sha256=receipt["specification_sha256"],
        binding_sha256=binding_sha256,
        preflight_sha256=receipt["preflight_sha256"],
        heartbeat=heartbeat,
    )
    return receipt


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.pilot_readiness_receipt posttrain",
        description="Assemble and validate one post-training run's readiness receipt.",
    )
    for name in (
        "specification",
        "binding",
        "preflight",
        "identity",
        "policy",
        "self-destruct-receipt",
        "confirmations",
        "data-fetch",
        "data-verification",
        "round-trip",
        "mac-copy",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--billing-origin-unix", type=float, required=True)
    parser.add_argument("--heartbeat", default=DEFAULT_HEARTBEAT)
    parser.add_argument("--connection", type=Path, help="default: next to --identity")
    parser.add_argument("--revision", type=Path, help="default: next to --identity")
    args = parser.parse_args(argv)
    try:
        receipt = build_receipt(
            specification=args.specification,
            binding=args.binding,
            binding_sha256=args.binding_sha256,
            preflight=args.preflight,
            identity=args.identity,
            policy=args.policy,
            self_destruct_receipt=args.self_destruct_receipt,
            confirmations=args.confirmations,
            data_fetch=args.data_fetch,
            data_verification=args.data_verification,
            round_trip=args.round_trip,
            mac_copy=args.mac_copy,
            revision=args.revision or args.identity.with_name("revision.json"),
            connection=args.connection or args.identity.with_name("pod-connection.json"),
            billing_origin_unix=args.billing_origin_unix,
            heartbeat=args.heartbeat,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"posttrain-readiness-receipt: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    atomic_json(args.output, receipt)
    fit = receipt["time_fit"]
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "run": receipt["run"],
                "remaining_seconds": round(fit["remaining_seconds"]),
                "planning_seconds_needed": round(fit["planning_seconds_needed"]),
                "fits_at_planning": fit["fits_at_planning"],
            }
        )
    )
    return 0
