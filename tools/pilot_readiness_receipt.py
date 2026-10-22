"""Assemble and validate the pilot readiness receipt from real artifacts.

The receipt binds the specification, data binding and A40 preflight by
SHA-256, carries the captured Pod identity, the watchdog policy, the armed
self-destruct deadline and the owner's explicit confirmations. Building it
from files instead of by hand removes the chance that a typo passes a gate.
``check_readiness`` is applied before writing, so an unresolved gate never
produces a receipt. Nothing here contacts Runpod or starts any resource.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools.gpu_pilot import TERMINATION_GUARANTEES, Specification, check_readiness
from tools.runpod_watchdog import PodIdentity, Policy, parse_timestamp

OWNER_GATES = (
    "owner_confirms_four_runpod_steps",
    "cloud_data_pre_staged",
    "ssh_transfer_tested",
    "external_watchdog_armed",
    "hardware_preflight_passed",
    "full_console_quote_confirmed",
    "staging_storage_cost_bound",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_receipt(
    *,
    specification: Path,
    binding: Path,
    preflight: Path,
    identity: Path,
    policy: Path,
    self_destruct_receipt: Path,
    confirmations: Path,
    billing_origin_unix: float,
    now: float | None = None,
) -> dict[str, Any]:
    spec = Specification.load(specification)
    pod_identity = PodIdentity.from_dict(_load(identity))
    policy_value = _load(policy)
    Policy(**policy_value)
    timer = _load(self_destruct_receipt)
    if timer.get("pod_id") != pod_identity.pod_id or timer.get("state") != "armed":
        raise ValueError("self-destruct receipt is not armed for this Pod")
    owner = _load(confirmations)
    guarantee = owner.get("termination_guarantee")
    if guarantee not in TERMINATION_GUARANTEES:
        raise ValueError("confirmations must state termination_guarantee")
    preflight_report = _load(preflight)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "created_at_unix": now if now is not None else time.time(),
        "specification_sha256": file_sha256(specification),
        "binding_sha256": file_sha256(binding),
        "preflight_sha256": file_sha256(preflight),
        "preflight_status": preflight_report.get("status"),
        "pod_identity": _load(identity),
        "pod_created_at_unix": parse_timestamp(pod_identity.created_at),
        "watchdog_policy": {**policy_value, "billing_started_at_unix": billing_origin_unix},
        "pod_self_destruct_armed": True,
        "self_destruct_deadline_unix": timer.get("deadline_unix"),
        "self_destruct_receipt_sha256": file_sha256(self_destruct_receipt),
        "termination_guarantee": guarantee,
        "termination_evidence": (
            f"in-pod self-destruct receipt {file_sha256(self_destruct_receipt)[:16]} armed at "
            f"{timer.get('armed_at_unix')} for deadline {timer.get('deadline_unix')}; "
            "Mac watchdog policy bound above; no provider-enforced Pod lifetime exists"
        ),
        "owner_accepts_best_effort_termination": owner.get("owner_accepts_best_effort_termination"),
    }
    for gate in OWNER_GATES:
        receipt[gate] = owner.get(gate)
    if receipt["preflight_status"] != "completed":
        raise ValueError("A40 preflight report is not completed")
    check_readiness(receipt, now=receipt["created_at_unix"], spec=spec)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "specification",
        "binding",
        "preflight",
        "identity",
        "policy",
        "self-destruct-receipt",
        "confirmations",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--billing-origin-unix", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_receipt(
            specification=args.specification,
            binding=args.binding,
            preflight=args.preflight,
            identity=args.identity,
            policy=args.policy,
            self_destruct_receipt=args.self_destruct_receipt,
            confirmations=args.confirmations,
            billing_origin_unix=args.billing_origin_unix,
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"pilot-readiness-receipt: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    atomic_json(args.output, receipt)
    print(
        json.dumps({"receipt": str(args.output), "termination": receipt["termination_guarantee"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
