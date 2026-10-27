"""Assemble and validate the pilot readiness receipt from real artifacts.

The receipt binds the specification, data binding and A40 preflight by
SHA-256, carries the captured Pod identity, the watchdog policy, the armed
self-destruct deadline and the owner's explicit confirmations. Building it
from files instead of by hand removes the chance that a typo passes a gate.
``check_readiness`` is applied before writing, so an unresolved gate never
produces a receipt. Nothing here contacts Runpod or starts any resource.

``rehearsal`` builds one receipt per rehearsal run (D-35) instead, in exactly the form
``tools.rehearsal_run run`` accepts. Besides the pilot's gates it requires the Pod's R2
round-trip receipt with an upload rate the transfer window can use, the fetch receipt and
re-verification of exactly the run's pinned data, the pinned shared initialization, an A40
preflight with the compile check when the specification compiles, a self-destruct receipt
the live timer rewrote within the last minute, a captured Pod price no higher than the
quote, a clean code revision (``revision.json``), and the owner's statement that the Pod's
R2 token is scoped to the one bucket. The specification is the worker's own file, checked
against the D-35 limits and the binding; the time left must pass the worker's launch rule
(``rehearsal_run.launch_fit``). ``check_rehearsal_readiness`` applies the D-35 limits
before anything is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from scratchcoder.common.config import finite_number
from scratchcoder.common.records import atomic_json
from scratchcoder.data.shards import file_sha256
from tools import rehearsal_ops
from tools.gpu_pilot import TERMINATION_GUARANTEES, Specification, check_readiness
from tools.rehearsal_binding import RUNS, load_binding, run_files
from tools.rehearsal_run import Specification as RehearsalSpecification
from tools.rehearsal_run import (
    binding_targets,
    launch_fit,
    lifetime_end,
    require_fit,
)
from tools.runpod_watchdog import CLOCK_SKEW_SECONDS, PodIdentity, Policy, parse_timestamp

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


# ---------------------------------------------------------------------------
# Rehearsal (D-35): one receipt per run, built on the Mac from the Pod's own evidence

REHEARSAL_KIND = rehearsal_ops.READINESS_KIND
REHEARSAL_OWNER_GATES = rehearsal_ops.OWNER_GATES
"""The owner's statements. A rehearsal moves checkpoints through R2, not SSH, so the
pilot's ``ssh_transfer_tested`` is replaced by the evidence gates below."""
REHEARSAL_EVIDENCE_GATES = rehearsal_ops.EVIDENCE_GATES
"""Set only from the Pod's receipts: the R2 round trip and the fetched, re-hashed data."""
SELF_DESTRUCT_FRESH_SECONDS = 60.0
"""The live timer rewrites its receipt every 5 s; one older than this at receipt time may
describe a timer that has since died, so it is refused (copy it again and retry)."""


def _number(value: object) -> float | None:
    """A finite number (never a bool), or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def self_destruct_live(timer: dict[str, Any], *, pod_id: str, now: float) -> float:
    """The armed deadline of a self-destruct receipt the live timer wrote just now.

    The timer rewrites the receipt every 5 s with its own ``pid`` and ``checked_at_unix``,
    so a receipt for this Pod in state ``armed``, naming a positive pid, checked within
    ``SELF_DESTRUCT_FRESH_SECONDS`` of ``now`` (and not after it beyond the clock skew),
    records that timer alive. A stale one, from a timer that died after arming, is refused.
    """
    if timer.get("pod_id") != pod_id or timer.get("state") != "armed":
        raise ValueError("self-destruct receipt is not armed for this Pod")
    pid = timer.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("the self-destruct receipt records no live timer pid")
    checked = _number(timer.get("checked_at_unix"))
    if checked is None or not now - SELF_DESTRUCT_FRESH_SECONDS <= checked:
        raise ValueError(
            "the self-destruct receipt was not refreshed by a live timer within the last "
            f"{SELF_DESTRUCT_FRESH_SECONDS:.0f} s; copy it from the Pod again"
        )
    armed = _number(timer.get("armed_at_unix"))
    if checked > now + CLOCK_SKEW_SECONDS or (armed is not None and checked < armed):
        raise ValueError("the self-destruct receipt's check time is inconsistent")
    return finite_number(timer.get("deadline_unix"), "self-destruct deadline")


def pod_cost_within_quote(connection: dict[str, Any], *, pod_id: str, quoted: float) -> float:
    """The captured Pod's hourly price (``pod-connection.json``), refused above ``quoted``.

    The Pod API's ``cost`` is the GPU price alone (the pilot's A40 read $0.59 while its disk
    billed separately), so ``quoted`` is the GPU quote: a GPU one cent above it, plus the
    40 GB disk, would carry a Pod past the $10 cap at its 60,000 s wall. The watchdog stops
    at ``max(API cost, total quote)``, and the worker's deadlines plan at the total quote.
    """
    if connection.get("pod_id") != pod_id:
        raise ValueError("the captured Pod connection record is for another Pod")
    cost = _number(connection.get("cost_usd_per_hour"))
    if cost is None or cost < 0:
        raise ValueError("the captured Pod connection record carries no hourly cost")
    if cost > quoted:
        raise ValueError(
            f"the Pod's captured GPU cost ${cost:.4f}/h is above the quoted ${quoted:.2f}/h"
        )
    return cost


def _rehearsal_constants() -> dict[str, Any]:
    return {
        "pod_lifetime_seconds": rehearsal_ops.POD_LIFETIME_SECONDS,
        "approved_cost_cap_usd": rehearsal_ops.APPROVED_COST_CAP_USD,
        "estimated_total_cost_usd": rehearsal_ops.ESTIMATED_TOTAL_COST_USD,
        "cost_multiplier": rehearsal_ops.COST_MULTIPLIER,
        "cost_stop_usd": rehearsal_ops.COST_STOP_USD,
        "watchdog_cost_stop_usd": rehearsal_ops.WATCHDOG_COST_STOP_USD,
        "idle_seconds": rehearsal_ops.IDLE_SECONDS,
        "max_transfer_seconds": rehearsal_ops.MAX_TRANSFER_SECONDS,
        "speed_check_seconds": rehearsal_ops.SPEED_CHECK_SECONDS,
        "reference_tokens_per_second": rehearsal_ops.REFERENCE_TOKENS_PER_SECOND,
        "minimum_speed_fraction": rehearsal_ops.MINIMUM_SPEED_FRACTION,
        "finalization_reserve_seconds": rehearsal_ops.FINALIZATION_RESERVE_SECONDS,
        "quoted_total_hourly": rehearsal_ops.QUOTED_TOTAL_HOURLY,
    }


def check_d35_specification(spec: RehearsalSpecification, *, run: str) -> None:
    """The worker's specification must be this run's and carry exactly the D-35 limits."""
    if spec.run != run:
        raise ValueError(f"the specification is run {spec.run}'s, not run {run}'s")
    limits = spec.limits
    approved = {
        "pod_seconds": (limits.pod_seconds, rehearsal_ops.POD_LIFETIME_SECONDS),
        "approved_cost_cap": (limits.approved_cost_cap, rehearsal_ops.APPROVED_COST_CAP_USD),
        "estimated_total_cost": (
            limits.estimated_total_cost,
            rehearsal_ops.ESTIMATED_TOTAL_COST_USD,
        ),
        "cost_multiplier": (limits.cost_multiplier, rehearsal_ops.COST_MULTIPLIER),
        "idle_seconds": (limits.idle_seconds, rehearsal_ops.IDLE_SECONDS),
        "speed_check_seconds": (limits.speed_check_seconds, rehearsal_ops.SPEED_CHECK_SECONDS),
        "minimum_speed_fraction": (
            limits.minimum_speed_fraction,
            rehearsal_ops.MINIMUM_SPEED_FRACTION,
        ),
        "finalization_reserve_seconds": (
            limits.finalization_reserve_seconds,
            rehearsal_ops.FINALIZATION_RESERVE_SECONDS,
        ),
        "quoted_total_hourly": (limits.quoted_total_hourly, rehearsal_ops.QUOTED_TOTAL_HOURLY),
        "expected_tokens_per_second": (
            spec.expected_tokens_per_second,
            rehearsal_ops.REFERENCE_TOKENS_PER_SECOND,
        ),
        "max_transfer_seconds": (spec.max_transfer_seconds, rehearsal_ops.MAX_TRANSFER_SECONDS),
    }
    differ = [name for name, (value, expected) in approved.items() if value != expected]
    if differ:
        raise ValueError(f"the specification's {differ} differ from the D-35 values")


def shards_staged_and_hash_checked(
    binding: dict[str, Any],
    *,
    run: str,
    binding_sha256: str,
    fetch_record: dict[str, Any],
    verification: dict[str, Any],
) -> bool:
    """True when the Pod fetched every file the run reads, pinned, and re-verified them.

    ``fetch_record`` is ``rehearsal_binding fetch``'s receipt; ``verification`` is the
    JSON that ``rehearsal_binding verify --run`` printed afterwards on the Pod.
    """
    if fetch_record.get("run") != run or fetch_record.get("binding_sha256") != binding_sha256:
        raise ValueError("the data fetch receipt is for another run or binding")
    build = binding["build"]
    if (fetch_record.get("tokens_identity"), fetch_record.get("build_identity")) != (
        build["tokens_identity"],
        build["build_identity"],
    ):
        raise ValueError("the data fetch receipt names another build")
    pinned = {
        item["path"]: (item["key"], item["size_bytes"], item["sha256"])
        for item in run_files(binding, run)
    }
    rows = fetch_record.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("the data fetch receipt lists no files")
    fetched = {
        row.get("path"): (row.get("key"), row.get("size_bytes"), row.get("sha256")) for row in rows
    }
    if fetched != pinned or any(row.get("method") not in {"downloaded", "present"} for row in rows):
        raise ValueError("the data fetch receipt does not cover exactly the run's pinned files")
    if verification.get("binding_sha256") != binding_sha256:
        raise ValueError("the data verification is for another binding")
    if verification.get("tokenizer") != binding["tokenizer"]["identity"]:
        raise ValueError("the data verification loaded another tokenizer")
    stages = verification.get("stages")
    expected = [(stage["name"], stage["order_sha256"]) for stage in binding["runs"][run]["stages"]]
    if not isinstance(stages, dict) or set(stages) != {run}:
        raise ValueError("the data verification must cover exactly this run")
    found = [(stage.get("name"), stage.get("order_sha256")) for stage in stages[run]]
    if found != expected:
        raise ValueError("the data verification's stage order differs from the binding")
    return True


def round_trip_passed(
    probe: dict[str, Any],
    *,
    run: str,
    pod_id: str,
    prefix: str | None = None,
    required: float = rehearsal_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND,
) -> bool:
    """True when the Pod's R2 round trip passed under this run's own prefix, fast enough.

    The probe must hold at least ``ROUND_TRIP_PROBE_BYTES`` and its measured upload rate
    (the bytes over the PUT's seconds) must reach ``REQUIRED_UPLOAD_BYTES_PER_SECOND``:
    slower, the final drain cannot move the stage-final milestone inside its 360 s. A
    post-training run gives its own ``prefix`` and ``required`` rate.
    """
    if prefix is None:
        prefix = rehearsal_ops.run_prefix(run, pod_id)
    if (
        probe.get("kind") != "r2-round-trip"
        or probe.get("status") != "passed"
        or probe.get("run") != run
        or probe.get("pod_id") != pod_id
        or probe.get("prefix") != prefix
        or not str(probe.get("key", "")).startswith(prefix + "preflight/")
    ):
        raise ValueError("the R2 round trip did not pass for this run's Pod and prefix")
    steps = probe.get("steps")
    if not isinstance(steps, dict) or set(steps) != {
        "put_seconds",
        "head_seconds",
        "get_seconds",
        "delete_seconds",
    }:
        raise ValueError("the R2 round trip receipt lacks a step")
    size = probe.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("the R2 round trip receipt records no probe size")
    if size < rehearsal_ops.ROUND_TRIP_PROBE_BYTES:
        raise ValueError(
            f"the R2 round trip probe of {size:,} bytes is below the "
            f"{rehearsal_ops.ROUND_TRIP_PROBE_BYTES:,} bytes that measure throughput"
        )
    rates = {}
    for name, step in (
        ("upload_bytes_per_second", "put_seconds"),
        ("download_bytes_per_second", "get_seconds"),
    ):
        rate, seconds = _number(probe.get(name)), _number(steps.get(step))
        if rate is None or seconds is None or rate <= 0 or seconds <= 0:
            raise ValueError(f"the R2 round trip receipt records no {name}")
        if not math.isclose(rate, size / seconds, rel_tol=1e-6):
            raise ValueError(f"the R2 round trip's {name} disagrees with its {step}")
        rates[name] = rate
    if rates["upload_bytes_per_second"] < required:
        raise ValueError(
            f"the Pod's measured R2 upload rate of {rates['upload_bytes_per_second'] / 1e6:.2f} "
            f"MB/s is below the {required / 1e6:.2f} MB/s the transfer window needs"
        )
    return True


def d35_lifetime_end(policy: Policy, *, origin: float, deadline: float) -> float:
    """``rehearsal_run.lifetime_end`` at the D-35 values the specification must carry."""
    return lifetime_end(
        policy,
        origin=origin,
        deadline=deadline,
        pod_seconds=rehearsal_ops.POD_LIFETIME_SECONDS,
        cost_stop_usd=rehearsal_ops.COST_STOP_USD,
        quoted_total_hourly=rehearsal_ops.QUOTED_TOTAL_HOURLY,
    )


def check_rehearsal_readiness(receipt: dict[str, Any], *, now: float, run: str) -> float:
    """Reject a rehearsal receipt with an unresolved gate or a limit already at risk.

    Returns the conservative billing origin. Every limit is checked against the D-35
    ceilings, the watchdog's transfer window must be the D-35 one, and the lifetime left
    must still pass the worker's launch rule (``rehearsal_run.require_fit``) against the
    plan the receipt recorded at the reference speed.
    """
    if receipt.get("kind") != REHEARSAL_KIND or receipt.get("run") != run:
        raise ValueError(f"not a rehearsal readiness receipt for run {run}")
    gates = (*REHEARSAL_OWNER_GATES, *REHEARSAL_EVIDENCE_GATES, "pod_self_destruct_armed")
    unresolved = [gate for gate in gates if receipt.get(gate) is not True]
    if unresolved:
        raise ValueError(f"rehearsal readiness gates are unresolved: {unresolved}")
    guarantee = receipt.get("termination_guarantee")
    if guarantee not in TERMINATION_GUARANTEES:
        raise ValueError("termination_guarantee must be provider_enforced or redundant_best_effort")
    if (
        guarantee == "redundant_best_effort"
        and receipt.get("owner_accepts_best_effort_termination") is not True
    ):
        raise ValueError("best-effort termination requires explicit owner acceptance")
    identity = PodIdentity.from_dict(receipt.get("pod_identity"))
    policy = Policy(**receipt.get("watchdog_policy", {}))
    if (
        policy.mode != "rehearsal"
        or policy.max_wall_seconds > rehearsal_ops.POD_LIFETIME_SECONDS
        or policy.max_idle_seconds > rehearsal_ops.IDLE_SECONDS
        or policy.max_cost_usd > rehearsal_ops.COST_STOP_USD
        or policy.quoted_total_hourly < rehearsal_ops.QUOTED_TOTAL_HOURLY
    ):
        raise ValueError("the watchdog policy is weaker than the D-35 limits")
    if policy.max_transfer_seconds != rehearsal_ops.MAX_TRANSFER_SECONDS:
        raise ValueError(
            "the watchdog policy's max_transfer_seconds must be the D-35 upload window of "
            f"{rehearsal_ops.MAX_TRANSFER_SECONDS} s"
        )
    if receipt.get("r2_run_prefix") != rehearsal_ops.run_prefix(run, identity.pod_id):
        raise ValueError("the receipt's R2 prefix is not this run's Pod prefix")
    rehearsal_ops.check_revision(receipt.get("revision"))  # A clean, well-formed revision.
    cost = _number(receipt.get("pod_cost_usd_per_hour"))
    if cost is None or not 0 <= cost <= rehearsal_ops.QUOTED_GPU_HOURLY:
        raise ValueError("the receipt's captured Pod cost is missing or above the D-35 GPU quote")
    for name in ("binding_sha256", "initialization_sha256", "specification_sha256"):
        value = receipt.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} is required")
    evidence = receipt.get("termination_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("termination evidence is required")
    created = parse_timestamp(identity.created_at)
    if receipt.get("pod_created_at_unix") != created or not 0 < created <= now:
        raise ValueError("pod timestamp differs from captured provider identity")
    origin = policy.billing_started_at_unix
    if origin is None or not 0 < origin <= created + CLOCK_SKEW_SECONDS:
        raise ValueError("conservative create-request billing origin is required")
    origin = min(origin, created)  # A fast Mac clock never shortens the accounted lifetime.
    deadline = finite_number(receipt.get("self_destruct_deadline_unix"), "self-destruct deadline")
    if not now < deadline <= origin + rehearsal_ops.POD_LIFETIME_SECONDS:
        raise ValueError("self-destruct deadline is outside the approved pod lifetime")
    elapsed = now - origin
    if (
        elapsed >= rehearsal_ops.POD_LIFETIME_SECONDS
        or elapsed / 3600 * rehearsal_ops.QUOTED_TOTAL_HOURLY >= rehearsal_ops.COST_STOP_USD
    ):
        raise ValueError("pod already exceeds an operating limit")
    remaining = d35_lifetime_end(policy, origin=origin, deadline=deadline) - now
    fit = receipt.get("time_fit")
    if not isinstance(fit, dict):
        raise ValueError("the receipt carries no time fit")
    needed = finite_number(fit.get("reference_seconds_needed"), "reference seconds needed")
    require_fit(remaining, needed)
    return origin


def build_rehearsal_receipt(
    *,
    run: str,
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
    initialization_fetch: Path,
    initialization_sha256: str,
    billing_origin_unix: float,
    connection: Path,
    revision: Path,
    now: float | None = None,
) -> dict[str, Any]:
    """Assemble one run's receipt from the Pod's evidence and the owner's confirmations.

    The result is exactly what ``tools.rehearsal_run run`` accepts: its kind, run, gate
    names, the four SHA-256 bindings, the Pod identity the worker takes its Pod ID from,
    the watchdog policy with the billing origin, the armed deadline and the code
    ``revision`` (``revision.json``, which the worker compares with its own copy).
    ``connection`` is the captured ``pod-connection.json``: its hourly cost, the API's GPU
    price alone, must not exceed the specification's GPU quote.
    """
    created_at = now if now is not None else time.time()
    # The worker's own file, parsed as the worker parses it and bound by its SHA-256.
    spec = RehearsalSpecification.load(specification)
    check_d35_specification(spec, run=run)
    bound = load_binding(binding, expected_sha256=binding_sha256)
    binding_targets(spec, bound)  # The specification plans the binding's stages and updates.
    pod_identity = PodIdentity.from_dict(_load(identity))
    policy_value = _load(policy)
    if policy_value.get("billing_started_at_unix") not in (None, billing_origin_unix):
        raise ValueError("the policy file's billing origin differs from --billing-origin-unix")
    watchdog = Policy(**{**policy_value, "billing_started_at_unix": billing_origin_unix})
    timer = _load(self_destruct_receipt)
    deadline = self_destruct_live(timer, pod_id=pod_identity.pod_id, now=created_at)
    cost = pod_cost_within_quote(
        _load(connection), pod_id=pod_identity.pod_id, quoted=spec.quoted_gpu_hourly
    )
    code = rehearsal_ops.check_revision(_load(revision))
    owner = _load(confirmations)
    guarantee = owner.get("termination_guarantee")
    if guarantee not in TERMINATION_GUARANTEES:
        raise ValueError("confirmations must state termination_guarantee")
    preflight_report = _load(preflight)
    fused = preflight_report.get("fused_attention")
    if (
        preflight_report.get("status") != "completed"
        or preflight_report.get("hardware_precision_and_resume_checked") is not True
        or preflight_report.get("device") != "cuda:0"
        or not isinstance(fused, dict)
        or fused.get("status") != "passed"
    ):
        raise ValueError("the A40 preflight with fused attention has not passed on cuda:0")
    compiled = preflight_report.get("compile_equivalence")
    passed = isinstance(compiled, dict) and compiled.get("status") == "passed"
    if spec.train.compile and not passed:
        raise ValueError(
            "the specification compiles, so the A40 preflight's compile_equivalence check "
            "must have passed (tools.pilot_preflight --compile-check)"
        )
    staged = shards_staged_and_hash_checked(
        bound,
        run=run,
        binding_sha256=binding_sha256,
        fetch_record=_load(data_fetch),
        verification=_load(data_verification),
    )
    probed = round_trip_passed(_load(round_trip), run=run, pod_id=pod_identity.pod_id)
    initialization = _load(initialization_fetch)
    if (
        initialization.get("kind") != "r2-pinned-object"
        or initialization.get("sha256") != initialization_sha256
        or initialization.get("method") not in {"downloaded", "present"}
    ):
        raise ValueError("the shared initialization was not fetched with its pinned SHA-256")
    origin = min(billing_origin_unix, parse_timestamp(pod_identity.created_at))
    end = d35_lifetime_end(watchdog, origin=origin, deadline=deadline)
    fit = launch_fit(spec, bound, remaining_seconds=end - created_at)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": REHEARSAL_KIND,
        "run": run,
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
        "termination_guarantee": guarantee,
        "termination_evidence": (
            f"in-pod self-destruct receipt {file_sha256(self_destruct_receipt)[:16]} armed at "
            f"{timer.get('armed_at_unix')} for deadline {deadline}; "
            "Mac watchdog policy bound above; no provider-enforced Pod lifetime exists"
        ),
        "owner_accepts_best_effort_termination": owner.get("owner_accepts_best_effort_termination"),
        "r2_run_prefix": rehearsal_ops.run_prefix(run, pod_identity.pod_id),
        "r2_round_trip_tested": probed,
        "r2_shards_staged_and_hash_checked": staged,
        "initialization_sha256": initialization_sha256,
        "initialization_key": initialization.get("key"),
        "revision": code,
        "pod_cost_usd_per_hour": cost,
        "evidence_sha256": {
            "data_fetch": file_sha256(data_fetch),
            "data_verification": file_sha256(data_verification),
            "round_trip": file_sha256(round_trip),
            "initialization_fetch": file_sha256(initialization_fetch),
            "connection": file_sha256(connection),
            "revision": file_sha256(revision),
        },
        "limits": _rehearsal_constants(),
        "time_fit": {**fit, "lifetime_end_unix": end, "checked_at_unix": created_at},
    }
    for gate in REHEARSAL_OWNER_GATES:
        receipt[gate] = owner.get(gate)
    check_rehearsal_readiness(receipt, now=created_at, run=run)
    return receipt


def rehearsal_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.pilot_readiness_receipt rehearsal",
        description="Assemble and validate one rehearsal run's readiness receipt.",
    )
    parser.add_argument("--run", choices=sorted(RUNS), required=True)
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
        "initialization-fetch",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--initialization-sha256", required=True)
    parser.add_argument("--billing-origin-unix", type=float, required=True)
    parser.add_argument(
        "--connection",
        type=Path,
        help="the captured pod-connection.json (default: next to --identity); its hourly "
        "cost must not exceed the quote",
    )
    parser.add_argument(
        "--revision",
        type=Path,
        help="the Mac's revision.json for the shipped code (default: next to --identity); "
        "a dirty tree is refused",
    )
    args = parser.parse_args(argv)
    try:
        receipt = build_rehearsal_receipt(
            run=args.run,
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
            initialization_fetch=args.initialization_fetch,
            initialization_sha256=args.initialization_sha256,
            billing_origin_unix=args.billing_origin_unix,
            # runpod_pod_identity capture writes both next to each other, as does the runbook.
            connection=args.connection or args.identity.with_name("pod-connection.json"),
            revision=args.revision or args.identity.with_name("revision.json"),
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"rehearsal-readiness-receipt: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    atomic_json(args.output, receipt)
    fit = receipt["time_fit"]
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "run": args.run,
                "termination": receipt["termination_guarantee"],
                "remaining_seconds": round(fit["remaining_seconds"]),
                "fits_at_reference": fit["fits_at_reference"],
                "fits_at_floor": fit["fits_at_floor"],
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["rehearsal"]:
        return rehearsal_main(arguments[1:])
    if arguments[:1] == ["posttrain"]:
        from tools.posttrain_readiness import main as posttrain_main

        return posttrain_main(arguments[1:])
    argv = arguments
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
