"""Run 2's readiness receipt: the gate-filled specification of record, the policy with its
derived ``awaiting_rewards`` extensions and the Mac's copy of run 1's SFT weights, in exactly
the form the run-2 worker accepts. Authored evidence; no Pod, no R2."""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from test_posttrain_readiness import CREATED, NOW, POD, REVISION, fetch_rows

from scratchcoder.common.records import atomic_json
from tools import posttrain_approved, posttrain_ops
from tools.posttrain_binding import _serialized, binding_sha256, staged_key
from tools.posttrain_readiness import DEFAULT_HEARTBEAT, build_receipt
from tools.posttrain_run import Specification, check_readiness

PIN = {"size_bytes": 4_000_000, "sha256": "7" * 64}
FILES = {
    "rl-tasks": "rl-tasks.jsonl",
    "rl-dev": "rl-dev.jsonl",
    "rl-facts": "rl-facts.json",
    "pool-dev": "dev-export.jsonl",
    "pool-test": "test-export.jsonl",
    "humaneval": "HumanEval.jsonl.gz",
    "humaneval-plus": "HumanEvalPlus-OriginFmt.jsonl.gz",
    "mbpp-plus": "MbppPlus-OriginFmt.jsonl.gz",
    "arc-easy-test": "arc-easy-test.jsonl",
}


def gated() -> dict[str, Any]:
    return posttrain_approved.run2_specification(
        {
            "ceiling_usd": 5.80,
            "sft-weights": {"key": f"posttrain-runs/1/{POD}/sft-weights/checkpoint.pt", **PIN},
            "rl-tasks": dict(PIN),
            "rl-dev": dict(PIN),
            "rl-facts": dict(PIN),
            "pool-test": dict(PIN),
        }
    )


def binding_of(raw: dict[str, Any], spec_path: Path) -> dict[str, Any]:
    inputs = {}
    for name, pin in raw["inputs"].items():
        if pin["key"] is None:
            file = FILES[name]
            entry = {
                "key": staged_key(pin["sha256"], file),
                "path": f"{name}/{file}",
                "origin": "staged",
            }
        else:
            file = pin["key"].rsplit("/", 1)[-1]
            entry = {"key": pin["key"], "path": f"{name}/{file}", "origin": "existing"}
        inputs[name] = {
            **entry,
            "kind": pin["kind"],
            "size_bytes": pin["size_bytes"],
            "sha256": pin["sha256"],
            "source": None,
        }
    return {
        "schema_version": 1,
        "kind": "posttrain-input-binding",
        "run": "2",
        "created_at": "2026-11-01T00:00:00Z",
        "binder": {"git_commit": "1" * 40, "dirty": False},
        "specification": {"path": str(spec_path), "sha256": "a" * 64},
        "inputs": inputs,
    }


def evidence(
    tmp_path: Path, *, raw: dict[str, Any] | None = None, **changes: Any
) -> dict[str, Any]:
    raw = gated() if raw is None else raw
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_path / "rehearsal-a40-s-post-2-gate.json"
    spec_path.write_text(json.dumps(raw, indent=2) + "\n")
    binding = binding_of(gated(), spec_path)
    spec = Specification.from_dict(gated())
    derived = spec.ceiling
    origin = int(CREATED) - 20
    values: dict[str, Any] = {
        "binding.json": _serialized(binding),
        "identity": {
            "id": POD,
            "name": "posttrain-a40-s-2",
            "gpuId": "NVIDIA A40",
            "gpuCount": 1,
            "dataCenterId": "test",
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(CREATED)),
        },
        "policy": posttrain_ops.watchdog_policy(
            derived, origin=origin, idle_baseline=NOW, bridge=spec.bridge_timeouts
        ),
        "self-destruct.json": {
            "pod_id": POD,
            "pid": 4242,
            "state": "armed",
            "armed_at_unix": CREATED + 60,
            "checked_at_unix": NOW - 2,
            "deadline_unix": derived.deadline_unix(origin),
            "dry_run": False,
            "idle_seconds": 900.0,
            "activity_files": [DEFAULT_HEARTBEAT, "/workspace/posttrain/setup-progress.json"],
        },
        "confirmations": {
            **dict.fromkeys(posttrain_ops.OWNER_GATES, True),
            "termination_guarantee": "redundant_best_effort",
            "owner_accepts_best_effort_termination": True,
        },
        "preflight": {
            "status": "completed",
            "hardware_precision_and_resume_checked": True,
            "device": "cuda:0",
            "hardware": {"device": "cuda:0", "name": "NVIDIA A40"},
            "fused_attention": {"status": "passed"},
            "compile_equivalence": {"status": "passed"},
        },
        "data-fetch": fetch_rows(binding, sorted(binding["inputs"])),
        "data-verification": {
            "binding_sha256": binding_sha256(binding),
            "run": "2",
            "inputs": {name: entry["sha256"] for name, entry in binding["inputs"].items()},
        },
        "mac-copy": fetch_rows(binding, ["sft-weights"]),
        "round-trip": {
            "kind": "r2-round-trip",
            "status": "passed",
            "run": "2",
            "pod_id": POD,
            "prefix": posttrain_ops.run_prefix("2", POD),
            "key": posttrain_ops.run_prefix("2", POD) + "preflight/round-trip-x.bin",
            "bytes": posttrain_ops.ROUND_TRIP_PROBE_BYTES,
            "steps": {
                "put_seconds": 20.0,
                "head_seconds": 0.1,
                "get_seconds": 10.0,
                "delete_seconds": 0.1,
            },
            "upload_bytes_per_second": posttrain_ops.ROUND_TRIP_PROBE_BYTES / 20.0,
            "download_bytes_per_second": posttrain_ops.ROUND_TRIP_PROBE_BYTES / 10.0,
        },
        "pod-connection.json": {"pod_id": POD, "cost_usd_per_hour": 0.59},
        "revision.json": REVISION,
    }
    for name, change in changes.items():
        if callable(change):
            change(values[name])
        else:
            values[name] = change
    files = {}
    for name, value in values.items():
        path = tmp_path / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            atomic_json(path, value)
        files[name] = path
    return {
        "specification": spec_path,
        "binding": files["binding.json"],
        "binding_sha256": binding_sha256(binding),
        "preflight": files["preflight"],
        "identity": files["identity"],
        "policy": files["policy"],
        "self_destruct_receipt": files["self-destruct.json"],
        "confirmations": files["confirmations"],
        "data_fetch": files["data-fetch"],
        "data_verification": files["data-verification"],
        "round_trip": files["round-trip"],
        "mac_copy": files["mac-copy"],
        "revision": files["revision.json"],
        "connection": files["pod-connection.json"],
        "billing_origin_unix": float(origin),
    }


def test_the_run_two_receipt_is_exactly_what_the_worker_accepts(tmp_path: Path) -> None:
    arguments = evidence(tmp_path)
    receipt = build_receipt(**arguments, now=NOW)
    assert receipt["run"] == "2" and receipt["r2_run_prefix"] == f"posttrain-runs/2/{POD}/"
    assert receipt["limits"]["wall_seconds"] == 34_800
    assert receipt["limits"]["self_destruct_seconds"] == 34_620
    assert receipt["limits"]["watchdog_cost_stop_usd"] == 5.79
    assert receipt["watchdog_policy"]["awaiting_dev_extension_seconds"] == 10_949.5
    assert receipt["bridge_timeouts"]["reward_timeout_seconds"] == 6_997.5
    assert receipt["time_fit"]["fits_at_planning"]
    launch = check_readiness(
        json.loads(json.dumps(receipt)),
        now=NOW,
        spec=Specification.load(arguments["specification"]),
        specification_sha256=receipt["specification_sha256"],
        binding_sha256=arguments["binding_sha256"],
        preflight_sha256=receipt["preflight_sha256"],
        heartbeat=DEFAULT_HEARTBEAT,
    )
    assert launch.end == arguments["billing_origin_unix"] + 34_620


def no_extension(policy: dict[str, Any]) -> None:
    policy.pop("awaiting_reward_extension_seconds")  # A run-1 policy: every wait idles out.


def longer_extension(policy: dict[str, Any]) -> None:
    policy["awaiting_dev_extension_seconds"] += 600


def parents_copied(copy: dict[str, Any]) -> None:
    copy["inputs"] = []


@pytest.mark.parametrize(
    "name, change, message",
    [
        ("policy", no_extension, "not the derived bound"),
        ("policy", longer_extension, "not the derived bound"),
        ("mac-copy", parents_copied, "exactly"),
    ],
)
def test_the_run_two_receipt_refuses_a_wrong_extension_or_a_missing_copy(
    tmp_path: Path, name: str, change: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_receipt(**evidence(tmp_path, **{name: change}), now=NOW)


def test_the_template_and_unapproved_values_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="still null"):
        build_receipt(
            **evidence(tmp_path / "a", raw=posttrain_approved.run2_specification()), now=NOW
        )
    changed = gated()
    changed["rl"]["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="approved run 2"):
        build_receipt(**evidence(tmp_path / "b", raw=changed), now=NOW)
