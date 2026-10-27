"""The post-training readiness receipt: built from the Pod's evidence under the ceiling-derived
limits, in exactly the form the run-1 worker accepts. Authored evidence; no Pod, no R2."""

import json
import time
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.common.records import atomic_json
from tools import pilot_readiness_receipt, posttrain_ops
from tools.posttrain_approved import PARENTS, run1_specification
from tools.posttrain_binding import _serialized, binding_sha256, staged_key
from tools.posttrain_readiness import DEFAULT_HEARTBEAT, build_receipt
from tools.posttrain_run import Specification, check_readiness

POD = "podpost01"
SPEC = Path(__file__).resolve().parents[1] / "configs" / "rehearsal-a40-s-post-1.json"
NOW = time.time()
CREATED = NOW - 1_500
REVISION = {"git_commit": "ab" * 20, "clean": True, "archive_sha256": "ef" * 32}
FILES = {
    "sft-train": "train.jsonl",
    "sft-code-dev": "code_dev.jsonl",
    "sft-language-dev": "language_dev.jsonl",
    "rl-candidates-list": "rl_candidates.jsonl",
    "pool-dev": "dev-export.jsonl",
    "pool-test": "test-export.jsonl",
    "rl-candidates": "candidates-export.jsonl",
    "humaneval": "HumanEval.jsonl.gz",
    "humaneval-plus": "HumanEvalPlus-OriginFmt.jsonl.gz",
    "mbpp-plus": "MbppPlus-OriginFmt.jsonl.gz",
    "arc-easy-test": "arc-easy-test.jsonl",
}


def binding_for_record() -> dict[str, Any]:
    """A binding of the specification of record; the pool test export is pinned at the go."""
    inputs = {}
    for name, pin in run1_specification()["inputs"].items():
        sha256 = pin["sha256"] or "7" * 64
        size = pin["size_bytes"] or 4_000_000
        if pin["key"] is None:
            file = FILES[name]
            entry = {"key": staged_key(sha256, file), "path": f"{name}/{file}", "origin": "staged"}
        else:
            file = pin["key"].rsplit("/", 1)[-1]
            entry = {"key": pin["key"], "path": f"{name}/{file}", "origin": "existing"}
        inputs[name] = {
            **entry,
            "kind": pin["kind"],
            "size_bytes": size,
            "sha256": sha256,
            "source": None,
        }
    return {
        "schema_version": 1,
        "kind": "posttrain-input-binding",
        "run": "1",
        "created_at": "2026-11-01T00:00:00Z",
        "binder": {"git_commit": "1" * 40, "dirty": False},
        "specification": {"path": str(SPEC), "sha256": "a" * 64},
        "inputs": inputs,
    }


def fetch_rows(binding: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {
        "binding_sha256": binding_sha256(binding),
        "inputs": [
            {
                "name": name,
                "method": "downloaded",
                **{k: binding["inputs"][name][k] for k in ("key", "path", "size_bytes", "sha256")},
            }
            for name in names
        ],
    }


def evidence(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    """Every file the receipt tool reads, as the Pod and the runbook would write them."""
    binding = binding_for_record()
    files: dict[str, Path] = {}

    def put(name: str, value: Any) -> Path:
        path = tmp_path / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            atomic_json(path, value)
        files[name] = path
        return path

    put("binding.json", _serialized(binding))
    identity = {
        "id": POD,
        "name": "posttrain-a40-s-1",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "test",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(CREATED)),
    }
    derived = Specification.load(SPEC).ceiling
    origin = int(CREATED) - 20
    values: dict[str, Any] = {
        "identity": identity,
        "policy": posttrain_ops.watchdog_policy(derived, origin=origin, idle_baseline=NOW),
        "self-destruct.json": {
            "pod_id": POD,
            "pid": 4242,
            "state": "armed",
            "armed_at_unix": CREATED + 60,
            "checked_at_unix": NOW - 2,
            "deadline_unix": derived.deadline_unix(origin),
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
            "run": "1",
            "inputs": {name: entry["sha256"] for name, entry in binding["inputs"].items()},
        },
        "mac-copy": fetch_rows(binding, list(PARENTS)),
        "round-trip": {
            "kind": "r2-round-trip",
            "status": "passed",
            "run": "1",
            "pod_id": POD,
            "prefix": posttrain_ops.run_prefix("1", POD),
            "key": posttrain_ops.run_prefix("1", POD) + "preflight/round-trip-x.bin",
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
    for name, value in values.items():
        put(name, value)
    return {
        "specification": SPEC,
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


def test_the_receipt_is_exactly_what_the_worker_accepts(tmp_path: Path) -> None:
    arguments = evidence(tmp_path)
    receipt = build_receipt(**arguments, now=NOW)
    assert receipt["kind"] == "posttrain-readiness" and receipt["run"] == "1"
    assert all(receipt[gate] is True for gate in posttrain_ops.READINESS_GATES)
    assert receipt["self_destruct_idle"]["idle_seconds"] == 900.0
    assert receipt["limits"]["wall_seconds"] == 18_000
    assert receipt["limits"]["self_destruct_seconds"] == 17_820
    assert receipt["time_fit"]["fits_at_planning"]
    launch = check_readiness(
        json.loads(json.dumps(receipt)),
        now=NOW,
        spec=Specification.load(SPEC),
        specification_sha256=receipt["specification_sha256"],
        binding_sha256=arguments["binding_sha256"],
        preflight_sha256=receipt["preflight_sha256"],
        heartbeat=DEFAULT_HEARTBEAT,
    )
    assert launch.pod_id == POD and launch.end == arguments["billing_origin_unix"] + 17_820


def slow_upload(probe: dict[str, Any]) -> None:
    probe["steps"]["put_seconds"] = 60.0  # 2.2 MB/s: below one weights label per drain.
    probe["upload_bytes_per_second"] = probe["bytes"] / 60.0


def weaker_policy(policy: dict[str, Any]) -> None:
    policy["max_cost_usd"] = 3.5


def no_idle(timer: dict[str, Any]) -> None:
    timer.pop("idle_seconds")


def no_heartbeat(timer: dict[str, Any]) -> None:
    timer["activity_files"] = ["/workspace/posttrain/setup-progress.json"]


def late_deadline(timer: dict[str, Any]) -> None:
    timer["deadline_unix"] += 600


def missing_parent(copy: dict[str, Any]) -> None:
    copy["inputs"] = copy["inputs"][:2]


def partial_fetch(fetch: dict[str, Any]) -> None:
    fetch["inputs"] = fetch["inputs"][1:]


def no_compile(preflight: dict[str, Any]) -> None:
    preflight["compile_equivalence"] = {"status": "skipped"}


@pytest.mark.parametrize(
    "name, change, message",
    [
        ("round-trip", slow_upload, "upload rate"),
        ("policy", weaker_policy, "weaker than the ceiling-derived"),
        ("self-destruct.json", no_idle, "idle limit"),
        ("self-destruct.json", no_heartbeat, "does not watch"),
        ("self-destruct.json", late_deadline, "ceiling-derived lifetime"),
        ("mac-copy", missing_parent, "exactly"),
        ("data-fetch", partial_fetch, "exactly"),
        ("preflight", no_compile, "compile_equivalence"),
        ("pod-connection.json", {"pod_id": POD, "cost_usd_per_hour": 0.65}, "above the quoted"),
    ],
)
def test_the_receipt_tool_refuses_missing_or_weaker_evidence(
    tmp_path: Path, name: str, change: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_receipt(**evidence(tmp_path, **{name: change}), now=NOW)


def test_a_specification_other_than_the_approved_run_is_refused(tmp_path: Path) -> None:
    arguments = evidence(tmp_path)
    raw = run1_specification()
    raw["limits"]["ceiling_usd"] = 3.5
    other = tmp_path / "other.json"
    other.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="approved run 1"):
        build_receipt(**{**arguments, "specification": other}, now=NOW)


def test_the_cli_writes_the_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = evidence(tmp_path)
    output = tmp_path / "readiness-receipt.json"
    argv = ["posttrain"]
    for key in (
        "specification",
        "binding",
        "preflight",
        "identity",
        "policy",
        "self_destruct_receipt",
        "confirmations",
        "data_fetch",
        "data_verification",
        "round_trip",
        "mac_copy",
        "revision",
        "connection",
    ):
        argv += [f"--{key.replace('_', '-')}", str(arguments[key])]
    argv += [
        "--binding-sha256",
        arguments["binding_sha256"],
        "--billing-origin-unix",
        str(arguments["billing_origin_unix"]),
        "--output",
        str(output),
    ]
    assert pilot_readiness_receipt.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fits_at_planning"] is True and output.is_file()
