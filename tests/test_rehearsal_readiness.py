"""Rehearsal readiness receipts are built from the Pod's own evidence under the D-35 limits."""

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scratchcoder.common.records import atomic_json
from scratchcoder.data.onepass import combine_digests
from scratchcoder.data.shards import file_sha256
from tools import pilot_readiness_receipt as receipt_tool
from tools import rehearsal_binding, rehearsal_ops, rehearsal_run
from tools.pilot_feasibility import (
    default_rehearsal_plans,
    planned_updates,
    rehearsal_plans_from_binding,
)
from tools.pilot_readiness_receipt import (
    build_rehearsal_receipt,
    check_rehearsal_readiness,
    main,
)
from tools.rehearsal_binding import (
    PACKING,
    RUNS,
    SUBSET_RULE,
    VALIDATION_STREAMS,
    load_binding,
    run_files,
)
from tools.rehearsal_run import (
    LAUNCH_RULE,
    Launch,
    Specification,
    binding_targets,
    check_readiness,
    planned_seconds,
)

POD = "podfixture01"
CREATED = 1_800_000_000.0
CREATED_AT = "2027-01-15T08:00:00+00:00"
PREFIX = "rehearsal-s-fixture/"
INIT = "cd" * 32
STREAMS = {
    "stage1-web": ("web", "train", 1_000_000_000, 1_200_000),
    "stage2-web": ("web", "train", 400_000_000, 480_000),
    "stage2-python": ("python", "train", 600_000_000, 700_000),
    "validation-web": ("web", "validation", 6_948_391, 7_341),
    "validation-python": ("python", "validation", 5_334_710, 4_848),
}
"""The S build's stage and validation sizes, so the checked-in specifications plan them."""
PROBE = rehearsal_ops.ROUND_TRIP_PROBE_BYTES
REVISION = {"git_commit": "ab" * 20, "clean": True, "archive_sha256": "ef" * 32}
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SPECS = {"A": CONFIGS / "rehearsal-a40-s-a.json", "B": CONFIGS / "rehearsal-a40-s-b.json"}
_A = json.loads(SPECS["A"].read_text(encoding="utf-8"))
A_LIMITS, A_R2, A_STAGES = _A["limits"], _A["r2"], _A["stages"]


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def synthetic_binding() -> dict[str, Any]:
    """A binding that passes ``check_binding``; no shard file exists behind it."""
    streams: dict[str, Any] = {}
    for name, (domain, split, tokens, documents) in STREAMS.items():
        file = f"shards/{name}/{name}-000.bin"
        streams[name] = {
            "domain": domain,
            "split": split,
            "quota_tokens": None,
            "tokens": tokens,
            "documents": documents,
            "multiset_sha256": digest("multiset " + name),
            "shards": [
                {
                    "file": file,
                    "sha256": digest(file),
                    "manifest_sha256": digest(file + ".json"),
                    "size_bytes": 2 * tokens,
                    "manifest_size_bytes": 100,
                    "documents": documents,
                    "tokens": tokens,
                    "key": PREFIX + file,
                    "manifest_key": PREFIX + file + ".json",
                }
            ],
        }
    runs: dict[str, Any] = {}
    for run, stages in RUNS.items():
        rows = []
        for stage_name, names in stages:
            shards = [shard for name in names for shard in streams[name]["shards"]]
            tokens = sum(shard["tokens"] for shard in shards)
            windows = -(-tokens // 1024)
            rows.append(
                {
                    "name": stage_name,
                    "streams": list(names),
                    "shards": [shard["file"] for shard in shards],
                    "order_seed": 17,
                    "order_sha256": digest(f"order {run} {stage_name}"),
                    "documents": sum(shard["documents"] for shard in shards),
                    "tokens": tokens,
                    "targets": tokens,
                    "windows": windows,
                    "planned_updates": -(-windows // 128),
                    "multiset_sha256": combine_digests(
                        *(streams[name]["multiset_sha256"] for name in names)
                    ),
                }
            )
        runs[run] = {
            "stages": rows,
            "documents": sum(row["documents"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
            "multiset_sha256": combine_digests(*(row["multiset_sha256"] for row in rows)),
        }
    return {
        "schema_version": 1,
        "kind": "rehearsal-data-binding",
        "created_at": "2026-10-25T00:00:00+00:00",
        "written_to": "/owner/binding.json",
        "binder": {"git_commit": "0" * 40, "dirty": False},
        "build": {
            "name": "rehearsal-s-fixture",
            "spec_sha256": digest("spec"),
            "report_sha256": digest("report"),
            "verification_sha256": digest("verification"),
            "tokens_identity": digest("tokens"),
            "build_identity": digest("build"),
            "git_commit": "1" * 40,
            "admission": {"holdout_sha256": digest("holdout")},
            "holdout_sha256": digest("holdout"),
            "documents": {"file": "documents.jsonl", "sha256": digest("documents")},
            "exclusions": {"file": "exclusions.jsonl", "sha256": digest("exclusions")},
        },
        "r2_prefix": PREFIX,
        "packing": PACKING,
        "tokenizer": {
            "path": "tokenizer/tokenizer.json",
            "source": "/owner/tokenizer.json",
            "key": PREFIX + "tokenizer/tokenizer.json",
            "sha256": digest("tokenizer file"),
            "size_bytes": 3_000_000,
            "identity": digest("tokenizer"),
            "vocab_size": 65_536,
            "bos_id": 256,
        },
        "stream_order": list(STREAMS),
        "streams": streams,
        "runs": runs,
        "same_multiset": {run: runs[run]["multiset_sha256"] for run in RUNS},
        "validation": {"streams": list(VALIDATION_STREAMS), "subset_rule": SUBSET_RULE},
    }


def fixtures(tmp_path: Path, run: str = "A", **change: Any) -> dict[str, Any]:
    binding = synthetic_binding()
    data = rehearsal_binding._serialized(binding)
    paths: dict[str, Path] = {
        name: tmp_path / f"{name}.json"
        for name in (
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
            "initialization_fetch",
            "connection",
            "revision",
        )
    }
    paths["binding"].write_bytes(data)
    binding_sha256 = hashlib.sha256(data).hexdigest()
    prefix = f"rehearsal-runs/{run}/{POD}/"
    values: dict[str, Any] = {
        "specification": json.loads(SPECS[run].read_text(encoding="utf-8")),
        "preflight": {
            "status": "completed",
            "hardware_precision_and_resume_checked": True,
            "device": "cuda:0",
            "hardware": {"device": "cuda:0", "name": "NVIDIA A40", "uuid": "GPU-fixture"},
            "fused_attention": {"status": "passed"},
            "compile_equivalence": {"status": "passed"},
        },
        "identity": {
            "id": POD,
            "name": f"rehearsal-a40-s-{run.lower()}",
            "gpuId": "NVIDIA A40",
            "gpuCount": 1,
            "dataCenterId": "CA-MTL-1",
            "createdAt": CREATED_AT,
        },
        "policy": {
            "mode": "rehearsal",
            "max_wall_seconds": 45_000,
            "max_cost_usd": 7.54,
            "max_idle_seconds": 600,
            "quoted_total_hourly": 0.6,
            "max_transfer_seconds": 900,
            "billing_started_at_unix": CREATED - 5,
            "idle_baseline_unix": CREATED + 100,
        },
        "self_destruct_receipt": {
            "pod_id": POD,
            "pid": 4242,
            "state": "armed",
            "armed_at_unix": CREATED + 300,
            "checked_at_unix": CREATED + 1_795,  # The live timer rewrites it every 5 s.
            "deadline_unix": CREATED - 5 + 45_000 - 180,
        },
        "confirmations": {
            "owner_confirms_four_runpod_steps": True,
            "cloud_data_pre_staged": True,
            "external_watchdog_armed": True,
            "hardware_preflight_passed": True,
            "full_console_quote_confirmed": True,
            "staging_storage_cost_bound": True,
            "r2_token_bucket_scoped": True,
            "termination_guarantee": "redundant_best_effort",
            "owner_accepts_best_effort_termination": True,
        },
        "data_fetch": {
            "run": run,
            "binding_sha256": binding_sha256,
            "tokens_identity": binding["build"]["tokens_identity"],
            "build_identity": binding["build"]["build_identity"],
            "files": [{**item, "method": "downloaded"} for item in run_files(binding, run)],
        },
        "data_verification": {
            "binding_sha256": binding_sha256,
            "tokenizer": binding["tokenizer"]["identity"],
            "streams": {},
            "stages": {
                run: [
                    {"name": stage["name"], "order_sha256": stage["order_sha256"]}
                    for stage in binding["runs"][run]["stages"]
                ]
            },
        },
        "round_trip": {
            "kind": "r2-round-trip",
            "status": "passed",
            "run": run,
            "pod_id": POD,
            "prefix": prefix,
            "key": prefix + "preflight/round-trip-0.bin",
            "bytes": PROBE,
            "steps": {
                "put_seconds": 10.0,
                "head_seconds": 0.1,
                "get_seconds": 5.0,
                "delete_seconds": 0.1,
            },
            "upload_bytes_per_second": PROBE / 10.0,  # 13.4 MB/s
            "download_bytes_per_second": PROBE / 5.0,
        },
        "initialization_fetch": {
            "kind": "r2-pinned-object",
            "key": f"rehearsal-init/{INIT}/rehearsal-init.pt",
            "sha256": INIT,
            "method": "downloaded",
        },
        "connection": {
            "schema_version": 1,
            "pod_id": POD,
            "status": "RUNNING",
            "cost_usd_per_hour": 0.6,
            "created_at": CREATED_AT,
        },
        "revision": dict(REVISION),
    }
    for name, update in change.items():
        if update is None:
            values[name] = {}
        else:
            values[name] = {**values[name], **update}
    for name, value in values.items():
        atomic_json(paths[name], value)
    return {
        **paths,
        "run": run,
        "binding_sha256": binding_sha256,
        "initialization_sha256": INIT,
        "billing_origin_unix": CREATED - 5,
        "now": CREATED + 1_800,
    }


def test_receipt_binds_the_pods_evidence_and_passes_the_d35_gate(tmp_path: Path) -> None:
    receipt = build_rehearsal_receipt(**fixtures(tmp_path))
    assert receipt["kind"] == "rehearsal-readiness" and receipt["run"] == "A"
    assert receipt["r2_round_trip_tested"] is True
    assert receipt["r2_shards_staged_and_hash_checked"] is True
    assert receipt["r2_token_bucket_scoped"] is True
    assert receipt["r2_run_prefix"] == f"rehearsal-runs/A/{POD}/"
    assert receipt["initialization_sha256"] == INIT
    assert receipt["watchdog_policy"]["billing_started_at_unix"] == CREATED - 5
    assert receipt["limits"]["cost_stop_usd"] == pytest.approx(7.55004)
    assert receipt["limits"]["max_transfer_seconds"] == 900
    assert receipt["watchdog_policy"]["max_transfer_seconds"] == 900
    fit = receipt["time_fit"]
    assert fit["remaining_seconds"] == pytest.approx(45_000 - 180 - 5 - 1_800)
    # The worker's launch rule, at the reference speed, from the worker's own plan.
    spec = Specification.load(SPECS["A"])
    targets, full = binding_targets(spec, synthetic_binding())
    plan = planned_seconds(spec, stage_targets=targets, full_validation_targets=full)
    assert fit["reference_seconds_needed"] == pytest.approx(plan["total_seconds"])
    assert fit["rule"] == LAUNCH_RULE and fit["fits_at_reference"]
    # 43,015 s left holds the run at 56,479 tok/s (38,107 s) but not at the floor (44,401 s).
    assert fit["fits_at_floor"] is False
    assert fit["floor_seconds_needed"] == pytest.approx(44_401, abs=1)
    assert "ssh_transfer_tested" not in receipt
    assert check_rehearsal_readiness(receipt, now=CREATED + 1_801, run="A") == CREATED - 5
    with pytest.raises(ValueError, match="run B"):
        check_rehearsal_readiness(receipt, now=CREATED + 1_801, run="B")


def test_receipt_records_the_code_revision_cost_and_upload_rate(tmp_path: Path) -> None:
    values = fixtures(tmp_path)
    receipt = build_rehearsal_receipt(**values)
    assert receipt["revision"] == REVISION  # The worker compares its own copy with this.
    assert receipt["pod_cost_usd_per_hour"] == 0.6
    for name in ("connection", "revision", "round_trip"):
        assert receipt["evidence_sha256"][name] == file_sha256(values[name])
    # 2 x 1.65 GB inside the 780 s drain with a factor-2 margin.
    required = rehearsal_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND
    assert required == pytest.approx(2 * 1.65e9 / (900 - 120) * 2)
    assert PROBE / 10.0 >= required  # The fixture's 13.4 MB/s passes.
    # The receipt is checked again before it is written, and wherever it is read.
    for change, message in (
        ({"revision": {**REVISION, "clean": False}}, "dirty"),
        ({"revision": None}, "exactly the keys"),
        ({"pod_cost_usd_per_hour": 0.62}, "captured Pod cost"),
        ({"pod_cost_usd_per_hour": None}, "captured Pod cost"),
    ):
        with pytest.raises(ValueError, match=message):
            check_rehearsal_readiness({**receipt, **change}, now=CREATED + 1_801, run="A")


def test_a_rate_at_the_requirement_passes_and_the_freshness_edge_is_inclusive(
    tmp_path: Path,
) -> None:
    seconds = PROBE / rehearsal_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND
    steps = {"put_seconds": seconds, "head_seconds": 0.1, "get_seconds": 5.0, "delete_seconds": 0.1}
    values = fixtures(
        tmp_path,
        round_trip={"steps": steps, "upload_bytes_per_second": PROBE / seconds},
        self_destruct_receipt={"checked_at_unix": CREATED + 1_800 - 60},
        connection={"cost_usd_per_hour": 0.59},
    )
    receipt = build_rehearsal_receipt(**values)
    assert receipt["r2_round_trip_tested"] is True and receipt["pod_cost_usd_per_hour"] == 0.59


def test_cli_reads_the_connection_and_revision_next_to_the_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    values = fixtures(tmp_path)
    monkeypatch.setattr(receipt_tool, "time", SimpleNamespace(time=lambda: values["now"]))
    controller = tmp_path / "ctrl"
    controller.mkdir()
    identity = controller / "pod-identity.json"
    identity.write_bytes(values["identity"].read_bytes())
    output = tmp_path / "receipt.json"
    argv = ["rehearsal", "--run", "A", "--output", str(output), "--identity", str(identity)]
    for name in (
        "specification",
        "binding",
        "preflight",
        "policy",
        "self_destruct_receipt",
        "confirmations",
        "data_fetch",
        "data_verification",
        "round_trip",
        "initialization_fetch",
        "binding_sha256",
        "initialization_sha256",
        "billing_origin_unix",
    ):
        argv += [f"--{name.replace('_', '-')}", str(values[name])]
    assert main(argv) == 1 and not output.exists()  # Neither file is next to it yet.
    assert "pod-connection.json" in capsys.readouterr().err
    (controller / "pod-connection.json").write_bytes(values["connection"].read_bytes())
    atomic_json(controller / "revision.json", {**REVISION, "clean": False})
    assert main(argv) == 1 and not output.exists()
    assert "dirty tree" in capsys.readouterr().err
    atomic_json(controller / "revision.json", REVISION)
    assert main(argv) == 0
    assert json.loads(output.read_text())["revision"] == REVISION
    # An explicit --revision wins over the default location.
    output.unlink()
    atomic_json(tmp_path / "dirty.json", {**REVISION, "clean": False})
    assert main([*argv, "--revision", str(tmp_path / "dirty.json")]) == 1
    assert not output.exists()


def test_run_b_receipt_covers_the_mixed_stage(tmp_path: Path) -> None:
    receipt = build_rehearsal_receipt(**fixtures(tmp_path, run="B"))
    assert receipt["run"] == "B" and receipt["r2_run_prefix"] == f"rehearsal-runs/B/{POD}/"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"confirmations": {"r2_token_bucket_scoped": False}}, "r2_token_bucket_scoped"),
        ({"round_trip": {"status": "failed"}}, "round trip"),
        ({"round_trip": {"prefix": "rehearsal-runs/B/podfixture01/"}}, "round trip"),
        ({"data_fetch": {"files": []}}, "pinned files"),
        ({"data_fetch": {"run": "B"}}, "another run"),
        ({"data_verification": {"stages": {"A": []}}}, "stage order"),
        ({"initialization_fetch": {"sha256": "ef" * 32}}, "initialization"),
        ({"preflight": {"fused_attention": {"status": "failed"}}}, "fused attention"),
        ({"preflight": {"compile_equivalence": {"status": "failed"}}}, "compile_equivalence"),
        ({"preflight": {"compile_equivalence": None}}, "--compile-check"),
        ({"policy": {"mode": "pilot", "max_wall_seconds": 5_400}}, "D-35"),
        ({"policy": {"max_cost_usd": 7.6}}, "D-35"),
        ({"policy": {"quoted_total_hourly": 0.5}}, "D-35"),
        ({"policy": {"max_transfer_seconds": 0}}, "max_transfer_seconds"),
        ({"policy": {"max_transfer_seconds": 1_800}}, "max_transfer_seconds"),
        ({"policy": {"billing_started_at_unix": CREATED + 30}}, "billing origin"),
        ({"specification": {"run": "B"}}, "run design"),
        ({"specification": {"limits": {**A_LIMITS, "idle_seconds": 500}}}, "D-35"),
        ({"specification": {"expected_tokens_per_second": 50_000}}, "D-35"),
        ({"specification": {"r2": {**A_R2, "max_transfer_seconds": 1_000}}}, "D-35"),
        ({"specification": {"stages": [{**A_STAGES[0], "updates": 7_631}, A_STAGES[1]]}}, "spec's"),
        ({"self_destruct_receipt": {"state": "dry_run_complete"}}, "not armed"),
        ({"self_destruct_receipt": {"deadline_unix": CREATED + 50_000}}, "lifetime"),
        ({"confirmations": {"owner_accepts_best_effort_termination": False}}, "acceptance"),
        # A timer that died after arming leaves a receipt nobody refreshes.
        ({"self_destruct_receipt": {"checked_at_unix": CREATED + 1_739}}, "live timer within"),
        ({"self_destruct_receipt": {"checked_at_unix": None}}, "live timer within"),
        ({"self_destruct_receipt": {"checked_at_unix": CREATED + 1_900}}, "inconsistent"),
        ({"self_destruct_receipt": {"checked_at_unix": CREATED + 200}}, "live timer within"),
        ({"self_destruct_receipt": {"pid": None}}, "pid"),
        ({"self_destruct_receipt": {"pid": 0}}, "pid"),
        ({"self_destruct_receipt": {"pid": True}}, "pid"),
        # The live Pod price must not exceed the quote the deadlines plan at.
        ({"connection": {"cost_usd_per_hour": 0.61}}, "above the quoted"),
        ({"connection": {"cost_usd_per_hour": None}}, "no hourly cost"),
        ({"connection": {"cost_usd_per_hour": True}}, "no hourly cost"),
        ({"connection": {"pod_id": "another-pod"}}, "another Pod"),
        # The round trip must measure throughput with a large probe, fast enough.
        ({"round_trip": {"bytes": 65_536}}, "below the 134,217,728 bytes"),
        (
            {
                "round_trip": {
                    "steps": {
                        "put_seconds": 20.0,
                        "head_seconds": 0.1,
                        "get_seconds": 5.0,
                        "delete_seconds": 0.1,
                    },
                    "upload_bytes_per_second": PROBE / 20.0,
                }
            },
            "upload rate of 6.71 MB/s is below the 8.46 MB/s",
        ),
        ({"round_trip": {"upload_bytes_per_second": None}}, "no upload_bytes_per_second"),
        ({"round_trip": {"download_bytes_per_second": 0}}, "no download_bytes_per_second"),
        ({"round_trip": {"upload_bytes_per_second": 1e9}}, "disagrees with its put_seconds"),
        # The code revision must be clean and well formed.
        ({"revision": {"clean": False}}, "dirty tree"),
        ({"revision": {"git_commit": "ab" * 10}}, "40 lower-case hex"),
        ({"revision": {"git_commit": "AB" * 20}}, "40 lower-case hex"),
        ({"revision": {"archive_sha256": "tarball"}}, "archive_sha256"),
        ({"revision": {"note": "extra"}}, "exactly the keys"),
        ({"revision": None}, "exactly the keys"),
    ],
)
def test_receipt_is_refused_when_a_gate_or_limit_fails(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_rehearsal_receipt(**fixtures(tmp_path, **change))


def test_receipt_is_refused_when_too_little_lifetime_is_left(tmp_path: Path) -> None:
    moment = CREATED + 45_000 - 180 - 1_700
    late = fixtures(tmp_path, self_destruct_receipt={"checked_at_unix": moment - 5})
    late["now"] = moment
    with pytest.raises(ValueError, match="cannot hold the run"):
        build_rehearsal_receipt(**late)
    (tmp_path / "fresh").mkdir()
    receipt = build_rehearsal_receipt(**fixtures(tmp_path / "fresh"))
    with pytest.raises(ValueError, match="cannot hold the run"):
        check_rehearsal_readiness(receipt, now=CREATED + 45_000 - 180 - 1_700, run="A")


def test_cli_writes_nothing_when_a_gate_fails_and_the_pilot_cli_is_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    values = fixtures(tmp_path, confirmations={"r2_token_bucket_scoped": False})
    monkeypatch.setattr(receipt_tool, "time", SimpleNamespace(time=lambda: values["now"]))
    output = tmp_path / "receipt.json"
    argv = ["rehearsal", "--run", "A", "--output", str(output)]
    for name in (
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
        "initialization_fetch",
        "connection",
        "revision",
        "binding_sha256",
        "initialization_sha256",
        "billing_origin_unix",
    ):
        argv += [f"--{name.replace('_', '-')}", str(values[name])]
    assert main(argv) == 1 and not output.exists()
    assert "r2_token_bucket_scoped" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--run", "A"])  # The pilot command line has no rehearsal flags.


def test_binding_plans_match_the_packing_and_the_s_build_defaults() -> None:
    plans = rehearsal_plans_from_binding(synthetic_binding())
    # 10^9 tokens per A stage: 976,563 windows of 1,024 tokens in 7,630 updates.
    assert plans["A"].stages == (("stage1", 7_630), ("stage2", 7_630))
    assert plans["B"].stages == (("mixed", 15_259),)
    assert plans["A"].validation_targets == 6_948_391 + 5_334_710
    defaults = default_rehearsal_plans()
    assert defaults["A"].stages == (("stage1", 7_630), ("stage2", 7_630))
    assert defaults["B"].stages == (("mixed", 15_259),)
    assert planned_updates(1024 * 128) == 1 and planned_updates(1024 * 128 + 1) == 2
    assert json.dumps(defaults["A"].updates) == "15260"


# The receipt tool's output, read by the worker ----------------------------------------


def worker_identity(values: dict[str, Any]) -> dict[str, str]:
    """The four SHA-256s the worker computes from its own command line."""
    return {
        "specification_sha256": file_sha256(values["specification"]),
        "binding_sha256": values["binding_sha256"],
        "preflight_sha256": file_sha256(values["preflight"]),
        "initialization_sha256": values["initialization_sha256"],
    }


@pytest.mark.parametrize("run", ["A", "B"])
def test_the_worker_accepts_exactly_the_receipt_the_tool_builds(tmp_path: Path, run: str) -> None:
    values = fixtures(tmp_path, run=run)
    path = tmp_path / "readiness-receipt.json"
    atomic_json(path, build_rehearsal_receipt(**values))  # What the receipt command writes.
    receipt = json.loads(path.read_text(encoding="utf-8"))
    spec = Specification.load(values["specification"])
    binding = load_binding(values["binding"], expected_sha256=values["binding_sha256"])
    later = values["now"] + 600  # The worker starts ten minutes after the receipt.

    def check(value: dict[str, Any], **change: Any) -> Launch:
        return check_readiness(
            value,
            now=later,
            spec=change.pop("spec", spec),
            binding=binding,
            **{**worker_identity(values), **change},
        )

    launch = check(receipt)
    # The Pod ID comes from the receipt's captured identity: no RUNPOD_POD_ID is needed.
    assert launch.pod_id == POD and receipt["r2_run_prefix"] == f"rehearsal-runs/{run}/{POD}/"
    # One billing origin, one armed deadline and one launch rule on both sides.
    assert launch.origin == CREATED - 5 == receipt["watchdog_policy"]["billing_started_at_unix"]
    assert launch.end == receipt["self_destruct_deadline_unix"]
    assert launch.end == receipt["time_fit"]["lifetime_end_unix"]
    assert launch.spec.limits.pod_seconds == launch.end - launch.origin
    assert launch.fit["rule"] == receipt["time_fit"]["rule"]
    needed = receipt["time_fit"]["reference_seconds_needed"]
    assert launch.fit["reference_seconds_needed"] == needed
    assert launch.fit["remaining_seconds"] == receipt["time_fit"]["remaining_seconds"] - 600
    # A tampered specification: the same content in other bytes is another file.
    resorted = tmp_path / "resorted-spec.json"
    resorted.write_text(json.dumps(json.loads(values["specification"].read_text()), sort_keys=True))
    assert Specification.load(resorted) == spec  # Key order never changes what it says.
    with pytest.raises(ValueError, match="specification differs"):
        check(receipt, specification_sha256=file_sha256(resorted))
    with pytest.raises(ValueError, match="binding identity differs"):
        check(receipt, binding_sha256="0" * 64)
    report = json.loads(values["preflight"].read_text())
    values["preflight"].write_text(json.dumps({**report, "note": "edited after the receipt"}))
    with pytest.raises(ValueError, match="preflight report differs"):
        check(receipt)
    with pytest.raises(ValueError, match="initialization identity differs"):
        check(receipt, preflight_sha256=receipt["preflight_sha256"], initialization_sha256="e" * 64)
    other = "B" if run == "A" else "A"
    with pytest.raises(ValueError, match=f"receipt for run {other}"):
        check(receipt, spec=Specification.load(SPECS[other]))


def test_the_worker_command_line_passes_every_gate_with_a_built_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = fixtures(tmp_path)
    receipt_path = tmp_path / "readiness-receipt.json"
    atomic_json(receipt_path, build_rehearsal_receipt(**values))
    output = tmp_path / "rehearsal-run"
    argv = ["run", "--output", str(output), "--root", str(tmp_path / "data")]
    for flag, value in (
        ("--specification", values["specification"]),
        ("--binding", values["binding"]),
        ("--binding-sha256", values["binding_sha256"]),
        ("--initialization", tmp_path / "initialization.pt"),
        ("--initialization-sha256", values["initialization_sha256"]),
        ("--readiness", receipt_path),
        ("--preflight", values["preflight"]),
    ):
        argv += [flag, str(value)]
    preflight = json.loads(values["preflight"].read_text())
    clock = SimpleNamespace(time=lambda: values["now"] + 600, monotonic=time.monotonic)
    monkeypatch.setattr(rehearsal_run, "time", clock)
    monkeypatch.setattr(rehearsal_run, "training_device", lambda name: torch.device("cpu"))
    monkeypatch.setattr(rehearsal_run, "hardware_description", lambda _: preflight["hardware"])
    for name in ("RUNPOD_POD_ID", "R2_ENDPOINT", "R2_BUCKET", *rehearsal_run.SECRET_VARIABLES):
        monkeypatch.delenv(name, raising=False)
    # The worker's launch does not source /etc/rp_environment: without RUNPOD_POD_ID it
    # takes the receipt's Pod and passes every gate, stopping first at the missing R2 key.
    with pytest.raises(ValueError, match="R2_POD_ACCESS_KEY_ID"):
        rehearsal_run.main(argv)
    monkeypatch.setenv("RUNPOD_POD_ID", POD)
    with pytest.raises(ValueError, match="R2_POD_ACCESS_KEY_ID"):
        rehearsal_run.main(argv)
    monkeypatch.setenv("RUNPOD_POD_ID", "another-pod")
    with pytest.raises(ValueError, match="not the one the receipt identifies"):
        rehearsal_run.main(argv)
    assert not output.exists()
