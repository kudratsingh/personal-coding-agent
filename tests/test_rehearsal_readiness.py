"""Rehearsal readiness receipts are built from the Pod's own evidence under the D-35 limits."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.common.records import atomic_json
from scratchcoder.data.onepass import combine_digests
from tools import rehearsal_binding
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
from tools.rehearsal_binding import PACKING, RUNS, SUBSET_RULE, VALIDATION_STREAMS, run_files

POD = "podfixture01"
CREATED = 1_800_000_000.0
CREATED_AT = "2027-01-15T08:00:00+00:00"
PREFIX = "rehearsal-s-fixture/"
INIT = "cd" * 32
STREAMS = {
    "stage1-web": ("web", "train", 3_000, 3),
    "stage2-web": ("web", "train", 1_000, 1),
    "stage2-python": ("python", "train", 2_000, 2),
    "validation-web": ("web", "validation", 500, 1),
    "validation-python": ("python", "validation", 400, 1),
}


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
        )
    }
    paths["binding"].write_bytes(data)
    binding_sha256 = hashlib.sha256(data).hexdigest()
    prefix = f"rehearsal-runs/{run}/{POD}/"
    values: dict[str, Any] = {
        "specification": {"run": run, "limits": {"pod_seconds": 45_000}},
        "preflight": {
            "status": "completed",
            "hardware_precision_and_resume_checked": True,
            "device": "cuda:0",
            "fused_attention": {"status": "passed"},
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
            "max_transfer_seconds": 0,
            "billing_started_at_unix": CREATED - 5,
            "idle_baseline_unix": CREATED + 100,
        },
        "self_destruct_receipt": {
            "pod_id": POD,
            "state": "armed",
            "armed_at_unix": CREATED + 300,
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
            "steps": {
                "put_seconds": 0.1,
                "head_seconds": 0.1,
                "get_seconds": 0.1,
                "delete_seconds": 0.1,
            },
        },
        "initialization_fetch": {
            "kind": "r2-pinned-object",
            "key": f"rehearsal-init/{INIT}/rehearsal-init.pt",
            "sha256": INIT,
            "method": "downloaded",
        },
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
    fit = receipt["time_fit"]
    assert fit["fits_at_reference"] and fit["fits_at_floor"]
    assert fit["remaining_seconds"] == pytest.approx(45_000 - 180 - 5 - 1_800)
    assert "ssh_transfer_tested" not in receipt
    assert check_rehearsal_readiness(receipt, now=CREATED + 1_801, run="A") == CREATED - 5
    with pytest.raises(ValueError, match="run B"):
        check_rehearsal_readiness(receipt, now=CREATED + 1_801, run="B")


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
        ({"policy": {"mode": "pilot", "max_wall_seconds": 5_400}}, "D-35"),
        ({"policy": {"max_cost_usd": 7.6}}, "D-35"),
        ({"policy": {"quoted_total_hourly": 0.5}}, "D-35"),
        ({"self_destruct_receipt": {"state": "dry_run_complete"}}, "not armed"),
        ({"self_destruct_receipt": {"deadline_unix": CREATED + 50_000}}, "lifetime"),
        ({"confirmations": {"owner_accepts_best_effort_termination": False}}, "acceptance"),
    ],
)
def test_receipt_is_refused_when_a_gate_or_limit_fails(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_rehearsal_receipt(**fixtures(tmp_path, **change))


def test_receipt_is_refused_when_too_little_lifetime_is_left(tmp_path: Path) -> None:
    late = fixtures(tmp_path)
    late["now"] = CREATED + 45_000 - 180 - 1_700
    with pytest.raises(ValueError, match="cannot hold the run"):
        build_rehearsal_receipt(**late)
    (tmp_path / "fresh").mkdir()
    receipt = build_rehearsal_receipt(**fixtures(tmp_path / "fresh"))
    with pytest.raises(ValueError, match="cannot hold the run"):
        check_rehearsal_readiness(receipt, now=CREATED + 45_000 - 180 - 1_700, run="A")


def test_cli_writes_nothing_when_a_gate_fails_and_the_pilot_cli_is_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = fixtures(tmp_path, confirmations={"r2_token_bucket_scoped": False})
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
    # 3,000 + 3,000 tokens: three and six 1,024-token windows, one partial update each.
    assert plans["A"].stages == (("stage1", 1), ("stage2", 1))
    assert plans["B"].stages == (("mixed", 1),)
    assert plans["A"].validation_targets == 900
    defaults = default_rehearsal_plans()
    assert defaults["A"].stages == (("stage1", 7_630), ("stage2", 7_630))
    assert defaults["B"].stages == (("mixed", 15_259),)
    assert planned_updates(1024 * 128) == 1 and planned_updates(1024 * 128 + 1) == 2
    assert json.dumps(defaults["A"].updates) == "15260"
