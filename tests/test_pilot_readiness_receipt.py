"""The readiness receipt is assembled from real artifacts and validated before writing."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.common.records import atomic_json
from scratchcoder.train.pilot import Limits
from tools.gpu_pilot import Specification
from tools.pilot_readiness_receipt import build_receipt, main


def specification() -> Specification:
    return Specification(
        RunConfig(
            model=ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=8),
            train=TrainConfig(steps=5, sequence_length=8, batch_size=2),
        ),
        Limits(60, 120, 300, 1, 1, 1, 1.2, 0.5, 60, 180, 500),
        (2, 4, 1),
        2,
        32,
        (0.0003, 0.0006, 0.0012),
        2,
        30,
        1,
        1,
    )


def fixtures(tmp_path: Path, *, armed: bool = True, accept: bool = True) -> dict[str, Any]:
    paths = {
        "specification": tmp_path / "spec.json",
        "binding": tmp_path / "binding.json",
        "preflight": tmp_path / "preflight.json",
        "identity": tmp_path / "pod-identity.json",
        "policy": tmp_path / "policy.json",
        "self_destruct_receipt": tmp_path / "self-destruct.json",
        "confirmations": tmp_path / "confirmations.json",
    }
    paths["specification"].write_text(json.dumps(asdict(specification())))
    paths["binding"].write_text("{}")
    atomic_json(paths["preflight"], {"status": "completed", "device": "cuda:0"})
    atomic_json(
        paths["identity"],
        {
            "id": "testpod",
            "name": "test",
            "gpuId": "NVIDIA A40",
            "gpuCount": 1,
            "dataCenterId": "US-XX-1",
            "createdAt": "1970-01-01T00:16:40+00:00",
        },
    )
    atomic_json(
        paths["policy"],
        {
            "max_wall_seconds": 300,
            "max_cost_usd": 1,
            "max_idle_seconds": 180,
            "quoted_total_hourly": 1,
        },
    )
    atomic_json(
        paths["self_destruct_receipt"],
        {
            "pod_id": "testpod",
            "state": "armed" if armed else "dry_run_complete",
            "armed_at_unix": 1000,
            "deadline_unix": 1290,
        },
    )
    atomic_json(
        paths["confirmations"],
        {
            "owner_confirms_four_runpod_steps": True,
            "cloud_data_pre_staged": True,
            "ssh_transfer_tested": True,
            "external_watchdog_armed": True,
            "hardware_preflight_passed": True,
            "full_console_quote_confirmed": True,
            "staging_storage_cost_bound": True,
            "termination_guarantee": "redundant_best_effort",
            "owner_accepts_best_effort_termination": accept,
        },
    )
    return paths


def test_receipt_binds_artifact_hashes_and_passes_the_gate(tmp_path: Path) -> None:
    paths = fixtures(tmp_path)
    receipt = build_receipt(**paths, billing_origin_unix=1000, now=1001)
    from scratchcoder.data.shards import file_sha256

    assert receipt["specification_sha256"] == file_sha256(paths["specification"])
    assert receipt["preflight_sha256"] == file_sha256(paths["preflight"])
    assert receipt["self_destruct_deadline_unix"] == 1290
    assert receipt["termination_guarantee"] == "redundant_best_effort"
    assert receipt["watchdog_policy"]["billing_started_at_unix"] == 1000
    assert receipt["pod_created_at_unix"] == 1000


def test_receipt_is_refused_without_an_armed_timer_or_owner_acceptance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not armed"):
        build_receipt(**fixtures(tmp_path, armed=False), billing_origin_unix=1000, now=1001)
    with pytest.raises(ValueError, match="owner acceptance"):
        build_receipt(**fixtures(tmp_path, accept=False), billing_origin_unix=1000, now=1001)


def test_cli_writes_nothing_when_a_gate_fails(tmp_path: Path) -> None:
    paths = fixtures(tmp_path, accept=False)
    output = tmp_path / "receipt.json"
    argv = [f"--{key.replace('_', '-')}={value}" for key, value in paths.items()] + [
        "--output",
        str(output),
        "--billing-origin-unix",
        "1000",
    ]
    assert main(argv) == 1
    assert not output.exists()
