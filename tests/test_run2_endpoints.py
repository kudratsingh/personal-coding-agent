"""Run 2's endpoints on the Mac: the plan, its lists, the unseal record and the scorecard's
inputs, on authored reports. A reused arm is checked against the specification's pins, run 1's
verified report and the Mac's own copies; nothing contacts R2."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.scorecard import load_inputs
from scratchcoder.eval.seal import UnsealRecord
from tools import posttrain_approved, posttrain_endpoints
from tools.posttrain_run import SFT_RESULT_FILES, SFT_RESULTS

RUN1, RUN2 = "podpost01", "podpost02"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def row(run: str, pod: str, name: str, sha256: str, size: int = 10) -> dict[str, Any]:
    label, file = name.split("/")
    return {
        "label": label,
        "name": file,
        "key": f"posttrain-runs/{run}/{pod}/{name}",
        "sha256": sha256,
        "bytes": size,
    }


def fixtures(root: Path, *, rl: str = "reuse") -> dict[str, Path]:
    """Run 1's fetched SFT artifacts, the launch specification and both verified reports.
    ``rl`` is RL's evaluation: ``reuse`` (its selected weights were the SFT weights), ``pod``
    or ``none``; the control was evaluated on the Pod."""
    artifacts = root / "run1-artifacts"
    pins: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(SFT_RESULTS):
        path = artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"run 1 artifact {index}\n")
        pins[name] = {
            "key": f"posttrain-runs/1/{RUN1}/{name}",
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    pin = {"size_bytes": 10, "sha256": "a" * 64}
    raw = posttrain_approved.run2_specification(
        {
            "ceiling_usd": 5.80,
            "gate_estimate_usd": 3.53,
            "sft-weights": {"key": f"posttrain-runs/1/{RUN1}/sft-weights/checkpoint.pt", **pin},
            "rl-tasks": dict(pin),
            "rl-dev": dict(pin),
            "rl-facts": dict(pin),
            "pool-test": dict(pin),
            "sft-results": pins,
        }
    )
    run1_rows = [
        row("1", RUN1, f"eval-{stage}-{name}/completions.jsonl", digest(f"{stage}-{name}"))
        for stage in ("base", "sft")
        for name in ("pool-test-greedy", "pool-test-n8")
    ]
    run1_rows = [r for r in run1_rows if r["label"] not in {n.split("/")[0] for n in SFT_RESULTS}]
    run1_rows += [
        row("1", RUN1, name, pins[name]["sha256"], pins[name]["size_bytes"]) for name in SFT_RESULTS
    ]
    run1 = {
        "kind": "posttrain-run-report",
        "run": "1",
        "pod_id": RUN1,
        "status": "completed",
        "inputs": {"b-mixed-final": "1" * 64},
        "sft": {"selected": {"sha256": "2" * 64}},
        "artifacts": run1_rows,
    }
    reused = {
        "evaluation": "reuses-run-1-sft",
        "selected": "rl-check-0000",
        "checkpoint_sha256": "3" * 64,
        "reason": "rl-check-0000 holds the SFT weights bit for bit",
        "reuses": {
            "endpoint": "sft",
            "results": {
                "arc-post-rl/arc.json": {"run_1": "arc-sft-sft/arc.json", **pins[SFT_RESULTS[0]]},
                **{
                    f"eval-rl-{name}/completions.jsonl": {
                        "run_1": f"eval-sft-{name}/completions.jsonl",
                        **pins[f"eval-sft-{name}/completions.jsonl"],
                    }
                    for name in SFT_RESULT_FILES
                },
            },
        },
    }
    evaluated = {"rl": rl == "pod", "more-sft": True}
    endpoints: dict[str, Any] = {"rule": "...", "more-sft": None, "rl": None}
    for arm, sha in (("rl", "4" * 64), ("more-sft", "5" * 64)):
        endpoints[arm] = {"evaluation": "pod", "selected": f"{arm}-check", "checkpoint_sha256": sha}
    if rl == "reuse":
        endpoints["rl"] = reused
    elif rl == "none":
        endpoints["rl"] = {"evaluation": "not-evaluated", "reason": "no RL"}
    run2_rows = [
        row("2", RUN2, f"eval-{arm}-{name}/completions.jsonl", digest(f"{arm}-{name}"))
        for arm, on in evaluated.items()
        if on
        for name in SFT_RESULT_FILES
    ]
    run2_rows += [
        row("2", RUN2, f"arc-post-{arm}/arc.json", digest(f"arc-{arm}"))
        for arm, on in evaluated.items()
        if on
    ]
    run2 = {
        "kind": "posttrain-run-report",
        "run": "2",
        "pod_id": RUN2,
        "status": "completed",
        "specification": raw,
        "selection": {"endpoints": endpoints},
        "artifacts": run2_rows,
    }
    paths = {
        "specification": root / "spec-2.json",
        "run1_report": root / "r1.json",
        "run2_report": root / "r2.json",
        "run1_artifacts": artifacts,
    }
    paths["specification"].write_text(json.dumps(raw, indent=2) + "\n")
    paths["run1_report"].write_text(json.dumps(run1))
    paths["run2_report"].write_text(json.dumps(run2))
    return paths


def test_a_reused_arm_is_planned_from_the_pinned_run_one_results(tmp_path: Path) -> None:
    value = posttrain_endpoints.plan(**fixtures(tmp_path))
    endpoints = value["endpoints"]
    assert {name: entry["source"] for name, entry in endpoints.items()} == {
        "base": "run 1",
        "sft": "run 1",
        "rl": "reuses-run-1-sft",
        "more-sft": "pod",
    }
    assert endpoints["rl"]["independent"] is False and len(endpoints["rl"]["checked"]) == 8
    # Only the arm evaluated on the Pod has run 2 files to execute, seal or score.
    assert value["lists"] == {
        "run2-pool-dev": ["eval-more-sft-pool-dev-greedy", "eval-more-sft-pool-dev-n8"],
        "run2-pool-test": ["eval-more-sft-pool-test-greedy", "eval-more-sft-pool-test-n8"],
        "run2-benchmarks": [
            "eval-more-sft-humaneval-greedy",
            "eval-more-sft-humaneval-plus-greedy",
            "eval-more-sft-mbpp-plus-greedy",
        ],
    }
    record = posttrain_endpoints.unseal_record(value)
    assert record["endpoints"]["rl"] == {
        "reuses": "sft",
        "selected_checkpoint_sha256": "3" * 64,
    }
    path = tmp_path / "unseal.json"
    path.write_text(json.dumps(record))
    loaded = UnsealRecord.load(path)
    assert [endpoint.name for endpoint in loaded.endpoints] == ["base", "sft", "more-sft"]
    assert loaded.reused_by("sft") == ["rl"]


@pytest.mark.parametrize("rl, executing", [("pod", 4), ("none", 3)])
def test_an_arm_on_the_pod_or_never_evaluated(tmp_path: Path, rl: str, executing: int) -> None:
    value = posttrain_endpoints.plan(**fixtures(tmp_path, rl=rl))
    record = posttrain_endpoints.unseal_record(value)
    path = tmp_path / "unseal.json"
    path.write_text(json.dumps(record))
    assert len(UnsealRecord.load(path).endpoints) == executing
    if rl == "none":
        assert record["endpoints"]["rl"] == {"not_evaluated": "no RL"}
        assert not any(label.startswith("eval-rl") for label in value["lists"]["run2-pool-dev"])


def test_an_arm_whose_evaluation_never_reached_r2_is_not_evaluated(tmp_path: Path) -> None:
    """A later stop (the projected finish, the reserve) can come between an arm's selection for
    the Pod and its evaluation: none of it is executed, and the record says why."""
    paths = fixtures(tmp_path, rl="pod")
    report = json.loads(paths["run2_report"].read_text())
    report["artifacts"] = [
        row for row in report["artifacts"] if not row["label"].startswith("eval-rl-")
    ]
    paths["run2_report"].write_text(json.dumps(report))
    value = posttrain_endpoints.plan(**paths)
    assert value["endpoints"]["rl"]["source"] == "not-evaluated"
    assert (
        "7 of its 8 evaluation artifacts are not in run 2's report"
        in value["endpoints"]["rl"]["reason"]
    )
    assert not any(label.startswith("eval-rl") for label in value["lists"]["run2-pool-test"])


def test_a_reuse_is_refused_unless_the_pins_reports_and_copies_agree(tmp_path: Path) -> None:
    paths = fixtures(tmp_path)
    copy = paths["run1_artifacts"] / "eval-sft-pool-dev-n8" / "completions.jsonl"
    original = copy.read_text()
    copy.write_text(original.upper())  # The same size, other bytes.
    with pytest.raises(ValueError, match="another SHA-256"):
        posttrain_endpoints.plan(**paths)
    copy.write_text(original)
    report = json.loads(paths["run2_report"].read_text())
    results = report["selection"]["endpoints"]["rl"]["reuses"]["results"]
    results["eval-rl-pool-dev-n8/completions.jsonl"]["sha256"] = "f" * 64
    paths["run2_report"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="another pin than sft_results"):
        posttrain_endpoints.plan(**paths)
    other = fixtures(tmp_path / "other")
    run1 = json.loads(other["run1_report"].read_text())
    for item in run1["artifacts"]:
        if item["label"] == "arc-sft-sft":
            item["sha256"] = "e" * 64
    other["run1_report"].write_text(json.dumps(run1))
    with pytest.raises(ValueError, match="not the artifact run 1's report lists"):
        posttrain_endpoints.plan(**other)
    spec = json.loads(other["specification"].read_text())
    spec["limits"]["gate_estimate_usd"] = None
    other["specification"].write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="filled launch specification"):
        posttrain_endpoints.plan(**other)


def test_the_scorecard_inputs_record_the_reuse_and_every_input(tmp_path: Path) -> None:
    value = posttrain_endpoints.plan(**fixtures(tmp_path))
    run1_shared, shared, artifacts = tmp_path / "run1", tmp_path / "run2", tmp_path / "art"
    for root, stage in (
        (run1_shared, "eval-base"),
        (run1_shared, "eval-sft"),
        (shared, "eval-more-sft"),
    ):
        for name in SFT_RESULT_FILES:
            if name.startswith("pool-"):
                path = root / "results" / f"{stage}-{name}.jsonl"
            else:
                path = root / "benchmarks-results" / f"{stage}-{name}" / "benchmark-x-0.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    for path in (
        run1_shared / "artifacts" / "arc-base-b-mixed-final" / "arc.json",
        run1_shared / "artifacts" / "arc-sft-sft" / "arc.json",
        artifacts / "arc-post-more-sft" / "arc.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    inputs = posttrain_endpoints.scorecard_inputs(
        value, run1_shared=run1_shared, shared=shared, artifacts=artifacts
    )
    assert set(inputs["checkpoints"]) == {"base", "sft", "more-sft"}
    assert inputs["reuses"]["rl"]["endpoint"] == "sft"
    assert (
        len(inputs["results"]) == 12 and len(inputs["benchmarks"]) == 9 and len(inputs["arc"]) == 3
    )
    assert not any("eval-rl" in path for path in inputs["results"] + inputs["benchmarks"])
    written = tmp_path / "scorecard-inputs.json"
    written.write_text(json.dumps(inputs))
    loaded = load_inputs(written)
    assert loaded.reuses == {"rl": "sft"} and loaded.checkpoints["more-sft"] == "5" * 64
    (run1_shared / "artifacts" / "arc-sft-sft" / "arc.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        posttrain_endpoints.scorecard_inputs(
            value, run1_shared=run1_shared, shared=shared, artifacts=artifacts
        )


def test_the_command_line_writes_once_and_lists_for_the_runbooks_loops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = fixtures(tmp_path)
    output = tmp_path / "endpoints.json"
    argv = ["plan", *(f"--{key.replace('_', '-')}={path}" for key, path in paths.items())]
    assert posttrain_endpoints.main([*argv, f"--output={output}"]) == 0
    assert json.loads(capsys.readouterr().out)["rl"] == "reuses-run-1-sft (sft)"
    assert posttrain_endpoints.main([*argv, f"--output={output}"]) == 1  # Never rewritten.
    capsys.readouterr()
    assert (
        posttrain_endpoints.main(["list", "--plan", str(output), "--list", "run2-pool-test"]) == 0
    )
    assert capsys.readouterr().out.split() == [
        "eval-more-sft-pool-test-greedy",
        "eval-more-sft-pool-test-n8",
    ]
    assert posttrain_endpoints.main(["list", "--plan", str(output), "--list", "other"]) == 1
    unseal = tmp_path / "unseal.json"
    assert posttrain_endpoints.main(["unseal", "--plan", str(output), "--output", str(unseal)]) == 0
    assert UnsealRecord.load(unseal).reused_by("sft") == ["rl"]
    assert posttrain_endpoints.main(["unseal", "--plan", str(output), "--output", str(unseal)]) == 1
