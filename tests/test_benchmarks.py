"""Benchmark scoring on Actions: canonical preflight, attempts, constant answers, the workflow."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from eval_fixtures import benchmark_tasks, write_benchmark_completions

from scratchcoder.data import humaneval
from scratchcoder.eval import protocols
from scratchcoder.eval.benchmarks import (
    EVALPLUS_LIMITS,
    artifact_name,
    benchmark_execution,
    release_manifest,
    results_filename,
    score_benchmark,
)
from scratchcoder.eval.completions import read_completions
from scratchcoder.sandbox.reward import FUNCTION_DEGENERATE_ANSWERS
from scratchcoder.sandbox.runner import Limits, Result, Runtime
from tools import benchmark_score
from tools.check_workflows import validate_workflow

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "benchmark-score.yml"


def fake_suite(code: str, tests: str, mode: str) -> Result:
    """Canonical and '# good' programs pass; constant answers pass only for f1 returning 2."""
    if code.rstrip().endswith("return x") or "# good" in code:
        return Result("passed")
    if "def f1(*args" in code and code.rstrip().endswith("return args[0]"):
        return Result("passed")
    return Result("failed")


def test_scoring_runs_canonical_attempts_and_constant_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)
    path, _ = write_benchmark_completions(
        tmp_path,
        tasks,
        {
            "HumanEval/0": "\n    return x  # good\n\ndef extra(): pass",
            "HumanEval/1": "\n    return None\n",
            "HumanEval/2": "\n    return -x\n",
        },
    )
    completions = read_completions(path)
    report = score_benchmark(completions, tasks, execute=fake_suite, workers=2)
    assert report["status"] == "completed" and report["canonical"]["passed"] == 3
    assert [item["status"] for item in report["attempts"]] == ["passed", "failed", "failed"]
    assert report["attempts"][0]["audit_flags"] == []
    assert (
        report["summary"]["passed_attempts"] == 1 and report["summary"]["unresolved_attempts"] == 0
    )
    baseline = report["constant_baseline"]
    by_task = {row["task_id"]: row for row in baseline["per_task"]}
    assert by_task["HumanEval/1"] == {
        "task_id": "HumanEval/1",
        "passed": True,
        "answer": "echo",
        "reason": "",
    }
    assert by_task["HumanEval/0"]["passed"] is False
    assert baseline["full_pass_share"] == pytest.approx(1 / 3) == baseline["mean_reward"]
    # The fixed values + the doctest example 1 (already a fixed value) + 5 echoes, per task.
    assert baseline["executed_calls"] == 3 * (len(FUNCTION_DEGENERATE_ANSWERS) + 5)
    assert report["completions_sha256"] == completions.sha256


def test_a_failing_canonical_preflight_blocks_model_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)
    path, _ = write_benchmark_completions(
        tmp_path, tasks, {task.id: "\n    pass\n" for task in tasks}
    )

    def broken(code: str, tests: str, mode: str) -> Result:
        return Result("failed") if "f2" in code else fake_suite(code, tests, mode)

    report = score_benchmark(read_completions(path), tasks, execute=broken)
    assert report["status"] == "canonical_failed"
    assert report["canonical"]["failed_tasks"] == ["HumanEval/2"] and "attempts" not in report


def test_host_failures_on_benchmarks_stay_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)
    path, _ = write_benchmark_completions(
        tmp_path, tasks, {task.id: "\n    return 7  # stall\n" for task in tasks}
    )

    def stalls(code: str, tests: str, mode: str) -> Result:
        if "# stall" in code:
            return Result("timeout", host_stalled=True)
        return fake_suite(code, tests, mode)

    report = score_benchmark(
        read_completions(path), tasks, execute=stalls, sleep=lambda _: None, constant_baseline=False
    )
    assert report["summary"]["unresolved_attempts"] == 3
    assert {item["status"] for item in report["attempts"]} == {"unresolved"}
    assert not any(item["passed"] for item in report["attempts"])
    assert "constant_baseline" not in report


def test_stored_base_protocol_programs_must_match_the_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)
    path, _ = write_benchmark_completions(
        tmp_path, tasks, {task.id: "\n    return x\n" for task in tasks}
    )
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["program"] = "def f0(x):\n    return x  # good"
    forged = tmp_path / "forged.jsonl"
    forged.write_text("\n".join([lines[0], json.dumps(row), *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="extraction"):
        score_benchmark(read_completions(forged), tasks, execute=fake_suite)


def test_suite_execution_settings_and_names() -> None:
    assert benchmark_execution("humaneval") == (Runtime(), Limits())
    image = "sha256:" + "a" * 64
    assert benchmark_execution("mbpp-plus", image) == (Runtime(image, True), EVALPLUS_LIMITS)
    assert EVALPLUS_LIMITS.timeout_seconds == 60 and EVALPLUS_LIMITS.memory_mb == 4096
    with pytest.raises(ValueError, match="EvalPlus image"):
        benchmark_execution("humaneval-plus")
    with pytest.raises(ValueError):
        benchmark_execution("pool")
    release_manifest("humaneval").validate()
    assert release_manifest("humaneval").sha256 == humaneval.DATA_SHA256
    assert release_manifest("mbpp-plus").size_bytes == 549938
    digest = "b" * 64
    assert artifact_name(digest) == "benchmark-results-" + digest
    assert results_filename("mbpp-plus", digest) == f"benchmark-mbpp-plus-{digest}.json"


def test_the_humaneval_release_is_pinned(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned"):
        humaneval.parse_release(b"not the release")
    path = tmp_path / "HumanEval.jsonl.gz"
    path.write_bytes(b"x" * 10)
    with pytest.raises(ValueError, match="size"):
        humaneval.read_humaneval(path)


def test_the_scoring_tool_refuses_other_bytes_and_writes_named_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = benchmark_tasks()
    monkeypatch.setattr(protocols, "read_humaneval", lambda path: tasks)
    path, release = write_benchmark_completions(
        tmp_path, tasks, {task.id: "\n    return x  # good\n" for task in tasks}
    )
    monkeypatch.setattr(
        benchmark_score,
        "run",
        lambda code, tests, limits, *, mode, runtime: fake_suite(code, tests, mode),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    arguments = [
        "--suite",
        "humaneval",
        "--completions",
        str(path),
        "--request-id",
        "f" * 32,
        "--release-path",
        str(release),
        "--output",
        str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="SHA-256"):
        benchmark_score.main([*arguments, "--sha256", "0" * 64])
    with pytest.raises(SystemExit):
        benchmark_score.main([*arguments, "--sha256", digest, "--suite", "mbpp-plus"])
    assert benchmark_score.main([*arguments, "--sha256", digest]) == 0
    saved = json.loads((tmp_path / "out" / results_filename("humaneval", digest)).read_text())
    assert saved["completions_sha256"] == digest and saved["request_id"] == "f" * 32
    assert saved["execution"]["runtime"]["site_packages"] is False
    printed = capsys.readouterr().out
    assert "BENCHMARK_SUMMARY=" in printed and "return x" not in printed


def workflow() -> dict[Any, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_workflow_passes_the_runner_policy_and_takes_the_dispatch_inputs() -> None:
    value = workflow()
    validate_workflow(value)
    trigger = value["on"] if "on" in value else value[True]
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "suite",
        "completions_url",
        "completions_sha256",
        "request_id",
        "constant_baseline",
    }
    assert inputs["suite"]["options"] == ["humaneval", "humaneval-plus", "mbpp-plus"]
    assert value["permissions"] == {"contents": "read"}
    assert "inputs.request_id" in value["run-name"]


def test_the_workflow_downloads_first_and_never_expands_inputs_in_scripts() -> None:
    steps = workflow()["jobs"]["score"]["steps"]
    first = steps[0]
    assert "curl" in first["run"] and "sha256sum --check --strict" in first["run"]
    assert "--max-redirs 0" in first["run"] and "--proto '=https'" in first["run"]
    for step in steps:
        assert "${{" not in step.get("run", ""), step.get("name")
    upload = steps[-1]
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "benchmark-results-${{ inputs.completions_sha256 }}"
    scoring = next(step for step in steps if "benchmark_score.py" in step.get("run", ""))
    assert "--sha256" in scoring["run"] and "--download" in scoring["run"]
