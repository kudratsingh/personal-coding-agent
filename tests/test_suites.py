import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scratchcoder.data.evalplus import origin_task, release
from scratchcoder.eval import suites
from scratchcoder.eval.suites import evaluate_suite
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Limits, Result, Runtime
from tools import evalplus_check


def test_canonical_suite_preserves_oracles_and_records_profile(tmp_path: Path) -> None:
    spec = replace(release("humaneval-plus"), task_count=1)
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a,b):\n",
        "canonical_solution": "    return a+b\n",
        "entry_point": "add",
        "test": "def check(candidate): assert candidate(2,3)==5",
    }
    task = origin_task(row, spec)

    def execute(code: str, tests: str, mode: str) -> Result:
        assert code == task.solution and tests == task.hidden_tests and mode == "function"
        return Result("passed")

    profile = Runtime("sha256:" + "a" * 64, True)
    reports = evaluate_suite(
        [task], spec, execute=execute, runtime=profile, limits=Limits(), output=tmp_path / "run"
    )
    report = reports["canonical"]
    assert report["metrics"]["greedy_pass_at_1"] == 1
    assert report["benchmark"]["runtime"]["image"] == profile.image
    assert report["benchmark"]["upstream_resource_policy_equivalent"] is False
    assert row["test"] not in json.dumps(report)
    with pytest.raises(RuntimeError, match="blocked"):
        evaluate_suite(
            [task],
            spec,
            execute=lambda *_: Result("failed", "TypeError: private-test-secret"),
            runtime=profile,
            limits=Limits(),
            output=tmp_path / "failed",
            checkpoint=tmp_path / "never-opened.pt",
            tokenizer_path=tmp_path / "never-opened.json",
        )
    assert (tmp_path / "failed/canonical/report.json").exists()
    failure = json.loads((tmp_path / "failed/canonical/report.json").read_text())
    assert failure["canonical_diagnostics"] == {"humaneval/0": "TypeError"}
    assert "private-test-secret" not in json.dumps(failure)
    assert not (tmp_path / "failed/checkpoint").exists()


def test_evalplus_command_routes_explicit_limits_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = replace(release("mbpp-plus"), task_count=1)
    task = origin_task(
        {
            "task_id": 2,
            "code": "def identity(x):return x",
            "prompt": "Return x",
            "source_file": "fixture",
            "test_imports": [],
            "test_list": ["assert identity(1)==1"],
            "test": "assert identity(2)==2",
        },
        spec,
    )
    monkeypatch.setattr(evalplus_check, "release", lambda _: spec)
    monkeypatch.setattr(evalplus_check, "read_release", lambda *_: [task])

    def execute(code: str, tests: str, limits: Limits, *, mode: str, runtime: Runtime) -> Result:
        assert mode == "function" and limits.input_bytes == 2 * 1024 * 1024
        assert runtime.site_packages
        return Result("passed")

    monkeypatch.setattr(evalplus_check, "run", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evalplus_check",
            "--suite",
            "mbpp-plus",
            "--release-path",
            str(tmp_path / "fixture.gz"),
            "--image-id",
            "sha256:" + "a" * 64,
            "--output",
            str(tmp_path / "evaluation"),
        ],
    )
    assert evalplus_check.main() == 0


def test_evalplus_command_rejects_correction_for_wrong_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evalplus_check",
            "--suite",
            "mbpp-plus",
            "--release-path",
            str(tmp_path / "must-not-open.gz"),
            "--image-id",
            "sha256:" + "a" * 64,
            "--output",
            str(tmp_path / "must-not-create"),
            "--correct-find-zero-oracle",
        ],
    )
    with pytest.raises(SystemExit) as error:
        evalplus_check.main()
    assert error.value.code == 2
    assert not (tmp_path / "must-not-create").exists()


def test_task32_excluded_before_both_stages_and_disclosed_in_every_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = release("humaneval-plus")
    tasks = [
        origin_task(
            {
                "task_id": f"HumanEval/{index}",
                "prompt": f"def fixture_{index}():\n",
                "canonical_solution": "    return 42\n",
                "entry_point": f"fixture_{index}",
                "test": f"def check(candidate): assert candidate() == {index}\n",
            },
            spec,
        )
        for index in range(164)
    ]
    by_tests = {task.hidden_tests: task for task in tasks}
    executed = []
    generated = []

    def execute(code: str, tests: str, mode: str) -> Result:
        task = by_tests[tests]
        assert task.id != "humaneval/32"
        assert code == task.solution and mode == task.mode
        executed.append(task.id)
        return Result("passed")

    def generate(task: Task, seed: int) -> str:
        assert task.id != "humaneval/32" and task.solution is not None
        generated.append(task.id)
        return task.solution

    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"authored fixture only")
    monkeypatch.setattr(
        suites, "load_checkpoint", lambda *_: (None, SimpleNamespace(sha256="fixture"))
    )
    monkeypatch.setattr(suites, "checkpoint_generator", lambda *_: generate)
    output = tmp_path / "excluded"
    reports = evaluate_suite(
        tasks,
        spec,
        execute=execute,
        runtime=Runtime(),
        limits=Limits(),
        output=output,
        checkpoint=checkpoint,
        tokenizer_path=tmp_path / "fixture.json",
        exclude_find_zero=True,
    )
    assert len(executed) == 326 and len(generated) == 163
    expected_ids = {task.id for task in tasks} - {"humaneval/32"}
    scorecards = [
        json.loads(line) for line in (output / "scorecard.jsonl").read_text().splitlines()
    ]
    for report in [*reports.values(), *scorecards]:
        assert report["task_count"] == 163
        assert report["metrics"]["greedy_pass_at_1"] == 1
        metadata = report["benchmark"]
        assert metadata["original_task_count"] == 164 and metadata["scored_task_count"] == 163
        assert metadata["excluded_tasks"] == [
            {"task_id": "humaneval/32", "reason": "known numerical-tolerance failure"}
        ]
        assert metadata["evaluation_scope"].endswith("not the full suite")
        assert metadata["oracle_corrections"] == []
        assert metadata["oracle_policy"] == "unmodified-OriginFmt-tests"
    for name, report in reports.items():
        assert {row["id"] for row in report["per_task"]} == expected_ids
        assert json.loads((output / name / "report.json").read_text()) == report


@pytest.mark.parametrize("suite,correction", [("mbpp-plus", False), ("humaneval-plus", True)])
def test_task32_exclusion_rejects_wrong_suite_or_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: str, correction: bool
) -> None:
    argv = [
        "evalplus_check",
        "--suite",
        suite,
        "--release-path",
        str(tmp_path / "must-not-open.gz"),
        "--image-id",
        "sha256:" + "a" * 64,
        "--output",
        str(tmp_path / "must-not-create"),
        "--exclude-find-zero",
    ]
    if correction:
        argv.append("--correct-find-zero-oracle")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        evalplus_check.main()
    assert error.value.code == 2
    assert not (tmp_path / "must-not-create").exists()
