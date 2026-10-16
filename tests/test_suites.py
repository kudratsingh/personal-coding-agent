import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scratchcoder.data.evalplus import origin_task, release
from scratchcoder.eval.suites import evaluate_suite
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
