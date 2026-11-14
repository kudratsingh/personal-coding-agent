"""The behavioural-equivalence harness, with a fake sandbox; the real one is Docker-gated."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data import clusters
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import Result
from tools import task_equivalence as harness

IMAGE_ID = "sha256:" + "ab" * 32


def task(identifier: str, solution: str, hidden: str, public: str, split: str = "dev") -> Task:
    return Task(
        id=identifier,
        prompt="fixture prompt",
        public_tests=public,
        hidden_tests=hidden,
        source="nvidia/OpenCodeInstruct",
        license="CC-BY-4.0",
        written_by="model",
        split=split,
        solution=solution,
    )


SQUARE_A = task(
    "a",
    "def square(x):\n    return x * x\n",
    "assert square(3) == 9\nassert square(-2) == 4",
    "assert square(1) == 1",
)
SQUARE_B = task(
    "b",
    "def sq(n):\n    return n ** 2\n",
    "assert sq(4) == 16\nassert sq(0) == 0",
    "assert sq(2) == 4",
    "train",
)


def test_plan_runs_each_reference_on_the_partners_tests_with_an_alias() -> None:
    plan = harness.plan_pair(SQUARE_A, SQUARE_B)
    assert plan.reasons == []
    assert [(c.direction, c.part) for c in plan.calls] == [
        ("a_on_b", "hidden"),
        ("a_on_b", "public"),
        ("b_on_a", "hidden"),
        ("b_on_a", "public"),
        ("a_on_a", "hidden"),
        ("b_on_b", "hidden"),
    ]
    forward = plan.calls[0]
    assert forward.alias == "\nsq = square\n" and forward.code.endswith("sq = square\n")
    assert forward.tests == SQUARE_B.hidden_tests
    assert plan.checks == {"a_on_b": 2, "b_on_a": 2}


def test_plan_records_why_a_pair_cannot_be_compared() -> None:
    io = Task(
        id="io",
        prompt="p",
        public_tests=json.dumps([{"input": "1\n", "output": "1\n"}]),
        hidden_tests=json.dumps([{"input": "2\n", "output": "2\n"}]),
        source="BAAI/TACO",
        license="Apache-2.0",
        written_by="human",
        split="train",
        mode="io",
        solution="print(input())",
    )
    assert harness.plan_pair(SQUARE_A, io).reasons == ["mode_mismatch"]
    two = task(
        "two",
        "def f(x):\n    return x\n\ndef g(x):\n    return x\n",
        "assert p(1) == 1\nassert q(1) == 1",
        "assert p(1) == 1",
    )
    partner = task(
        "partner",
        "def p(x):\n    return x\n\ndef q(x):\n    return x\n",
        "assert p(1) == 1\nassert q(1) == 1",
        "assert p(2) == 2",
    )
    plan = harness.plan_pair(two, partner)
    # Two called names cannot be mapped onto two other names; the reverse needs no alias.
    assert plan.reasons == ["a_on_b:called_names_unmapped"]
    assert {c.direction for c in plan.calls} == {"b_on_a", "a_on_a", "b_on_b"}
    assert harness.check_count("assert 1\nx = 2\nassert 2", "function") == 2
    assert harness.check_count(json.dumps([{"input": "", "output": ""}] * 3), "io") == 3


@pytest.mark.parametrize(
    ("statuses", "verdict"),
    [
        ({"hidden": ["passed"] * 3, "public": ["passed"] * 3}, "pass"),
        ({"hidden": ["failed"] * 3, "public": ["timeout"] * 3}, "fail"),
        ({"hidden": ["failed"] * 3, "public": ["passed"] * 3}, "partial"),
        ({"hidden": ["passed", "failed", "passed"], "public": ["passed"] * 3}, "nondeterministic"),
    ],
)
def test_direction_verdicts(statuses: dict[str, list[str]], verdict: str) -> None:
    assert harness.direction_verdict(statuses) == verdict


def test_undecided_statuses_are_never_classified() -> None:
    with pytest.raises(ValueError):
        harness.direction_verdict({"hidden": ["infrastructure_error"]})


CHECKS = {"a_on_b": 2, "b_on_a": 2}


@pytest.mark.parametrize(
    ("verdicts", "checks", "outcome", "reasons"),
    [
        (["pass", "pass", "pass", "pass"], CHECKS, "equivalent", []),
        (["fail", "fail", "pass", "pass"], CHECKS, "not_equivalent", []),
        (["pass", "fail", "pass", "pass"], CHECKS, "unclear", ["partial_agreement_one_direction"]),
        (
            ["partial", "pass", "pass", "pass"],
            CHECKS,
            "unclear",
            ["partial_agreement_public_vs_hidden"],
        ),
        (["pass", "nondeterministic", "pass", "pass"], CHECKS, "unclear", ["nondeterministic"]),
        (["pass", "pass", "fail", "pass"], CHECKS, "unclear", ["reference_control_failed"]),
        (
            ["pass", "pass", "pass", "pass"],
            {"a_on_b": 1, "b_on_a": 2},
            "unclear",
            ["weak_tests:a_on_b"],
        ),
    ],
)
def test_outcomes(
    verdicts: list[str], checks: dict[str, int], outcome: str, reasons: list[str]
) -> None:
    names = ["a_on_b", "b_on_a", "a_on_a", "b_on_b"]
    assert harness.classify(dict(zip(names, verdicts, strict=True)), checks) == (outcome, reasons)
    assert harness.classify({}, checks, ["mode_mismatch"]) == ("unclear", ["mode_mismatch"])


def fake_sandbox(rule: Any) -> Any:
    calls: list[harness.Call] = []

    def execute(call: harness.Call) -> Result:
        calls.append(call)
        return Result("passed" if rule(call) else "failed")

    execute.calls = calls  # type: ignore[attr-defined]
    return execute


def test_execute_pair_repeats_every_call_and_classifies() -> None:
    plan = harness.plan_pair(SQUARE_A, SQUARE_B)
    execute = fake_sandbox(lambda call: True)
    row = harness.execute_pair(plan, execute, repeats=3)
    assert row["outcome"] == "equivalent" and len(execute.calls) == 18
    assert row["aliases"] == {"a_on_b": "sq = square", "b_on_a": "square = sq"}
    row = harness.execute_pair(
        plan, fake_sandbox(lambda call: call.direction in {"a_on_a", "b_on_b"})
    )
    assert row["outcome"] == "not_equivalent"
    flips = iter([True, False, True] * 10)
    row = harness.execute_pair(plan, fake_sandbox(lambda call: next(flips)))
    assert row["outcome"] == "unclear" and "nondeterministic" in row["reasons"]


def test_undecided_calls_are_retried_then_stop_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "RETRY_DELAY_SECONDS", 0.0)
    answers = iter([Result("infrastructure_error"), Result("passed")])
    assert harness.decided(lambda: next(answers)).passed
    stalled = Result("timeout", host_stalled=True)
    with pytest.raises(RuntimeError, match="nothing is scored"):
        harness.decided(lambda: stalled)


def receipt(tmp_path: Path, **changes: Any) -> Path:
    path = tmp_path / "window.json"
    record = {
        "kind": "mac-execution-window-check",
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": {"image": {"found_id": IMAGE_ID, "expected_id": IMAGE_ID}},
        "command": [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "tools.task_equivalence",
            "run",
            "--window-receipt",
            str(path),
        ],
    }
    record.update(changes)
    path.write_text(json.dumps(record))
    return path


def test_window_receipt_is_required_fresh_passing_and_for_this_command(tmp_path: Path) -> None:
    now = datetime.now(UTC) + timedelta(seconds=5)
    options: dict[str, Any] = {"image_id": IMAGE_ID, "environ": {}, "dotenv": [], "now": now}
    assert harness.require_window(receipt(tmp_path), **options)["status"] == "passed"
    refusals: list[dict[str, Any]] = [
        {"status": "failed"},
        {"kind": "other"},
        {"checks": {"image": {"found_id": "sha256:" + "cd" * 32, "expected_id": IMAGE_ID}}},
        {"command": ["python", "-m", "tools.task_equivalence", "plan"]},
        {"command": ["python", "-m", "tools.task_equivalence", "run", "--window-receipt", "x"]},
        {"checked_at": (now - timedelta(hours=1)).isoformat()},
    ]
    for change in refusals:
        with pytest.raises(PermissionError):
            harness.require_window(receipt(tmp_path, **change), **options)
    with pytest.raises(PermissionError, match="withheld"):
        harness.require_window(receipt(tmp_path), **{**options, "environ": {"SSH_AUTH_SOCK": "/x"}})
    with pytest.raises(PermissionError, match="withheld"):
        harness.require_window(
            receipt(tmp_path), **{**options, "environ": {"MY_KEY": "v"}, "dotenv": ["MY_KEY"]}
        )


def manifest_files(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    items = [
        clusters.Item("a", "dev", "nvidia/OpenCodeInstruct", "p", SQUARE_A.solution),
        clusters.Item("b", "train", "nvidia/OpenCodeInstruct", "p", SQUARE_B.solution),
    ]
    record = clusters.cluster_record(items, [0, 1], clusters.find_edges(items))
    manifest = {"format": clusters.MANIFEST_FORMAT, "clusters": [record]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    tasks = tmp_path / "pool.jsonl"
    tasks.write_text("".join(json.dumps({**task.__dict__}) + "\n" for task in (SQUARE_A, SQUARE_B)))
    return path, tasks, manifest


def test_run_refuses_without_a_window_and_never_reaches_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, tasks, _ = manifest_files(tmp_path)

    def forbidden(*_: Any, **__: Any) -> Result:
        raise AssertionError("the sandbox must not be reached")

    monkeypatch.setattr(harness, "run", forbidden)
    stale = receipt(tmp_path, checked_at="2026-01-01T00:00:00+00:00")
    arguments = [
        "run",
        "--manifest",
        str(manifest),
        "--tasks",
        str(tasks),
        "--output",
        str(tmp_path / "behavior.jsonl"),
        "--window-receipt",
        str(stale),
        "--image-id",
        IMAGE_ID,
    ]
    for name in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PermissionError):
        harness.main(arguments)
    assert not (tmp_path / "behavior.jsonl").exists()


def test_plan_command_executes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, tasks, _ = manifest_files(tmp_path)
    monkeypatch.setattr(harness, "run", lambda *_, **__: pytest.fail("executed"))
    assert harness.main(["plan", "--manifest", str(manifest), "--tasks", str(tasks)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "pairs": 1,
        "calls_per_repeat": 6,
        "sandbox_calls": 18,
        "pairs_with_precomputed_reasons": 0,
        "executed": 0,
    }


def test_merge_fills_behaviour_and_leaves_adjudication_to_a_reviewer(tmp_path: Path) -> None:
    _, _, manifest = manifest_files(tmp_path)
    row = harness.execute_pair(harness.plan_pair(SQUARE_A, SQUARE_B), fake_sandbox(lambda c: True))
    filled = harness.merge(manifest, [{"kind": "pair", **row}], "sha")
    record = manifest["clusters"][0]
    assert filled == 1 and record["behavior"][0]["outcome"] == "equivalent"
    assert record["adjudication"]["status"] == "pending"
    clusters.adjudicate(
        manifest,
        record["cluster_id"],
        status="confirmed_equivalent",
        reviewer="owner",
        decided_at="2026-11-14",
        rationale="same contract and tests pass both ways",
    )
    assert harness.merge(manifest, [{"kind": "pair", **row}], "sha") == 0


@pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires the disposable Docker sandbox; run only inside a passing Mac window",
)
def test_real_sandbox_pair() -> None:  # pragma: no cover - Docker-gated
    from scratchcoder.sandbox.runner import Runtime, run
    from tools.task_pool import limits_for

    runtime = Runtime()

    def execute(call: harness.Call) -> Result:
        limits = limits_for(harness.VERIFICATION, call.tests, call.mode)
        return run(call.code, call.tests, limits, mode=call.mode, runtime=runtime)

    plan = harness.plan_pair(SQUARE_A, SQUARE_B)
    assert harness.execute_pair(plan, execute)["outcome"] == "equivalent"
