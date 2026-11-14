"""Admission: static gate, negative controls, reference runs, the identity-bound cache."""

import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data import admission
from scratchcoder.data.admission import Identities, Run, decide, plan, static_gate, usable
from scratchcoder.data.verifier_scan import scan_task
from scratchcoder.eval.tasks import Task

IDS = Identities("e" * 64, "python@sha256:" + "1" * 64, "sha256:" + "2" * 64)


def task(
    hidden: str,
    solution: str | None = "def f(x):\n    return x + 1\n",
    prompt: str = "Write f(x) that returns x plus one.",
    mode: str = "function",
    task_id: str = "t/1",
) -> Task:
    return Task(
        id=task_id,
        prompt=prompt,
        public_tests="assert True" if mode == "function" else hidden,
        hidden_tests=hidden,
        source="synthetic",
        license="project-authored",
        written_by="human",
        split="dev",
        mode=mode,
        solution=solution,
    )


def scan(item: Task) -> Any:
    from dataclasses import asdict

    return scan_task(asdict(item))


def outcomes(item: Task, statuses: dict[tuple[str, int], str]) -> dict[Any, Any]:
    return {
        (item.id, kind, index): {"status": status, "resolved": True}
        for (kind, index), status in statuses.items()
    }


SOUND = {
    ("reference", 0): "passed",
    ("reference", 1): "passed",
    ("reference", 2): "passed",
    ("none_stub", 0): "failed",
    ("echo_stub", 0): "failed",
}

ALL_FAIL = {
    **SOUND,
    ("reference", 0): "failed",
    ("reference", 1): "failed",
    ("reference", 2): "timeout",
}


def test_a_sound_task_is_admitted() -> None:
    item = task("assert f(1) == 2\nassert f(2) == 3\nassert f(5) == 6")
    decision = decide(item, scan(item), outcomes(item, SOUND))
    assert decision.status == "admitted" and decision.defects == ()
    assert decision.reference_statuses == ("passed",) * 3


def test_the_plan_has_three_reference_runs_and_two_controls() -> None:
    item = task("assert f(1) == 2")
    runs = plan(item)
    assert [(run.kind, run.index) for run in runs] == [
        ("reference", 0),
        ("reference", 1),
        ("reference", 2),
        ("none_stub", 0),
        ("echo_stub", 0),
    ]
    io = task('[{"input": "1\\n", "output": "2"}]', "print(int(input()) + 1)", mode="io")
    assert [run.kind for run in plan(io)][3:] == ["empty_output", "echo_input"]
    assert plan(replace(item, solution=None)) == []


def test_controls_define_every_called_candidate_name() -> None:
    item = task("import math\nassert f(g(1)) == 2\nassert Box(1).size() == 1\nassert len([1]) == 1")
    stub = admission.control_program(item, "none_stub")
    assert "def Box(" in stub and "def f(" in stub and "def g(" in stub
    assert "def len(" not in stub and "def math(" not in stub
    # Checked as text: control programs run only in the sandbox.
    tree = ast.parse(stub)
    assert all(
        isinstance(node, ast.FunctionDef) and ast.unparse(node.body[0]) == "return None"
        for node in tree.body
    )
    echo = ast.parse(admission.control_program(item, "echo_stub"))
    assert ast.unparse(echo.body[0].body[0]) == "return args[0] if args else None"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "statuses,status,defects,warnings",
    [
        (ALL_FAIL, "excluded", ("reference_failed",), ()),
        ({**SOUND, ("reference", 1): "failed"}, "excluded", ("nondeterministic_reference",), ()),
        ({**SOUND, ("reference", 2): "timeout"}, "admitted", (), ("timing_unstable_reference",)),
        ({**SOUND, ("none_stub", 0): "passed"}, "excluded", ("none_stub_passed",), ()),
        ({**SOUND, ("echo_stub", 0): "passed"}, "admitted", (), ("echo_passed",)),
    ],
)  # fmt: skip
def test_executed_findings(
    statuses: dict[Any, str], status: str, defects: tuple[str, ...], warnings: tuple[str, ...]
) -> None:
    item = task("assert f(1) == 2\nassert f(2) == 3\nassert f(5) == 6")
    decision = decide(item, scan(item), outcomes(item, statuses))
    assert (decision.status, decision.defects, decision.warnings) == (status, defects, warnings)


def test_a_missing_or_unresolved_run_keeps_the_task_pending() -> None:
    item = task("assert f(1) == 2\nassert f(2) == 3\nassert f(5) == 6")
    partial = {key: value for key, value in SOUND.items() if key != ("reference", 2)}
    decision = decide(item, scan(item), outcomes(item, partial))
    assert decision.status == "pending" and decision.pending == ("reference:2",)


def test_static_defects_and_warnings() -> None:
    none_only = task("assert f(1) is None\nassert f(2) == None")
    assert static_gate(scan(none_only))[0] == ["none_only_checks"]
    independent = task("assert abs(-1) == 1")
    assert "no_candidate_checks" in static_gate(scan(independent))[0]
    raise_prompt = task(
        "assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4",
        prompt="Write f(x); raise ValueError for negative x.",
    )
    defects, warnings = static_gate(scan(raise_prompt))
    assert defects == [] and "raise_in_prompt" in warnings


def test_prompt_required_globals_and_two_check_tasks_stay_admitted() -> None:
    named = task(
        "assert f(1) == 2\nassert LIMIT == 10",
        solution="LIMIT = 10\ndef f(x):\n    return x + 1\n",
        prompt="Define LIMIT = 10 and write f(x) that returns x plus one.",
    )
    decision = decide(named, scan(named), outcomes(named, SOUND))
    assert decision.status == "admitted"
    assert "prompt_named_global" in decision.warnings and "two_or_fewer_checks" in decision.warnings
    hidden_global = replace(named, prompt="Write f(x) that returns x plus one.")
    assert "reference_only_global" in decide(hidden_global, scan(hidden_global), {}).defects


def test_reviewed_exclusions_and_warnings_are_kept_with_their_reasons() -> None:
    item = task("assert f(1) == 2\nassert f(2) == 3\nassert f(5) == 6")
    decision = decide(
        item,
        scan(item),
        outcomes(item, SOUND),
        reviewed=("reviewed:reference_dependent_tests",),
        reviewed_warnings=("provisional:untested_specified_behaviour",),
    )
    assert decision.status == "excluded"
    assert decision.defects == ("reviewed:reference_dependent_tests",)
    assert decision.warnings == ("provisional:untested_specified_behaviour",)


def test_the_cache_binds_every_outcome_to_its_identities(tmp_path: Path) -> None:
    item = task("assert f(1) == 2")
    run = Run(item.id, "reference", 0, item.solution or "", item.hidden_tests, item.mode)
    digest = admission.task_sha256(item)
    cache = tmp_path / "cache.jsonl"
    rows = [
        admission.cache_row(
            run, digest, IDS, status="passed", resolved=True, host_failures=0, elapsed_seconds=1
        ),
        admission.cache_row(
            replace(run, index=1),
            digest,
            IDS,
            status="infrastructure_error",
            resolved=False,
            host_failures=4,
            elapsed_seconds=1,
        ),
        admission.cache_row(
            replace(run, index=2),
            digest,
            replace(IDS, evaluator_sha256="f" * 64),
            status="passed",
            resolved=True,
            host_failures=0,
            elapsed_seconds=1,
        ),
    ]
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"torn', encoding="utf-8")
    found = usable(admission.read_cache(cache), {item.id: item}, IDS)
    assert set(found) == {(item.id, "reference", 0)}  # unresolved and other-evaluator rows unused
    changed = replace(item, hidden_tests="assert f(1) == 2\nassert f(0) == 1")
    assert usable(admission.read_cache(cache), {item.id: changed}, IDS) == {}
    assert len(admission.read_cache(cache)) == 3  # every outcome stays recorded


def test_union_counts_count_tasks_not_reasons() -> None:
    item = task("assert f(1) is None")
    decisions = [
        decide(item, scan(item), {}, reviewed=("reviewed:x",)),
        decide(replace(item, id="t/2"), scan(item), {}),
    ]
    counts = admission.union_counts(decisions)
    assert counts["excluded"] == 2
    assert counts["excluded_by_reason_overlapping"] == {"none_only_checks": 2, "reviewed:x": 1}
