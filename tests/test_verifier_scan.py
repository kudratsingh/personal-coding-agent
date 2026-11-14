"""Static verifier scan: one hit and one honest non-hit per rule, the CLI and its sealed guard.

Every fixture is synthetic text written for these tests. The scan parses text only; nothing
here runs a fixture.
"""

import ast
import builtins
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data import verifier_scan
from scratchcoder.data.verifier_scan import (
    RULES,
    RULESET_VERSION,
    TaskScan,
    few_declared_checks,
    scan_task,
    summarize,
)
from scratchcoder.sandbox import reward
from tools import verifier_scan as cli


def task(
    prompt: str,
    tests: str,
    solution: str | None = "def f(x):\n    return x\n",
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "fixture/1",
        "prompt": prompt,
        "hidden_tests": tests,
        "mode": "function",
        "solution": solution,
    }
    record.update(extra)
    return record


def hits(record: dict[str, Any]) -> set[tuple[str, str | None]]:
    return {(hit.rule, hit.tag) for hit in scan_task(record).hits}


def rules(record: dict[str, Any]) -> set[str]:
    return {rule for rule, _ in hits(record)}


# ---- R1 and R2 ----------------------------------------------------------------------


def test_r1_counts_declared_checks_and_reports_total() -> None:
    record = task("p", "assert f(1) == 1\nassert f(2) == 2\nassert f(3) == 3\n")
    scan = scan_task(record)
    assert (scan.declared_checks, scan.total_asserts) == (3, 3)
    assert not few_declared_checks(scan)
    assert few_declared_checks(scan_task(task("p", "assert f(1) == 1\n")))
    assert few_declared_checks(scan_task(task("p", "assert True is True\nassert f(1) == 1\n")))


def test_r1_ignores_imports_as_reference_names() -> None:
    record = task(
        "p", "import math\nassert math.floor(2.5) == 2\n", "import math\ndef f(x): return x\n"
    )
    assert scan_task(record).declared_checks == 0


def test_r2_candidate_independent_assert() -> None:
    assert ("R2", None) in hits(
        task("p", "import math\nassert math.sqrt(4) == 2\nassert f(1) == 1\n")
    )
    assert "R2" not in rules(task("p", "assert f(1) == 1\nassert f(2) == 2\n"))


def test_r1_r2_fall_back_to_entry_point_without_reference() -> None:
    scan = scan_task(task("p", "assert f(1) == 1\n", None, entry_point="f"))
    assert scan.declared_checks == 1
    assert "reference_missing" in scan.notes


# ---- R3 and R4 ----------------------------------------------------------------------


def test_r3_none_only_and_r4_side_effect_prompt() -> None:
    tests = "assert f(1) is None\nassert None == f(2)\nassert f(3) == None\n"
    assert rules(task("Compute and return the value.", tests)) >= {"R3"}
    assert "R4" not in rules(task("Compute and return the value.", tests))
    found = rules(task("The function prints the total to the screen.", tests))
    assert {"R3", "R4"} <= found


def test_r3_r4_honest_none_return_is_not_a_hit() -> None:
    # A function that legitimately returns None for invalid input, checked beside real values.
    prompt = "Return the integer square root of n, or None for a negative n. Logs nothing."
    tests = "assert f(-4) is None\nassert f(9) == 3\nassert f(16) == 4\n"
    solution = "def f(n):\n    if n < 0:\n        return None\n    return int(n ** 0.5)\n"
    found = rules(task(prompt, tests, solution))
    assert not found & {"R3", "R4"}


def test_r4_bare_write_imperative_is_not_a_side_effect_keyword() -> None:
    assert "R4" not in rules(task("Write a function that returns None.", "assert f(1) is None\n"))


# ---- R5 -----------------------------------------------------------------------------


def test_r5_raise_in_prompt() -> None:
    assert ("R5", None) in hits(
        task("Raise a ValueError when the input is empty.", "assert f(1) == 1\n")
    )
    assert ("R5", None) in hits(task("It throws an exception on bad input.", "assert f(1) == 1\n"))
    assert "R5" not in rules(task("Return the largest element.", "assert f(1) == 1\n"))


def test_r5_applies_to_io_tasks() -> None:
    io = {
        "id": "io/1",
        "prompt": "Print an error and raise an exception on bad input.",
        "mode": "io",
        "hidden_tests": json.dumps([{"input": "1\n", "output": "1\n"}]),
    }
    assert ("R5", None) in hits(io)


# ---- R6 -----------------------------------------------------------------------------


def test_r6_tautological_compare() -> None:
    assert ("R6", None) in hits(task("p", "assert f(1) == 1\nassert 3 == 3\n"))
    assert ("R6", None) in hits(task("p", "assert [1, 2] == [1, 2]\n"))
    assert "R6" not in rules(task("p", "assert f(1) == 1\nassert f(2) == [2, 2]\n"))


# ---- R7 -----------------------------------------------------------------------------

STRING_TESTS = "assert f(['b', 'a'], ['a', 'b', 'c']) == ['a', 'b']\n"
INT_TESTS = "assert f([2, 1], [1, 2, 3]) == [1, 2]\n"


def test_r7_list_of_set_call_hash_order_candidate() -> None:
    solution = "def f(a, b):\n    return list(set(a) & set(b))\n"
    assert ("R7", "hash_order_candidate") in hits(task("p", STRING_TESTS, solution))


def test_r7_impl_order_dependent_for_ints() -> None:
    solution = "def f(a, b):\n    return list(set(a).intersection(b))\n"
    found = hits(task("p", INT_TESTS, solution))
    assert ("R7", "impl_order_dependent") in found
    assert ("R7", "hash_order_candidate") not in found


@pytest.mark.parametrize(
    "body",
    [
        "    common = set(a).union(b)\n    return list(common)\n",
        "    seen = set()\n    for x in a:\n        seen.add(x)\n    return tuple(seen)\n",
        "    left = set(a)\n    right = set(b)\n    both = left & right\n    return list(both)\n",
        "    both = set(a).difference(b)\n    return [x for x in both]\n",
        "    both = set(a).symmetric_difference(b)\n    return list(both)\n",
        "    return list({x for x in a})\n",
        "    return list({1, 2} | set(a))\n",
        "    return list(x for x in set(a))\n",
    ],
)
def test_r7_covers_bound_names_operators_and_literals(body: str) -> None:
    solution = "def f(a, b):\n" + body
    assert ("R7", "impl_order_dependent") in hits(task("p", INT_TESTS, solution))


@pytest.mark.parametrize(
    ("solution", "tests"),
    [
        ("def f(a, b):\n    return sorted(set(a) & set(b))\n", INT_TESTS),
        ("def f(a, b):\n    return sorted([x for x in set(a)])\n", INT_TESTS),
        ("def f(a, b):\n    return list(set(a) & set(b))\n", "assert f([1], [1]) == [1]\n"),
        (
            "def f(a, b):\n    return list(set(a) & set(b))\n",
            "assert f([2, 1], [1, 2]) == [1, 2] or f([2, 1], [1, 2]) == [2, 1]\n",
        ),
        (
            "def f(a, b):\n    return list(set(a) & set(b))\n",
            "assert sorted(f([2, 1], [1, 2])) == [1, 2]\n",
        ),
        ("def f(a, b):\n    return list(a)\n", INT_TESTS),
    ],
)
def test_r7_honest_non_hits(solution: str, tests: str) -> None:
    assert "R7" not in rules(task("p", tests, solution))


# ---- R8 -----------------------------------------------------------------------------


def test_r8_reference_only_global_and_prompt_named_global() -> None:
    solution = "LIMIT = 5\nTABLE = {1: 2}\ndef f(x):\n    return x + LIMIT\n"
    tests = "assert f(1) == LIMIT + 1\n"
    assert ("R8", "reference_only_global") in hits(task("Add the limit.", tests, solution))
    assert ("R8", "prompt_named_global") in hits(task("Add LIMIT to x.", tests, solution))
    assert ("R8", "reference_only_global") not in hits(task("Add LIMIT to x.", tests, solution))


def test_r8_non_hits_for_defs_and_test_bound_names() -> None:
    solution = "LIMIT = 5\ndef helper():\n    return 1\ndef f(x):\n    return x\n"
    assert "R8" not in rules(task("p", "assert f(1) == helper()\n", solution))
    assert "R8" not in rules(task("p", "LIMIT = 5\nassert f(1) == LIMIT - 4\n", solution))
    assert "R8" not in rules(task("p", "assert f(1) == 1\n", solution))


# ---- R9 -----------------------------------------------------------------------------


def test_r9_rounding_clause_hit_and_honest_floats() -> None:
    prompt = "Return the mean, rounded to 2 decimal places."
    assert ("R9", None) in hits(task(prompt, "assert f(1) == 3.14159\n"))
    assert ("R9", None) in hits(
        task("Round the result to two decimal places.", "assert f(1) == 2.5001\n")
    )
    assert "R9" not in rules(task(prompt, "assert f(1) == 4.0\nassert f(2) == 1.67\n"))
    assert "R9" not in rules(task("Return the mean.", "assert f(1) == 3.14159\n"))


def test_r9_ignores_floats_that_are_arguments() -> None:
    prompt = "Return the mean, rounded to 2 decimal places."
    assert "R9" not in rules(task(prompt, "assert f(3.14159) == 3.14\n"))


def test_r9_io_mode_reads_expected_output_text() -> None:
    def io(output: str) -> dict[str, Any]:
        return {
            "id": "io/2",
            "prompt": "Print the average rounded to 2 decimal places.",
            "mode": "io",
            "hidden_tests": json.dumps([{"input": "1\n", "output": output}]),
        }

    assert ("R9", None) in hits(io("3.14159\n"))
    assert "R9" not in rules(io("3.14\n"))
    assert "R9" not in rules(io("4.0\n"))


# ---- R10 ----------------------------------------------------------------------------


def test_r10_order_free_clause() -> None:
    prompt = "Return the common elements in any order."
    assert ("R10", None) in hits(task(prompt, "assert f([2, 1]) == [1, 2]\n"))
    assert "R10" not in rules(task("Return the common elements.", "assert f([2, 1]) == [1, 2]\n"))


@pytest.mark.parametrize(
    "tests",
    [
        "assert sorted(f([2, 1])) == [1, 2]\n",
        "assert set(f([2, 1])) == {1, 2}\n",
        "assert f([2, 1]) == [1, 2] or f([2, 1]) == [2, 1]\n",
        "assert f([1]) == [1]\n",
    ],
)
def test_r10_honest_non_hits(tests: str) -> None:
    assert "R10" not in rules(task("The order does not matter.", tests))


# ---- R11 ----------------------------------------------------------------------------


def test_r11_builtin_outer() -> None:
    found = hits(task("p", "assert not isinstance(f(1), str)\nassert callable(f) is True\n"))
    assert ("R11", "builtin_outer") in found


def test_r11_module_attr_outer() -> None:
    found = hits(task("p", "import math\nassert math.isclose(f(1), 1.0) is True\n"))
    assert ("R11", "module_attr_outer") in found


@pytest.mark.parametrize(
    "tests",
    [
        "assert f(1) is True\n",
        "assert not f(0)\n",
        "assert f(1) == 1\n",
        "assert isinstance(f(1), int)\n",
        "import math\nassert math.isclose(f(1), 1.0)\n",
        "assert len(f(1)) < 2\n",
        "assert len(f(1)) == 0j\n",
        "assert len(f(1)) == len(f(2))\n",
    ],
)
def test_r11_honest_non_hits(tests: str) -> None:
    assert "R11" not in rules(task("p", tests))


def test_r11_not_flagged_when_reference_defines_the_outer_name() -> None:
    solution = "def isinstance(a, b):\n    return True\ndef f(x):\n    return x\n"
    assert "R11" not in rules(task("p", "assert isinstance(f(1), int) is True\n", solution))


_PARITY_SHAPES = [
    "assert f(1) is True",
    "assert True is f(1)",
    "assert f(1) is False",
    "assert False is f(1)",
    "assert f(1) is None",
    "assert None is f(1)",
    "assert not f(1)",
    "assert not not f(1)",
    "assert not (f(1) == 2)",
    "assert f(1) == 0",
    "assert 0 == f(1)",
    "assert f(1) == ''",
    "assert f(1) == []",
    "assert f(1) == None",
    "assert f(1) == 1",
    "assert 5 == len(f(4))",
    "assert len(f(4)) == 5",
    "assert f(1) == [1]",
    "assert f(1) == {1: (2, 3)}",
    "assert f(1) == {1, 2}",
    "assert f(1) == b'x'",
    "assert f(1) == -1",
    "assert f(1) == 0j",
    "assert f(1) == 1j",
    "assert f(1) == ...",
    "assert f(1) == [1j]",
    "assert f(1) == 1 == 1",
    "assert f(1) == f(2)",
    "assert 1 == 1",
    "assert f(1) is 5",
    "assert f(1) is not None",
    "assert f(1) is f(2)",
    "assert f(1) < 3",
    "assert f(1) in [1, 2]",
    "assert f(1) == 0 or g(1)",
    "assert f(1) and g(1)",
    "assert f(1) == (g(1) == 2)",
    "assert (f(1) == 2) is True",
    "assert (f(1) > 2) == True",
    "assert isinstance(f(1), int)",
    "assert isinstance(f(1), int) is True",
    "assert math.isclose(f(1), 1.0) is True",
    "assert f(1)",
    "assert True",
    "assert None is None",
]


@pytest.mark.parametrize("test", _PARITY_SHAPES)
def test_check_forms_mirror_the_reward_channel_parser_exactly(test: str) -> None:
    tree = ast.parse(test)
    statement = tree.body[0]
    assert isinstance(statement, ast.Assert)
    mine = verifier_scan._check_call(statement.test)
    predicate = reward._parse_predicate(statement.test)
    assert (mine is None) == (predicate is None)
    if mine is not None and predicate is not None:
        assert predicate.call == ast.unparse(mine)
    # Whole-task parseability agrees with parse_function_checks (small inputs, no size limit).
    assert verifier_scan._task_parseable(tree) == (
        reward.parse_function_checks(test + "\n") is not None
    )


@pytest.mark.parametrize(
    "tests",
    [
        "import math\nassert f(1) == 1\n",
        "from math import pi\nassert not f(1)\n",
        "assert f(1) == 1\nassert f(2) == 2\n",
        "assert f(1) == 1\nx = 3\n",
        "assert f(1) == 1\nassert f(2) < 3\n",
        "import math\n",
        "",
        "def helper():\n    pass\nassert f(1) == 1\n",
        "for i in range(2):\n    assert f(i) == i\n",
    ],
)
def test_task_parseable_mirrors_parse_function_checks(tests: str) -> None:
    assert verifier_scan._task_parseable(ast.parse(tests)) == (
        reward.parse_function_checks(tests) is not None
    )


def test_r11_builtin_outer_with_representable_literal() -> None:
    assert ("R11", "builtin_outer") in hits(task("p", "assert len(f(4)) == 5\n"))
    assert ("R11", "builtin_outer") in hits(task("p", "assert 5 == len(f(4))\n"))
    assert ("R11", "builtin_outer") in hits(task("p", "assert None is len(f(4))\n"))


def test_r11_task_parseable_flag_and_split_summary() -> None:
    reachable = scan_task(task("p", "import math\nassert len(f(4)) == 5\n"))
    unreachable = scan_task(task("p", "assert len(f(4)) == 5\nassert f(1) < 3\n"))
    assert [hit.task_parseable for hit in reachable.hits if hit.rule == "R11"] == [True]
    assert [hit.task_parseable for hit in unreachable.hits if hit.rule == "R11"] == [False]
    assert reachable.hits[0].to_dict()["task_parseable"] is True
    counts = summarize([reachable, unreachable])["tag_counts"]
    base = "redefinable_outer_bool_check/builtin_outer"
    assert counts[base] == 2
    assert counts[f"{base}/task_parseable=true"] == 1
    assert counts[f"{base}/task_parseable=false"] == 1


def test_non_r11_hits_have_no_task_parseable_flag() -> None:
    scan = scan_task(task("p", "assert f(1) is None\n"))
    assert all(hit.task_parseable is None for hit in scan.hits if hit.rule != "R11")


# ---- R7 scope and in-place sort ----------------------------------------------------


def test_r7_set_binding_is_per_function_scope() -> None:
    solution = (
        "def g(a):\n    s = set(a)\n    return len(s)\n"
        "def f(a):\n    s = list(a)\n    return [x for x in s]\n"
    )
    assert "R7" not in rules(task("p", "assert f([2, 1]) == [2, 1]\n", solution))


def test_r7_module_scope_walk_skips_nested_function_bodies() -> None:
    solution = (
        "s = [3, 1]\n"
        "def f(a):\n    s = set(a)\n    return len(s)\n"
        "def h():\n    return [x for x in s]\n"
    )
    assert "R7" not in rules(task("p", "assert h() == [3, 1]\n", solution))


def test_r7_set_binding_still_found_inside_its_own_function() -> None:
    solution = "def f(a):\n    s = set(a)\n    return [x for x in s]\n"
    assert "R7" in rules(task("p", "assert f([2, 1]) == [1, 2]\n", solution))


def test_r7_list_sorted_in_place_is_not_a_hit() -> None:
    solution = "def f(a):\n    s = list(set(a))\n    s.sort()\n    return s\n"
    assert "R7" not in rules(task("p", "assert f([2, 1]) == [1, 2]\n", solution))


def test_r7_for_loop_append_over_set_is_a_hit() -> None:
    solution = (
        "def f(a):\n    out = []\n    for x in set(a):\n        out.append(x)\n    return out\n"
    )
    assert "R7" in rules(task("p", "assert f([2, 1]) == [1, 2]\n", solution))


def test_r7_for_loop_append_sorted_afterwards_is_not_a_hit() -> None:
    solution = (
        "def f(a):\n    out = []\n    for x in set(a):\n        out.append(x)\n"
        "    out.sort()\n    return out\n"
    )
    assert "R7" not in rules(task("p", "assert f([2, 1]) == [1, 2]\n", solution))


# ---- scan errors and minor heuristics -----------------------------------------------


def test_scan_task_records_recursion_error_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_asserts: Any) -> Any:
        raise RecursionError

    monkeypatch.setattr(verifier_scan, "_r6", boom)
    scan = scan_task(task("p", "assert f(1) == 1\n"))
    assert scan.notes == ("scan_error",)
    assert scan.hits == ()
    assert summarize([scan])["notes"] == {"scan_error": 1}


def test_scan_task_survives_pathologically_deep_tests() -> None:
    depth = 20000
    deep = "assert f(1) == " + "-" * depth + "1\n"
    scan = scan_task(task("p", deep))
    assert scan.notes in ((), ("scan_error",), ("hidden_tests_unparseable",))


def test_scan_task_memory_error_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_asserts: Any) -> Any:
        raise MemoryError

    monkeypatch.setattr(verifier_scan, "_r6", boom)
    assert scan_task(task("p", "assert f(1) == 1\n")).notes == ("scan_error",)


@pytest.mark.parametrize(
    "prompt",
    ["It runs in O(log(n)) time.", "Use log n steps.", "Stay under log in time bounds."],
)
def test_r4_math_log_is_not_a_side_effect_keyword(prompt: str) -> None:
    assert "R4" not in rules(task(prompt, "assert f(1) is None\n"))


def test_r4_log_verb_still_matches() -> None:
    assert "R4" in rules(task("Log the value.", "assert f(1) is None\n"))


def test_r5_with_the_exception_of_is_not_a_raise() -> None:
    assert "R5" not in rules(task("Sum all items with the exception of zeros.", "assert f(1)\n"))
    assert "R5" in rules(task("Raise an exception.", "assert f(1)\n"))


def test_r10_order_of_results_not_important() -> None:
    tests = "assert f(1) == [1, 2]\n"
    prompt = "The order of the results is not important."
    assert "R10" in rules(task(prompt, tests))


# ---- records, determinism, errors ---------------------------------------------------


def test_unparseable_inputs_are_noted_not_fatal() -> None:
    scan = scan_task(task("p", "assert (", "def f(:\n"))
    assert "hidden_tests_unparseable" in scan.notes
    scan = scan_task(task("p", "assert f(1) == 1\n", "def f(:\n"))
    assert "reference_unparseable" in scan.notes


def test_missing_fields_raise() -> None:
    with pytest.raises(ValueError, match="text field"):
        scan_task({"id": "x", "prompt": "p"})
    with pytest.raises(ValueError, match="unknown task mode"):
        scan_task(task("p", "assert f(1) == 1\n", mode="other"))


def test_output_is_deterministic_and_ordered() -> None:
    record = task(
        "Raise an error. Return the mean, rounded to 2 decimal places, in any order.",
        "import math\nassert 1 == 1\nassert math.sqrt(f(1)) is None\nassert f(2) == [2, 1]\n"
        "assert f(3) == 3.14159\n",
        "def f(x):\n    return list(set(x))\n",
    )
    first = scan_task(record).to_row()
    assert first == scan_task(record).to_row()
    assert json.dumps(first) == json.dumps(scan_task(record).to_row())
    order = [int(hit["rule"][1:]) for hit in first["hits"]]
    assert order == sorted(order)
    assert {hit["name"] for hit in first["hits"]} <= set(RULES.values())


def test_summary_counts_and_buckets() -> None:
    scans: list[TaskScan] = [
        scan_task(task("p", "assert f(1) is None\n")),
        scan_task(task("p", "assert f(1) == 1\nassert f(2) == 2\nassert f(3) == 3\n")),
    ]
    summary = summarize(scans)
    assert summary["tasks"] == 2
    assert summary["tasks_with_hits"] == 1
    assert summary["declared_checks_le2"] == 1
    assert summary["declared_checks_histogram"] == {"1": 1, "3+": 1}
    assert summary["rule_counts"] == {"none_only": 1}
    assert RULESET_VERSION.startswith("verifier-scan-")


def test_module_never_names_code_executing_facilities() -> None:
    source = Path(verifier_scan.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"exec", "eval", "compile", "__import__", "subprocess", "importlib", "os", "runpy"}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            # ``re.compile`` builds a pattern, not code.
            if not (
                node.attr == "compile"
                and isinstance(node.value, ast.Name)
                and node.value.id == "re"
            ):
                used.add(node.attr)
        elif isinstance(node, ast.Import):
            used.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            used.add((node.module or "").split(".")[0])
    assert not used & forbidden
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    cli_names = {node.id for node in ast.walk(ast.parse(cli_source)) if isinstance(node, ast.Name)}
    assert not cli_names & {"exec", "eval", "compile", "subprocess"}


# ---- sealed guard and CLI -----------------------------------------------------------

SEALED = [
    "test.jsonl",
    "holdout.jsonl",
    "rl_only.jsonl",
    "Test.jsonl",
    "eval-code-test-v1.jsonl",
    "my_holdout_copy.jsonl",
    "dev_test.jsonl",
    "test-split.jsonl",
    "rl-only.jsonl",
]
ALLOWED = ["dev.jsonl", "train.jsonl", "rl_candidates.jsonl", "language_dev.jsonl", "latest.jsonl"]


@pytest.mark.parametrize("name", SEALED)
def test_sealed_names_are_refused(name: str) -> None:
    assert cli.sealed_reason(Path("/nonexistent") / name) is not None


@pytest.mark.parametrize("name", ALLOWED)
def test_allowed_names_pass_the_guard(name: str) -> None:
    assert cli.sealed_reason(Path("/nonexistent") / name) is None


def test_holdout_directory_is_refused() -> None:
    assert cli.sealed_reason(Path("/nonexistent/holdout-set/dev.jsonl")) is not None


def test_symlink_to_sealed_file_is_refused(tmp_path: Path) -> None:
    sealed = tmp_path / "holdout.jsonl"
    sealed.write_text("{}\n")
    link = tmp_path / "innocent.jsonl"
    link.symlink_to(sealed)
    assert cli.sealed_reason(link) is not None


def _forbid_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def refuse(*args: object, **kwargs: object) -> None:
        calls.append(repr(args))
        raise AssertionError("a file was opened")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    monkeypatch.setattr(Path, "read_text", refuse)
    return calls


@pytest.mark.parametrize("name", ["test.jsonl", "holdout.jsonl", "rl_only.jsonl"])
def test_cli_refuses_sealed_file_without_opening_anything(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sealed = tmp_path / name
    sealed.write_text("not read\n")
    ordinary = tmp_path / "train.jsonl"
    ordinary.write_text("not read either\n")
    out = tmp_path / "out"
    calls = _forbid_open(monkeypatch)
    code = cli.main([str(ordinary), str(sealed), "--out", str(out)])
    assert code != 0
    assert calls == []
    assert not out.exists()
    assert "refusing" in capsys.readouterr().err


def test_cli_refuses_sealed_pool_in_id_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = tmp_path / "rl_candidates.jsonl"
    pool = tmp_path / "holdout.jsonl"
    ids.write_text("x\n")
    pool.write_text("x\n")
    calls = _forbid_open(monkeypatch)
    assert cli.main(["--id-join", str(ids), str(pool), "--out", str(tmp_path / "o")]) != 0
    assert calls == []


def _pool_row(task_id: str, tests: str, solution: str | None = "def f(x):\n    return x\n") -> str:
    row = {
        "id": task_id,
        "prompt": "Return the value.",
        "public_tests": "assert f(1) == 1",
        "hidden_tests": tests,
        "source": "fixture",
        "license": "x",
        "written_by": "model",
        "split": "train",
        "mode": "function",
        "solution": solution,
    }
    return json.dumps(row)


def test_cli_end_to_end_with_id_join(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(
            [
                _pool_row("a/1", "assert f(1) is None\n"),
                _pool_row("a/2", "assert f(1) == 1\nassert f(2) == 2\nassert f(3) == 3\n"),
                _pool_row("a/3", "assert f(1) == 1\n", None),
            ]
        )
        + "\n"
    )
    candidates = tmp_path / "rl_candidates.jsonl"
    candidates.write_text(
        json.dumps({"id": "a/3", "source": "x"})
        + "\n"
        + json.dumps({"id": "gone/9", "source": "x"})
        + "\n"
    )
    out = tmp_path / "out"
    code = cli.main(
        [
            str(train),
            "--id-join",
            str(candidates),
            str(train),
            "--out",
            str(out),
            "--commit",
            "abc123",
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in (out / "scan-train.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == ["a/1"]
    assert {hit["rule"] for hit in rows[0]["hits"]} == {"R3"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["ruleset_version"] == RULESET_VERSION
    assert summary["tool_commit"] == "abc123"
    assert summary["summaries"]["train"]["rule_counts"] == {"none_only": 1}
    assert summary["summaries"]["train"]["declared_checks_le2"] == 1
    joined = summary["summaries"]["rl_candidates"]
    assert joined["unjoined_ids"] == ["gone/9"]
    assert joined["tasks"] == 1
    assert all(len(entry["sha256"]) == 64 for entry in summary["inputs"])
    record = json.loads((out / "record.json").read_text())
    names = [Path(path).name for path in record["files_opened"]]
    assert sorted(names) == sorted(["train.jsonl", "rl_candidates.jsonl", "train.jsonl"])
    # Deterministic: a second run into a fresh directory gives byte-identical outputs.
    again = tmp_path / "out2"
    cli.main(
        [
            str(train),
            "--id-join",
            str(candidates),
            str(train),
            "--out",
            str(again),
            "--commit",
            "abc123",
        ]
    )
    for name in ("scan-train.jsonl", "scan-rl_candidates.jsonl", "summary.json"):
        assert (out / name).read_bytes() == (again / name).read_bytes()


def test_cli_does_not_overwrite_existing_scan(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(_pool_row("a/1", "assert f(1) is None\n") + "\n")
    out = tmp_path / "out"
    assert cli.main([str(train), "--out", str(out)]) == 0
    before = (out / "scan-train.jsonl").read_bytes()
    assert cli.main([str(train), "--out", str(out)]) == 3
    assert (out / "scan-train.jsonl").read_bytes() == before


def test_cli_existing_summary_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(_pool_row("a/1", "assert f(1) is None\n") + "\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("{}")
    assert cli.main([str(train), "--out", str(out)]) == 3
    assert "summary.json" in capsys.readouterr().err
    assert not (out / "scan-train.jsonl").exists()


def test_cli_records_absolute_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text(_pool_row("a/1", "assert f(1) is None\n") + "\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["train.jsonl", "--out", "out"]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["inputs"][0]["path"] == str(train.resolve())
    assert len(summary["inputs"][0]["sha256"]) == 64
    record = json.loads((tmp_path / "out" / "record.json").read_text())
    assert record["files_opened"] == [str(train.resolve())]


def test_cli_refuses_hard_link_to_sealed_named_sibling(tmp_path: Path) -> None:
    sealed = tmp_path / "holdout.jsonl"
    sealed.write_text("{}\n")
    alias = tmp_path / "innocent.jsonl"
    os.link(sealed, alias)
    assert cli.sealed_reason(alias) is not None
    assert cli.main([str(alias), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_hard_link_check_allows_unrelated_siblings(tmp_path: Path) -> None:
    (tmp_path / "holdout.jsonl").write_text("{}\n")
    other = tmp_path / "train.jsonl"
    other.write_text("{}\n")
    assert cli.sealed_reason(other) is None
