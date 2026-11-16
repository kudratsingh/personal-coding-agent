"""Agent harness v0, host side: static locate, the reply rules and span edits, the packed
program, feedback parsing and the prompt budgeter. Nothing here executes workspace code."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from agent_fixtures import ARITH, CORRECTNESS, FIXED_DOUBLE, STEPS, TESTS, packed_sources

from scratchcoder.agent import execution, prompt, workspace
from scratchcoder.eval import harness
from scratchcoder.sandbox.runner import Limits, Result

FILES = {"arith.py": ARITH, "steps.py": STEPS, "test_arith.py": TESTS}
ALLOWED = ("arith.py", "steps.py")


class Bytes:
    """A byte-level encoder: one token per UTF-8 byte (never a special token)."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(token for token in ids if token < 256).decode()


def shown(*keys: str) -> list[workspace.Function]:
    found = []
    for key in keys:
        path, name = key.split("::")
        found.append(workspace.functions(FILES[path], path)[name])
    return found


# ----------------------------------------------------------------------------- locate


def test_locate_follows_calls_across_modules_nearest_first() -> None:
    found = workspace.locate(FILES, ["test_arith.py"], ["test_scaled"], ALLOWED)
    assert [(item.function.key, item.distance) for item in found] == [
        ("arith.py::scaled", 0),
        ("steps.py::clamp", 1),
        ("arith.py::double", 1),
    ]
    only_double = workspace.locate(FILES, ["test_arith.py"], ["test_double"], ALLOWED)
    assert [item.function.key for item in only_double] == ["arith.py::double"]


def test_locate_keeps_only_allowed_files_and_falls_back_to_every_test() -> None:
    found = workspace.locate(FILES, ["test_arith.py"], ["test_unknown"], ["arith.py"])
    assert [item.function.key for item in found] == ["arith.py::double", "arith.py::scaled"]


def test_locate_resolves_module_attribute_calls() -> None:
    files = {
        "pkg.py": "def f(x):\n    return x\n",
        "test_pkg.py": "import pkg\n\n\ndef test_f():\n    assert pkg.f(1) == 1\n",
    }
    found = workspace.locate(files, ["test_pkg.py"], None, ["pkg.py"])
    assert [item.function.key for item in found] == ["pkg.py::f"]


def test_reserved_and_standard_library_module_names_are_refused() -> None:
    for name in ("json.py", "_sc_x.py", "a/b.py", "x.txt"):
        with pytest.raises(ValueError):
            workspace.check_file_name(name)
    workspace.check_file_name("textkit.py")


# ----------------------------------------------------------------------------- reply rules


def test_a_reply_replaces_exactly_the_function_span() -> None:
    edit = workspace.parse_reply(FIXED_DOUBLE, shown("arith.py::double"), FILES, ALLOWED)
    after = workspace.apply_function(FILES, edit, ALLOWED)
    assert after["arith.py"] == ARITH.replace("return x + x + 1", "return x + x")
    assert after["steps.py"] == STEPS and after["test_arith.py"] == TESTS
    diff = workspace.unified_diff(FILES, after)
    assert diff.startswith("--- a/arith.py\n+++ b/arith.py\n")
    assert "-    return x + x + 1\n+    return x + x\n" in diff
    assert workspace.touched(FILES, after) == ["arith.py"]


def test_decorators_are_part_of_the_span() -> None:
    files = {"m.py": "import functools\n\n\n@functools.cache\ndef f(x):\n    return 1\n"}
    function = workspace.functions(files["m.py"], "m.py")["f"]
    assert (function.start, function.end) == (4, 6)
    edit = workspace.Edit("m.py", "f", "def f(x):\n    return 2")
    assert workspace.apply_function(files, edit, ["m.py"])["m.py"] == (
        "import functools\n\n\ndef f(x):\n    return 2\n"
    )


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (None, "no_python_block"),
        ("def double(x)\n    return x", "unparsable"),
        ("x = 1", "no_function"),
        ("def triple(x):\n    return 3 * x", "unknown_name"),
        (
            "def double(x):\n    return 2 * x\n\ndef scaled(x, a, b):\n    return 0",
            "multiple_functions",
        ),
        ("import math\n\ndef double(x):\n    return math.fsum([x, x])", "extra_statements"),
        (
            "def two(x):\n    return x + x\n\ndef double(x):\n    return two(x)",
            "multiple_functions",
        ),
        ("K = 2\n\ndef double(x):\n    return K * x", "extra_statements"),
    ],
)
def test_replies_the_host_refuses(code: str | None, reason: str) -> None:
    with pytest.raises(workspace.EditRefused) as refused:
        workspace.parse_reply(code, shown("arith.py::double", "arith.py::scaled"), FILES, ALLOWED)
    assert refused.value.reason == reason


def test_independent_extra_statements_are_dropped_and_recorded() -> None:
    code = "import re\nfrom steps import clamp\n\n"
    code += "def double(x):\n    return clamp(x + x, 0, 99)\n\n"
    code += "def helper():\n    pass\n\nprint(double(2))\nassert double(1) == 2\n"
    edit = workspace.parse_reply(code, shown("arith.py::double"), FILES, ALLOWED)
    assert edit.source == "def double(x):\n    return clamp(x + x, 0, 99)"
    assert edit.dropped == (
        "import re",
        "from steps import clamp",
        "def helper():",
        "print(double(2))",
        "assert double(1) == 2",
    )


def test_test_files_and_disallowed_files_are_never_edited() -> None:
    tests = workspace.functions(TESTS, "test_arith.py")["test_double"]
    with pytest.raises(workspace.EditRefused) as refused:
        workspace.parse_reply(
            "def test_double():\n    pass", [tests], FILES, [*ALLOWED, "test_arith.py"]
        )
    assert refused.value.reason == "not_allowed"
    with pytest.raises(workspace.EditRefused) as refused:
        edit = workspace.Edit("steps.py", "clamp", "def clamp(): pass")
        workspace.apply_function(FILES, edit, ["arith.py"])
    assert refused.value.reason == "not_allowed"


def test_an_edit_that_breaks_the_module_is_refused() -> None:
    edit = workspace.Edit("arith.py", "double", "def double(x):\n  return 1\n   x = (")
    with pytest.raises(workspace.EditRefused) as refused:
        workspace.apply_function(FILES, edit, ALLOWED)
    assert refused.value.reason == "breaks_module"


# ----------------------------------------------------------------------------- packing


def test_the_packed_program_serves_each_module_from_its_source() -> None:
    program, tests = execution.public_call(FILES, ["test_arith.py"])
    assert packed_sources(program) == {"arith": ARITH, "steps": STEPS, "test_arith": TESTS}
    assert "_sc_sys.meta_path.insert(0, _ScWorkspace())" in program
    assert "for _sc_name in ['test_arith']:" in tests and "raise SystemExit(1)" in tests
    graded, checks = execution.check_call(FILES, CORRECTNESS)
    assert packed_sources(graded) == {"arith": ARITH, "steps": STEPS} and checks == CORRECTNESS


def test_the_container_script_compiles_and_keeps_the_hook_names() -> None:
    # compile() builds code objects and runs nothing. The hook's methods look the helper
    # names up at import time, so the prelude must not delete them.
    for program, tests in (
        execution.public_call(FILES, ["test_arith.py"]),
        execution.check_call(FILES, CORRECTNESS),
    ):
        compile(program + "\n" + tests + "\nprint('marker')\n", "/tmp/candidate.py", "exec")
        assert "\ndel " not in program


def test_isolated_execution_receives_exactly_the_packed_program_and_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, str, Limits, str]] = []

    def fake_run(code: str, tests: str, limits: Limits, *, mode: str) -> Result:
        received.append((code, tests, limits, mode))
        return Result("passed", "", 0, 0.1)

    monkeypatch.setattr(harness, "run", fake_run)
    program, checks = execution.check_call(FILES, CORRECTNESS)
    outcome = execution.run_call(program, checks, execute=harness.isolated_execution)
    assert outcome.passed and outcome.resolved
    assert received == [(program, checks, Limits(), "function")]
    feedback_program, driver = execution.public_call(FILES, ["test_arith.py"])
    execution.run_call(feedback_program, driver, execute=harness.isolated_execution)
    assert received[1][:2] == (feedback_program, driver) and received[1][3] == "function"


def test_a_persistent_host_failure_is_unresolved() -> None:
    waits: list[float] = []
    outcome = execution.run_call(
        "x", "y", execute=lambda *_: Result("unavailable"), sleep=waits.append
    )
    assert not outcome.resolved and not outcome.passed and len(waits) == harness.HOST_RETRIES


# ----------------------------------------------------------------------------- feedback


def test_feedback_rows_are_bounded_and_malformed_rows_skipped() -> None:
    prefix = execution.REPORT_PREFIX
    stdout = "\n".join(
        [
            "noise",
            prefix + "not json",
            prefix + json.dumps({"module": 3, "test": "t", "outcome": "pass"}),
            prefix + json.dumps({"module": "m", "test": "t1", "outcome": "pass"}),
            prefix
            + json.dumps(
                {
                    "module": "m",
                    "test": "t2",
                    "outcome": "fail",
                    "type": "E",
                    "message": "x" * 5000,
                    "frames": [["a.py", "1", "f"], ["b.py", 2, "g"]],
                }
            ),
        ]
    )
    rows = execution.parse_report(stdout)
    assert [(row.test, row.passed) for row in rows] == [("t1", True), ("t2", False)]
    assert len(rows[1].message) == execution.MAX_MESSAGE_CHARS
    assert rows[1].frames == (("b.py", 2, "g"),)


def test_condense_names_the_test_line_exception_and_last_workspace_frame() -> None:
    frames = [
        ["/tmp/candidate.py", 40, "<module>"],
        ["test_arith.py", 10, "test_scaled"],
        ["arith.py", 13, "scaled"],
        ["steps.py", 6, "clamp"],
    ]
    stdout = execution.REPORT_PREFIX + json.dumps(
        {
            "module": "test_arith",
            "test": "test_scaled",
            "outcome": "fail",
            "type": "TypeError",
            "message": "bad operand\nsecond line",
            "frames": frames,
        }
    )
    text = execution.condense(execution.SandboxRun("failed", True, 0, 0.0, stdout), FILES)
    assert text.split("\n") == [
        "RESULT run_tests: failed (1 of 1 tests)",
        "FAIL test_scaled (test_arith.py:10)",
        "    assert scaled(1, 0, 10) == 2",
        "TypeError: bad operand",
        "  in steps.py:6 clamp: return max(low, min(value, high))",
    ]
    timeout = execution.condense(execution.SandboxRun("timeout", True, 0, 2.0, ""), FILES)
    assert "timeout" in timeout
    passed = execution.condense(execution.SandboxRun("passed", True, 0, 0.0, ""), FILES)
    assert passed.startswith("RESULT run_tests: passed")


# ----------------------------------------------------------------------------- prompt


def blocks(observation: str = "RESULT run_tests: failed", **changes: object) -> prompt.Blocks:
    values: dict[str, object] = {
        "task": "Fix double.",
        "map": workspace.repository_map(FILES),
        "snippets": [prompt.Snippet("arith.py::double", FIXED_DOUBLE)],
        "observation": observation,
        "previous": (),
    }
    values.update(changes)
    return prompt.Blocks(**values)  # type: ignore[arg-type]


def test_the_prompt_is_one_chat_v1_user_turn_counted_exactly() -> None:
    built = prompt.build(Bytes(), blocks())
    assert built.text.startswith("Fix double.\n\nMAP\narith.py: double, scaled\nsteps.py: clamp")
    assert built.text.endswith(prompt.INSTRUCTION)
    # <|bos|><|user_start|> text <|user_end|><|assistant_start|>
    assert built.tokens == len(built.text.encode()) + 4
    assert built.block_tokens["task"] == len("Fix double.")
    assert built.shown == ("arith.py::double",) and not built.truncations
    assert built.generation_cap() == prompt.GENERATION_CAP


def test_over_budget_blocks_are_truncated_then_dropped_in_order() -> None:
    long_function = "def double(x):\n" + "".join(f"    y{i} = {i}\n" for i in range(40))
    snippets = [
        prompt.Snippet("arith.py::double", long_function.rstrip()),
        prompt.Snippet("arith.py::scaled", "def scaled(x, low, high):\n    return 0"),
    ]
    built = prompt.build(
        Bytes(), blocks("E" * 400, snippets=snippets, previous=["arith.py:", "-a", "+b"] * 20)
    )
    assert "function:arith.py::double" in built.truncations
    assert "previous_dropped" in built.truncations and "observation" in built.truncations
    assert built.block_tokens["observation"] <= prompt.BUDGETS["observation"]
    assert built.tokens <= prompt.PROMPT_BUDGET
    assert prompt.ELLIPSIS in built.text


def test_a_task_over_its_budget_is_a_context_limit() -> None:
    with pytest.raises(prompt.ContextLimit):
        prompt.build(Bytes(), blocks(task="x" * 500))
    assert issubclass(prompt.ContextLimit, harness.ContextLimitError)


def test_a_lone_carriage_return_cannot_smuggle_code_past_the_function() -> None:
    # ast treats a lone "\r" as a line end; the host must see the same lines, so OTHER and g
    # are dropped (and recorded), never written into the module.
    code = "def double(x):\r    return x + x\rOTHER = 5\rg = lambda y: y\r"
    edit = workspace.parse_reply(code, shown("arith.py::double"), FILES, ALLOWED)
    assert edit.source == "def double(x):\n    return x + x"
    assert edit.dropped == ("OTHER = 5", "g = lambda y: y")
    after = workspace.apply_function(FILES, edit, ALLOWED)
    assert "OTHER" not in after["arith.py"] and "lambda" not in after["arith.py"]
    crlf = workspace.parse_reply(
        "def double(x):\r\n    return x + x\r\n", shown("arith.py::double"), FILES, ALLOWED
    )
    assert crlf.source == "def double(x):\n    return x + x"


@pytest.mark.parametrize(
    "source",
    [
        "def double(x):\n    return x + x\nOTHER = 5",
        "def double(x):\n    return x + x\n\ndef extra():\n    pass",
        "def double(x):\r    return x\rOTHER = 5",
        "X = 1\ndef double(x):\n    return x",
        "def double(x):\n    return x\n# trailing comment",
    ],
)
def test_apply_refuses_any_change_outside_exactly_one_function(source: str) -> None:
    with pytest.raises(workspace.EditRefused):
        workspace.apply_function(FILES, workspace.Edit("arith.py", "double", source), ALLOWED)


def test_apply_compares_every_other_statement_as_a_syntax_tree() -> None:
    # A function source that closes the def and opens a new one under the same name would
    # change the statement count; one that rewrites a neighbour changes its tree.
    two = "def double(x):\n    return 1\n\n\ndef double(x):\n    return 2"
    with pytest.raises(workspace.EditRefused, match="breaks_module"):
        workspace.apply_function(FILES, workspace.Edit("arith.py", "double", two), ALLOWED)
    ok = workspace.apply_function(
        FILES, workspace.Edit("arith.py", "double", "def double(x):\n    return x + x"), ALLOWED
    )
    assert ok["arith.py"] == ARITH.replace(
        '    """Twice ``x``."""\n    return x + x + 1', "    return x + x"
    )
