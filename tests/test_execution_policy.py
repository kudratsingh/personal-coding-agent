"""The execution-policy identity: concrete limits, code hashes and their blind spots."""

import ast
from typing import Any

import pytest

from scratchcoder.eval import execution_policy
from scratchcoder.eval.execution_policy import (
    VERDICT_CODE,
    callable_name,
    deadline_rule,
    policy_differences,
    policy_sha256,
    unit_sha256,
)
from scratchcoder.eval.harness import isolated_execution
from scratchcoder.sandbox import budget

HISTORICAL_RULE = "function: 2 s; I/O: 2 s + 0.25 s per case, at most 60 s; 128 MiB"


def current() -> dict[str, Any]:
    return execution_policy.execution_policy(
        isolated_execution, 3, unexecuted_statuses=("no_program", "context_limit")
    )


def test_the_recorded_limits_are_the_deadline_rule() -> None:
    policy = current()
    assert deadline_rule() == policy["deadline_rule"] == HISTORICAL_RULE
    assert policy["limits"]["function"]["timeout_seconds"] == 2.0
    assert policy["limits"]["function"]["memory_mb"] == 128
    io = policy["limits"]["io_by_cases"]
    assert {cases: io[cases]["timeout_seconds"] for cases in io} == {
        str(cases): min(budget.MAX_SECONDS, 2.0 + 0.25 * cases)
        for cases in execution_policy.IO_PROBE_CASES
    }
    assert io["99"]["timeout_seconds"] == 26.75 and io["1000"]["timeout_seconds"] == 60.0
    assert policy["host_backoff_seconds"] == [2.0, 4.0, 8.0]
    assert policy["execute"] == "scratchcoder.eval.harness.isolated_execution"
    assert policy["unexecuted_statuses"] == ["context_limit", "no_program"]
    assert policy_sha256(policy) == policy_sha256(current())  # Deterministic.


SOURCE = '''"""Module docstring."""
import json

LIMIT = 3
"""Attribute docstring."""


def verdict(x):
    """Function docstring."""
    # A comment.
    return x > LIMIT


def unrelated():
    return json.dumps(1)
'''


def edited(*pairs: tuple[str, str]) -> str:
    text = SOURCE
    for old, new in pairs:
        assert old in text
        text = text.replace(old, new)
    return text


@pytest.mark.parametrize(
    "change",
    [
        ('"""Module docstring."""', '"""Another module docstring."""'),
        ('"""Attribute docstring."""', '"""Changed."""'),
        ('"""Function docstring."""', '"""Changed, and longer."""'),
        ("    # A comment.\n", ""),
        ("def verdict(x):", "\n\n# Moved down.\ndef verdict(x):"),
    ],
)
def test_documentation_never_changes_a_unit(change: tuple[str, str]) -> None:
    for names in (None, ("LIMIT", "verdict")):
        assert unit_sha256("m", names, edited(change)) == unit_sha256("m", names, SOURCE)


@pytest.mark.parametrize(
    ("change", "names", "differs"),
    [
        (("return x > LIMIT", "return x >= LIMIT"), ("LIMIT", "verdict"), True),
        (("LIMIT = 3", "LIMIT = 4"), ("LIMIT", "verdict"), True),
        (("import json", "import json as j"), ("LIMIT", "verdict"), True),  # Imports count.
        (("json.dumps(1)", "json.dumps(2)"), ("LIMIT", "verdict"), False),  # Not listed.
        (("json.dumps(1)", "json.dumps(2)"), None, True),  # The whole module is listed.
    ],
)
def test_code_changes_in_a_listed_unit_change_it(
    change: tuple[str, str], names: tuple[str, ...] | None, differs: bool
) -> None:
    assert (unit_sha256("m", names, edited(change)) != unit_sha256("m", names, SOURCE)) is differs


def test_a_missing_listed_definition_is_an_error() -> None:
    with pytest.raises(ValueError, match="verdict not found"):
        unit_sha256("m", ("LIMIT", "verdict"), edited(("def verdict(", "def judge(")))


def test_every_listed_unit_exists_and_imports_only_listed_verdict_code() -> None:
    """The listed definitions use scratchcoder code only from other listed units, so the
    enumeration is closed under the imports that the verdict path actually uses."""
    listed = dict(VERDICT_CODE)
    for module, names in VERDICT_CODE:
        unit_sha256(module, names)  # Exists, with every listed name.
        tree = ast.parse(execution_policy.module_source(module))
        imported: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("scratchcoder"):
                imported |= {alias.asname or alias.name: node.module or "" for alias in node.names}
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("scratchcoder") for a in node.names), module
        used = {
            child.id
            for node in tree.body
            if names is None or any(name in names for name in _defined(node))
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
        }
        for name in sorted(used & set(imported)):
            source = imported[name]
            assert source in listed, f"{module} uses {name} from unlisted {source}"
            source_names = listed[source]
            assert source_names is None or name in source_names, f"{source}.{name} unlisted"


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    return set()


class Instance:
    def __call__(self, code: str, tests: str, mode: str) -> None:
        return None


def test_callable_names() -> None:
    assert callable_name(isolated_execution) == "scratchcoder.eval.harness.isolated_execution"
    assert callable_name(Instance()) == f"{__name__}.Instance instance"
    assert callable_name(lambda: None).endswith("<lambda>")


def test_differences_are_named() -> None:
    policy = current()
    other = {**policy, "host_retries": 1, "code": {**policy["code"], "python": "3.13"}}
    other["code"]["units"] = {**policy["code"]["units"], "scratchcoder.sandbox.runner": "0"}
    assert policy_differences(other, policy) == [
        "code: python version",
        "code: scratchcoder.sandbox.runner",
        "host_retries",
    ]
    assert policy_differences(None, policy) == ["the whole policy (none recorded)"]
    assert policy_differences(policy, current()) == []


def test_negative_retries_are_refused() -> None:
    with pytest.raises(ValueError, match="retries"):
        execution_policy.execution_policy(isolated_execution, -1, unexecuted_statuses=())
