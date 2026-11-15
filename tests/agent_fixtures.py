"""Generic agent-loop fixtures: a tiny two-module repository, its case files, and a scripted
executor. Nothing here executes workspace code; the executor decides by comparing the packed
sources (read back with ``ast.literal_eval`` from the trusted prelude) with known texts."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path

from scratchcoder.agent import execution, workspace
from scratchcoder.agent.cases import build_registry
from scratchcoder.sandbox.runner import Result

ARITH = '''"""Arithmetic helpers."""

from steps import clamp


def double(x):
    """Twice ``x``."""
    return x + x + 1


def scaled(x, low, high):
    """``double(x)`` limited to [low, high]."""
    return clamp(double(x), low, high)
'''
STEPS = '''"""Step helpers."""


def clamp(value, low, high):
    """``value`` limited to [low, high]."""
    return max(low, min(value, high))
'''
TESTS = """from arith import double, scaled


def test_double():
    got = double(2)
    assert got == 4, repr(got)


def test_scaled():
    assert scaled(1, 0, 10) == 2
"""
CORRECTNESS = "from arith import double\n\nassert double(3) == 6\nassert double(0) == 0\n"
PRESERVATION = "from steps import clamp\n\nassert clamp(5, 0, 3) == 3\n"
FIXED_DOUBLE = 'def double(x):\n    """Twice ``x``."""\n    return x + x'
REFERENCE = ARITH.replace("    return x + x + 1\n", "    return x + x\n")


def write_case(root: Path, case_id: str = "X01", **overrides: object) -> Path:
    directory = root / f"{case_id}-double"
    for sub in ("repo", "hidden", "reference"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    (directory / "repo" / "arith.py").write_text(ARITH, encoding="utf-8")
    (directory / "repo" / "steps.py").write_text(STEPS, encoding="utf-8")
    (directory / "repo" / "test_arith.py").write_text(TESTS, encoding="utf-8")
    (directory / "hidden" / "correctness.py").write_text(CORRECTNESS, encoding="utf-8")
    (directory / "hidden" / "preservation.py").write_text(PRESERVATION, encoding="utf-8")
    (directory / "reference" / "arith.py").write_text(
        str(overrides.pop("reference_text", REFERENCE)), encoding="utf-8"
    )
    spec = {
        "schema": "agent-case-v1",
        "id": case_id,
        "title": "double adds one",
        "category": "bug_fix",
        "workflow_type": "W1",
        "structure": "single module",
        "task": "Bug: double(2) returns 5 instead of 4. Fix the bug.",
        "split": "dev",
        "repo": "repo",
        "allowed_files": ["arith.py", "steps.py"],
        "public_tests": ["test_arith.py"],
        "public_fails_before_fix": True,
        "hidden": {
            "correctness": "hidden/correctness.py",
            "preservation": "hidden/preservation.py",
        },
        "reference": {"arith.py": "reference/arith.py"},
        "targets": ["arith.py::double"],
        "budgets": {
            "max_edit_turns": 3,
            "max_feedback_runs": 6,
            "max_generated_tokens": 2000,
            "max_wall_seconds": 600,
        },
        "provenance": {"author": "test fixture"},
        **overrides,
    }
    (directory / "case.json").write_text(json.dumps(spec), encoding="utf-8")
    return directory


def write_registry(root: Path) -> None:
    (root / "registry.json").write_text(json.dumps(build_registry(root)), encoding="utf-8")


def packed_sources(program: str) -> dict[str, str]:
    """The module sources a packed program serves (parsed as a literal, never run)."""
    for node in ast.parse(program).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_SC_SOURCES":
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise AssertionError("not a packed program")


def _double_fixed(sources: dict[str, str]) -> bool:
    text = sources["arith"]
    found = workspace.functions(text, "arith.py").get("double")
    return found is not None and workspace.function_source(text, found) == FIXED_DOUBLE


def row(test: str, passed: bool, frames: Sequence[list[object]] = ()) -> str:
    body: dict[str, object] = {"module": "test_arith", "test": test, "outcome": "pass"}
    if not passed:
        body.update(outcome="fail", type="AssertionError", message="5", frames=list(frames))
    return execution.REPORT_PREFIX + json.dumps(body)


class ScriptedExecutor:
    """Returns scripted :class:`Result`s: the public run, the correctness checks and the
    preservation checks pass exactly when ``double`` equals the reference version. Records
    every call it receives. ``host_failures`` makes the first N calls host failures."""

    def __init__(self, host_failures: int = 0, preservation_passes: bool = True) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.host_failures = host_failures
        self.preservation_passes = preservation_passes

    def __call__(self, program: str, tests: str, mode: str) -> Result:
        self.calls.append((program, tests, mode))
        if self.host_failures:
            self.host_failures -= 1
            return Result("infrastructure_error", detail="runner_error")
        sources = packed_sources(program)
        fixed = _double_fixed(sources)
        if tests == CORRECTNESS:
            return Result("passed" if fixed else "failed", elapsed_seconds=0.5)
        if tests == PRESERVATION:
            return Result("passed" if self.preservation_passes else "failed", elapsed_seconds=0.5)
        assert tests == execution.public_driver(["test_arith.py"])
        frames = [["/tmp/candidate.py", 3, "<module>"], ["test_arith.py", 6, "test_double"]]
        stdout = "\n".join([row("test_double", fixed, frames), row("test_scaled", fixed, frames)])
        return Result("passed" if fixed else "failed", stdout + "\n", 0, 0.5)
