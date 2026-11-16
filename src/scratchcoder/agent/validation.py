"""Sandbox validation of a dev case before any policy runs on it (plan section 4.4, v0 subset).

Every call goes through the evaluation path of record. A case is valid only if every call
resolved and matched its expectation:

- original repository: the public tests fail (or pass, as the spec states), the correctness
  checks fail (so the null patch is unsolved) and the preservation checks pass;
- reference fix, in ``REFERENCE_RUNS`` fresh containers: public, correctness and preservation
  all pass;
- negative controls on each target function (body replaced by ``return None``, and by
  returning its first argument): the correctness checks do not pass.

An invalid case is reported and excluded from scoring, never scored.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Mapping
from typing import Any

from scratchcoder.agent import execution, workspace
from scratchcoder.agent.cases import Case

REFERENCE_RUNS = 3


def control_source(text: str, path: str, name: str, kind: str) -> str:
    """The target function with its body replaced: ``none`` returns None, ``echo`` returns
    its first parameter (or None when it has none)."""
    tree = workspace.parse(text, path)
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    replaced = copy.deepcopy(node)
    replaced.decorator_list = []
    params = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
    value: ast.expr = (
        ast.Name(params[0], ast.Load()) if kind == "echo" and params else ast.Constant(None)
    )
    replaced.body = [ast.Return(value)]
    return ast.unparse(ast.fix_missing_locations(replaced))


def validate_case(
    case: Case,
    *,
    execute: execution.Execute,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def check(label: str, files: Mapping[str, str], kind: str, expect_pass: bool) -> None:
        if kind == "public":
            program, tests = execution.public_call(files, case.public_tests)
        else:
            checks = case.correctness if kind == "correctness" else case.preservation
            program, tests = execution.check_call(files, checks)
        run = execution.run_call(program, tests, execute=execute, sleep=sleep)
        calls.append(
            {
                "label": label,
                "kind": kind,
                "expect": "passed" if expect_pass else "not_passed",
                "status": run.status,
                "resolved": run.resolved,
                "host_failures": run.host_failures,
                "seconds": round(run.seconds, 4),
                "ok": run.resolved and run.passed == expect_pass,
            }
        )

    original = dict(case.files)
    check("original", original, "public", not case.public_fails_before_fix)
    check("original", original, "correctness", False)
    check("original", original, "preservation", True)
    reference = case.reference_files()
    for index in range(REFERENCE_RUNS):
        for kind in ("public", "correctness", "preservation"):
            check(f"reference-{index + 1}", reference, kind, True)
    for key in case.targets:
        path, name = key.split("::")
        for kind in ("none", "echo"):
            source = control_source(original[path], path, name, kind)
            edited = workspace.apply_function(
                original, workspace.Edit(path, name, source), case.allowed_files
            )
            check(f"control-{kind}:{key}", edited, "correctness", False)
    return {
        "case": case.id,
        "valid": all(call["ok"] for call in calls),
        "resolved": all(call["resolved"] for call in calls),
        "calls": calls,
    }
