"""The identity of what decides an evaluation verdict, so a resumed results file never mixes two.

A results file may be written over several runs (windows). Every verdict in it must come
from one execution policy: the same limits, the same rule for what counts as a verdict and
what counts as a host failure, and the same code producing both. :func:`execution_policy`
records that policy in two ways at once:

- **By value.** The limits the evaluation runner really applies (a function call, and I/O
  calls with several case counts, through :func:`scratchcoder.eval.harness.execution_limits`),
  the scored statuses, the retry count and backoff, the input limit the caller passes, the
  image, the host watchdog and the protections the supervisor must establish. A changed
  constant, or a function replaced in memory, shows here.
- **By code.** A SHA-256 of the syntax tree of every unit in :data:`VERDICT_CODE`, with
  docstrings removed (comments never reach the tree). Editing a docstring or a comment, or
  any file not listed, leaves the identity unchanged; editing the code of a listed unit
  changes it.

:data:`VERDICT_CODE` is a deliberate enumeration, not an import scan: the sandbox runner,
its in-container supervisor and the deadline budget as whole modules, plus the named
definitions and every import of the harness and parallel modules that turn a sandbox result
into a verdict, a retry or ``oversized``, and of the executor that decides which attempts
run, what an unexecuted attempt scores and which status counts as passed. A renamed or
removed listed definition is an error, so a refactor has to update the list on purpose.
Changing the list or the record's shape bumps :data:`EXECUTION_POLICY_VERSION`.

The reward channel's static audit only flags passing attempts; it never changes a verdict.
Its identity (:func:`audit_policy`) is recorded beside the execution policy, and in every
window, so mixed flag counts can be seen, but it is not part of the verdict identity.

The syntax-tree dump depends on the Python minor version, which is recorded with the hashes:
running under another Python changes the identity.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scratchcoder.eval import harness, parallel
from scratchcoder.sandbox import budget, program, runner

EXECUTION_POLICY_VERSION = 2
"""Bump when :data:`VERDICT_CODE`, the probes or the shape of the policy record change.

Version 2 (2026-11-14) adds ``supervisor_protections``: the path-of-record supervisor now
refuses to run a candidate unless it is non-dumpable and handles the catchable signals, and
the host accepts only one nonce-bearing result line. A results file started under version 1
(the unprotected supervisor) is refused on resume."""

CodeUnits = Sequence[tuple[str, tuple[str, ...] | None]]

VERDICT_CODE: CodeUnits = (
    # The runner: limits, the hardened docker command, the host watchdog, the status mapping.
    ("scratchcoder.sandbox.runner", None),
    # The supervisor source sent into the container: the deadline, the marker, the I/O check.
    ("scratchcoder.sandbox.program", None),
    # The I/O deadline rule.
    ("scratchcoder.sandbox.budget", None),
    # The limits per call, the default execute, and what is a verdict versus a host failure.
    (
        "scratchcoder.eval.harness",
        (
            "SCORED_STATUSES",
            "HOST_RETRIES",
            "HOST_BACKOFF_SECONDS",
            "HOST_BACKOFF_MAX_SECONDS",
            "execution_limits",
            "isolated_execution",
            "is_host_failure",
            "host_backoff",
        ),
    ),
    # The oversized rule, the retry loop and the scheduler that applies the input limit.
    (
        "scratchcoder.eval.parallel",
        (
            "Execute",
            "OVERSIZED",
            "INPUT_LIMIT",
            "Call",
            "Outcome",
            "_size",
            "resolve_call",
            "run_calls",
        ),
    ),
    # The executor's decisions: unexecuted statuses, the call per attempt, the input limit it
    # passes, which outcomes are scored and which status is a pass.
    (
        "scratchcoder.eval.executor",
        (
            "NO_PROGRAM",
            "CONTEXT_LIMIT",
            "CALL_INPUT_LIMIT",
            "attempt_plan",
            "scored_status",
            "attempt_passed",
            "run_verdict_calls",
        ),
    ),
)
"""The code that turns a candidate program into a verdict: (module, names). ``None`` takes the
whole module; a tuple takes those top-level definitions plus every top-level import."""

AUDIT_CODE: CodeUnits = (("scratchcoder.sandbox.reward", None),)
"""The static audit's code. The whole module, which is broader than ``audit_attempt`` and
its helpers; the identity is only recorded, so a broad hash costs nothing."""

IO_PROBE_CASES = (0, 1, 40, 99, 100, 1000)
"""I/O case counts whose limits are recorded: the base, one case, the sandbox test's 40, the
pool's 99, the runner's maximum of 100, and a count far past the 60 s cap."""


def module_source(name: str) -> str:
    """The source text of module ``name``, located without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise ValueError(f"{name}: no Python source to identify")
    return Path(spec.origin).read_text(encoding="utf-8")


class _DropStrings(ast.NodeTransformer):
    """Remove every statement that is only a string literal: docstrings, and the attribute
    docstrings written under a constant. Such a statement does nothing when it runs."""

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def _defined_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return {node.name.id}
    return set()


def unit_sha256(module: str, names: tuple[str, ...] | None, source: str | None = None) -> str:
    """SHA-256 of one code unit's syntax tree without docstrings or source positions."""
    tree = _DropStrings().visit(ast.parse(module_source(module) if source is None else source))
    assert isinstance(tree, ast.Module)
    if names is None:
        nodes: list[ast.stmt] = tree.body
    else:
        nodes, found = [], set()
        for node in tree.body:
            defined = _defined_names(node) & set(names)
            if defined or isinstance(node, ast.Import | ast.ImportFrom):
                nodes.append(node)
                found |= defined
        missing = sorted(set(names) - found)
        if missing:
            raise ValueError(
                f"{module}: {', '.join(missing)} not found; the verdict code moved, so update "
                "the enumeration and bump EXECUTION_POLICY_VERSION"
            )
    text = "\n".join(ast.dump(node) for node in nodes)
    return hashlib.sha256(text.encode()).hexdigest()


def code_identity(units: CodeUnits) -> dict[str, Any]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "units": {module: unit_sha256(module, names) for module, names in units},
    }


def callable_name(execute: Callable[..., Any]) -> str:
    """The qualified name of ``execute``, or of its class for a callable instance."""
    qualname = getattr(execute, "__qualname__", None)
    if isinstance(qualname, str):
        return f"{getattr(execute, '__module__', '?')}.{qualname}"
    kind = type(execute)
    return f"{kind.__module__}.{kind.__qualname__} instance"


def deadline_rule() -> str:
    """The deadline rule in words, built from the same values the limits come from."""
    function = harness.execution_limits("assert True", "function")
    return (
        f"function: {function.timeout_seconds:g} s; "
        f"I/O: {budget.IO_BASE_SECONDS:g} s + {budget.IO_SECONDS_PER_CASE:g} s per case, "
        f"at most {budget.MAX_SECONDS:g} s; {function.memory_mb} MiB"
    )


def _io_tests(cases: int) -> str:
    return json.dumps([{"input": "", "output": ""}] * cases)


def execution_policy(
    execute: Callable[..., Any],
    retries: int,
    *,
    unexecuted_statuses: Sequence[str],
    input_limit: int,
) -> dict[str, Any]:
    """The verdict identity of an execution with ``execute`` and ``retries`` host retries.

    ``unexecuted_statuses`` are the statuses the caller writes without running anything (the
    executor's ``no_program`` and ``context_limit``); ``input_limit`` is the limit the caller
    passes to :func:`scratchcoder.eval.parallel.run_calls`."""
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    return {
        "version": EXECUTION_POLICY_VERSION,
        "execute": callable_name(execute),
        "image": runner.IMAGE,
        "deadline_rule": deadline_rule(),
        "limits": {
            "function": asdict(harness.execution_limits("assert True", "function")),
            "io_by_cases": {
                str(cases): asdict(harness.execution_limits(_io_tests(cases), "io"))
                for cases in IO_PROBE_CASES
            },
        },
        "host_watchdog_seconds": runner.HOST_WATCHDOG_SECONDS,
        "supervisor_protections": list(program.REQUIRED_PROTECTIONS),
        "scored_statuses": sorted(harness.SCORED_STATUSES),
        "host_retries": retries,
        "host_backoff_seconds": [harness.host_backoff(n) for n in range(1, retries + 1)],
        "input_limit": input_limit,
        "oversized_status": parallel.OVERSIZED,
        "unexecuted_statuses": sorted(unexecuted_statuses),
        "code": code_identity(VERDICT_CODE),
    }


def audit_policy() -> dict[str, Any]:
    """The identity of the static audit recorded on passing attempts (not a verdict input)."""
    return {
        "function": "scratchcoder.sandbox.reward.audit_attempt",
        "applies_to": "passing attempts with a program",
        "code": code_identity(AUDIT_CODE),
    }


def policy_sha256(policy: Mapping[str, Any]) -> str:
    text = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def policy_differences(recorded: Any, current: Mapping[str, Any]) -> list[str]:
    """The parts of ``current`` that ``recorded`` does not match, named for an error message."""
    if not isinstance(recorded, Mapping):
        return ["the whole policy (none recorded)"]
    changed: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        old, new = recorded.get(key), current.get(key)
        if old == new:
            continue
        if key == "code" and isinstance(old, Mapping) and isinstance(new, Mapping):
            if old.get("python") != new.get("python"):
                changed.append("code: python version")
            old_units, new_units = old.get("units") or {}, new.get("units") or {}
            changed.extend(
                f"code: {unit}"
                for unit in sorted(set(old_units) | set(new_units))
                if old_units.get(unit) != new_units.get(unit)
            )
        else:
            changed.append(key)
    return changed
