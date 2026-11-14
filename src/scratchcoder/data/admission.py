"""Task admission: a static gate, targeted negative controls and fresh-process reference checks.

A task is admitted to a development set or an RL pool only when nothing establishes a defect
in it. The module separates three kinds of evidence:

- **Established defects** exclude a task. Static ones are mechanical facts of the task text
  (:func:`static_gate`): every hidden check compares with ``None`` (``none_only_checks``); no
  check mentions a name the reference defines (``no_candidate_checks``); a test reads a
  module-level value only the reference assigns and the prompt never names
  (``reference_only_global``). Executed ones come from the checks below: the reference fails
  every one of its three fresh-process runs (``reference_failed``), or gives a different
  verdict between runs without a timeout being involved (``nondeterministic_reference``); a
  program that returns ``None`` for every call passes (``none_stub_passed``); a program that
  prints nothing passes an I/O task (``empty_output_passed``). Reviewed exclusions (overlap
  adjudication, read-confirmed quarantines, owner rulings) are supplied by the caller with
  their reasons.
- **Warnings** keep a task admitted but labelled: heuristic scan flags (a raise named in the
  prompt, a set-order or order-free pattern, a prompt-named global, a rounding clause, two or
  fewer checks), an echo program passing (the task may legitimately be an identity), and a
  verdict that varied only with timeouts (load-sensitive timing, not an established defect).
  Prompt-required globals and two-check tasks are therefore kept.
- **Pending**: a check is missing or unresolved (a host failure that outlasted its retries).
  Such a task is neither admitted nor excluded until a later run resolves it.

Every execution goes through the evaluation path of record
(:func:`scratchcoder.eval.harness.isolated_execution`, one fresh container and interpreter
per call, so each reference run has its own string-hash seed). Every outcome, including
unresolved ones, is appended to a cache row keyed by the task's content hash, the
evaluator's execution-policy hash and the runtime image, the run's kind and index. A row
whose identities differ from the current ones is never reused. Passing these checks does not
prove a task deterministic or its tests adequate; it only means no defect was established.
"""

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scratchcoder.data.verifier_scan import TaskScan, few_declared_checks
from scratchcoder.eval.tasks import Task

ADMISSION_VERSION = "admission-v1"
REFERENCE_RUNS = 3

STATIC_DEFECTS = {
    "R3": "none_only_checks",
    "R8:reference_only_global": "reference_only_global",
}
"""Scan hits that are established defects by themselves (mechanical facts of the text)."""

WARNING_RULES = {
    "R4": "side_effect_prompt_none_only",
    "R5": "raise_in_prompt",
    "R6": "tautological_compare",
    "R7": "set_order",
    "R8:prompt_named_global": "prompt_named_global",
    "R9": "rounding_clause",
    "R10": "order_free_clause",
    "R11": "redefinable_outer_bool_check",
}
"""Scan hits that stay provisional warnings unless a review or an execution establishes the
defect. R2 on some (not all) checks is a warning too: ``partly_candidate_independent``."""

EXECUTED_DEFECTS = (
    "reference_failed",
    "nondeterministic_reference",
    "none_stub_passed",
    "empty_output_passed",
)
EXECUTED_WARNINGS = ("echo_passed", "timing_unstable_reference")

FUNCTION_CONTROLS = ("none_stub", "echo_stub")
IO_CONTROLS = ("empty_output", "echo_input")

VERDICTS = frozenset({"passed", "failed", "timeout", "output_limit", "resource_limit", "oversized"})
"""Resolved outcomes. ``oversized`` (program and tests exceed the sandbox input limit) is the
executor's own verdict for a call it never starts; for a reference it means the task cannot be
evaluated as built."""


# --------------------------------------------------------------------------------------
# Identities and the run plan.
# --------------------------------------------------------------------------------------


def task_sha256(task: Task) -> str:
    """The content identity of a task: every field, canonically serialized."""
    text = json.dumps(asdict(task), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry_names(hidden_tests: str) -> list[str]:
    """The names the hidden tests call that are neither builtins nor imported: the
    candidate's functions and classes, which a control program must define."""
    import builtins

    tree = ast.parse(hidden_tests)
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
    names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return sorted(names - set(dir(builtins)) - imported)


def control_program(task: Task, kind: str) -> str:
    """A trivial program that a sound task's tests must reject."""
    if task.mode == "io":
        if kind == "empty_output":
            return "pass\n"
        if kind == "echo_input":
            return "import sys\nsys.stdout.write(sys.stdin.read())\n"
    else:
        names = _entry_names(task.hidden_tests) or ["solution"]
        if kind == "none_stub":
            body = "    return None\n"
        elif kind == "echo_stub":
            body = "    return args[0] if args else None\n"
        else:
            raise ValueError(f"unknown control {kind!r}")
        return "".join(f"def {name}(*args, **kwargs):\n{body}" for name in names)
    raise ValueError(f"unknown control {kind!r} for an {task.mode} task")


@dataclass(frozen=True)
class Run:
    """One planned execution: the task, the program kind (``reference`` or a control) and
    the run index (0 to 2 for the reference, 0 for a control)."""

    task_id: str
    kind: str
    index: int
    code: str
    tests: str
    mode: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.task_id, self.kind, self.index)


def plan(task: Task) -> list[Run]:
    """The reference runs and the negative controls of one task."""
    if task.solution is None:
        return []
    runs = [
        Run(task.id, "reference", index, task.solution, task.hidden_tests, task.mode)
        for index in range(REFERENCE_RUNS)
    ]
    for kind in IO_CONTROLS if task.mode == "io" else FUNCTION_CONTROLS:
        runs.append(
            Run(task.id, kind, 0, control_program(task, kind), task.hidden_tests, task.mode)
        )
    return runs


# --------------------------------------------------------------------------------------
# The cache: every outcome, bound to its identities.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Identities:
    """What a cached outcome depends on besides the task: the evaluator (the execution-policy
    SHA-256 of the path of record) and the runtime (the pinned image and its local ID)."""

    evaluator_sha256: str
    image: str
    image_id: str
    admission: str = ADMISSION_VERSION


def cache_row(
    run: Run,
    task_hash: str,
    identities: Identities,
    *,
    status: str,
    resolved: bool,
    host_failures: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "task_id": run.task_id,
        "task_sha256": task_hash,
        "kind": run.kind,
        "index": run.index,
        "code_sha256": hashlib.sha256(run.code.encode("utf-8")).hexdigest(),
        "status": status,
        "resolved": resolved,
        "host_failures": host_failures,
        "elapsed_seconds": round(elapsed_seconds, 3),
        **asdict(identities),
    }


def read_cache(path: Path) -> list[dict[str, Any]]:
    """Complete rows only; a torn final line (an interrupted write) is ignored."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows = []
    for line in raw[: raw.rfind(b"\n") + 1].splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or "task_id" not in row:
            raise ValueError(f"{path}: malformed admission row")
        rows.append(row)
    return rows


def usable(
    rows: Iterable[Mapping[str, Any]],
    tasks: Mapping[str, Task],
    identities: Identities,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """The latest resolved outcome per run whose task, evaluator and runtime identities
    match the current ones. Unresolved rows and rows of other identities are kept in the
    file (every outcome is recorded) but never satisfy a run."""
    current = asdict(identities)
    hashes = {task_id: task_sha256(task) for task_id, task in tasks.items()}
    found: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in hashes or row.get("task_sha256") != hashes[task_id]:
            continue
        if any(row.get(name) != value for name, value in current.items()):
            continue
        if not row.get("resolved") or row.get("status") not in VERDICTS:
            continue
        found[(task_id, row["kind"], int(row["index"]))] = row
    return found


# --------------------------------------------------------------------------------------
# Decisions.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A task's admission outcome: ``admitted``, ``excluded`` or ``pending``."""

    id: str
    status: str
    defects: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    reference_statuses: tuple[str, ...] = ()
    control_statuses: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {**asdict(self), "admission": ADMISSION_VERSION}


def static_gate(scan: TaskScan) -> tuple[list[str], list[str]]:
    """Established static defects and provisional warnings from a task's scan."""
    defects: list[str] = []
    warnings: list[str] = []
    r2 = 0
    for hit in scan.hits:
        tagged = f"{hit.rule}:{hit.tag}" if hit.tag else hit.rule
        if tagged in STATIC_DEFECTS or hit.rule in STATIC_DEFECTS:
            defects.append(STATIC_DEFECTS.get(tagged) or STATIC_DEFECTS[hit.rule])
        elif hit.rule == "R2":
            r2 += 1
        elif tagged in WARNING_RULES or hit.rule in WARNING_RULES:
            warnings.append(WARNING_RULES.get(tagged) or WARNING_RULES[hit.rule])
    if r2:
        if scan.declared_checks == 0:
            defects.append("no_candidate_checks")
        else:
            warnings.append("partly_candidate_independent")
    if few_declared_checks(scan) and scan.declared_checks:
        warnings.append("two_or_fewer_checks")
    return sorted(set(defects)), sorted(set(warnings))


def executed_findings(
    task: Task, outcomes: Mapping[tuple[str, str, int], Mapping[str, Any]]
) -> tuple[list[str], list[str], list[str], tuple[str, ...], dict[str, str]]:
    """Established defects, warnings, missing runs, reference statuses and control
    statuses from one task's cached outcomes."""
    defects: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    if task.solution is None:
        return ["no_reference"], [], [], (), {}
    statuses: list[str] = []
    for index in range(REFERENCE_RUNS):
        row = outcomes.get((task.id, "reference", index))
        if row is None:
            missing.append(f"reference:{index}")
        else:
            statuses.append(str(row["status"]))
    controls: dict[str, str] = {}
    for kind in IO_CONTROLS if task.mode == "io" else FUNCTION_CONTROLS:
        row = outcomes.get((task.id, kind, 0))
        if row is None:
            missing.append(kind)
        else:
            controls[kind] = str(row["status"])
    if len(statuses) == REFERENCE_RUNS:
        passes = statuses.count("passed")
        if passes == 0:
            defects.append("reference_failed")
        elif passes < REFERENCE_RUNS:
            others = {status for status in statuses if status != "passed"}
            if others <= {"timeout"}:
                warnings.append("timing_unstable_reference")
            else:
                defects.append("nondeterministic_reference")
    if controls.get("none_stub") == "passed":
        defects.append("none_stub_passed")
    if controls.get("empty_output") == "passed":
        defects.append("empty_output_passed")
    if controls.get("echo_stub") == "passed" or controls.get("echo_input") == "passed":
        warnings.append("echo_passed")
    return defects, warnings, missing, tuple(statuses), controls


def decide(
    task: Task,
    scan: TaskScan,
    outcomes: Mapping[tuple[str, str, int], Mapping[str, Any]],
    reviewed: Sequence[str] = (),
    reviewed_warnings: Sequence[str] = (),
) -> Decision:
    """Combine the static gate, the executed checks and reviewed exclusions or warnings."""
    static_defects, static_warnings = static_gate(scan)
    run_defects, run_warnings, missing, references, controls = executed_findings(task, outcomes)
    defects = sorted({*static_defects, *run_defects, *reviewed})
    warnings = sorted({*static_warnings, *run_warnings, *reviewed_warnings} - set(defects))
    if defects:
        status = "excluded"
    elif missing:
        status = "pending"
    else:
        status = "admitted"
    return Decision(
        task.id,
        status,
        tuple(defects),
        tuple(warnings),
        tuple(missing),
        references,
        controls,
    )


def union_counts(decisions: Sequence[Decision]) -> dict[str, Any]:
    """Counts by status and, for exclusions, by reason; reasons overlap, so the excluded
    total is the size of the union of task IDs, not the sum of the reason counts."""
    by_reason: dict[str, int] = {}
    by_warning: dict[str, int] = {}
    for decision in decisions:
        for reason in decision.defects:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        if decision.status == "admitted":
            for warning in decision.warnings:
                by_warning[warning] = by_warning.get(warning, 0) + 1
    return {
        "tasks": len(decisions),
        "admitted": sum(d.status == "admitted" for d in decisions),
        "excluded": sum(d.status == "excluded" for d in decisions),
        "pending": sum(d.status == "pending" for d in decisions),
        "excluded_by_reason_overlapping": dict(sorted(by_reason.items())),
        "admitted_with_warning_by_warning_overlapping": dict(sorted(by_warning.items())),
        "admitted_with_any_warning": sum(
            d.status == "admitted" and bool(d.warnings) for d in decisions
        ),
    }
