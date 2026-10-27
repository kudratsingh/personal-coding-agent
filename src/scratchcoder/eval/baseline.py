"""Constant-answer baselines: what an answer that solves nothing would score on a suite.

Owner ruling (2026-10-26): every evaluation suite reports, beside each model's score, the
mean reward a constant answer would earn, so a score near the constant baseline is
visible. A constant answer is one of the reward channel's degenerate answers
(``scratchcoder.sandbox.reward.degenerate_baseline``): a fixed return value or printed line
(``FUNCTION_DEGENERATE_ANSWERS`` / ``IO_DEGENERATE_ANSWERS``), a value copied from the
task's public tests or examples, or echoing the first argument or the input. Each task
takes its best answer. A suite reports the mean best-constant reward (the fraction of a
task's tests passed) and the share of tasks a constant answer fully passes, the figure
comparable with pass@1.

Pool tasks the reward channel can parse are scored on the host by ``degenerate_baseline``,
which runs no code. For the others, this module writes constant-answer programs (trusted
code, but still run through the sandbox like any candidate) and executes them through the
evaluation runner, one call per answer. A task that cannot be covered either way is
counted and reported as uncovered, never dropped. Benchmark programs are built here and
scored where the benchmark runs (GitHub Actions). ARC-Easy's baseline is the accuracy of
the best single fixed answer position, beside chance, both from the labels alone.

The pool's test split holds sealed hidden tests, so its baseline may only be computed
inside the logged unseal step that executes the test completions (``allow_test``).
"""

import ast
import builtins
import doctest
import json
import re
import textwrap
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.harness import HOST_RETRIES, isolated_execution
from scratchcoder.eval.parallel import OVERSIZED, Call, Execute, Outcome, run_calls
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.reward import (
    FUNCTION_DEGENERATE_ANSWERS,
    IO_DEGENERATE_ANSWERS,
    degenerate_baseline,
)

DEFINITION = (
    "best per task of the reward channel's degenerate answers: a fixed return value or "
    "printed line, a value copied from the public tests or examples, or echoing the first "
    "argument or the input"
)
CHECK_MARK = "__scratchcoder_constant_check__"
MAX_PUBLIC_VALUES = 8
"""At most this many distinct public example values are tried per task."""
ECHO = "echo"


@dataclass(frozen=True)
class Answer:
    """One constant answer: ``label`` names it; ``source`` is the Python expression a function
    returns, or the text an I/O program prints; ``echo`` answers return/print the input."""

    label: str
    source: str
    echo: bool = False


@dataclass(frozen=True)
class TaskBaseline:
    """A task's best constant answer. ``method`` is ``reward-channel`` (host, no code run),
    ``executed`` (constant programs through the evaluation runner) or ``uncovered``."""

    task_id: str
    source: str
    written_by: str
    method: str
    reward: float | None
    passed: int
    total: int
    answer: str
    reason: str = ""

    @property
    def full_pass(self) -> bool | None:
        return None if self.reward is None else self.total > 0 and self.passed == self.total


def _literal(value: object) -> str | None:
    """Python source for ``value`` that evaluates back to it, or None."""
    source = repr(value)
    try:
        again = ast.literal_eval(source)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return source if type(again) is type(value) and again == value else None


def _public_expected(source: str) -> list[object]:
    """Expected values of ``assert CALL == LITERAL`` lines in public test text."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    values: list[object] = []
    for node in tree.body:
        test = node.test if isinstance(node, ast.Assert) else None
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.left, ast.Call)
        ):
            try:
                values.append(ast.literal_eval(test.comparators[0]))
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
    return values


def _docstrings(prompt: str) -> list[str]:
    """The docstrings of a HumanEval-style prompt (completed with a ``pass`` body)."""
    try:
        tree = ast.parse(prompt + "    pass\n")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            docstring = ast.get_docstring(node)
            if docstring:
                found.append(docstring)
    return found


def _doctest_expected(prompt: str) -> list[object]:
    """Expected values of the ``>>>`` examples in a benchmark prompt's docstrings."""
    try:
        examples = [
            example
            for docstring in _docstrings(prompt)
            for example in doctest.DocTestParser().get_examples(docstring)
        ]
    except ValueError:
        return []
    values: list[object] = []
    for example in examples:
        try:
            values.append(ast.literal_eval(example.want.strip()))
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
    return values


def function_answers(public_values: Sequence[object]) -> list[Answer]:
    """The fixed values, then distinct public example values, then echo."""
    answers = [Answer(f"constant {value!r}", repr(value)) for value in FUNCTION_DEGENERATE_ANSWERS]
    seen = {answer.source for answer in answers}
    for value in public_values:
        source = _literal(value)
        if source is None or source in seen:
            continue
        seen.add(source)
        answers.append(Answer(f"public example {value!r:.60}", source))
        if len(seen) - len(FUNCTION_DEGENERATE_ANSWERS) >= MAX_PUBLIC_VALUES:
            break
    answers.append(Answer(ECHO, "args[0]", echo=True))
    return answers


def io_answers(task: Task) -> list[Answer]:
    answers = [Answer(f"constant {text!r}", text) for text in IO_DEGENERATE_ANSWERS]
    seen = {text.strip() for text in IO_DEGENERATE_ANSWERS}
    try:
        cases = json.loads(task.public_tests)
    except ValueError:
        cases = []
    for case in cases if isinstance(cases, list) else []:
        output = case.get("output") if isinstance(case, dict) else None
        if isinstance(output, str) and output.strip() not in seen:
            seen.add(output.strip())
            answers.append(Answer(f"public example {output.strip()!r:.60}", output))
    answers.append(Answer(ECHO, "", echo=True))
    return answers


def candidate_names(tests: str) -> list[str]:
    """Names the tests call as plain functions that the candidate must define: not builtins,
    not imported and not defined by the tests themselves."""
    tree = ast.parse(tests)
    foreign = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            foreign.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            foreign.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            foreign.update(
                name.id
                for target in targets
                for name in ast.walk(target)
                if isinstance(name, ast.Name)
            )
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return sorted(called - foreign)


def constant_function_program(names: Sequence[str], answer: Answer) -> str:
    """Every named function returns the answer (or its first argument, for echo)."""
    return "".join(
        f"def {name}(*args, **kwargs):\n    return {answer.source}\n\n\n" for name in names
    )


def counting_tests(hidden_tests: str) -> tuple[str, int]:
    """The hidden tests with each top-level statement run on its own: an ``assert`` prints
    ``CHECK_MARK index 1`` when it holds and ``CHECK_MARK index 0`` when it raises; other
    statements (imports, helpers) run unguarded by the count. Returns the test text and
    the number of asserts, the task's test count."""
    tree = ast.parse(hidden_tests)
    blocks: list[str] = []
    total = 0
    for node in tree.body:
        body = textwrap.indent(ast.unparse(node), "    ")
        if isinstance(node, ast.Assert):
            blocks.append(
                f"try:\n{body}\nexcept BaseException:\n"
                f"    print({CHECK_MARK!r}, {total}, 0, flush=True)\n"
                f"else:\n    print({CHECK_MARK!r}, {total}, 1, flush=True)\n"
            )
            total += 1
        else:
            blocks.append(f"try:\n{body}\nexcept BaseException:\n    pass\n")
    return "".join(blocks), total


def counted_passes(stdout: str, total: int) -> int:
    """The asserts reported as holding; an assert with no report (cut off) failed."""
    verdicts: dict[int, bool] = {}
    for match in re.finditer(rf"(?m)^{CHECK_MARK} (\d+) ([01])$", stdout):
        index = int(match[1])
        if index < total and index not in verdicts:
            verdicts[index] = match[2] == "1"
    return sum(verdicts.values())


def io_program(answer: Answer) -> str:
    if answer.echo:
        return "import sys\nsys.stdout.write(sys.stdin.read())\n"
    return f"print({answer.source!r})\n"


def _io_passes(answer: Answer, cases: Sequence[Mapping[str, str]]) -> int:
    """Cases the answer passes under the runner's rule (stripped stdout equals output)."""
    return sum(
        (case["input"] if answer.echo else answer.source).strip() == case["output"].strip()
        for case in cases
    )


@dataclass
class _Pending:
    task: Task
    answers: list[Answer]
    total: int
    outcomes: dict[int, Outcome]


def _summary(rows: Sequence[TaskBaseline]) -> dict[str, Any]:
    def stats(group: Sequence[TaskBaseline]) -> dict[str, Any]:
        scored = [row for row in group if row.reward is not None]
        return {
            "tasks": len(group),
            "covered": len(scored),
            "uncovered": len(group) - len(scored),
            "mean_reward": (
                sum(row.reward or 0.0 for row in scored) / len(scored) if scored else None
            ),
            "full_pass_share": (
                sum(bool(row.full_pass) for row in scored) / len(scored) if scored else None
            ),
        }

    by_source: dict[str, list[TaskBaseline]] = defaultdict(list)
    by_author: dict[str, list[TaskBaseline]] = defaultdict(list)
    methods: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[row.source].append(row)
        by_author[row.written_by].append(row)
        methods[row.method] += 1
    return {
        **stats(rows),
        "by_source": {name: stats(group) for name, group in sorted(by_source.items())},
        "by_authorship": {name: stats(group) for name, group in sorted(by_author.items())},
        "methods": dict(sorted(methods.items())),
        "uncovered_tasks": [
            {"task_id": row.task_id, "reason": row.reason} for row in rows if row.reward is None
        ],
    }


def pool_constant_baseline(
    tasks: Sequence[Task],
    *,
    execute: Execute = isolated_execution,
    workers: int = 8,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    allow_test: bool = False,
) -> dict[str, Any]:
    """The constant-answer baseline of pool tasks, per task and summarized.

    ``allow_test`` is set only by the logged unseal step; without it a test-split task is
    refused before anything is read from its hidden tests.
    """
    if any(task.split == "test" for task in tasks) and not allow_test:
        raise PermissionError(
            "the pool test split's baseline reads sealed hidden tests; it is computed only "
            "inside the logged unseal step (eval-execute with an unseal record)"
        )
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("duplicate task IDs")
    rows: dict[str, TaskBaseline] = {}
    pending: dict[str, _Pending] = {}
    calls: list[Call] = []

    for task in sorted(tasks, key=lambda item: item.id):
        parsed = degenerate_baseline(task)
        if parsed is not None:
            rows[task.id] = TaskBaseline(
                task.id,
                task.source,
                task.written_by,
                "reward-channel",
                parsed.reward,
                parsed.passed,
                parsed.total,
                parsed.answer,
            )
            continue
        if task.mode == "io":
            cases = json.loads(task.hidden_tests)
            answers = io_answers(task)
            pending[task.id] = _Pending(task, answers, len(cases), {})
            for index, answer in enumerate(answers):
                calls.append(Call((task.id, index), io_program(answer), task.hidden_tests, "io"))
            continue
        try:
            names = candidate_names(task.hidden_tests)
            tests, total = counting_tests(task.hidden_tests)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            rows[task.id] = _uncovered(task, "hidden tests do not parse")
            continue
        if not names or not total:
            reason = "the hidden tests call no candidate function" if total else "no asserts"
            rows[task.id] = _uncovered(task, reason, total)
            continue
        answers = function_answers(_public_expected(task.public_tests))
        pending[task.id] = _Pending(task, answers, total, {})
        for index, answer in enumerate(answers):
            calls.append(
                Call((task.id, index), constant_function_program(names, answer), tests, "function")
            )

    seconds = 0.0

    def collect(outcome: Outcome) -> None:
        nonlocal seconds
        seconds += outcome.elapsed_seconds
        task_id, index = _key(outcome.key)
        pending[task_id].outcomes[index] = outcome

    run_calls(calls, execute, on_outcome=collect, workers=workers, retries=retries, sleep=sleep)
    for task_id, item in pending.items():
        rows[task_id] = _executed(item)
    ordered = [rows[task.id] for task in sorted(tasks, key=lambda item: item.id)]
    return {
        "kind": "constant-baseline",
        "suite": "pool",
        "definition": DEFINITION,
        "summary": _summary(ordered),
        "executed_calls": len(calls),
        "sandbox_seconds": seconds,
        "per_task": [asdict(row) for row in ordered],
    }


def _key(key: Hashable) -> tuple[str, int]:
    if not isinstance(key, tuple) or len(key) != 2:
        raise ValueError("unexpected call key")
    task_id, index = key
    return str(task_id), int(index)


def _uncovered(task: Task, reason: str, total: int = 0) -> TaskBaseline:
    return TaskBaseline(
        task.id, task.source, task.written_by, "uncovered", None, 0, total, "none", reason
    )


def _executed(item: _Pending) -> TaskBaseline:
    task = item.task
    best_passed, best_label = 0, "none"
    cases = json.loads(task.hidden_tests) if task.mode == "io" else []
    for index, answer in enumerate(item.answers):
        outcome = item.outcomes.get(index)
        if outcome is None or not outcome.resolved:
            return _uncovered(task, "a constant-answer call stayed unresolved", item.total)
        if outcome.status == OVERSIZED:
            return _uncovered(task, "the runner refused the input", item.total)
        if task.mode == "io":
            passed = _io_passes(answer, cases)
            if (passed == len(cases)) != outcome.passed:
                reason = "the sandbox verdict disagrees with the output comparison"
                return _uncovered(task, reason, item.total)
        else:
            passed = counted_passes(outcome.stdout, item.total)
        if passed > best_passed:
            best_passed, best_label = passed, answer.label
    return TaskBaseline(
        task.id,
        task.source,
        task.written_by,
        "executed",
        best_passed / item.total,
        best_passed,
        item.total,
        best_label,
    )


def arc_constant_baseline(tasks: Sequence[ChoiceTask]) -> dict[str, Any]:
    """The best single fixed answer position and chance, from the answer labels alone."""
    if not tasks:
        raise ValueError("ARC baseline needs tasks")
    positions = [task.labels.index(task.answer) for task in tasks]
    width = max(len(task.labels) for task in tasks)
    accuracy = {
        position: sum(found == position for found in positions) / len(tasks)
        for position in range(width)
    }
    best = max(accuracy, key=lambda position: (accuracy[position], -position))
    return {
        "kind": "constant-baseline",
        "suite": "arc-easy",
        "definition": "always answer the same position; chance is the mean of 1 / choices",
        "tasks": len(tasks),
        "best_position": best,
        "best_position_accuracy": accuracy[best],
        "accuracy_by_position": {str(position): value for position, value in accuracy.items()},
        "chance": sum(1 / len(task.labels) for task in tasks) / len(tasks),
    }


def benchmark_answers(task: Task, suite: str) -> list[Answer]:
    """Constant answers for a HumanEval-family or MBPP+ task (examples from the prompt's
    doctests, or from MBPP+'s public asserts)."""
    if suite == "mbpp-plus":
        return function_answers(_public_expected(task.public_tests))
    return function_answers(_doctest_expected(task.prompt))


def benchmark_constant_programs(task: Task, suite: str) -> list[tuple[str, str]]:
    """(label, program) pairs, one per constant answer, for a benchmark task.

    HumanEval and HumanEval+: the prompt (keeping its imports and helper functions), a
    ``pass`` body, then the entry point redefined to return the constant. MBPP+: the
    public test imports, then the function its public asserts call. A task whose entry
    point cannot be found gets no programs (it is reported as uncovered).
    """
    answers = benchmark_answers(task, suite)
    if suite == "mbpp-plus":
        imports = "\n".join(
            line for line in task.public_tests.splitlines() if line.startswith(("import ", "from "))
        )
        names = candidate_names(task.public_tests)
        if not names:
            return []
        prefix = imports + "\n\n" if imports else ""
        return [
            (answer.label, prefix + constant_function_program(names, answer)) for answer in answers
        ]
    entry = task.entry_point
    if entry is None:
        return []
    prefix = ""
    try:
        ast.parse(task.prompt + "    pass\n")
        prefix = task.prompt + "    pass\n\n\n"
    except SyntaxError:
        prefix = ""
    return [
        (answer.label, prefix + constant_function_program([entry], answer)) for answer in answers
    ]


def pool_baseline_report(
    tasks_path: Path,
    *,
    split: str,
    execute: Execute = isolated_execution,
    workers: int = 8,
    retries: int = HOST_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    allow_test: bool = False,
) -> dict[str, Any]:
    """:func:`pool_constant_baseline` for a pool task file, with its identity recorded."""
    from scratchcoder.eval.protocols import load_suite_tasks

    if split == "test" and not allow_test:
        raise PermissionError(
            "the pool test split's baseline is computed only inside the logged unseal step"
        )
    tasks, _ = load_suite_tasks("pool", split, tasks_path)
    report = pool_constant_baseline(
        tasks,
        execute=execute,
        workers=workers,
        retries=retries,
        sleep=sleep,
        allow_test=allow_test,
    )
    commit, dirty = git_state()
    return {
        **report,
        "split": split,
        "task_file_sha256": file_sha256(tasks_path),
        "git_commit": commit,
        "dirty": dirty,
        "created_at": utc_now(),
    }


def save_baseline(path: Path, report: dict[str, Any]) -> None:
    """Publish a baseline report once; an existing file is never replaced."""
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    atomic_json(path, report)
