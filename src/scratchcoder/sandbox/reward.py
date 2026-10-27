"""Out-of-process reward channel for RL rewards and RL screening.

This path is used ONLY to score reinforcement-learning rewards and to screen RL
candidates. Evaluation scoring of record keeps using :func:`scratchcoder.sandbox.runner.run`
and is untouched.

The hole this closes: the evaluation path puts the candidate, the hidden tests and a
completion marker in one program, so generated code can forge a pass (read the marker from
code-object constants, print it from an ``excepthook``, call ``os._exit(0)``), and an object
whose ``__eq__`` always returns ``True`` passes any assertion. RL would optimise against any
such hole.

The fix: the *expected* values never enter the container. For a supported hidden assertion
``assert CALL == LITERAL`` the host parses ``LITERAL`` with :func:`ast.literal_eval` and keeps
it; only ``CALL``'s source goes into the container with the candidate. Inside, the trusted
supervisor (``reward_program.py``) runs the candidate in a child process, evaluates each
``CALL`` there, and reports a type-preserving encoding of its value. The host decodes that and
compares type and value against the literal it kept. Forging the reported value earns nothing,
because the candidate never learns which value would pass; a value that cannot be encoded
(an always-equal object, a function, a live handle) is a failure. The reward denominator comes
from the task file, never from the container. For I/O tasks every case runs under its own
deadline and the per-attempt budget; the container returns bounded stdout and the host compares
with expected outputs it never shared.
"""

import ast
import math
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any
from uuid import uuid4

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.runner import ENDPOINT, Limits, Runtime, command

# Per-attempt and per-call defaults (seconds). The proposal fixes the per-attempt budget at 8 s.
DEFAULT_BUDGET_SECONDS = 8.0
DEFAULT_PER_CALL_SECONDS = 4.0
HOST_SLACK_SECONDS = 15.0
"""Extra wall time the host waits past the budget before declaring a host stall, covering
amd64 emulation startup. The container supervisor enforces the budget well within this."""

# Attempt statuses the caller sees.
COMPLETED = "completed"
"""The container ran; ``reward`` is the fraction of the task file's tests that passed."""
RESOURCE_KILL = "resource_kill"
"""The container was killed for resource use (OOM); the attempt scores ``reward`` 0.0."""
NOT_PARSEABLE = "not_parseable"
"""The task's hidden tests are outside the supported forms; ``reward`` is ``None``."""
HOST_TIMEOUT = "host_timeout"
"""The docker CLI or the host reader never returned in time; a host failure to retry."""
UNAVAILABLE = "unavailable"
"""Docker is missing, the daemon is down or the pinned image is absent; a host failure."""
HOST_ERROR = "host_error"
"""The supervisor produced no parseable framing; a host failure to retry."""

HOST_FAILURES = frozenset({HOST_TIMEOUT, UNAVAILABLE, HOST_ERROR})
"""Statuses that must be retried and never scored as a reward (not even 0.0)."""


@dataclass(frozen=True)
class RewardResult:
    """The outcome of one scored attempt. ``reward`` is ``None`` exactly for host failures
    and unparseable tasks, so a host failure can never be mistaken for a zero reward."""

    status: str
    reward: float | None
    passed: int = 0
    total: int = 0
    audit_flags: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    detail: str = ""

    @property
    def is_host_failure(self) -> bool:
        return self.status in HOST_FAILURES


@dataclass(frozen=True)
class Check:
    """One parsed hidden assertion: ``call`` is evaluated in the container, and the host
    applies ``predicate`` ("eq" or "falsy") to the reported value against ``expected``."""

    call: str
    predicate: str
    expected: object = None


@dataclass(frozen=True)
class FunctionChecks:
    setup: str
    checks: tuple[Check, ...]

    @property
    def total(self) -> int:
        return len(self.checks)


@dataclass(frozen=True)
class IoCases:
    inputs: tuple[str, ...]
    expected: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.inputs)


# --------------------------------------------------------------------------------------
# Host-side parsing of hidden tests (pure).
# --------------------------------------------------------------------------------------


def _safe_literal(node: ast.expr) -> tuple[bool, object]:
    """Return ``(True, value)`` when ``node`` is a safe literal of a representable type."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None
    try:
        _canonical_from_value(value)
    except TypeError:
        return False, None
    return True, value


def _parse_predicate(test: ast.expr) -> Check | None:
    """Translate one assertion's test expression into a :class:`Check`, or ``None`` when it
    is outside the supported forms."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        operator = test.ops[0]
        left, right = test.left, test.comparators[0]
        if isinstance(operator, ast.Eq):
            left_ok, left_value = _safe_literal(left)
            right_ok, right_value = _safe_literal(right)
            if right_ok and not left_ok:
                return Check(ast.unparse(left), "eq", right_value)
            if left_ok and not right_ok:
                return Check(ast.unparse(right), "eq", left_value)
            return None
        if isinstance(operator, ast.Is):
            for expression, other in ((left, right), (right, left)):
                if (
                    isinstance(other, ast.Constant)
                    and other.value in (None, True, False)
                    and not isinstance(expression, ast.Constant)
                ):
                    return Check(ast.unparse(expression), "eq", other.value)
            return None
        return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if isinstance(test.operand, ast.Constant):
            return None
        return Check(ast.unparse(test.operand), "falsy")
    return None


def parse_function_checks(hidden_tests: str) -> FunctionChecks | None:
    """Parse a function task's hidden assertions; ``None`` means the task is not parseable
    for the reward channel. Any top-level statement other than an import or a supported
    ``assert`` makes the whole task unparseable, so a partial parse never over-rewards."""
    try:
        tree = ast.parse(hidden_tests)
    except SyntaxError:
        return None
    setup_lines: list[str] = []
    checks: list[Check] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            setup_lines.append(ast.unparse(node))
            continue
        if not isinstance(node, ast.Assert):
            return None
        check = _parse_predicate(node.test)
        if check is None:
            return None
        checks.append(check)
    if not checks:
        return None
    return FunctionChecks("\n".join(setup_lines), tuple(checks))


def parse_io_cases(hidden_tests: str) -> IoCases:
    cases = loads(hidden_tests)
    return IoCases(tuple(case["input"] for case in cases), tuple(case["output"] for case in cases))


def parse_task(task: Task) -> FunctionChecks | IoCases | None:
    """The parsed hidden tests of a task, or ``None`` if unparseable for this mode."""
    if task.mode == "io":
        return parse_io_cases(task.hidden_tests)
    return parse_function_checks(task.hidden_tests)


def is_parseable(task: Task) -> bool:
    """Whether the reward channel can score ``task``. RL screening keeps only parseable
    tasks (I/O tasks always are; a function task is parseable iff every hidden assertion is
    a supported form)."""
    return parse_task(task) is not None


# --------------------------------------------------------------------------------------
# Type-preserving value comparison (host side).
# --------------------------------------------------------------------------------------


def _canonical_from_value(value: object, depth: int = 0) -> object:
    """A hashable, type-tagged canonical form of a Python value kept by the host. ``bool`` is
    checked before ``int`` so ``True`` never compares equal to ``1``."""
    if depth > 32:
        raise TypeError("value nested too deeply")
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, list):
        return ("list", tuple(_canonical_from_value(item, depth + 1) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_from_value(item, depth + 1) for item in value))
    if isinstance(value, set):
        return ("set", frozenset(_canonical_from_value(item, depth + 1) for item in value))
    if isinstance(value, frozenset):
        return ("frozenset", frozenset(_canonical_from_value(item, depth + 1) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            frozenset(
                (_canonical_from_value(key, depth + 1), _canonical_from_value(val, depth + 1))
                for key, val in value.items()
            ),
        )
    raise TypeError("unsupported value type")


def _canonical_from_wire(node: object, depth: int = 0) -> object:
    """The same canonical form, rebuilt from the container's type-tagged encoding."""
    if depth > 64 or not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("malformed encoded value")
    kind = node[0]
    if kind == "none":
        return ("none",)
    if kind == "bool":
        return ("bool", bool(node[1]))
    if kind == "int":
        if not isinstance(node[1], int) or isinstance(node[1], bool):
            raise ValueError("malformed int")
        return ("int", node[1])
    if kind == "float":
        if not isinstance(node[1], (int, float)) or isinstance(node[1], bool):
            raise ValueError("malformed float")
        return ("float", float(node[1]))
    if kind == "float_s":
        return ("float", {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[node[1]])
    if kind == "str":
        if not isinstance(node[1], str):
            raise ValueError("malformed str")
        return ("str", node[1])
    if kind == "bytes":
        return ("bytes", bytes(node[1]))
    if kind == "list":
        return ("list", tuple(_canonical_from_wire(item, depth + 1) for item in node[1]))
    if kind == "tuple":
        return ("tuple", tuple(_canonical_from_wire(item, depth + 1) for item in node[1]))
    if kind == "set":
        return ("set", frozenset(_canonical_from_wire(item, depth + 1) for item in node[1]))
    if kind == "frozenset":
        return ("frozenset", frozenset(_canonical_from_wire(item, depth + 1) for item in node[1]))
    if kind == "dict":
        return (
            "dict",
            frozenset(
                (_canonical_from_wire(pair[0], depth + 1), _canonical_from_wire(pair[1], depth + 1))
                for pair in node[1]
            ),
        )
    raise ValueError("unknown encoded kind")


def _wire_truthy(node: object) -> bool:
    """Truthiness of an encoded value, for the ``assert not CALL`` form."""
    if not isinstance(node, list) or not node:
        raise ValueError("malformed encoded value")
    kind = node[0]
    if kind == "none":
        return False
    if kind in ("bool", "int", "str"):
        return bool(node[1])
    if kind == "float":
        return float(node[1]) != 0.0
    if kind == "float_s":
        return True
    if kind in ("bytes", "list", "tuple", "set", "frozenset", "dict"):
        return bool(node[1])
    raise ValueError("unknown encoded kind")


def check_passes(check: Check, reported: dict[str, Any]) -> bool:
    """Whether one reported call result satisfies its check. Any malformed, errored or
    unserialisable report is a failure; equality is strict in type and value."""
    if not reported.get("ok"):
        return False
    try:
        if check.predicate == "falsy":
            return not _wire_truthy(reported["v"])
        return _canonical_from_wire(reported["v"]) == _canonical_from_value(check.expected)
    except (ValueError, KeyError, TypeError):
        return False


# --------------------------------------------------------------------------------------
# Static audit (pure, no execution).
# --------------------------------------------------------------------------------------

_AUDIT_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("_getframe", "_getframe"),
    ("co_consts", "co_consts"),
    ("excepthook", "excepthook"),
    ("_exit", "os_exit"),
    ("__file__", "dunder_file"),
    ("/proc", "proc_path"),
)
_AUDIT_MODULES = {
    "inspect": "inspect",
    "atexit": "atexit",
    "gc": "gc",
    "ctypes": "ctypes",
    "signal": "signal",
    "builtins": "builtins",
    "sys": None,
}


def audit_attempt(code: str) -> tuple[str, ...]:
    """Flag risky identifiers and patterns in an attempt without running it. Returns the
    sorted distinct flags; an empty tuple means nothing flagged. These feed the RL
    per-update audit count and the leakage stop; a flag is a signal, not a verdict."""
    flags: set[str] = set()
    for needle, flag in _AUDIT_SUBSTRINGS:
        if needle in code:
            flags.add(flag)
    try:
        tree: ast.AST | None = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                for name in names:
                    root = name.split(".")[0]
                    if root in _AUDIT_MODULES and _AUDIT_MODULES[root] is not None:
                        flags.add(str(_AUDIT_MODULES[root]))
            elif isinstance(node, ast.Attribute):
                if node.attr in ("_getframe", "co_consts", "excepthook", "_exit"):
                    flags.add({"_exit": "os_exit"}.get(node.attr, node.attr))
                if node.attr == "modules" and _is_name(node.value, "sys"):
                    flags.add("sys_modules")
            elif isinstance(node, ast.Name) and node.id in ("__builtins__", "builtins"):
                flags.add("builtins")
            elif isinstance(node, ast.FunctionDef) and node.name == "__eq__":
                flags.add("eq_override")
            elif isinstance(node, ast.Call) and _is_name(node.func, "open"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    target = node.args[0].value
                    if isinstance(target, str) and target.startswith("/tmp"):
                        flags.add("open_tmp")
    return tuple(sorted(flags))


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


# --------------------------------------------------------------------------------------
# Container invocation and host-status classification.
# --------------------------------------------------------------------------------------

ContainerRunner = Callable[[dict[str, Any]], "ContainerOutcome"]


@dataclass(frozen=True)
class ContainerOutcome:
    """The classified result of one container invocation, before reward computation."""

    host_status: str
    report: dict[str, Any] | None
    exit_code: int | None = None
    elapsed_seconds: float = 0.0


REWARD_SUPERVISOR = Path(__file__).with_name("reward_program.py").read_text(encoding="utf-8")


def run_reward_container(
    payload: dict[str, Any],
    *,
    limits: Limits,
    runtime: Runtime | None = None,
    docker: str | None = None,
) -> ContainerOutcome:
    """Run the reward supervisor in the hardened container and classify the host outcome.

    Host failures (``unavailable``, ``host_timeout``, ``host_error``) are reported distinctly
    from candidate outcomes, and an exit 137 with no report is a resource kill (OOM), not a
    candidate timeout. Every invocation removes its own randomly named container.
    """
    runtime = runtime or Runtime()
    resolved = docker or shutil.which("docker")
    if resolved is None:
        return ContainerOutcome(UNAVAILABLE, None)
    name = "scratchcoder-reward-" + uuid4().hex
    budget = float(payload["budget"])
    encoded = dumps(payload).encode()
    started = time.monotonic()
    captured = bytearray()
    process: subprocess.Popen[bytes] | None = None
    try:
        with TemporaryFile() as input_stream:
            input_stream.write(encoded)
            input_stream.seek(0)
            arguments = command(
                name, limits, docker=resolved, runtime=runtime, supervisor=REWARD_SUPERVISOR
            )
            process = subprocess.Popen(
                arguments,
                stdin=input_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            assert process.stdout is not None
            deadline = budget + HOST_SLACK_SECONDS
            host_stalled = False
            while True:
                if time.monotonic() - started > deadline:
                    host_stalled = True
                    process.kill()
                    break
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                if len(captured) < limits.output_bytes * 4 + 262144:
                    captured.extend(chunk)
            process.wait(timeout=5)
            process.stdout.close()
        elapsed = time.monotonic() - started
        if host_stalled:
            return ContainerOutcome(HOST_TIMEOUT, None, process.returncode, elapsed)
        if process.returncode == 125:
            return ContainerOutcome(UNAVAILABLE, None, 125, elapsed)
        if process.returncode == 0:
            try:
                report = loads(bytes(captured))
            except (JSONDecodeError, ValueError):
                return ContainerOutcome(HOST_ERROR, None, 0, elapsed)
            if not isinstance(report, dict) or report.get("status") != "ran":
                return ContainerOutcome(HOST_ERROR, None, 0, elapsed)
            return ContainerOutcome(COMPLETED, report, 0, elapsed)
        if process.returncode == 137:
            return ContainerOutcome(RESOURCE_KILL, None, 137, elapsed)
        return ContainerOutcome(HOST_ERROR, None, process.returncode, elapsed)
    except (OSError, ValueError, subprocess.SubprocessError):
        return ContainerOutcome(HOST_ERROR, None, None, time.monotonic() - started)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        try:
            subprocess.run(
                [resolved, "--host", ENDPOINT, "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass


# --------------------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------------------


def _function_payload(
    code: str, checks: FunctionChecks, *, budget: float, per_call: float, limits: Limits
) -> dict[str, Any]:
    return {
        "mode": "function",
        "code": code,
        "setup": checks.setup,
        "calls": [check.call for check in checks.checks],
        "per_call": per_call,
        "budget": budget,
        "output_bytes": limits.output_bytes,
        "site_packages": False,
    }


def _io_payload(
    code: str, cases: IoCases, *, budget: float, per_case: float, limits: Limits
) -> dict[str, Any]:
    return {
        "mode": "io",
        "code": code,
        "cases": list(cases.inputs),
        "per_case": per_case,
        "budget": budget,
        "output_bytes": limits.output_bytes,
        "site_packages": False,
    }


def _score_function(checks: FunctionChecks, report: dict[str, Any]) -> int:
    rows = report.get("calls")
    if not isinstance(rows, list):
        return 0
    by_index = {
        row["i"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("i"), int)
    }
    passed = 0
    for index, check in enumerate(checks.checks):
        row = by_index.get(index)
        if isinstance(row, dict) and check_passes(check, row):
            passed += 1
    return passed


def _score_io(cases: IoCases, report: dict[str, Any]) -> int:
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases.inputs):
        return 0
    passed = 0
    for row, expected in zip(rows, cases.expected, strict=True):
        if (
            isinstance(row, dict)
            and row.get("ran") is True
            and isinstance(row.get("stdout"), str)
            and row["stdout"].strip() == expected.strip()
        ):
            passed += 1
    return passed


def score_attempt(
    task: Task,
    code: str,
    *,
    budget: float = DEFAULT_BUDGET_SECONDS,
    per_call: float = DEFAULT_PER_CALL_SECONDS,
    limits: Limits | None = None,
    container: ContainerRunner | None = None,
) -> RewardResult:
    """Score one candidate completion for ``task``. The reward is ``passed / total`` with
    ``total`` taken from the task file. Unparseable tasks and host failures return a reward
    of ``None`` (never 0.0), so the RL loop can skip or retry instead of optimising against a
    host problem. A resource kill counts as a failed attempt (reward 0.0)."""
    limits = limits or Limits(timeout_seconds=min(60.0, max(1.0, budget)))
    flags = audit_attempt(code)
    parsed = parse_task(task)
    if parsed is None:
        return RewardResult(NOT_PARSEABLE, None, audit_flags=flags)
    total = parsed.total

    def default_runner(payload: dict[str, Any]) -> ContainerOutcome:
        return run_reward_container(payload, limits=limits)

    runner = container if container is not None else default_runner
    if isinstance(parsed, FunctionChecks):
        payload = _function_payload(code, parsed, budget=budget, per_call=per_call, limits=limits)
    else:
        payload = _io_payload(code, parsed, budget=budget, per_case=per_call, limits=limits)
    outcome = runner(payload)
    if outcome.host_status in HOST_FAILURES:
        return RewardResult(
            outcome.host_status, None, 0, total, flags, outcome.elapsed_seconds, "host failure"
        )
    if outcome.host_status == RESOURCE_KILL or outcome.report is None:
        return RewardResult(
            RESOURCE_KILL, 0.0, 0, total, flags, outcome.elapsed_seconds, "resource kill"
        )
    if isinstance(parsed, FunctionChecks):
        passed = _score_function(parsed, outcome.report)
    else:
        passed = _score_io(parsed, outcome.report)
    passed = max(0, min(passed, total))
    reward = passed / total if total else 0.0
    return RewardResult(COMPLETED, reward, passed, total, flags, outcome.elapsed_seconds)


def score_with_host_retries(
    task: Task,
    code: str,
    *,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> RewardResult:
    """Score an attempt, retrying host failures up to ``attempts`` times with linear backoff.
    If every try is a host failure, the last host-failure result is returned (reward ``None``),
    so the caller skips the attempt or triggers the reward-wait stop; a host failure is never
    scored as reward 0.0."""
    result = score_attempt(task, code, **kwargs)
    tries = 1
    while result.is_host_failure and tries < attempts:
        sleep(backoff_seconds * tries)
        result = score_attempt(task, code, **kwargs)
        tries += 1
    return result


@dataclass
class AuditTally:
    """Running counts for the RL per-update monitoring and the leakage stop."""

    attempts: int = 0
    flagged: int = 0
    flag_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, flags: Sequence[str]) -> None:
        self.attempts += 1
        if flags:
            self.flagged += 1
        for flag in flags:
            self.flag_counts[flag] = self.flag_counts.get(flag, 0) + 1

    @property
    def rate(self) -> float:
        return self.flagged / self.attempts if self.attempts else 0.0
