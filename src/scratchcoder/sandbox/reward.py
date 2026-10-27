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

Host failures are separated from candidate failures by framing: the supervisor writes a
``started`` frame tagged with a random per-invocation nonce before any candidate process
exists. A problem before that frame (no docker binary, an unreachable daemon, an absent
image, a hung CLI) is a host failure that is retried and never scored. Once ``started`` has
arrived, a lost, malformed or oversized report is attributed to the candidate and scores
0.0, so a candidate cannot convert its own failure into a skipped attempt.
"""

import ast
import base64
import math
import os
import re
import secrets
import selectors
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any
from uuid import uuid4

from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox.budget import io_seconds
from scratchcoder.sandbox.reward_program import LineSplitter
from scratchcoder.sandbox.runner import ENDPOINT, Limits, Runtime, command

FUNCTION_BUDGET_SECONDS = 8.0
"""Per-attempt wall budget for a function task. An I/O task's budget is the pool
verification's rule instead (:func:`attempt_budget`)."""
PER_CALL_SECONDS = 4.0
"""Deadline for one hidden-assertion call of a function task, inside the attempt budget."""
HOST_SLACK_SECONDS = 15.0
"""Extra wall time the host waits past the budget before declaring a host stall, covering
amd64 emulation startup. The container supervisor enforces the budget well within this."""
MAX_IO_CASES = 100
"""The same case-count bound the evaluation runner enforces."""

# Attempt statuses the caller sees.
COMPLETED = "completed"
"""The container ran; ``reward`` is the fraction of the task file's tests that passed."""
RESOURCE_KILL = "resource_kill"
"""The candidate hit a resource limit: the container's out-of-memory killer fired during
the attempt (calls or cases it killed count as failed), or the container exited 137 after
the candidate started and before it reported (the attempt scores 0.0)."""
MALFORMED_REPORT = "malformed_report"
"""The candidate started but the report was lost or malformed (for example the candidate
disrupted the supervisor). Attributed to the candidate: the attempt scores 0.0."""
NOT_PARSEABLE = "not_parseable"
"""The task's hidden tests are outside the supported forms; ``reward`` is ``None``."""
HOST_TIMEOUT = "host_timeout"
"""The docker CLI or the host reader never returned in time; a host failure to retry."""
UNAVAILABLE = "unavailable"
"""Docker is missing, the daemon is unreachable or the pinned image is absent; a host
failure."""
HOST_ERROR = "host_error"
"""The container never reported that the candidate started; a host failure to retry."""

HOST_FAILURES = frozenset({HOST_TIMEOUT, UNAVAILABLE, HOST_ERROR})
"""Statuses that must be retried and never scored as a reward (not even 0.0)."""

_DAEMON_UNREACHABLE = (
    "cannot connect to the docker daemon",
    "failed to connect to the docker api",
    "error during connect",
    "is the docker daemon running",
)


@dataclass(frozen=True)
class RewardResult:
    """The outcome of one scored attempt. ``reward`` is ``None`` exactly for host failures
    and unparseable tasks, so a host failure can never be mistaken for a zero reward.

    ``unfinished`` counts calls or cases that produced no result in time (a per-call or
    per-case timeout, or the per-attempt budget running out); a correct solution with a
    nonzero ``unfinished`` was under-scored by the budget. ``outcomes`` holds one pass flag
    per hidden check, in task-file order. ``budget_seconds`` is the per-attempt wall budget
    the attempt ran under (``None`` when nothing ran: an unparseable task)."""

    status: str
    reward: float | None
    passed: int = 0
    total: int = 0
    audit_flags: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    detail: str = ""
    unfinished: int = 0
    outcomes: tuple[bool, ...] = ()
    budget_seconds: float | None = None

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
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None
    try:
        _canonical_from_value(value)
    except TypeError:
        return False, None
    return True, value


def _call_is_safe(node: ast.expr) -> bool:
    """A ``CALL`` must not compare anything itself: a comparison inside it would carry the
    expected value into the container and be judged there by candidate objects."""
    if isinstance(node, (ast.BoolOp, ast.Constant)):
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return False
    return not any(isinstance(sub, ast.Compare) for sub in ast.walk(node))


def _parse_predicate(test: ast.expr) -> Check | None:
    """Translate one assertion's test expression into a :class:`Check`, or ``None`` when it
    is outside the supported forms."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        operator = test.ops[0]
        left, right = test.left, test.comparators[0]
        if isinstance(operator, ast.Eq):
            left_ok, left_value = _safe_literal(left)
            right_ok, right_value = _safe_literal(right)
            if right_ok and not left_ok and _call_is_safe(left):
                return Check(ast.unparse(left), "eq", right_value)
            if left_ok and not right_ok and _call_is_safe(right):
                return Check(ast.unparse(right), "eq", left_value)
            return None
        if isinstance(operator, ast.Is):
            for expression, other in ((left, right), (right, left)):
                # Identity, not equality: ``0 in (None, True, False)`` would accept ``is 0``.
                if (
                    isinstance(other, ast.Constant)
                    and (other.value is None or other.value is True or other.value is False)
                    and _call_is_safe(expression)
                ):
                    return Check(ast.unparse(expression), "eq", other.value)
            return None
        return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        if not _call_is_safe(test.operand):
            return None
        return Check(ast.unparse(test.operand), "falsy")
    return None


def _size(text: str) -> int:
    return len(text.encode("utf-8", errors="surrogatepass"))


def parse_function_checks(
    hidden_tests: str, *, limits: Limits | None = None
) -> FunctionChecks | None:
    """Parse a function task's hidden assertions; ``None`` means the task is not parseable
    for the reward channel. Any top-level statement other than an import or a supported
    ``assert`` makes the whole task unparseable, so a partial parse never over-rewards. So
    does an expected value too large to be reported within one result row, or call sources
    larger than the sandbox input limit."""
    limits = limits or Limits()
    row_bytes = limits.output_bytes
    try:
        tree = ast.parse(hidden_tests)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
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
        if check.predicate == "eq" and _honest_row_bytes(len(checks), check.expected) > row_bytes:
            return None
        checks.append(check)
    if not checks:
        return None
    setup = "\n".join(setup_lines)
    if _size(setup) + sum(_size(check.call) for check in checks) > limits.input_bytes:
        return None
    return FunctionChecks(setup, tuple(checks))


def parse_io_cases(hidden_tests: str, *, limits: Limits | None = None) -> IoCases | None:
    """Parse an I/O task's cases; ``None`` when the cases are malformed, more than the
    evaluation runner allows, an input is larger than the sandbox input limit, or an expected
    output is larger than the per-case output cap (no stdout could ever match it). Expected
    outputs never enter the container, so their total size is not limited."""
    limits = limits or Limits()
    try:
        cases = loads(hidden_tests)
    except ValueError:
        return None
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_IO_CASES:
        return None
    inputs: list[str] = []
    expected: list[str] = []
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("input"), str)
            or not isinstance(case.get("output"), str)
        ):
            return None
        if _size(case["input"]) > limits.input_bytes:
            return None
        if _size(case["output"].strip()) > limits.output_bytes:
            return None
        inputs.append(case["input"])
        expected.append(case["output"])
    return IoCases(tuple(inputs), tuple(expected))


def parse_task(task: Task, *, limits: Limits | None = None) -> FunctionChecks | IoCases | None:
    """The parsed hidden tests of a task, or ``None`` if unparseable for this mode."""
    if task.mode == "io":
        return parse_io_cases(task.hidden_tests, limits=limits)
    return parse_function_checks(task.hidden_tests, limits=limits)


def is_parseable(task: Task, *, limits: Limits | None = None) -> bool:
    """Whether the reward channel can score ``task``. RL screening keeps only parseable
    tasks: a function task is parseable iff every hidden assertion is a supported form; an
    I/O task iff its cases are well formed and every expected output fits the output cap."""
    return parse_task(task, limits=limits) is not None


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


def _wire_from_value(value: object) -> object:
    """The encoding the honest driver would report for ``value`` (used to size rows)."""
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float_s", "nan"]
        if math.isinf(value):
            return ["float_s", "inf" if value > 0 else "-inf"]
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", list(value)]
    if isinstance(value, dict):
        return ["dict", [[_wire_from_value(k), _wire_from_value(v)] for k, v in value.items()]]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [type(value).__name__, [_wire_from_value(item) for item in value]]
    raise TypeError("unsupported value type")


def _honest_row_bytes(index: int, value: object) -> int:
    """The length of the result row an honest driver writes for ``value``; too large to
    report (or to encode at all) is ``sys.maxsize``."""
    try:
        return len(dumps({"i": index, "ok": True, "v": _wire_from_value(value)}))
    except (ValueError, TypeError, RecursionError):
        return sys.maxsize


_SCALAR_TYPES: dict[str, type] = {"bool": bool, "int": int, "float": float, "str": str}
_SPECIAL_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _canonical_from_wire(node: object, depth: int = 0) -> object:
    """The same canonical form, rebuilt from the container's type-tagged encoding. The
    encoding is untrusted: every node's shape, length and payload type is checked, and any
    deviation raises ``ValueError``."""
    if depth > 64 or type(node) is not list or not node or type(node[0]) is not str:
        raise ValueError("malformed encoded value")
    kind = node[0]
    if kind == "none":
        if len(node) != 1:
            raise ValueError("malformed none")
        return ("none",)
    if len(node) != 2:
        raise ValueError("malformed encoded value")
    payload = node[1]
    if kind in _SCALAR_TYPES:
        if type(payload) is not _SCALAR_TYPES[kind]:
            raise ValueError("malformed scalar")
        return (kind, payload)
    if kind == "float_s":
        if type(payload) is not str or payload not in _SPECIAL_FLOATS:
            raise ValueError("malformed special float")
        return ("float", _SPECIAL_FLOATS[payload])
    if type(payload) is not list:
        raise ValueError("malformed container")
    if kind == "bytes":
        if any(type(item) is not int or not 0 <= item <= 255 for item in payload):
            raise ValueError("malformed bytes")
        return ("bytes", bytes(payload))
    if kind in ("list", "tuple"):
        return (kind, tuple(_canonical_from_wire(item, depth + 1) for item in payload))
    if kind in ("set", "frozenset"):
        return (kind, frozenset(_canonical_from_wire(item, depth + 1) for item in payload))
    if kind == "dict":
        pairs = []
        for pair in payload:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("malformed dict item")
            pairs.append(
                (_canonical_from_wire(pair[0], depth + 1), _canonical_from_wire(pair[1], depth + 1))
            )
        return ("dict", frozenset(pairs))
    raise ValueError("unknown encoded kind")


def _truthy(canonical: object) -> bool:
    """Truthiness of a canonical value, for the ``assert not CALL`` form."""
    assert isinstance(canonical, tuple)
    if canonical[0] == "none":
        return False
    return bool(canonical[1])


def check_passes(check: Check, reported: object) -> bool:
    """Whether one reported call result satisfies its check. Any malformed, errored or
    unserialisable report is a failure, whatever exception decoding it raises; equality is
    strict in type and value."""
    try:
        if not isinstance(reported, dict) or reported.get("ok") is not True:
            return False
        canonical = _canonical_from_wire(reported.get("v"))
        if check.predicate == "falsy":
            return not _truthy(canonical)
        return canonical == _canonical_from_value(check.expected)
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Static audit (pure, no execution).
# --------------------------------------------------------------------------------------

_TEXT_FLAGS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"_getframe"), "_getframe"),
    (re.compile(r"co_consts"), "co_consts"),
    (re.compile(r"excepthook"), "excepthook"),
    # ``os._exit`` or ``from os import _exit``, not ``__exit__`` or ``can_exit``.
    (re.compile(r"(?<!\w)_exit(?!\w)"), "os_exit"),
    (re.compile(r"__file__"), "dunder_file"),
    (re.compile(r"/proc(?!\w)"), "proc_path"),
    (re.compile(r"/tmp(?!\w)"), "tmp_path"),
    # Comparison and truth overrides, however they are attached (def, assignment, type()).
    (re.compile(r"__(?:eq|ne|bool)__"), "eq_override"),
    (re.compile(r"__builtins__"), "builtins"),
)
_AUDIT_MODULES = {
    "inspect": "inspect",
    "atexit": "atexit",
    "gc": "gc",
    "ctypes": "ctypes",
    "signal": "signal",
    "builtins": "builtins",
}
_ATTRIBUTE_FLAGS = {
    "_getframe": "_getframe",
    "co_consts": "co_consts",
    "excepthook": "excepthook",
    "_exit": "os_exit",
    "__eq__": "eq_override",
    "__ne__": "eq_override",
    "__bool__": "eq_override",
    "__file__": "dunder_file",
}
_ATTRIBUTE_CALLS = ("getattr", "setattr", "delattr", "hasattr")


def _fold(node: ast.AST) -> str | None:
    """The string value of a constant string expression (``'a' + 'b'``, an f-string
    without placeholders), or ``None``."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = [_fold(value) for value in node.values]
        return None if any(part is None for part in parts) else "".join(map(str, parts))
    return None


def _flag_text(text: str, flags: set[str]) -> None:
    for pattern, flag in _TEXT_FLAGS:
        if pattern.search(text):
            flags.add(flag)


def _flag_module(name: str | None, flags: set[str]) -> None:
    if name is None:
        flags.add("dynamic_import")
        return
    flag = _AUDIT_MODULES.get(name.split(".")[0])
    if flag is not None:
        flags.add(flag)


def _audit_tree(tree: ast.AST, flags: set[str]) -> None:
    sys_names = {"sys"}
    os_names = {"os", "posix"}
    importers = {"__import__", "import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _flag_module(alias.name, flags)
                if alias.name == "sys":
                    sys_names.add(alias.asname or "sys")
                elif alias.name in ("os", "posix"):
                    os_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            _flag_module(node.module, flags)
            for alias in node.names:
                if node.module == "sys" and alias.name == "modules":
                    flags.add("sys_modules")
                elif node.module in ("os", "posix") and alias.name in ("kill", "killpg"):
                    flags.add("os_kill")
                elif node.module == "importlib" and alias.name == "import_module":
                    importers.add(alias.asname or alias.name)

    def is_module(node: ast.AST, names: set[str], module: str) -> bool:
        if isinstance(node, ast.Name):
            return node.id in names
        return (
            isinstance(node, ast.Call)
            and _callee(node) in importers
            and bool(node.args)
            and _fold(node.args[0]) == module
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            folded = _fold(node)
            if folded is not None:
                _flag_text(folded, flags)
        elif isinstance(node, ast.Attribute):
            if node.attr in _ATTRIBUTE_FLAGS:
                flags.add(_ATTRIBUTE_FLAGS[node.attr])
            if node.attr == "modules" and is_module(node.value, sys_names, "sys"):
                flags.add("sys_modules")
            if node.attr in ("kill", "killpg") and is_module(node.value, os_names, "os"):
                flags.add("os_kill")
        elif isinstance(node, ast.Name) and node.id == "builtins":
            flags.add("builtins")
        elif isinstance(node, ast.Call):
            callee = _callee(node)
            if callee in importers and node.args:
                _flag_module(_fold(node.args[0]), flags)
            elif callee in _ATTRIBUTE_CALLS and len(node.args) >= 2:
                name = _fold(node.args[1])
                if name in _ATTRIBUTE_FLAGS:
                    flags.add(_ATTRIBUTE_FLAGS[name])
                if name == "modules" and is_module(node.args[0], sys_names, "sys"):
                    flags.add("sys_modules")
                if name in ("kill", "killpg") and is_module(node.args[0], os_names, "os"):
                    flags.add("os_kill")


def _callee(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def audit_attempt(code: str) -> tuple[str, ...]:
    """Flag risky identifiers and patterns in an attempt without running it. Returns the
    sorted distinct flags; an empty tuple means nothing flagged. These feed the RL
    per-update audit count and the leakage stop; a flag is a signal, not a verdict.

    The text scan runs even on code that does not parse. The syntax-tree scan adds what
    text alone misses: module aliases (``import sys as s; s.modules``), dynamic imports
    (``__import__``/``importlib.import_module``, flagged ``dynamic_import`` when the name is
    not a constant), names assembled from string pieces, and ``os.kill``."""
    flags: set[str] = set()
    _flag_text(code, flags)
    try:
        _audit_tree(ast.parse(code), flags)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        pass
    return tuple(sorted(flags))


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
    detail: str = ""


REWARD_SUPERVISOR = Path(__file__).with_name("reward_program.py").read_text(encoding="utf-8")


def _max_frame_bytes(payload: dict[str, Any]) -> int:
    """The largest final frame the supervisor can legitimately write for ``payload``: every
    value is base64 of a bounded byte string, so this is fixed before the run starts."""
    if payload["mode"] == "function":
        count, item_bytes, overhead = len(payload["calls"]), int(payload["row_bytes"]), 16
    else:
        count, item_bytes, overhead = len(payload["cases"]), int(payload["output_bytes"]), 96
    return count * (4 * math.ceil(item_bytes / 3) + overhead) + 65536


class _FrameReader:
    """Collect the supervisor's nonce-tagged frames from the container's output; any other
    text (docker CLI messages, junk) is kept only as a short diagnostic prefix."""

    def __init__(self, nonce: str, max_line: int) -> None:
        self.nonce = nonce
        self.token = nonce.encode()
        self.started = False
        self.final: dict[str, Any] | None = None
        self.text = bytearray()
        self.lines = LineSplitter(max_line, self._line)

    def _line(self, line: bytes) -> None:
        if self.token in line:
            try:
                frame = loads(line)
            except Exception:
                frame = None
            if isinstance(frame, dict) and frame.get("nonce") == self.nonce:
                if frame.get("frame") == "started":
                    self.started = True
                elif frame.get("frame") == "final" and self.started and self.final is None:
                    report = frame.get("report")
                    self.final = report if isinstance(report, dict) else {}
                return
        if len(self.text) < 4096:
            self.text.extend(line[: 4096 - len(self.text)] + b"\n")

    @property
    def daemon_unreachable(self) -> bool:
        text = self.text.decode("utf-8", errors="replace").lower()
        return any(marker in text for marker in _DAEMON_UNREACHABLE)


def run_reward_container(
    payload: dict[str, Any],
    *,
    limits: Limits,
    runtime: Runtime | None = None,
    docker: str | None = None,
) -> ContainerOutcome:
    """Run the reward supervisor in the hardened container and classify the host outcome.

    - No docker binary, exit 125 or an unreachable daemon before the candidate started:
      ``unavailable``. Any other failure before the ``started`` frame: ``host_error``. A CLI
      that has not finished ``budget + HOST_SLACK_SECONDS`` after launch: ``host_timeout``.
      These are host failures.
    - After ``started``: a valid final frame is ``completed``; exit 137 without one is
      ``resource_kill``; anything else is ``malformed_report``. These are scored.

    Output is read without blocking under the deadline, and every invocation removes its own
    randomly named container.
    """
    runtime = runtime or Runtime()
    resolved = docker or shutil.which("docker")
    if resolved is None:
        return ContainerOutcome(UNAVAILABLE, None, detail="docker binary not found")
    nonce = secrets.token_hex(16)
    name = "scratchcoder-reward-" + uuid4().hex
    deadline = float(payload["budget"]) + HOST_SLACK_SECONDS
    reader = _FrameReader(nonce, _max_frame_bytes(payload))
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    stalled = False
    try:
        with TemporaryFile() as input_stream:
            input_stream.write(dumps({**payload, "nonce": nonce}).encode())
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
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                remaining = deadline - (time.monotonic() - started)
                if remaining <= 0:
                    stalled = True
                    break
                for key, _ in selector.select(timeout=min(0.05, remaining)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        reader.lines.feed(chunk)
        finally:
            selector.close()
        if not stalled:
            try:
                process.wait(timeout=max(0.0, deadline - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                stalled = True
        reader.lines.finish()
        elapsed = time.monotonic() - started
        code = process.returncode
        if stalled:
            return ContainerOutcome(HOST_TIMEOUT, None, code, elapsed, "host deadline passed")
        if not reader.started:
            if code == 125 or reader.daemon_unreachable:
                return ContainerOutcome(UNAVAILABLE, None, code, elapsed, "docker unavailable")
            return ContainerOutcome(HOST_ERROR, None, code, elapsed, "candidate never started")
        if reader.final is not None and reader.final.get("status") == "ran":
            return ContainerOutcome(COMPLETED, reader.final, code, elapsed)
        if code == 137:
            return ContainerOutcome(RESOURCE_KILL, None, code, elapsed, "container killed")
        detail = "no final report" if reader.final is None else "supervisor failed"
        return ContainerOutcome(MALFORMED_REPORT, None, code, elapsed, detail)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return ContainerOutcome(
            HOST_ERROR, None, None, time.monotonic() - started, type(error).__name__
        )
    finally:
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
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
        "row_bytes": limits.output_bytes,
        "site_packages": False,
    }


def _io_payload(
    code: str, cases: IoCases, *, budget: float, per_case: float | None, limits: Limits
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


def _decode_row(raw: object, index: int) -> dict[str, Any] | None:
    if type(raw) is not str:
        return None
    try:
        row = loads(base64.b64decode(raw, validate=True))
    except Exception:
        return None
    if not isinstance(row, dict) or type(row.get("i")) is not int or row["i"] != index:
        return None
    return row


def _score_function(checks: FunctionChecks, report: dict[str, Any]) -> tuple[list[bool], int]:
    rows = report.get("rows")
    if type(rows) is not list or len(rows) != checks.total:
        return [False] * checks.total, 0
    outcomes: list[bool] = []
    unfinished = 0
    for index, (check, raw) in enumerate(zip(checks.checks, rows, strict=True)):
        row = _decode_row(raw, index)
        if row is None:
            unfinished += raw is None
            outcomes.append(False)
            continue
        unfinished += row.get("e") == "call_timeout"
        outcomes.append(check_passes(check, row))
    return outcomes, unfinished


def _score_io(cases: IoCases, report: dict[str, Any]) -> tuple[list[bool], int]:
    rows = report.get("cases")
    if type(rows) is not list or len(rows) != cases.total:
        return [False] * cases.total, 0
    outcomes: list[bool] = []
    unfinished = 0
    for row, expected in zip(rows, cases.expected, strict=True):
        if not isinstance(row, dict):
            outcomes.append(False)
            continue
        unfinished += row.get("e") in ("case_timeout", "budget")
        out = row.get("out")
        if row.get("ran") is not True or type(out) is not str:
            outcomes.append(False)
            continue
        try:
            stdout = base64.b64decode(out, validate=True).decode("utf-8", errors="replace")
        except Exception:
            outcomes.append(False)
            continue
        outcomes.append(stdout.strip() == expected.strip())
    return outcomes, unfinished


def attempt_budget(parsed: FunctionChecks | IoCases) -> float:
    """The per-attempt wall budget for parsed hidden tests: ``FUNCTION_BUDGET_SECONDS`` for a
    function task, and for an I/O task the pool verification's rule
    (:func:`scratchcoder.sandbox.budget.io_seconds`: 2 s plus 0.25 s per case, at most 60 s),
    so the reward channel gives an I/O task exactly the deadline its reference passed under."""
    if isinstance(parsed, IoCases):
        return io_seconds(parsed.total)
    return FUNCTION_BUDGET_SECONDS


def budget_for(task: Task, *, limits: Limits | None = None) -> float | None:
    """The per-attempt wall budget ``score_attempt`` applies to ``task`` by default, or
    ``None`` when the task is not parseable for this channel."""
    parsed = parse_task(task, limits=limits)
    return None if parsed is None else attempt_budget(parsed)


def score_attempt(
    task: Task,
    code: str,
    *,
    budget: float | None = None,
    per_call: float = PER_CALL_SECONDS,
    per_case: float | None = None,
    limits: Limits | None = None,
    container: ContainerRunner | None = None,
) -> RewardResult:
    """Score one candidate completion for ``task``. The reward is ``passed / total`` with
    ``total`` taken from the task file. Unparseable tasks and host failures return a reward
    of ``None`` (never 0.0), so the RL loop can skip or retry instead of optimising against a
    host problem. Everything that happens after the candidate started is scored: a resource
    kill or a disrupted report counts as a failed attempt.

    ``budget`` defaults to :func:`attempt_budget` (the value used is recorded in the result's
    ``budget_seconds``). ``per_call`` bounds each function-mode call. ``per_case`` bounds each
    I/O case; the default ``None`` lets each case use whatever the attempt budget has left, as
    the pool verification's I/O program does, so a case is never cut short while the budget
    lasts. Without explicit ``limits`` the container's CPU limit is the budget, as in the pool
    verification."""
    flags = audit_attempt(code)
    parsed = parse_task(task, limits=limits)
    if parsed is None:
        return RewardResult(NOT_PARSEABLE, None, audit_flags=flags)
    total = parsed.total
    if budget is None:
        budget = attempt_budget(parsed)
    run_limits = limits or Limits(timeout_seconds=min(60.0, max(1.0, budget)))

    def default_runner(payload: dict[str, Any]) -> ContainerOutcome:
        return run_reward_container(payload, limits=run_limits)

    runner = container if container is not None else default_runner
    if isinstance(parsed, FunctionChecks):
        payload = _function_payload(
            code, parsed, budget=budget, per_call=per_call, limits=run_limits
        )
    else:
        payload = _io_payload(code, parsed, budget=budget, per_case=per_case, limits=run_limits)
    outcome = runner(payload)
    elapsed = outcome.elapsed_seconds
    if outcome.host_status in HOST_FAILURES:
        detail = outcome.detail or "host failure"
        return RewardResult(
            outcome.host_status, None, 0, total, flags, elapsed, detail, budget_seconds=budget
        )
    failed = (False,) * total
    if outcome.report is None:
        status = outcome.host_status
        if status not in (RESOURCE_KILL, MALFORMED_REPORT):
            status = MALFORMED_REPORT
        return RewardResult(
            status, 0.0, 0, total, flags, elapsed, outcome.detail, 0, failed, budget
        )
    try:
        if isinstance(parsed, FunctionChecks):
            outcomes, unfinished = _score_function(parsed, outcome.report)
        else:
            outcomes, unfinished = _score_io(parsed, outcome.report)
    except Exception:
        return RewardResult(
            MALFORMED_REPORT, 0.0, 0, total, flags, elapsed, "bad report", 0, failed, budget
        )
    passed = sum(outcomes)
    # A complete report is labelled a resource kill only on in-container evidence (the
    # container's out-of-memory kill count rose during the attempt).
    oom_kills = outcome.report.get("oom_kills")
    killed = type(oom_kills) is int and oom_kills > 0
    return RewardResult(
        RESOURCE_KILL if killed else COMPLETED,
        passed / total,
        passed,
        total,
        flags,
        elapsed,
        "out-of-memory kill" if killed else "",
        unfinished,
        tuple(outcomes),
        budget,
    )


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
    """Running counts for the RL per-update monitoring and the leakage stop.

    The leakage stop is defined over *rewarded* attempts (reward above zero), so the tally
    keeps that denominator separately: ``rewarded_rate`` is the stop's rate, ``rate`` the
    share over every observed attempt."""

    attempts: int = 0
    flagged: int = 0
    rewarded: int = 0
    rewarded_flagged: int = 0
    flag_counts: dict[str, int] = field(default_factory=dict)

    def observe(self, flags: Sequence[str], *, rewarded: bool = False) -> None:
        self.attempts += 1
        if flags:
            self.flagged += 1
        if rewarded:
            self.rewarded += 1
            if flags:
                self.rewarded_flagged += 1
        for flag in flags:
            self.flag_counts[flag] = self.flag_counts.get(flag, 0) + 1

    def observe_result(self, result: RewardResult) -> None:
        """Observe a scored attempt; it counts as rewarded when its reward is above zero."""
        self.observe(result.audit_flags, rewarded=result.reward is not None and result.reward > 0)

    @property
    def rate(self) -> float:
        return self.flagged / self.attempts if self.attempts else 0.0

    @property
    def rewarded_rate(self) -> float:
        return self.rewarded_flagged / self.rewarded if self.rewarded else 0.0
