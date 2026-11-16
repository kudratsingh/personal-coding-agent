"""Running a workspace through the evaluation path of record, and reading its feedback.

The sandbox takes one program and one test string (``runner.run``, function mode). A
workspace of several modules is therefore **packed** into one program: a trusted prelude
installs an import hook that serves each workspace module from its source text, so
``from textkit import slugify`` in the tests resolves to the workspace file, compiled under its
own file name (tracebacks name ``textkit.py``). Only workspace module names are served; they
never shadow the standard library (:func:`scratchcoder.agent.workspace.check_file_name`).

Two kinds of call go through :func:`scratchcoder.eval.harness.isolated_execution`, one fresh
container each:

- **Feedback** (:func:`public_call`): the packed workspace (tests included) with a trusted
  driver as the test string. The driver runs every ``test_*`` function of the public test
  modules, prints one bounded JSON row per test after a fixed prefix, and exits 1 if any
  failed, so the supervisor's status is ``passed`` only when every public test passed. The
  rows are **untrusted** (workspace code runs in the same process and can print anything):
  they become model context and the review packet, never a grade.
- **Grading** (:func:`check_call`): the packed source modules (no tests) with a hidden check
  script as the test string. The verdict is the supervisor's typed, nonce-framed status;
  nothing the candidate prints is read for it.

Host failures are retried by :func:`scratchcoder.eval.parallel.resolve_call` and, if they
persist, reported as unresolved: never a pass or a fail.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from scratchcoder.agent import workspace
from scratchcoder.eval import harness
from scratchcoder.eval.parallel import Call, resolve_call
from scratchcoder.sandbox.runner import Result

Execute = Callable[[str, str, str], Result]
MODE = "function"
REPORT_PREFIX = "@@scratchcoder-test "
MAX_ROWS = 64
MAX_MESSAGE_CHARS = 400
MAX_FRAMES = 8

_PRELUDE = """\
import importlib.abc as _sc_abc
import importlib.util as _sc_util
import sys as _sc_sys

_SC_SOURCES = {sources}


class _ScWorkspace(_sc_abc.MetaPathFinder, _sc_abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name in _SC_SOURCES:
            return _sc_util.spec_from_loader(name, self, origin=name + ".py")
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = module.__name__ + ".py"
        code = compile(_SC_SOURCES[module.__name__], module.__file__, "exec")
        exec(code, module.__dict__)


_sc_sys.meta_path.insert(0, _ScWorkspace())
"""

_DRIVER = """\
import importlib as _sc_importlib
import json as _sc_json
import traceback as _sc_traceback

_sc_failures = 0


def _sc_text(value):
    try:
        return str(value)[:{limit}]
    except BaseException:
        return "<unprintable>"


def _sc_row(row):
    print({prefix!r} + _sc_json.dumps(row), flush=True)


for _sc_name in {modules}:
    try:
        _sc_module = _sc_importlib.import_module(_sc_name)
    except BaseException as _sc_error:
        _sc_failures += 1
        _sc_frames = _sc_traceback.extract_tb(_sc_error.__traceback__)[-{frames}:]
        _sc_row({{"module": _sc_name, "test": None, "outcome": "fail",
                 "type": type(_sc_error).__name__, "message": _sc_text(_sc_error),
                 "frames": [[f.filename, f.lineno, f.name] for f in _sc_frames]}})
        continue
    for _sc_test_name, _sc_test in list(vars(_sc_module).items()):
        if not _sc_test_name.startswith("test_") or not callable(_sc_test):
            continue
        if getattr(_sc_test, "__module__", None) != _sc_name:
            continue
        try:
            _sc_test()
        except BaseException as _sc_error:
            _sc_failures += 1
            _sc_frames = _sc_traceback.extract_tb(_sc_error.__traceback__)[-{frames}:]
            _sc_row({{"module": _sc_name, "test": _sc_test_name, "outcome": "fail",
                     "type": type(_sc_error).__name__, "message": _sc_text(_sc_error),
                     "frames": [[f.filename, f.lineno, f.name] for f in _sc_frames]}})
        else:
            _sc_row({{"module": _sc_name, "test": _sc_test_name, "outcome": "pass"}})
if _sc_failures:
    raise SystemExit(1)
"""


def pack(files: Mapping[str, str]) -> str:
    """The trusted prelude serving ``files`` (name -> source) as importable modules."""
    sources = {workspace.module_name(path): files[path] for path in sorted(files)}
    for path in files:
        workspace.check_file_name(path)
    return _PRELUDE.format(sources=repr(sources))


def public_driver(test_paths: Sequence[str]) -> str:
    """The trusted test string that runs the public tests and reports each one."""
    modules = [workspace.module_name(path) for path in test_paths]
    return _DRIVER.format(
        limit=MAX_MESSAGE_CHARS, prefix=REPORT_PREFIX, modules=repr(modules), frames=MAX_FRAMES
    )


def public_call(files: Mapping[str, str], test_paths: Sequence[str]) -> tuple[str, str]:
    """(program, tests) of a feedback run: the whole workspace and the public driver."""
    return pack(files), public_driver(test_paths)


def check_call(files: Mapping[str, str], checks: str) -> tuple[str, str]:
    """(program, tests) of a grading run: the source modules and a hidden check script."""
    sources = {path: text for path, text in files.items() if not workspace.is_test_file(path)}
    return pack(sources), checks


@dataclass(frozen=True)
class SandboxRun:
    """One resolved (or unresolved) sandbox call. ``status`` is the supervisor's verdict
    when ``resolved``; otherwise the last host status, which is not a verdict."""

    status: str
    resolved: bool
    host_failures: int
    seconds: float
    stdout: str

    @property
    def passed(self) -> bool:
        return self.resolved and self.status == "passed"


def run_call(
    program: str,
    tests: str,
    *,
    execute: Execute = harness.isolated_execution,
    sleep: Callable[[float], None] | None = None,
) -> SandboxRun:
    """Execute one call through the path of record, retrying host failures."""
    call = Call("agent", program, tests, MODE)
    outcome = (
        resolve_call(call, execute) if sleep is None else resolve_call(call, execute, sleep=sleep)
    )
    return SandboxRun(
        outcome.status,
        outcome.resolved,
        outcome.host_failures,
        outcome.elapsed_seconds,
        outcome.stdout,
    )


@dataclass(frozen=True)
class TestRow:
    module: str
    test: str | None
    passed: bool
    error_type: str = ""
    message: str = ""
    frames: tuple[tuple[str, int, str], ...] = ()


def _row(raw: object) -> TestRow | None:
    if not isinstance(raw, dict):
        return None
    module, test, outcome = raw.get("module"), raw.get("test"), raw.get("outcome")
    if not isinstance(module, str) or not (test is None or isinstance(test, str)):
        return None
    if outcome == "pass":
        return TestRow(module[:80], None if test is None else test[:80], True)
    if outcome != "fail":
        return None
    frames = []
    for frame in raw.get("frames", [])[:MAX_FRAMES] if isinstance(raw.get("frames"), list) else []:
        if (
            isinstance(frame, list)
            and len(frame) == 3
            and isinstance(frame[0], str)
            and type(frame[1]) is int
            and isinstance(frame[2], str)
        ):
            frames.append((frame[0][:80], frame[1], frame[2][:80]))
    error_type, message = raw.get("type"), raw.get("message")
    return TestRow(
        module[:80],
        None if test is None else test[:80],
        False,
        error_type[:80] if isinstance(error_type, str) else "",
        message[:MAX_MESSAGE_CHARS] if isinstance(message, str) else "",
        tuple(frames),
    )


def parse_report(stdout: str) -> list[TestRow]:
    """The driver's rows in ``stdout``: untrusted, so every field is type-checked and bounded,
    and malformed lines are skipped."""
    rows = []
    for line in stdout.splitlines():
        if not line.startswith(REPORT_PREFIX):
            continue
        try:
            row = _row(json.loads(line[len(REPORT_PREFIX) :]))
        except (ValueError, RecursionError):
            continue
        if row is not None:
            rows.append(row)
        if len(rows) >= MAX_ROWS:
            break
    return rows


def failing_tests(rows: Sequence[TestRow]) -> list[str]:
    return [row.test for row in rows if not row.passed and row.test is not None]


def _line(files: Mapping[str, str], path: str, number: int) -> str:
    if path not in files:
        return ""
    lines = files[path].split("\n")
    return lines[number - 1].strip() if 1 <= number <= len(lines) else ""


def condense(run: SandboxRun, files: Mapping[str, str], *, limit: int = 2) -> str:
    """The observation for the next turn: status, then for up to ``limit`` failing tests the
    test's own line, the exception, and the last frame inside a non-test workspace module.
    Line text comes from the host's copy of the files, never from the sandbox."""
    if not run.resolved:
        return "RESULT run_tests: not run (sandbox unavailable)"
    if run.status == "timeout":
        return "RESULT run_tests: timeout (the tests did not finish in time)"
    if run.status in {"output_limit", "resource_limit", "oversized"}:
        return f"RESULT run_tests: {run.status.replace('_', ' ')}"
    rows = parse_report(run.stdout)
    failed = [row for row in rows if not row.passed]
    if run.status == "passed":
        return f"RESULT run_tests: passed ({len(rows)} tests)"
    if not failed:
        return "RESULT run_tests: failed before any test reported"
    lines = [f"RESULT run_tests: failed ({len(failed)} of {len(rows)} tests)"]
    for row in failed[:limit]:
        test_file = row.module + ".py"
        own = [frame for frame in row.frames if frame[0] == test_file]
        where = f" ({test_file}:{own[-1][1]})" if own else ""
        lines.append(f"FAIL {row.test or 'import ' + row.module}{where}")
        if own:
            lines.append("    " + _line(files, test_file, own[-1][1]))
        message = row.message.strip().split("\n")[0]
        lines.append(f"{row.error_type}: {message}" if message else row.error_type)
        inner = [
            frame
            for frame in row.frames
            if frame[0] in files and not workspace.is_test_file(frame[0])
        ]
        if inner:
            path, number, name = inner[-1]
            lines.append(f"  in {path}:{number} {name}: {_line(files, path, number)}")
    if len(failed) > limit:
        lines.append(f"(+{len(failed) - limit} more failing)")
    return "\n".join(lines)
