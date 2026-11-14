"""The path-of-record supervisor's protections and the host's response validation.

The pure tests drive :func:`scratchcoder.sandbox.runner.run` against a fake container (a
trusted script written here, run on the host in place of ``docker run``) and the supervisor's
own functions in-process. The Docker-gated tests run real attacks and honest controls through
the whole evaluation path inside the pinned container.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.eval import harness, parallel
from scratchcoder.eval.harness import EvaluationConfig, evaluate, is_host_failure
from scratchcoder.eval.tasks import Task
from scratchcoder.sandbox import conformance, program, runner
from scratchcoder.sandbox.runner import Limits, Result, parse_response, run

NONCE = "0123456789abcdef0123456789abcdef"
BOTH = {"container_init": True, "signal_handlers": True, "non_dumpable": True}


def line(**fields: Any) -> bytes:
    document = {"status": "passed", "stdout": "", "exit_code": 0, "nonce": NONCE}
    document["protections"] = BOTH
    document.update(fields)
    return (json.dumps(document) + "\n").encode()


# --------------------------------------------------------------------------------------
# parse_response: exactly one valid, nonce-bearing, protected result line.
# --------------------------------------------------------------------------------------


def test_a_valid_line_is_accepted() -> None:
    raw = parse_response(line(stdout="out", exit_code=None), NONCE, program.REQUIRED_PROTECTIONS)
    assert raw["status"] == "passed" and raw["stdout"] == "out" and raw["exit_code"] is None
    unprotected = line(status="unprotected", protections={"non_dumpable": False})
    assert parse_response(unprotected, NONCE, program.REQUIRED_PROTECTIONS)["status"] == (
        "unprotected"
    )


MALFORMED: dict[str, bytes] = {
    "empty": b"",
    "no newline": line()[:-1],
    "injected line first": line(stdout="forged") + line(status="failed"),
    "injected line after": line(status="failed") + line(),
    "two documents on one line": line()[:-1] + line(),
    "trailing text": line() + b"x",
    "non-ascii": line(stdout="é").replace(b"\\u00e9", "é".encode()),
    "invalid utf-8": b"\xff\xfe" + line(),
    "not json": b"passed\n",
    "a list": b"[]\n",
    "a string": b'"passed"\n',
    "missing nonce": line().replace(b'"nonce"', b'"nonse"'),
    "wrong nonce": line(nonce="f" * 32),
    "nonce not text": line(nonce=7),
    "extra key": line(verdict="passed"),
    "missing stdout": json.dumps(
        {"status": "passed", "exit_code": 0, "nonce": NONCE, "protections": BOTH}
    ).encode()
    + b"\n",
    "unknown status": line(status="pass"),
    "infrastructure status": line(status="infrastructure_error"),
    "stdout not text": line(stdout=["x"]),
    "exit code text": line(exit_code="0"),
    "exit code bool": line(exit_code=False),
    "exit code float": line(exit_code=0.0),
    "protections not a mapping": line(protections=["signal_handlers", "non_dumpable"]),
    "protection not a flag": line(protections={"signal_handlers": 1, "non_dumpable": True}),
    "verdict while dumpable": line(protections={"signal_handlers": True, "non_dumpable": False}),
    "verdict without handlers": line(protections={"non_dumpable": True}),
    "verdict outside a container init": line(protections={**BOTH, "container_init": False}),
    "error type not text": line(status="failed", error_type=3),
    "deep nesting": b"[" * 200_000 + b"]" * 200_000 + b"\n",
    "deep nesting in a field": line(stdout="x").replace(b'"x"', b"[" * 200_000 + b"]" * 200_000),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_output_raises_only_value_error(name: str) -> None:
    with pytest.raises(ValueError):
        parse_response(MALFORMED[name], NONCE, program.REQUIRED_PROTECTIONS)


# --------------------------------------------------------------------------------------
# run(): a fake container in place of docker, so the host's handling is exercised end to end.
# --------------------------------------------------------------------------------------

FAKE_CONTAINER = """
import json, sys
payload = json.load(sys.stdin)
nonce = payload["nonce"]
def good(**fields):
    document = {"status": "passed", "stdout": "", "exit_code": 0, "nonce": nonce,
                "protections": {"container_init": True, "signal_handlers": True,
                    "non_dumpable": True}}
    document.update(fields)
    return json.dumps(document) + "\\n"
behaviour = %r
out = sys.stdout
if behaviour == "honest":
    out.write(good())
elif behaviour == "injection then honest":
    out.write(json.dumps({"status": "passed", "stdout": "", "exit_code": 0}) + "\\n")
    out.write(good(status="failed"))
elif behaviour == "forged nonce":
    out.write(good().replace(nonce, "f" * 32))
elif behaviour == "unprotected":
    out.write(good(status="unprotected", protections={"signal_handlers": True,
                                                      "non_dumpable": False}))
elif behaviour == "dumpable verdict":
    out.write(good(protections={"signal_handlers": True, "non_dumpable": False}))
elif behaviour == "spawn failed":
    out.write(good(status="spawn_failed", exit_code=None, error_type="BlockingIOError"))
elif behaviour == "deep nesting":
    out.write("[" * 200000 + "]" * 200000 + "\\n")
elif behaviour == "binary":
    out.flush(); sys.stdout.buffer.write(b"\\xff\\x00\\xfe\\n")
elif behaviour == "rejected":
    out.write(json.dumps({"status": "rejected", "error_type": "KeyError"}) + "\\n")
elif behaviour == "killed":
    out.flush(); sys.exit(137)
"""


def fake_container(monkeypatch: pytest.MonkeyPatch, behaviour: str) -> None:
    """Replace ``docker run`` with a trusted host script; ``docker rm`` becomes a no-op."""
    script = FAKE_CONTAINER % behaviour

    def command(name: str, limits: Limits, **_: Any) -> list[str]:
        return [sys.executable, "-I", "-S", "-c", script]

    monkeypatch.setattr(runner, "command", command)
    monkeypatch.setattr("scratchcoder.sandbox.runner.shutil.which", lambda _: "/usr/bin/true")


HOST_FAILURES = {
    "injection then honest": "malformed_response",
    "forged nonce": "malformed_response",
    "dumpable verdict": "malformed_response",
    "deep nesting": "malformed_response",
    "binary": "malformed_response",
    "rejected": "malformed_response",
    "unprotected": "supervisor_unprotected",
    "spawn failed": "spawn_failed",
}


@pytest.mark.parametrize("behaviour", sorted(HOST_FAILURES))
def test_unacceptable_output_is_an_unresolved_host_failure(
    monkeypatch: pytest.MonkeyPatch, behaviour: str
) -> None:
    fake_container(monkeypatch, behaviour)
    result = run("def add(a, b):\n    return a + b", "assert add(1, 2) == 3")
    assert result.status == "infrastructure_error" and not result.passed
    assert result.detail == HOST_FAILURES[behaviour]
    assert is_host_failure(result)  # Retried, never scored as a pass or a failure.


def test_a_valid_response_is_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_container(monkeypatch, "honest")
    result = run("def add(a, b):\n    return a + b", "assert add(1, 2) == 3")
    assert result.status == "passed" and result.detail == "" and not is_host_failure(result)


def test_a_killed_container_keeps_its_status_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_container(monkeypatch, "killed")
    assert run("x = 1", "assert True").status == "resource_limit"


def test_an_unexpected_host_error_is_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError("host side")

    monkeypatch.setattr("scratchcoder.sandbox.runner.shutil.which", lambda _: "/usr/bin/true")
    monkeypatch.setattr(runner, "command", broken)
    result = run("x = 1", "assert True")
    assert result.status == "infrastructure_error" and result.detail == "runner_error"


def test_infrastructure_errors_stop_the_harness_unscored() -> None:
    """A genuine host failure is never turned into a candidate failure: the harness refuses
    to report a score rather than count it either way."""
    task = conformance.by_name()["io_sum_honest"].task()
    for detail in ("malformed_response", "supervisor_unprotected", "spawn_failed"):

        def execute(code: str, tests: str, mode: str, detail: str = detail) -> Result:
            return Result("infrastructure_error", detail=detail)

        with pytest.raises(RuntimeError, match="isolation is unavailable or failed"):
            evaluate(
                [task],
                lambda task, seed: "print(1)",
                config=EvaluationConfig(),
                checkpoint_hash="c",
                tokenizer_hash="t",
                execute=execute,
            )


# --------------------------------------------------------------------------------------
# The supervisor itself (in-process; nothing here runs a candidate).
# --------------------------------------------------------------------------------------

PAYLOAD = {
    "nonce": NONCE,
    "code": "raise SystemExit('a candidate must not run in this test')",
    "tests": "assert True",
    "mode": "function",
    "timeout": 1.0,
    "output_bytes": 4096,
    "site_packages": False,
}


@pytest.mark.parametrize(
    "protections",
    [
        {"signal_handlers": True, "non_dumpable": False},
        {"signal_handlers": False, "non_dumpable": True},
        {"signal_handlers": True},
        {},
    ],
)
def test_nothing_runs_without_every_protection(
    monkeypatch: pytest.MonkeyPatch, protections: dict[str, bool]
) -> None:
    def forbidden(payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("supervise ran without its protections")

    monkeypatch.setattr(program, "supervise", forbidden)
    assert program.report(dict(PAYLOAD), protections)["status"] == "unprotected"


def test_a_failure_before_any_candidate_is_spawn_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_process(*args: Any, **kwargs: Any) -> Any:
        raise BlockingIOError("EAGAIN")

    monkeypatch.setattr(program, "CANDIDATE_PATH", str(tmp_path / "candidate.py"))
    monkeypatch.setattr(program, "_candidate_started", False)
    monkeypatch.setattr("scratchcoder.sandbox.program.subprocess.Popen", no_process)
    result = program.report(dict(PAYLOAD), dict(BOTH))
    assert result["status"] == "spawn_failed" and result["error_type"] == "BlockingIOError"


@pytest.mark.parametrize("planted", ["symlink", "file"])
def test_the_candidate_file_is_created_never_reused(tmp_path: Path, planted: str) -> None:
    """A path that already exists (a planted symlink, FIFO or file) is refused, so the
    supervisor never writes through anything a candidate could have prepared."""
    path = tmp_path / "candidate.py"
    target = tmp_path / "elsewhere"
    if planted == "symlink":
        path.symlink_to(target)
    else:
        path.write_text("planted", encoding="utf-8")
    with pytest.raises(OSError):
        program._write_candidate(str(path), "print(1)")
    assert not target.exists()
    if planted == "file":
        assert path.read_text(encoding="utf-8") == "planted"


def test_the_candidate_file_is_written_once_per_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writes: list[str] = []
    monkeypatch.setattr(program, "_write_candidate", lambda path, text: writes.append(path))

    def no_process(*args: Any, **kwargs: Any) -> Any:
        raise BlockingIOError("EAGAIN")

    monkeypatch.setattr("scratchcoder.sandbox.program.subprocess.Popen", no_process)
    io_payload = {**PAYLOAD, "mode": "io", "tests": IO_CASE.tests}
    with pytest.raises(BlockingIOError):
        program.supervise(io_payload, str(tmp_path / "candidate.py"))
    assert len(writes) == 1


def test_the_sweep_kills_everything_else_and_reaps_as_the_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``kill`` is replaced before ``getpid`` is, so no mistake here can signal real processes.
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "scratchcoder.sandbox.program.os.kill", lambda pid, sig: signals.append((pid, sig))
    )
    reaped = iter([(41, 0), (0, 0), (42, 0)])

    def waitpid(pid: int, flags: int) -> tuple[int, int]:
        try:
            return next(reaped)
        except StopIteration:
            raise ChildProcessError from None

    monkeypatch.setattr("scratchcoder.sandbox.program.os.waitpid", waitpid)
    monkeypatch.setattr("scratchcoder.sandbox.program.time.sleep", lambda seconds: None)
    monkeypatch.setattr("scratchcoder.sandbox.program.sys.platform", "linux")
    monkeypatch.setattr("scratchcoder.sandbox.program.os.getpid", lambda: 1)
    program._sweep()
    assert signals and all(pid == -1 and sig == signal.SIGKILL for pid, sig in signals)
    assert len(signals) == 2  # Once at the start, once more while a child was still alive.


def test_the_sweep_does_nothing_outside_a_container_init(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(pid: int, sig: int) -> None:
        raise AssertionError("the sweep signalled a process outside a container")

    monkeypatch.setattr("scratchcoder.sandbox.program.os.kill", forbidden)
    program._sweep()  # This process is not PID 1, so nothing is signalled.


def test_a_failure_after_a_candidate_started_is_a_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def started_then_failed(payload: dict[str, Any]) -> dict[str, Any]:
        program._candidate_started = True
        raise BlockingIOError("EAGAIN")  # e.g. the candidate's leftovers hold the pid limit

    monkeypatch.setattr(program, "_candidate_started", False)  # Restored after the test.
    monkeypatch.setattr(program, "supervise", started_then_failed)
    assert program.report(dict(PAYLOAD), dict(BOTH))["status"] == "failed"


def test_main_reports_unprotected_with_the_nonce(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    dumpable = {"signal_handlers": True, "non_dumpable": False}
    monkeypatch.setattr(program, "protect", lambda: dumpable)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PAYLOAD)))
    program.main()
    output = capsys.readouterr().out.encode()
    raw = parse_response(output, NONCE, program.REQUIRED_PROTECTIONS)
    assert raw["status"] == "unprotected" and raw["protections"]["non_dumpable"] is False


def test_main_without_a_nonce_is_never_accepted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    monkeypatch.setattr(program, "protect", lambda: dict(BOTH))
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    program.main()
    with pytest.raises(ValueError):
        parse_response(capsys.readouterr().out.encode(), NONCE, program.REQUIRED_PROTECTIONS)


def test_protect_installs_handlers_and_drops_dumpability_on_linux() -> None:
    """Run in a separate interpreter: the protections change that process for good."""
    probe = (
        "import json, signal\n"
        "from scratchcoder.sandbox import program\n"
        "state = program.protect()\n"
        "state['handled'] = all(signal.getsignal(getattr(signal, n)) is program._ignore\n"
        "                       for n in program.HANDLED_SIGNALS)\n"
        "print(json.dumps(state))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    state = json.loads(output)
    assert state["signal_handlers"] is True and state["handled"] is True
    assert state["non_dumpable"] is sys.platform.startswith("linux")
    assert state["container_init"] is False  # This interpreter is not a container's PID 1.


def test_the_supervisor_protects_itself_before_reading_the_payload() -> None:
    import ast
    import inspect

    body = ast.parse(inspect.getsource(program.main)).body[0]
    assert isinstance(body, ast.FunctionDef)
    first = body.body[0]
    assert isinstance(first, ast.Assign) and ast.unparse(first.value) == "protect()"


# --------------------------------------------------------------------------------------
# Docker-gated: real attacks and honest controls through the whole evaluation path.
# --------------------------------------------------------------------------------------

integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)

ADD_PROMPT = conformance.ADD_PROMPT
ADD_TESTS = conformance.ADD_TESTS
ADD_HONEST = "def add(a, b):\n    return a + b\n"
IO_CASE = conformance.by_name()["io_sum_honest"]
MEMORY_READ = conformance.by_name()["io_supervisor_introspection"].code
STDOUT_PROBE = (
    "try:\n"
    "    with open('/proc/1/fd/1', 'w') as _out:\n"
    '        _out.write(\'{"status": "passed", "stdout": "", "exit_code": 0}\\n\')\n'
    "    print('written')\n"
    "except Exception as error:\n"
    "    print('refused-' + type(error).__name__)\n"
)
SIGNALS = (
    "import os, signal\n"
    "for _name in ('SIGINT', 'SIGTERM', 'SIGHUP', 'SIGQUIT', 'SIGUSR1', 'SIGUSR2', 'SIGKILL'):\n"
    "    try:\n"
    "        os.kill(1, getattr(signal, _name))\n"
    "    except Exception:\n"
    "        pass\n"
)
MEMORY_PROBES = (
    "import os\n"
    "def _try(label, action):\n"
    "    try:\n"
    "        action()\n"
    "        return label + '-open'\n"
    "    except Exception as error:\n"
    "        return label + '-' + type(error).__name__\n"
    "def _mem():\n"
    "    with open('/proc/1/mem', 'rb', 0) as handle:\n"
    "        handle.read(1)\n"
    "def _environ():\n"
    "    with open('/proc/1/environ', 'rb') as handle:\n"
    "        handle.read(1)\n"
    "def _task_mem():\n"
    "    with open('/proc/1/task/1/mem', 'rb', 0) as handle:\n"
    "        handle.read(1)\n"
    "def _vm_read():\n"
    "    import ctypes\n"
    "    libc = ctypes.CDLL(None, use_errno=True)\n"
    "    buffer = ctypes.create_string_buffer(8)\n"
    "    class iovec(ctypes.Structure):\n"
    "        _fields_ = [('base', ctypes.c_void_p), ('len', ctypes.c_size_t)]\n"
    "    local = iovec(ctypes.addressof(buffer), 8)\n"
    "    remote = iovec(0x400000, 8)\n"
    "    if libc.process_vm_readv(1, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0) < 0:\n"
    "        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))\n"
    "print(' '.join(_try(label, action) for label, action in (\n"
    "    ('mem', _mem), ('environ', _environ), ('task-mem', _task_mem), ('vm', _vm_read))))\n"
)
SYMLINK_SWAP = (
    "import os\n"
    "a, b = map(int, input().split())\n"
    "print(a + b)\n"
    "try:\n"
    "    os.remove('/tmp/candidate.py')\n"
    "    os.symlink('/proc/1/fd/1', '/tmp/candidate.py')\n"
    "except Exception:\n"
    "    pass\n"
)
FIFO_SWAP = (
    "import os\n"
    "a, b = map(int, input().split())\n"
    "print(a + b)\n"
    "try:\n"
    "    os.remove('/tmp/candidate.py')\n"
    "    os.mkfifo('/tmp/candidate.py')\n"
    "except Exception:\n"
    "    pass\n"
)
STARVE = (
    "import os, time\n"
    "try:\n"
    "    os.setpriority(os.PRIO_PROCESS, 1, 19)\n"
    "except Exception:\n"
    "    pass\n"
    "for _ in range(4):\n"
    "    if os.fork() == 0:\n"
    "        while True:\n"
    "            if os.fork() == 0:\n"
    "                pass\n"
    "            else:\n"
    "                started = time.process_time()\n"
    "                while time.process_time() - started < 0.5:\n"
    "                    pass\n"
    "                os._exit(0)\n"
    "while True:\n"
    "    pass\n"
)
"""A same-uid candidate lowers PID 1's priority and keeps CPU-bound processes running, each
replaced before its CPU limit. Predicted ``timeout`` (a scored verdict); a host stall would be
a finding: an availability residual of the path of record."""
LEFTOVER = (
    "import os, time\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    os.close(0); os.close(1); os.close(2)\n"
    "    time.sleep(120)\n"
    "    os._exit(0)\n"
)


def _task(name: str, mode: str, tests: str, prompt: str = ADD_PROMPT) -> Task:
    return Task(
        id=f"hardening/{name}",
        prompt=prompt,
        public_tests=tests,
        hidden_tests=tests,
        source="synthetic supervisor hardening case",
        license="project-authored",
        written_by="human",
        split="dev",
        mode=mode,
    )


# Ordered by ID, so every attack runs before the honest candidates it must not disturb. Each
# entry carries its predicted status: the attacks are scored verdicts, never passes and never
# host failures.
LIVE: list[tuple[Task, str, str]] = [
    (_task("a-memory-read", "io", IO_CASE.tests, IO_CASE.prompt), MEMORY_READ, "failed"),
    (
        _task("b-stdout-injection", "function", ADD_TESTS),
        conformance.PID1_STDOUT_INJECTION,
        "failed",
    ),
    (_task("c-stdout-probe-io", "io", IO_CASE.tests, IO_CASE.prompt), STDOUT_PROBE, "failed"),
    (_task("d-signals-then-honest", "function", ADD_TESTS), SIGNALS + ADD_HONEST, "passed"),
    (
        _task("e-leftover-then-honest", "io", IO_CASE.tests, IO_CASE.prompt),
        LEFTOVER + IO_CASE.code,
        "passed",
    ),
    (_task("e1-symlink-swap", "io", IO_CASE.tests, IO_CASE.prompt), SYMLINK_SWAP, "failed"),
    (_task("e2-fifo-swap", "io", IO_CASE.tests, IO_CASE.prompt), FIFO_SWAP, "timeout"),
    (_task("f-honest-io", "io", IO_CASE.tests, IO_CASE.prompt), IO_CASE.code, "passed"),
    (_task("g-honest-function", "function", ADD_TESTS), ADD_HONEST, "passed"),
]
RECORD_ENV = "SCRATCHCODER_HARDENING_RECORD"


def _observe(test: str, case: str, predicted: str, result: Result | None, **extra: Any) -> None:
    """Append one observed outcome beside its prediction, before any assertion, when
    ``SCRATCHCODER_HARDENING_RECORD`` names a file (keep it outside the repository)."""
    target = os.environ.get(RECORD_ENV)
    if not target:
        return
    row: dict[str, Any] = {"test": test, "case": case, "predicted": predicted, **extra}
    if result is not None:
        row.update(
            {
                "status": result.status,
                "detail": result.detail,
                "exit_code": result.exit_code,
                "host_stalled": result.host_stalled,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "stdout": result.stdout[:200],
            }
        )
    with open(target, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _containers() -> set[str]:
    output = subprocess.run(
        [
            "docker",
            "--host",
            runner.ENDPOINT,
            "ps",
            "-a",
            "--filter",
            "name=scratchcoder-",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    return {name for name in output.split() if name}


@integration
def test_attacks_neither_pass_nor_stop_the_harness() -> None:
    before = _containers()
    codes = {task.id: code for task, code, _ in LIVE}
    report = evaluate(
        [task for task, _, _ in LIVE],
        lambda task, seed: codes[task.id],
        config=EvaluationConfig(),
        checkpoint_hash="hardening-test",
        tokenizer_hash="hardening-test",
    )
    observed = {row["id"]: row["attempts"][0]["status"] for row in report["per_task"]}
    for task, _, predicted in LIVE:
        _observe("harness", task.id, predicted, None, status=observed.get(task.id))
    assert report["status"] == "completed" and report["task_count"] == len(LIVE)
    assert observed == {task.id: predicted for task, _, predicted in LIVE}
    assert _containers() <= before


@integration
def test_attacks_neither_pass_nor_stop_the_executor_scheduler() -> None:
    before = _containers()
    outcomes: list[parallel.Outcome] = []
    calls = [parallel.Call(task.id, code, task.hidden_tests, task.mode) for task, code, _ in LIVE]
    delivered = parallel.run_calls(
        calls, harness.isolated_execution, on_outcome=outcomes.append, workers=2
    )
    by_key = {outcome.key: outcome for outcome in outcomes}
    for task, _, predicted in LIVE:
        outcome = by_key.get(task.id)
        _observe(
            "scheduler",
            task.id,
            predicted,
            None,
            status=None if outcome is None else outcome.status,
            resolved=None if outcome is None else outcome.resolved,
            host_failures=None if outcome is None else outcome.host_failures,
        )
    assert delivered == len(LIVE) and all(outcome.resolved for outcome in outcomes)
    assert {key: outcome.status for key, outcome in by_key.items()} == {
        task.id: predicted for task, _, predicted in LIVE
    }
    assert all(outcome.host_failures == 0 for outcome in outcomes)
    assert _containers() <= before


@integration
def test_the_supervisor_memory_and_result_channel_are_unreachable() -> None:
    limits = harness.execution_limits(IO_CASE.tests, "io")
    read = run(MEMORY_READ, IO_CASE.tests, limits, mode="io")
    _observe("unreachable", "maps-scan", "failed: TAG-maps-unreadable-PermissionError", read)
    probes = run(MEMORY_PROBES, IO_CASE.tests, limits, mode="io")
    predicted_probes = "mem-PermissionError environ-PermissionError task-mem-PermissionError"
    _observe(
        "unreachable", "direct-probes", f"failed: {predicted_probes} vm-PermissionError", probes
    )
    probe = run(STDOUT_PROBE, IO_CASE.tests, limits, mode="io")
    _observe("unreachable", "stdout-probe", "failed: refused-PermissionError", probe)
    assert read.status == "failed"
    assert read.stdout.startswith("TAG-maps-unreadable-PermissionError"), read.stdout
    assert probes.status == "failed"
    words = probes.stdout.split()
    assert words[:3] == predicted_probes.split(), probes.stdout
    assert words[3].startswith("vm-") and words[3] != "vm-open", probes.stdout
    assert probe.status == "failed" and probe.stdout.strip() == "refused-PermissionError"


@integration
def test_no_container_outlives_its_call() -> None:
    before = _containers()
    result = run(
        LEFTOVER + IO_CASE.code,
        IO_CASE.tests,
        harness.execution_limits(IO_CASE.tests, "io"),
        mode="io",
    )
    _observe("cleanup", "leftover-sleepers", "passed", result)
    assert result.passed, result
    assert _containers() <= before


@integration
def test_priority_starvation_is_a_verdict_not_a_stall() -> None:
    """Observed once and recorded. Predicted ``timeout``; a passing verdict fails the test,
    and a host stall is recorded as an availability finding (not rerun)."""
    result = run(STARVE, ADD_TESTS, harness.execution_limits(ADD_TESTS, "function"))
    _observe("starvation", "nice-19-cpu-burners", "timeout", result)
    assert not result.passed
    assert result.status == "timeout" and not result.host_stalled, result
