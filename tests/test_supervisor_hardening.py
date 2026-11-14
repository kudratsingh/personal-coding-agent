"""The path-of-record supervisor's protections and the host's response validation.

The pure tests drive :func:`scratchcoder.sandbox.runner.run` against a fake container (a
trusted script written here, run on the host in place of ``docker run``) and the supervisor's
own functions in-process. The Docker-gated tests run real attacks and honest controls through
the whole evaluation path inside the pinned container.
"""

import json
import os
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
BOTH = {"signal_handlers": True, "non_dumpable": True}


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
                "protections": {"signal_handlers": True, "non_dumpable": True}}
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

    monkeypatch.setattr(program, "Path", lambda _: tmp_path / "candidate.py")
    monkeypatch.setattr("scratchcoder.sandbox.program.subprocess.Popen", no_process)
    result = program.report(dict(PAYLOAD), dict(BOTH))
    assert result["status"] == "spawn_failed" and result["error_type"] == "BlockingIOError"


def test_a_failure_after_a_candidate_started_is_a_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def started_then_failed(payload: dict[str, Any]) -> dict[str, Any]:
        program._candidate_started = True
        raise BlockingIOError("EAGAIN")  # e.g. the candidate's leftovers hold the pid limit

    monkeypatch.setattr(program, "supervise", started_then_failed)
    assert program.report(dict(PAYLOAD), dict(BOTH))["status"] == "failed"
    monkeypatch.setattr(program, "_candidate_started", False)


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


# Ordered by ID, so every attack runs before the honest candidates it must not disturb.
LIVE: list[tuple[Task, str, bool]] = [
    (_task("a-memory-read", "io", IO_CASE.tests, IO_CASE.prompt), MEMORY_READ, False),
    (_task("b-stdout-injection", "function", ADD_TESTS), conformance.PID1_STDOUT_INJECTION, False),
    (_task("c-stdout-probe-io", "io", IO_CASE.tests, IO_CASE.prompt), STDOUT_PROBE, False),
    (_task("d-signals-then-honest", "function", ADD_TESTS), SIGNALS + ADD_HONEST, True),
    (
        _task("e-leftover-then-honest", "io", IO_CASE.tests, IO_CASE.prompt),
        LEFTOVER + IO_CASE.code,
        True,
    ),
    (_task("f-honest-io", "io", IO_CASE.tests, IO_CASE.prompt), IO_CASE.code, True),
    (_task("g-honest-function", "function", ADD_TESTS), ADD_HONEST, True),
]


def _containers() -> list[str]:
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
    return [name for name in output.split() if name]


@integration
def test_attacks_neither_pass_nor_stop_the_harness() -> None:
    codes = {task.id: code for task, code, _ in LIVE}
    report = evaluate(
        [task for task, _, _ in LIVE],
        lambda task, seed: codes[task.id],
        config=EvaluationConfig(),
        checkpoint_hash="hardening-test",
        tokenizer_hash="hardening-test",
    )
    assert report["status"] == "completed" and report["task_count"] == len(LIVE)
    observed = {row["id"]: row["attempts"][0]["status"] for row in report["per_task"]}
    expected = {task.id: ("passed" if honest else "failed") for task, _, honest in LIVE}
    assert observed == expected


@integration
def test_attacks_neither_pass_nor_stop_the_executor_scheduler() -> None:
    outcomes: list[parallel.Outcome] = []
    calls = [parallel.Call(task.id, code, task.hidden_tests, task.mode) for task, code, _ in LIVE]
    delivered = parallel.run_calls(
        calls, harness.isolated_execution, on_outcome=outcomes.append, workers=2
    )
    assert delivered == len(LIVE) and all(outcome.resolved for outcome in outcomes)
    observed = {outcome.key: outcome.passed for outcome in outcomes}
    assert observed == {task.id: honest for task, _, honest in LIVE}
    assert all(outcome.host_failures == 0 for outcome in outcomes)


@integration
def test_the_supervisor_memory_and_result_channel_are_unreachable() -> None:
    read = run(MEMORY_READ, IO_CASE.tests, harness.execution_limits(IO_CASE.tests, "io"), mode="io")
    assert read.status == "failed"
    assert read.stdout.startswith("TAG-maps-unreadable-PermissionError"), read.stdout
    limits = harness.execution_limits(IO_CASE.tests, "io")
    probe = run(STDOUT_PROBE, IO_CASE.tests, limits, mode="io")
    assert probe.status == "failed" and probe.stdout.strip() == "refused-PermissionError"


@integration
def test_no_container_outlives_its_call() -> None:
    result = run(
        LEFTOVER + IO_CASE.code,
        IO_CASE.tests,
        harness.execution_limits(IO_CASE.tests, "io"),
        mode="io",
    )
    assert result.passed, result
    assert _containers() == []
