"""Trusted supervisor source, executed ONLY inside the isolated container.

It runs as PID 1 under the same uid as the candidate, so before any candidate process exists
it establishes two protections and refuses to run anything if either is missing:

- **Signal handlers.** The catchable signals a same-user candidate could send to PID 1 get a
  no-op handler. A pid-namespace init never receives signals it has no handler for, so the
  remaining signals (``SIGKILL`` included) are already dropped by the kernel. A handler, not
  ``SIG_IGN``, is used so the candidate starts with the default dispositions after ``exec``.
- **Non-dumpable.** ``prctl(PR_SET_DUMPABLE, 0)``, checked with ``PR_GET_DUMPABLE``. Without
  it a candidate could read ``/proc/1/mem`` (the expected I/O outputs, the result nonce) and
  write to ``/proc/1/fd/1`` (the result channel). With it the kernel's ptrace access check
  refuses both to an unprivileged process.

These are the protections the reward supervisor (``reward_program.py``) already applies. If
either cannot be established the supervisor reports ``unprotected`` and runs nothing; the
host treats that as a host failure, never as a verdict.

The candidate's source is written to ``/tmp/candidate.py`` once, before any candidate process
exists, with ``O_EXCL | O_NOFOLLOW``, and never reopened: ``/tmp`` is writable by the candidate,
so a later write could follow a symlink the candidate planted (to ``/proc/1/fd/1``, which PID 1
itself may open) or block on a FIFO. After every case the supervisor kills every other process
in the container, so nothing a case leaves behind runs into the next case.

The result is one JSON line carrying the host's random nonce. The host accepts nothing else:
see :func:`scratchcoder.sandbox.runner.parse_response`.
"""

import json
import os
import secrets
import selectors
import signal
import subprocess
import sys
import time
from tempfile import TemporaryFile
from typing import Any

HANDLED_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2")
REQUIRED_PROTECTIONS = ("signal_handlers", "non_dumpable")
"""Every protection must hold before a candidate process may start."""
VERDICT_STATUSES = ("passed", "failed", "timeout", "output_limit")
"""Statuses that judge the candidate. The others name a supervisor-side failure."""
HOST_FAILURE_STATUSES = ("unprotected", "spawn_failed")
"""``unprotected``: a required protection could not be established, so nothing ran.
``spawn_failed``: the first candidate process could not be started, so no candidate code ran."""
_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4
CANDIDATE_PATH = "/tmp/candidate.py"
SWEEP_SECONDS = 1.0

_candidate_started = False
"""Set once a candidate process may exist; a failure before that point is the host's."""


def _ignore(signum: int, frame: object) -> None:
    return None


def protect() -> dict[str, bool]:
    """Install the signal handlers and drop dumpability; report which protections hold."""
    handlers = True
    for name in HANDLED_SIGNALS:
        try:
            number = getattr(signal, name)
            signal.signal(number, _ignore)
            handlers = handlers and signal.getsignal(number) is _ignore
        except (AttributeError, OSError, ValueError):
            handlers = False
    dumpable_off = False
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            dumpable_off = (
                libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) == 0
                and libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) == 0
            )
        except Exception:
            dumpable_off = False
    return {"signal_handlers": handlers, "non_dumpable": dumpable_off}


def _write_candidate(path: str, program: str) -> None:
    """Create the candidate file; refuse to follow or reuse anything already at ``path``."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(program)


def _sweep() -> None:
    """Kill every process except PID 1 and reap them. Runs only as PID 1 of a Linux pid
    namespace (the container's init); anywhere else it does nothing."""
    if os.getpid() != 1 or not sys.platform.startswith("linux"):
        return
    give_up = time.monotonic() + SWEEP_SECONDS
    try:
        os.kill(-1, signal.SIGKILL)
    except OSError:
        pass
    while time.monotonic() < give_up:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.005)


def supervise(payload: dict[str, Any], candidate_path: str | None = None) -> dict[str, Any]:
    global _candidate_started
    candidate_path = candidate_path or CANDIDATE_PATH
    deadline = time.monotonic() + payload["timeout"]
    cases = (
        [{"input": "", "output": ""}]
        if payload["mode"] == "function"
        else json.loads(payload["tests"])
    )
    marker = "__scratchcoder_completed_" + secrets.token_hex(24)
    program = payload["code"]
    if payload["mode"] == "function":
        program += "\n" + payload["tests"] + "\nprint(" + repr(marker) + ")\n"
    _write_candidate(candidate_path, program)  # Once, before any candidate process exists.
    last_output = ""
    for case in cases:
        with TemporaryFile() as input_stream:
            input_stream.write(case["input"].encode())
            input_stream.seek(0)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    *([] if payload["site_packages"] else ["-S"]),
                    candidate_path,
                ],
                stdin=input_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "LANG": "C.UTF-8",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
            _candidate_started = True
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            captured = bytearray()
            status = "failed"
            try:
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        status = "timeout"
                        break
                    for key, _ in selector.select(timeout=0.05):
                        block = os.read(key.fd, 4096)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        captured.extend(block)
                        if len(captured) > payload["output_bytes"]:
                            status = "output_limit"
                            break
                    if status == "output_limit":
                        break
                if status in {"timeout", "output_limit"}:
                    process.kill()
                process.wait(timeout=1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=1)
                selector.close()
                process.stdout.close()
                _sweep()  # After the case's own process is reaped, so its status is kept.
            last_output = bytes(captured[: payload["output_bytes"]]).decode(
                "utf-8", errors="replace"
            )
            if status in {"timeout", "output_limit"}:
                return {"status": status, "stdout": last_output, "exit_code": process.returncode}
            if process.returncode != 0:
                return {"status": "failed", "stdout": last_output, "exit_code": process.returncode}
            if payload["mode"] == "function":
                if not last_output.rstrip().endswith(marker):
                    return {"status": "failed", "stdout": last_output, "exit_code": 0}
                last_output = last_output.rsplit(marker, 1)[0]
            elif last_output.strip() != case["output"].strip():
                return {"status": "failed", "stdout": last_output, "exit_code": 0}
    return {"status": "passed", "stdout": last_output, "exit_code": 0}


def report(payload: dict[str, Any], protections: dict[str, bool]) -> dict[str, Any]:
    """The result for one call. Nothing runs unless every required protection holds."""
    global _candidate_started
    _candidate_started = False
    if not all(protections.get(name) is True for name in REQUIRED_PROTECTIONS):
        return {"status": "unprotected", "stdout": "", "exit_code": None}
    try:
        return supervise(payload)
    except BaseException as error:
        return {
            "status": "failed" if _candidate_started else "spawn_failed",
            "stdout": "",
            "exit_code": 1 if _candidate_started else None,
            "error_type": type(error).__name__,
        }


def main() -> None:
    protections = protect()  # Before the payload is read and before any candidate exists.
    try:
        payload = json.load(sys.stdin)
        nonce = payload["nonce"]
        if not isinstance(nonce, str):
            raise ValueError("missing nonce")
    except Exception as error:
        # No nonce to answer with: the host cannot accept this line and treats it as its own
        # failure.
        sys.stdout.write(json.dumps({"status": "rejected", "error_type": type(error).__name__}))
        sys.stdout.write("\n")
        return
    result = report(payload, protections)
    result["nonce"] = nonce
    result["protections"] = protections
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
