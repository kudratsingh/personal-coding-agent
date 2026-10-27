"""Trusted reward supervisor, executed ONLY as PID 1 inside the isolated container.

Unlike ``program.py`` (the evaluation path of record), this supervisor NEVER receives
the expected values of a task. The host passes only the candidate code plus the *left*
side of each hidden assertion (function mode) or each case's stdin (I/O mode). The
candidate runs in separate child processes; the supervisor collects whatever they report
and forwards it, unjudged, to the host, which compares it against the secret expected
values it kept to itself.

The result channel is therefore treated as UNTRUSTED: a candidate can forge any value on
it, but cannot know which value would pass, so forgery earns no more than an honest
solution would. The supervisor's jobs are framing and containment:

- **Framing.** It writes a ``started`` frame, tagged with the host's random nonce, before
  any candidate process exists, and one ``final`` frame with the report at the end. The
  host attributes a lost, malformed or oversized report after ``started`` to the candidate,
  so a candidate cannot turn its own failure into a retried host failure. Two outcomes after
  ``started`` remain host failures: a ``spawn_failed`` report, written only when the
  candidate's first process could not be started (so no candidate code ever ran), and a
  container that is still running when the host's stall deadline passes.
- **Hardening.** PID 1 ignores the catchable signals a same-user candidate could send
  (a pid-namespace init never receives signals it has no handler for) and marks itself
  non-dumpable, so a candidate cannot open ``/proc/1/fd``, ``/proc/1/mem`` or
  ``/proc/1/environ``, and cannot write ``/proc/1/oom_score_adj``.
- **Bounds.** Every buffer is bounded, and every wait ends at the per-attempt wall budget
  (and, in function mode, the per-call limit) or within ``OVERRUN_SECONDS`` after it, far
  inside the host's stall slack; report values travel base64-encoded so the report size is
  fixed by the payload, whatever bytes the candidate produced.
- **Cleanup.** After the run, and after every I/O case, it kills every other process in
  the container, so no straggler can hold the pid limit or a pipe into the next case.

Function mode keeps the evaluation program's judgement of the process as a whole. That
program runs the candidate, the tests and ``print(marker)`` as one script and accepts the run
only if the output reaches the end of file in time and ends with the marker. The driver here
ends with the same ``print(marker)`` (a random marker of the same length), and the supervisor
reads the candidate's output to the end of file, counts the marker toward the output cap and
reports whether the output ended with it and whether the stream was still open when the
budget ended; the host fails every check on any of those. The results themselves travel on a
separate pipe, so a candidate cannot forge them through its own output. The marker lives in
the driver's process, which the candidate shares, so a candidate that inspects its frames can
read it and print it again; the end-of-output check holds only against a candidate that does
not.

I/O cases run from a fork server: a clean interpreter is started once per attempt, and each
case runs in a fresh fork of it, so a case costs a fork instead of an interpreter start and
no case inherits another case's memory. Files a case leaves in ``/tmp`` do persist (and
count against the container's memory limit), exactly as between the cases of the evaluation
runner's I/O program. As in that program, the cases run in turn under one shared deadline
(the attempt budget), unless the host sets a per-case limit.
"""

import base64
import codecs
import json
import os
import secrets
import selectors
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

_CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}

# Catchable signals that PID 1 handles with a no-op. A handler (not SIG_IGN) is used so
# children start with the default dispositions after exec. Signals left at SIG_DFL are
# already dropped by the kernel for a pid-namespace init.
_HANDLED_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2")
_PR_SET_DUMPABLE = 4
_READ_BYTES = 16384
_ROW_PREFIX = b'{"i": '
"""Every honest driver row starts with this; other lines are dropped without parsing."""
_CANDIDATE_PATH = "/tmp/candidate.py"
"""Where the evaluation program keeps the candidate's source; the reward channel runs the
candidate under the same name."""

MARKER_PREFIX = "__scratchcoder_completed_"
"""The start of the completion marker; with 48 hex digits it is as long as the evaluation
program's, so it takes the same share of the output cap."""
DRAIN_SECONDS = 0.25
"""The most the supervisor spends reading the result pipe's last rows after the output
stream ends or the budget runs out."""
GRACE_SECONDS = 2.0
"""How long the supervisor waits for a killed process to be reaped, or for a fork-server
answer, before giving up on it."""
SWEEP_SECONDS = 1.0
"""How long one sweep of leftover processes may take."""
OVERRUN_SECONDS = 3 * GRACE_SECONDS + SWEEP_SECONDS
"""The most the supervisor's own work can run past the attempt budget: the I/O path's worst
case is two missed fork-server answers, closing the fork server and one sweep. The host's
stall slack (``reward.HOST_SLACK_SECONDS``) is well above this plus container start-up, so a
candidate cannot push an attempt into a host stall by itself."""

_candidate_started = False
"""Set once a candidate process may exist. A failure before that point is the host's: the
supervisor reports ``spawn_failed`` and the host retries the attempt."""

# Source of the per-attempt driver that runs in the candidate child for function mode.
# It reads its configuration (candidate code, import setup, call expressions, per-call
# budget, row limit, and the inherited result-pipe descriptor number) as one JSON document
# on stdin and writes one result line per call to that descriptor. It is a best-effort
# convenience for honest candidates; its integrity is not a security assumption, because
# the host compares every reported value against secret expectations it never shared.
DRIVER = r"""
import builtins, io, json, os, signal, sys

_write = os.write
_dumps = json.dumps
_result_fd = 2
_row_bytes = 65536
_dict_keys = type({}.keys())


def _items(values, depth):
    return [_encode(item, depth + 1) for item in values]


def _encode(value, depth=0):
    if depth > 32:
        raise TypeError("value nested too deeply")
    if value is None:
        return ["none"]
    if value is True or value is False:
        return ["bool", value]
    # Subclasses are read through the base type's own storage, so an overridden method
    # cannot change what is reported: a Counter reports its items as a dict, a named tuple
    # as a tuple. The base-type calls reject objects that only pretend to be the type.
    if isinstance(value, int):
        return ["int", int.__int__(value)]
    if isinstance(value, float):
        number = float.__float__(value)
        if number != number:
            return ["float_s", "nan"]
        if number in (float("inf"), float("-inf")):
            return ["float_s", "inf" if number > 0 else "-inf"]
        return ["float", number]
    if isinstance(value, str):
        return ["str", str.__str__(value)]
    if isinstance(value, bytes):
        return ["bytes", list(bytes.__bytes__(value))]
    # A bytearray equals a literal exactly when its bytes do (``bytearray(b'ab') == b'ab'``),
    # so it is reported as bytes.
    if isinstance(value, bytearray):
        return ["bytes", list(bytearray.__iter__(value))]
    if isinstance(value, list):
        return ["list", _items(list.__iter__(value), depth)]
    if isinstance(value, tuple):
        return ["tuple", _items(tuple.__iter__(value), depth)]
    if isinstance(value, set):
        return ["set", _items(set.__iter__(value), depth)]
    if isinstance(value, frozenset):
        return ["frozenset", _items(frozenset.__iter__(value), depth)]
    if isinstance(value, dict):
        return [
            "dict",
            [[_encode(k, depth + 1), _encode(v, depth + 1)] for k, v in dict.items(value)],
        ]
    # A keys view compares like the set of its keys (``{1: 0}.keys() == {1}``), so it is
    # reported as a frozenset; iterating it reads the dict's own storage.
    if type(value) is _dict_keys:
        return ["frozenset", _items(iter(value), depth)]
    raise TypeError("unsupported value type")


def _emit(line):
    try:
        _write(_result_fd, (line + "\n").encode())
    except Exception:
        pass


def _row(index, value):
    try:
        line = _dumps({"i": index, "ok": True, "v": _encode(value)})
    except Exception:
        return _dumps({"i": index, "ok": False, "e": "unserializable"})
    if len(line) > _row_bytes:
        return _dumps({"i": index, "ok": False, "e": "too_large"})
    return line


def _empty_stdin():
    # The configuration arrived on stdin. The candidate gets an empty stdin instead, as in
    # the evaluation program (whose function-mode stdin is an empty file), so it cannot
    # rewind stdin to re-read the configuration and its call sources.
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    old = sys.stdin
    buffer = io.open(0, "rb", -1, closefd=False)
    buffer.raw.name = "<stdin>"
    stream = io.TextIOWrapper(
        buffer, old.encoding, old.errors, "\n", old.line_buffering, old.write_through
    )
    stream.mode = "r"
    sys.stdin = sys.__stdin__ = stream


def main():
    global _result_fd, _row_bytes
    config = json.load(sys.stdin)
    _result_fd = config["result_fd"]
    _row_bytes = config["row_bytes"]
    calls = config["calls"]
    per_call = config["per_call"]
    code = config["code"]
    setup = config["setup"]
    marker = config["marker"]
    del config
    _empty_stdin()
    # As in the evaluation program, the candidate runs as the main script
    # ``/tmp/candidate.py``: a fresh ``__main__`` module in ``sys.modules`` (so its
    # ``if __name__ == "__main__":`` block runs and pickling its classes works), ``__file__``,
    # ``sys.argv`` and the builtins module, with tracebacks and ``inspect`` reading the
    # source the supervisor wrote there. The hidden tests' imports come after it.
    driver = sys.modules["__main__"]  # keeps the driver's own module alive
    script = type(sys)("__main__")
    script.__file__ = "/tmp/candidate.py"
    script.__cached__ = None
    script.__builtins__ = builtins
    sys.modules["__main__"] = script
    sys.argv[:] = ["/tmp/candidate.py"]
    namespace = vars(script)
    try:
        # The source is compiled as UTF-8 bytes, as ``python /tmp/candidate.py`` reads the
        # file: a byte-order mark is dropped and an encoding line is honoured (or rejected).
        exec(compile(code.encode("utf-8"), "/tmp/candidate.py", "exec"), namespace)
        exec(compile(setup, "<setup>", "exec"), namespace)
    except BaseException:
        for index in range(len(calls)):
            _emit(_dumps({"i": index, "ok": False, "e": "candidate_error"}))
        return

    class _Deadline(Exception):
        pass

    def _alarm(signum, frame):
        raise _Deadline()

    try:
        signal.signal(signal.SIGALRM, _alarm)
    except Exception:
        pass
    for index, source in enumerate(calls):
        try:
            signal.setitimer(signal.ITIMER_REAL, per_call)
            value = eval(compile(source, "<call>", "eval"), namespace)
            signal.setitimer(signal.ITIMER_REAL, 0)
        except _Deadline:
            _emit(_dumps({"i": index, "ok": False, "e": "call_timeout"}))
            continue
        except BaseException as error:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            _emit(_dumps({"i": index, "ok": False, "e": type(error).__name__}))
            continue
        _emit(_row(index, value))
    # The evaluation program ends the candidate-plus-tests script with this statement and
    # accepts the run only if the output ends with the marker: output printed after it (an
    # ``atexit`` callback, a destructor's complaint), a closed or replaced stdout and a
    # stream a leftover process holds open all fail there, and fail here the same way.
    print(marker)


main()
"""

# Source of the I/O-mode fork server. It reads the candidate code from stdin once, then for
# every request on its socket receives the case's stdin and stdout descriptors, forks, and
# reports the child's pid and exit status. The forked child restores what a fresh
# ``python -I [-S] /tmp/candidate.py`` would see (standard streams, ``sys.argv``, a new
# ``__main__``, the startup module set), runs the candidate, and then never returns to the
# loop: the candidate's exception, ``SystemExit`` or normal end propagates to the top of this
# program, so the interpreter itself prints the traceback, picks the exit status and runs
# its normal shutdown (joining threads, ``atexit`` callbacks, flushing and finalising every
# stream), exactly as it does for a script.
ZYGOTE = r"""
import sys
_fresh = frozenset(sys.modules)
import io, os, socket

_SystemExit = SystemExit
_exit = os._exit
_code = sys.stdin.buffer.read()
_spec = [
    (stream.encoding, stream.errors, stream.line_buffering, stream.write_through)
    for stream in (sys.stdin, sys.stdout, sys.stderr)
]
_sock = socket.socket(fileno=int(sys.argv[1]))


def _stream(fd, write, spec, name):
    encoding, errors, line_buffering, write_through = spec
    buffer = io.open(fd, "wb" if write else "rb", -1, closefd=False)
    buffer.raw.name = name
    stream = io.TextIOWrapper(buffer, encoding, errors, "\n", line_buffering, write_through)
    stream.mode = "w" if write else "r"
    return stream


def _case(stdin_fd, stdout_fd):
    try:
        _sock.detach()
        os.dup2(stdin_fd, 0)
        os.dup2(stdout_fd, 1)
        os.dup2(stdout_fd, 2)
        os.closerange(3, 65536)
        for name in set(sys.modules) - _fresh:
            del sys.modules[name]
        sys.stdin = sys.__stdin__ = _stream(0, False, _spec[0], "<stdin>")
        sys.stdout = sys.__stdout__ = _stream(1, True, _spec[1], "<stdout>")
        sys.stderr = sys.__stderr__ = _stream(2, True, _spec[2], "<stderr>")
        sys.argv[:] = ["/tmp/candidate.py"]
        main = type(sys)("__main__")
        main.__file__ = "/tmp/candidate.py"
        main.__cached__ = None
        main.__builtins__ = __builtins__
        sys.modules["__main__"] = main
        program = compile(_code, "/tmp/candidate.py", "exec")
    except SyntaxError:
        raise
    except BaseException:
        _exit(1)
    exec(program, vars(main))
    raise _SystemExit(0)


def _main():
    while True:
        try:
            message, fds, _flags, _address = socket.recv_fds(_sock, 1, 2)
        except OSError:
            return
        if not message or len(fds) != 2:
            return
        pid = os.fork()
        if pid == 0:
            _case(fds[0], fds[1])
            _exit(1)
        for fd in fds:
            os.close(fd)
        _sock.sendall(b"P" + pid.to_bytes(4, "little"))
        _, wait_status = os.waitpid(pid, 0)
        code = os.waitstatus_to_exitcode(wait_status)
        _sock.sendall(b"S" + code.to_bytes(4, "little", signed=True))


_main()
"""


# --------------------------------------------------------------------------------------
# Pure helpers (also imported and unit-tested on the host; they never run candidate code).
# --------------------------------------------------------------------------------------


class LineSplitter:
    """Split a byte stream into lines of at most ``limit`` bytes. A longer line is dropped
    whole, so one oversized line can never displace the lines after it."""

    def __init__(self, limit: int, handle: Callable[[bytes], None]) -> None:
        self.limit = limit
        self.handle = handle
        self.pending = bytearray()
        self.dropping = False

    def feed(self, data: bytes) -> None:
        start = 0
        while True:
            end = data.find(b"\n", start)
            piece = data[start:] if end < 0 else data[start:end]
            if not self.dropping:
                if len(self.pending) + len(piece) > self.limit:
                    self.dropping = True
                    self.pending.clear()
                else:
                    self.pending.extend(piece)
            if end < 0:
                return
            if not self.dropping:
                self.handle(bytes(self.pending))
            self.pending.clear()
            self.dropping = False
            start = end + 1

    def finish(self) -> None:
        """Hand over a final unterminated line (if it fit) and reset."""
        if self.pending and not self.dropping:
            self.handle(bytes(self.pending))
        self.pending.clear()
        self.dropping = False


class MarkerWatcher:
    """Follow a byte stream in blocks and say whether it ends with ``marker`` followed by
    nothing but whitespace, which is the evaluation program's test of a function run's
    output (``output.rstrip().endswith(marker)``). The last marker counts, and a marker split
    between two blocks is found."""

    def __init__(self, marker: bytes) -> None:
        self.marker = marker
        self.window = b""  # the end of the stream so far, long enough to hold half a marker
        self.seen = False
        self.stray = False  # something other than whitespace came after the last marker
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, data: bytes) -> None:
        joined = self.window + data
        at = joined.rfind(self.marker)
        if at >= 0:
            self.seen = True
            self.stray = False
            self.decoder.reset()
            data = joined[at + len(self.marker) :]
        if self.seen and data and self.decoder.decode(data).strip():
            self.stray = True
        keep = len(self.marker) - 1
        self.window = joined[-keep:] if keep else b""

    def complete(self) -> bool:
        """Whether the stream so far ends with the marker and then only whitespace."""
        if not self.seen or self.stray:
            return False
        state = self.decoder.getstate()
        ending = self.decoder.decode(b"", final=True)  # half a character is not whitespace
        self.decoder.setstate(state)
        return not ending.strip()


def admit_row(rows: dict[int, bytes], count: int, line: bytes) -> None:
    """Keep the first well-framed row for each in-range call index. Anything else (junk,
    deep nesting, a duplicate or out-of-range index) is ignored; values are not judged."""
    if len(rows) >= count or not line.startswith(_ROW_PREFIX):
        return
    try:
        row = json.loads(line)
    except Exception:
        return
    if not isinstance(row, dict):
        return
    index = row.get("i")
    if type(index) is int and 0 <= index < count and index not in rows:
        rows[index] = line


def encode_bytes(data: bytes) -> str:
    """Base64 keeps the report size a fixed function of the raw byte count."""
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------------------------
# Container-only helpers.
# --------------------------------------------------------------------------------------


def _ignore(signum: int, frame: object) -> None:
    return None


def _harden() -> bool:
    """Make PID 1 deaf to candidate signals and unreadable through ``/proc``."""
    for name in _HANDLED_SIGNALS:
        try:
            signal.signal(getattr(signal, name), _ignore)
        except (AttributeError, OSError, ValueError):
            pass
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return bool(libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) == 0)
    except Exception:
        return False


def _reap() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _kill_others(keep: set[int], *, limit_seconds: float = SWEEP_SECONDS) -> None:
    """Kill every process except PID 1 and ``keep``, then reap orphans. Runs only as PID 1
    of a Linux pid namespace (the container's init); anywhere else it does nothing."""
    if os.getpid() != 1 or not sys.platform.startswith("linux"):
        return
    give_up = time.monotonic() + limit_seconds
    while True:
        if not keep:
            try:
                os.kill(-1, signal.SIGKILL)
            except OSError:
                pass
            _reap()
            return
        alive = False
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == 1 or pid in keep:
                continue
            try:
                with open(f"/proc/{pid}/stat", "rb") as handle:
                    state = handle.read().rsplit(b")", 1)[1].split()[0]
            except (OSError, IndexError):
                continue
            if state == b"Z":
                continue
            alive = True
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        _reap()
        if not alive or time.monotonic() >= give_up:
            return
        time.sleep(0.005)


def _oom_kills() -> int | None:
    try:
        with open("/sys/fs/cgroup/memory.events", encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
    except (OSError, ValueError):
        return None
    return None


def _memory_file(data: bytes) -> int:
    """An anonymous in-memory file holding ``data``, positioned at the start. It is not on
    ``/tmp``, so a candidate filling ``/tmp`` cannot stop the next case from starting."""
    if sys.platform != "linux":
        raise OSError("in-memory files exist only inside the Linux container")
    fd = os.memfd_create("scratchcoder")
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.lseek(fd, 0, os.SEEK_SET)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _interpreter(site_packages: bool) -> list[str]:
    return [sys.executable, "-I", *([] if site_packages else ["-S"])]


def _write_candidate(code: str) -> None:
    """Write the candidate's source where the evaluation program keeps it, for programs that
    read their own source (tracebacks, ``inspect``). The candidate itself runs from memory,
    so a failed write changes nothing else. Only the container's init writes it."""
    if os.getpid() != 1 or not sys.platform.startswith("linux"):
        return
    try:
        with open(_CANDIDATE_PATH, "w", encoding="utf-8") as handle:
            handle.write(code)
    except Exception:
        pass


def _mark_candidate_started() -> None:
    global _candidate_started
    _candidate_started = True


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _finish(process: subprocess.Popen[bytes], deadline: float) -> int | None:
    """Wait for ``process`` until ``deadline``, then kill it; never raises on a timeout."""
    try:
        return process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        process.kill()
    try:
        return process.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return None


# --------------------------------------------------------------------------------------
# Function mode.
# --------------------------------------------------------------------------------------


def _drain(fd: int, handle: Callable[[bytes], None]) -> None:
    """Read what is already waiting on ``fd`` without blocking, for at most a moment: a
    leftover process that keeps writing cannot hold the supervisor here."""
    os.set_blocking(fd, False)
    give_up = time.monotonic() + DRAIN_SECONDS
    while time.monotonic() < give_up:
        try:
            block = os.read(fd, _READ_BYTES)
        except BlockingIOError:
            return
        if not block:
            return
        handle(block)


def _run_function(payload: dict[str, Any], deadline: float) -> dict[str, Any]:
    calls = payload["calls"]
    count = len(calls)
    per_call = min(payload["per_call"], payload["budget"])
    cap = payload["output_bytes"]
    marker = MARKER_PREFIX + secrets.token_hex(24)
    watcher = MarkerWatcher(marker.encode())
    _write_candidate(payload["code"])
    read_fd, write_fd = os.pipe()
    rows: dict[int, bytes] = {}
    lines = LineSplitter(payload["row_bytes"], lambda line: admit_row(rows, count, line))
    process: subprocess.Popen[bytes] | None = None
    child_rc: int | None = None
    output_bytes = 0
    output_limit = False
    output_closed = False
    try:
        try:
            # The driver writes results to the inherited write end at its own descriptor
            # number; pass_fds keeps it open across exec.
            os.set_inheritable(write_fd, True)
            config = json.dumps(
                {
                    "code": payload["code"],
                    "setup": payload["setup"],
                    "calls": calls,
                    "per_call": per_call,
                    "result_fd": write_fd,
                    "row_bytes": payload["row_bytes"],
                    "marker": marker,
                }
            ).encode()
            config_fd = _memory_file(config)
            try:
                process = subprocess.Popen(
                    [*_interpreter(payload["site_packages"]), "-c", DRIVER],
                    stdin=config_fd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    pass_fds=(write_fd,),
                    env=_CHILD_ENV,
                )
            finally:
                os.close(config_fd)
            # From here on the candidate may run, so every failure is the candidate's.
            _mark_candidate_started()
        finally:
            os.close(write_fd)
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(read_fd, selectors.EVENT_READ, "result")
        try:
            # As in the evaluation program, read until the output stream ends: it ends when
            # the process and everything it started have closed it, so a leftover process
            # that holds it open keeps this loop going until the budget ends.
            while not output_closed and not output_limit:
                now = time.monotonic()
                if now >= deadline:
                    break
                for key, _ in selector.select(timeout=min(0.05, deadline - now)):
                    block = os.read(key.fd, _READ_BYTES)
                    if not block:
                        selector.unregister(key.fileobj)
                        output_closed = output_closed or key.data == "stdout"
                    elif key.data == "result":
                        lines.feed(block)
                    else:
                        # The candidate's own stdout and stderr are discarded but counted:
                        # as in the evaluation program, more than the output cap fails the
                        # whole attempt. The completion marker counts too.
                        output_bytes += len(block)
                        watcher.feed(block)
                        if output_bytes > cap:
                            output_limit = True
                            break
        finally:
            selector.close()
        if output_limit:
            process.kill()
        child_rc = _finish(process, deadline)
        _drain(read_fd, lines.feed)
    finally:
        os.close(read_fd)
        if process is not None and process.stdout is not None:
            process.stdout.close()
    return {
        "status": "ran",
        "mode": "function",
        "n": count,
        "rows": [encode_bytes(rows[index]) if index in rows else None for index in range(count)],
        "child_rc": child_rc,
        "output_limit": output_limit,
        "output_closed": output_closed,
        "marker": watcher.complete(),
    }


# --------------------------------------------------------------------------------------
# I/O mode.
# --------------------------------------------------------------------------------------


def _read_capped(fd: int, cap: int, deadline: float) -> tuple[bytes, str]:
    """Read ``fd`` to end of file; stop at ``deadline`` or once more than ``cap`` bytes
    arrive (an output-limit failure, as in the evaluation path)."""
    captured = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return bytes(captured), "case_timeout"
            for _key, _ in selector.select(timeout=min(0.05, remaining)):
                block = os.read(fd, _READ_BYTES)
                if not block:
                    return bytes(captured), "ok"
                captured.extend(block)
                if len(captured) > cap:
                    return bytes(captured[:cap]), "output_limit"
    finally:
        selector.close()


class _Zygote:
    """The supervisor's handle on one fork server."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.sock, remote = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            code_fd = _memory_file(payload["code"].encode("utf-8"))
            try:
                self.process = subprocess.Popen(
                    [
                        *_interpreter(payload["site_packages"]),
                        "-c",
                        ZYGOTE,
                        str(remote.fileno()),
                    ],
                    stdin=code_fd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(remote.fileno(),),
                    env=_CHILD_ENV,
                )
            finally:
                os.close(code_fd)
        except BaseException:
            self.sock.close()
            raise
        finally:
            remote.close()

    @property
    def pid(self) -> int:
        return self.process.pid

    def _receive(self, tag: bytes, deadline: float) -> int:
        data = b""
        while len(data) < 5:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("fork server did not answer")
            self.sock.settimeout(remaining)
            chunk = self.sock.recv(5 - len(data))
            if not chunk:
                raise ConnectionError("fork server exited")
            data += chunk
        if data[:1] != tag:
            raise ConnectionError("fork server protocol error")
        return int.from_bytes(data[1:], "little", signed=True)

    def run_case(self, text: str, cap: int, case_deadline: float) -> dict[str, Any]:
        stdin_fd = _memory_file(text.encode("utf-8"))
        read_fd, write_fd = os.pipe()
        try:
            try:
                # Once the request may have reached the fork server, a forked candidate may
                # run, so every later failure is the candidate's.
                _mark_candidate_started()
                socket.send_fds(self.sock, [b"R"], [stdin_fd, write_fd])
            finally:
                os.close(stdin_fd)
                os.close(write_fd)
            pid = self._receive(b"P", case_deadline)
            captured, status = _read_capped(read_fd, cap, case_deadline)
        finally:
            os.close(read_fd)
        if status != "ok":
            _kill(pid)
        try:
            grace = time.monotonic() + GRACE_SECONDS
            code = self._receive(b"S", case_deadline if status == "ok" else grace)
        except TimeoutError:
            # Output ended but the process kept running past the case deadline.
            _kill(pid)
            status = "case_timeout"
            code = self._receive(b"S", time.monotonic() + GRACE_SECONDS)
        if status == "ok" and code == 0:
            return {"ran": True, "rc": code, "out": encode_bytes(captured)}
        return {"ran": False, "rc": code, "e": "exit" if status == "ok" else status}

    def close(self) -> None:
        self.sock.close()
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass


def _run_io(payload: dict[str, Any], deadline: float) -> dict[str, Any]:
    cases = payload["cases"]
    # None: every case may use whatever the attempt budget has left, as in ``program.py``.
    per_case = payload["per_case"]
    cap = payload["output_bytes"]
    _write_candidate(payload["code"])
    zygote: _Zygote | None = None
    out: list[dict[str, Any]] = []
    try:
        for text in cases:
            if time.monotonic() >= deadline:
                out.append({"ran": False, "e": "budget"})
                continue
            try:
                if zygote is None:
                    zygote = _Zygote(payload)
                case_deadline = (
                    deadline if per_case is None else min(deadline, time.monotonic() + per_case)
                )
                out.append(zygote.run_case(text, cap, case_deadline))
            except Exception:
                if not _candidate_started:
                    # No fork server could take a first case: nothing of the candidate ran.
                    raise
                # A broken or killed fork server fails this case; the next case gets a new one.
                out.append({"ran": False, "e": "runner"})
                if zygote is not None:
                    zygote.close()
                    zygote = None
            _kill_others({zygote.pid} if zygote is not None else set())
    finally:
        if zygote is not None:
            zygote.close()
    return {"status": "ran", "mode": "io", "n": len(cases), "cases": out}


# --------------------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------------------


def supervise(payload: dict[str, Any]) -> dict[str, Any]:
    global _candidate_started
    _candidate_started = False
    deadline = time.monotonic() + payload["budget"]
    if payload["mode"] == "function":
        return _run_function(payload, deadline)
    return _run_io(payload, deadline)


def run_report(payload: dict[str, Any]) -> dict[str, Any]:
    """The final report for one attempt. A supervisor failure before any candidate process
    could exist (the candidate's process or the first fork server could not be started, for
    example on ``EAGAIN`` or ``ENOMEM``) is ``spawn_failed``, which the host retries as a host
    failure; a failure after that point is ``supervisor_error``, which the host scores."""
    try:
        return supervise(payload)
    except BaseException as error:
        status = "supervisor_error" if _candidate_started else "spawn_failed"
        return {"status": status, "error_type": type(error).__name__}


def _write_frame(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame) + "\n")
    sys.stdout.flush()


def main() -> None:
    hardened = _harden()
    try:
        payload = json.load(sys.stdin)
        nonce = payload["nonce"]
        if not isinstance(nonce, str):
            raise ValueError("missing nonce")
    except Exception as error:
        # No nonce means the host never sees "started": a host-side failure.
        _write_frame({"frame": "rejected", "error_type": type(error).__name__})
        return
    _write_frame({"frame": "started", "nonce": nonce, "hardened": hardened})
    oom_before = _oom_kills()
    report = run_report(payload)
    _kill_others(set())
    oom_after = _oom_kills()
    if oom_before is not None and oom_after is not None:
        report["oom_kills"] = oom_after - oom_before
    _write_frame({"frame": "final", "nonce": nonce, "report": report})


if __name__ == "__main__":
    main()
