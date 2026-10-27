"""Trusted reward supervisor, executed ONLY as PID 1 inside the isolated container.

Unlike ``program.py`` (the evaluation path of record), this supervisor NEVER receives
the expected values of a task. The host passes only the candidate code plus the *left*
side of each hidden assertion (function mode) or each case's stdin (I/O mode). The
candidate runs in a separate child process; the supervisor collects whatever the child
reports and forwards it, unjudged, to the host, which compares it against the secret
expected values it kept to itself.

The result channel is therefore treated as UNTRUSTED: a candidate can forge any value on
it, but cannot know which value would pass, so forgery earns no more than an honest
solution would. The supervisor's only jobs are isolation-preserving framing: keep the
candidate's stdout off the container's stdout, bound every buffer, enforce a per-call and
an overall per-attempt wall budget, and emit one clean JSON object to the host.
"""

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any

_CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}

# Source of the per-attempt driver that runs in the candidate child for function mode.
# It reads its configuration (candidate code, import setup, call expressions, per-call
# budget, and the inherited result-pipe descriptor number) as one JSON document on stdin and
# writes one result line per call to that descriptor. It is a best-effort convenience for
# honest candidates; its integrity is not a security assumption, because the host compares
# every reported value against secret expectations it never shared with the container.
DRIVER = r"""
import json, os, signal, sys

_write = os.write
_dumps = json.dumps
_result_fd = 2


def _encode(value, depth=0):
    if depth > 32:
        raise TypeError("value nested too deeply")
    kind = type(value)
    if value is None:
        return ["none"]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", value]
    if kind is float:
        if value != value:
            return ["float_s", "nan"]
        if value in (float("inf"), float("-inf")):
            return ["float_s", "inf" if value > 0 else "-inf"]
        return ["float", value]
    if kind is str:
        return ["str", value]
    if kind is bytes:
        return ["bytes", list(value)]
    if kind is list:
        return ["list", [_encode(item, depth + 1) for item in value]]
    if kind is tuple:
        return ["tuple", [_encode(item, depth + 1) for item in value]]
    if kind is set:
        return ["set", [_encode(item, depth + 1) for item in value]]
    if kind is frozenset:
        return ["frozenset", [_encode(item, depth + 1) for item in value]]
    if kind is dict:
        return ["dict", [[_encode(k, depth + 1), _encode(v, depth + 1)] for k, v in value.items()]]
    raise TypeError("unsupported value type")


def _emit(obj):
    try:
        _write(_result_fd, (_dumps(obj) + "\n").encode())
    except Exception:
        pass


def main():
    global _result_fd
    config = json.load(sys.stdin)
    _result_fd = config["result_fd"]
    calls = config["calls"]
    per_call = config["per_call"]
    namespace = {"__name__": "__candidate__"}
    try:
        exec(compile(config["setup"], "<setup>", "exec"), namespace)
        exec(compile(config["code"], "<candidate>", "exec"), namespace)
    except BaseException:
        for index in range(len(calls)):
            _emit({"i": index, "ok": False, "e": "candidate_error"})
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
            _emit({"i": index, "ok": False, "e": "call_timeout"})
            continue
        except BaseException as error:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            _emit({"i": index, "ok": False, "e": type(error).__name__})
            continue
        try:
            encoded = _encode(value)
        except TypeError:
            _emit({"i": index, "ok": False, "e": "unserializable"})
            continue
        _emit({"i": index, "ok": True, "v": encoded})


main()
"""


def _child_arguments(site_packages: bool) -> list[str]:
    return [sys.executable, "-I", *([] if site_packages else ["-S"]), "-c", DRIVER]


def _drain(
    process: subprocess.Popen[bytes],
    sources: dict[str, Any],
    caps: dict[str, int],
    buffers: dict[str, bytearray],
    deadline: float,
) -> None:
    """Read every registered source until EOF or the per-attempt wall deadline."""
    selector = selectors.DefaultSelector()
    for name, source in sources.items():
        selector.register(source, selectors.EVENT_READ, name)
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                break
            for key, _ in selector.select(timeout=min(0.05, remaining)):
                block = os.read(key.fd, 4096)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[key.data]
                if len(buffer) < caps[key.data]:
                    buffer.extend(block[: caps[key.data] - len(buffer)])
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
    finally:
        selector.close()


def _run_function(payload: dict[str, Any], deadline: float) -> dict[str, Any]:
    calls = payload["calls"]
    per_call = min(payload["per_call"], payload["budget"])
    read_fd, write_fd = os.pipe()
    # The driver writes results to the inherited write end at its own descriptor number;
    # pass_fds keeps it open across exec, so no remapping is needed (the fd-closing step runs
    # after any preexec_fn and would otherwise close a remapped descriptor).
    os.set_inheritable(write_fd, True)
    config = json.dumps(
        {
            "code": payload["code"],
            "setup": payload["setup"],
            "calls": calls,
            "per_call": per_call,
            "result_fd": write_fd,
        }
    ).encode()
    result_reader = os.fdopen(read_fd, "rb", buffering=0)
    buffers = {"stdout": bytearray(), "result": bytearray()}
    caps = {"stdout": payload["output_bytes"], "result": payload["output_bytes"] * 2 + 65536}
    with TemporaryFile() as config_stream:
        config_stream.write(config)
        config_stream.seek(0)
        process = subprocess.Popen(
            _child_arguments(payload["site_packages"]),
            stdin=config_stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(write_fd,),
            env=_CHILD_ENV,
        )
    os.close(write_fd)
    assert process.stdout is not None
    sources = {"stdout": process.stdout, "result": result_reader}
    try:
        _drain(process, sources, caps, buffers, deadline)
    finally:
        result_reader.close()
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
    reported: dict[int, dict[str, Any]] = {}
    for line in bytes(buffers["result"]).split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        index = row.get("i") if isinstance(row, dict) else None
        if isinstance(index, int) and 0 <= index < len(calls) and index not in reported:
            reported[index] = row
    calls_out = [
        reported.get(index, {"i": index, "ok": False, "e": "unrun"}) for index in range(len(calls))
    ]
    return {"status": "ran", "mode": "function", "n": len(calls), "calls": calls_out}


def _run_io(payload: dict[str, Any], deadline: float) -> dict[str, Any]:
    cases = payload["cases"]
    per_case = payload["per_case"]
    Path("/tmp/candidate.py").write_text(payload["code"], encoding="utf-8")
    arguments = [
        sys.executable,
        "-I",
        *([] if payload["site_packages"] else ["-S"]),
        "/tmp/candidate.py",
    ]
    out: list[dict[str, Any]] = []
    for text in cases:
        if time.monotonic() >= deadline:
            out.append({"ran": False, "e": "budget"})
            continue
        case_deadline = min(deadline, time.monotonic() + per_case)
        captured = bytearray()
        with TemporaryFile() as input_stream:
            input_stream.write(text.encode())
            input_stream.seek(0)
            process = subprocess.Popen(
                arguments,
                stdin=input_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_CHILD_ENV,
            )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        status = "ok"
        try:
            while selector.get_map():
                if time.monotonic() >= case_deadline:
                    status = "case_timeout"
                    break
                for key, _ in selector.select(timeout=0.05):
                    block = os.read(key.fd, 4096)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    if len(captured) < payload["output_bytes"]:
                        captured.extend(block[: payload["output_bytes"] - len(captured)])
            if status == "case_timeout" or len(captured) >= payload["output_bytes"]:
                process.kill()
            process.wait(timeout=2)
        finally:
            selector.close()
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
        decoded = bytes(captured).decode("utf-8", errors="replace")
        if status == "case_timeout":
            out.append({"ran": False, "e": "case_timeout", "stdout": decoded})
        elif process.returncode != 0:
            out.append({"ran": False, "e": "exit", "stdout": decoded})
        else:
            out.append({"ran": True, "stdout": decoded})
    return {"status": "ran", "mode": "io", "n": len(cases), "cases": out}


def supervise(payload: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + payload["budget"]
    if payload["mode"] == "function":
        return _run_function(payload, deadline)
    return _run_io(payload, deadline)


if __name__ == "__main__":
    try:
        print(json.dumps(supervise(json.load(sys.stdin))))
    except BaseException as error:
        print(json.dumps({"status": "supervisor_error", "error_type": type(error).__name__}))
