"""Fail-closed Docker isolation; generated programs never run on the host."""

import json
import math
import os
import re
import secrets
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any
from uuid import uuid4

from scratchcoder.sandbox.program import REQUIRED_PROTECTIONS

IMAGE = "python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe"
ENDPOINT = "unix:///var/run/docker.sock"
HOST_WATCHDOG_SECONDS = 10.0
"""How long past the candidate deadline the host waits for the container before giving up.
The supervisor enforces the candidate deadline itself, so reaching this is a host stall."""
RESPONSE_STATUSES = frozenset({"passed", "failed", "timeout", "output_limit"})
"""Supervisor statuses that are a verdict on the candidate."""
SUPERVISOR_HOST_FAILURES = {
    "unprotected": "supervisor_unprotected",
    "spawn_failed": "spawn_failed",
}
"""Supervisor statuses that say nothing about the candidate (no candidate code ran), mapped to
the ``detail`` of the ``infrastructure_error`` result the host reports for them."""
_RESPONSE_KEYS = frozenset({"status", "stdout", "exit_code", "nonce", "protections"})
_OPTIONAL_KEYS = frozenset({"error_type"})


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 2.0
    memory_mb: int = 128
    processes: int = 16
    output_bytes: int = 65536
    input_bytes: int = 128000

    def __post_init__(self) -> None:
        if not 0.1 <= self.timeout_seconds <= 60 or not 64 <= self.memory_mb <= 4096:
            raise ValueError("invalid timeout or memory limit")
        if not 4 <= self.processes <= 64 or not 1024 <= self.output_bytes <= 1024 * 1024:
            raise ValueError("invalid process or output limit")
        if type(self.input_bytes) is not int or not 1024 <= self.input_bytes <= 8 * 1024 * 1024:
            raise ValueError("invalid input byte limit")


@dataclass(frozen=True)
class Runtime:
    image: str = IMAGE
    site_packages: bool = False

    def __post_init__(self) -> None:
        if self.image != IMAGE and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image):
            raise ValueError("runtime must use the default digest or an immutable local image ID")
        if type(self.site_packages) is not bool:
            raise ValueError("site_packages must be an explicit boolean")


@dataclass(frozen=True)
class Result:
    status: str
    stdout: str = ""
    exit_code: int | None = None
    elapsed_seconds: float = 0.0
    host_stalled: bool = False
    """True when the host watchdog, not the supervisor, ended the call. The status stays
    ``timeout`` so the status vocabulary is unchanged, but such a result is not a verdict on
    the candidate: callers must retry it and never score it (as a pass, a failure or a
    timeout)."""
    detail: str = ""
    """Why an ``infrastructure_error`` happened: ``malformed_response`` (the container's output
    was not exactly one valid, nonce-bearing result line), ``supervisor_unprotected`` (a
    required supervisor protection could not be established, so nothing ran),
    ``spawn_failed`` (the first candidate process could not start), ``container_exit`` (the
    docker CLI ended with a status other than 0, 125 or 137, or by a signal; see ``exit_code``) or
    ``runner_error`` (the host side failed). Empty for every other status. It is diagnostic only: an
    ``infrastructure_error`` is never a verdict, whatever its detail."""

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def parse_response(output: bytes, nonce: str, protections: tuple[str, ...]) -> dict[str, Any]:
    """Validate the supervisor's output and return its result document.

    The output must be exactly one ASCII line, ending in a newline, holding a JSON object with
    the expected keys and types, the host's ``nonce`` and every protection in ``protections``
    reported as established. Anything else (extra lines, a second document, a wrong nonce, a
    missing protection, a type error, deep nesting) raises :class:`ValueError`; no other
    exception escapes. The caller turns that into an ``infrastructure_error``: such output
    is never a verdict, because a candidate that cannot reach the result channel cannot be
    its cause."""
    try:
        if not output.endswith(b"\n") or output.count(b"\n") != 1 or not output.isascii():
            raise ValueError("not exactly one ASCII line")
        raw = json.loads(output)
        if not isinstance(raw, dict):
            raise ValueError("not an object")
        keys = set(raw)
        if not _RESPONSE_KEYS <= keys or not keys <= _RESPONSE_KEYS | _OPTIONAL_KEYS:
            raise ValueError("unexpected keys")
        if not isinstance(raw["nonce"], str) or not secrets.compare_digest(raw["nonce"], nonce):
            raise ValueError("wrong nonce")
        status = raw["status"]
        if status not in RESPONSE_STATUSES and status not in SUPERVISOR_HOST_FAILURES:
            raise ValueError("unknown status")
        if not isinstance(raw["stdout"], str):
            raise ValueError("stdout is not text")
        exit_code = raw["exit_code"]
        if exit_code is not None and type(exit_code) is not int:
            raise ValueError("exit code is not an integer")
        reported = raw["protections"]
        if not isinstance(reported, dict) or any(
            type(value) is not bool for value in reported.values()
        ):
            raise ValueError("protections are not flags")
        if status != "unprotected" and not all(reported.get(name) is True for name in protections):
            raise ValueError("a verdict without every required protection")
        if "error_type" in raw and not isinstance(raw["error_type"], str):
            raise ValueError("error type is not text")
        return raw
    except ValueError:
        raise
    except Exception as error:  # RecursionError, TypeError, MemoryError, ...
        raise ValueError(f"unparseable response: {type(error).__name__}") from None


def command(
    name: str,
    limits: Limits,
    *,
    docker: str = "docker",
    runtime: Runtime | None = None,
    supervisor: str | None = None,
) -> list[str]:
    """Build the fixed hardened ``docker run`` argument vector.

    ``supervisor`` is the trusted supervisor source executed as PID 1 inside the
    container; the default is ``program.py`` (the evaluation path of record). The
    reward channel passes its own supervisor here. Every isolation flag is identical
    regardless of which supervisor runs.
    """
    runtime = runtime or Runtime()
    file_bytes = max(1048576, limits.input_bytes + 4096)
    if supervisor is None:
        supervisor = Path(__file__).with_name("program.py").read_text(encoding="utf-8")
    return [
        docker,
        "--host",
        ENDPOINT,
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--platform",
        "linux/amd64",
        "--network=none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit",
        str(limits.processes),
        "--memory",
        f"{limits.memory_mb}m",
        "--memory-swap",
        f"{limits.memory_mb}m",
        "--cpus",
        "1",
        "--ulimit",
        f"cpu={math.ceil(limits.timeout_seconds)}:{math.ceil(limits.timeout_seconds)}",
        "--ulimit",
        f"fsize={file_bytes}:{file_bytes}",
        "--ulimit",
        "nofile=64:64",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--workdir",
        "/tmp",
        "--log-driver=none",
        "-i",
        runtime.image,
        "python",
        "-I",
        "-S",
        "-c",
        supervisor,
    ]


def run(
    code: str,
    tests: str,
    limits: Limits | None = None,
    *,
    mode: str = "function",
    runtime: Runtime | None = None,
) -> Result:
    limits = limits or Limits()
    runtime = runtime or Runtime()
    if (
        mode not in {"function", "io"}
        or not tests.strip()
        or len((code + tests).encode()) > limits.input_bytes
    ):
        raise ValueError("invalid test mode, missing tests, or oversized input")
    if mode == "io":
        cases = json.loads(tests)
        if (
            not isinstance(cases, list)
            or not 1 <= len(cases) <= 100
            or any(
                not isinstance(case, dict)
                or set(case) != {"input", "output"}
                or not all(isinstance(value, str) for value in case.values())
                for case in cases
            )
        ):
            raise ValueError("I/O tests require input/output string pairs")
    docker = shutil.which("docker")
    if docker is None:
        return Result("unavailable")
    name = "scratchcoder-" + uuid4().hex
    nonce = secrets.token_hex(16)
    payload = json.dumps(
        {
            "nonce": nonce,
            "code": code,
            "tests": tests,
            "mode": mode,
            "timeout": limits.timeout_seconds,
            "output_bytes": limits.output_bytes,
            "site_packages": runtime.site_packages,
        }
    ).encode()
    started = time.monotonic()
    status = "infrastructure_error"
    captured = bytearray()
    process: subprocess.Popen[bytes] | None = None
    host_stalled = False
    try:
        with TemporaryFile() as input_stream:
            input_stream.write(payload)
            input_stream.seek(0)
            process = subprocess.Popen(
                command(name, limits, docker=docker, runtime=runtime),
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
                    if time.monotonic() - started > limits.timeout_seconds + HOST_WATCHDOG_SECONDS:
                        status = "timeout"
                        host_stalled = True
                        break
                    for key, _ in selector.select(timeout=0.05):
                        chunk = os.read(key.fd, 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            captured.extend(chunk)
                    if len(captured) > limits.output_bytes * 8 + 65536:
                        status = "output_limit"
                        break
                if process.poll() is None:
                    if status in {"timeout", "output_limit"}:
                        process.kill()
                    process.wait(timeout=2)
            finally:
                selector.close()
                process.stdout.close()
            if process.returncode == 0 and status not in {"timeout", "output_limit"}:
                try:
                    raw = parse_response(bytes(captured), nonce, REQUIRED_PROTECTIONS)
                except ValueError:
                    return Result(
                        "infrastructure_error",
                        elapsed_seconds=time.monotonic() - started,
                        detail="malformed_response",
                    )
                if raw["status"] in SUPERVISOR_HOST_FAILURES:
                    return Result(
                        "infrastructure_error",
                        elapsed_seconds=time.monotonic() - started,
                        detail=SUPERVISOR_HOST_FAILURES[raw["status"]],
                    )
                return Result(
                    raw["status"],
                    raw["stdout"][: limits.output_bytes],
                    raw["exit_code"],
                    time.monotonic() - started,
                )
            if process.returncode == 125:
                status = "unavailable"
            elif process.returncode == 137:
                status = "resource_limit"
            return Result(
                status,
                exit_code=process.returncode,
                elapsed_seconds=time.monotonic() - started,
                host_stalled=host_stalled,
                detail="container_exit" if status == "infrastructure_error" else "",
            )
    except Exception:
        # Every host-side failure, whatever its type, is unresolved, never a verdict.
        return Result(
            "infrastructure_error",
            elapsed_seconds=time.monotonic() - started,
            detail="runner_error",
        )
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        try:
            subprocess.run(
                [docker, "--host", ENDPOINT, "rm", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
