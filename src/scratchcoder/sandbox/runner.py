"""Fail-closed Docker isolation; generated programs never run on the host."""

import json
import math
import os
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from uuid import uuid4

IMAGE = "python@sha256:2b4f19dae3a777dfc3b76730bda1e82e1f66ab2a2686fa93ca78edbfb4f04ffe"
ENDPOINT = "unix:///var/run/docker.sock"


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 2.0
    memory_mb: int = 128
    processes: int = 16
    output_bytes: int = 65536

    def __post_init__(self) -> None:
        if not 0.1 <= self.timeout_seconds <= 60 or not 64 <= self.memory_mb <= 1024:
            raise ValueError("invalid timeout or memory limit")
        if not 4 <= self.processes <= 64 or not 1024 <= self.output_bytes <= 1024 * 1024:
            raise ValueError("invalid process or output limit")


@dataclass(frozen=True)
class Result:
    status: str
    stdout: str = ""
    exit_code: int | None = None
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def command(name: str, limits: Limits, *, docker: str = "docker") -> list[str]:
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
        "fsize=1048576:1048576",
        "--ulimit",
        "nofile=64:64",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--workdir",
        "/tmp",
        "--log-driver=none",
        "-i",
        IMAGE,
        "python",
        "-I",
        "-S",
        "-c",
        supervisor,
    ]


def run(code: str, tests: str, limits: Limits | None = None, *, mode: str = "function") -> Result:
    limits = limits or Limits()
    if mode not in {"function", "io"} or not tests.strip() or len((code + tests).encode()) > 128000:
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
    payload = json.dumps(
        {
            "code": code,
            "tests": tests,
            "mode": mode,
            "timeout": limits.timeout_seconds,
            "output_bytes": limits.output_bytes,
        }
    ).encode()
    started = time.monotonic()
    status = "infrastructure_error"
    captured = bytearray()
    process: subprocess.Popen[bytes] | None = None
    try:
        with TemporaryFile() as input_stream:
            input_stream.write(payload)
            input_stream.seek(0)
            process = subprocess.Popen(
                command(name, limits, docker=docker),
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
                    if time.monotonic() - started > limits.timeout_seconds + 10:
                        status = "timeout"
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
                raw = json.loads(captured)
                if raw["status"] not in {"passed", "failed", "timeout", "output_limit"}:
                    raise ValueError("invalid sandbox response")
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
                status, exit_code=process.returncode, elapsed_seconds=time.monotonic() - started
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        return Result("infrastructure_error", elapsed_seconds=time.monotonic() - started)
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
