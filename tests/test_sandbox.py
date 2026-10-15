import json
import os

import pytest

from scratchcoder.sandbox.runner import IMAGE, Limits, command, run


def test_command_has_no_host_mounts_secrets_or_network() -> None:
    args = command("scratchcoder-test", Limits())
    for required in (
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--pids-limit",
        "--security-opt=no-new-privileges",
        "--pull=never",
        IMAGE,
    ):
        assert required in args
    assert "--mount" not in args and "-v" not in args and "--env-file" not in args
    assert "--user" in args and "65534:65534" in args


def test_missing_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scratchcoder.sandbox.runner.shutil.which", lambda _: None)
    assert run("raise RuntimeError('must not execute here')", "assert True").status == "unavailable"


integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)


@integration
def test_function_and_io_execution() -> None:
    assert run("def add(a,b): return a+b", "assert add(1,2) == 3").passed
    assert not run("def add(a,b): return 0", "assert add(1,2) == 3").passed
    assert not run("import os; os._exit(0)", "assert False").passed
    assert run(
        "print(int(input()) + 1)", json.dumps([{"input": "2\n", "output": "3"}]), mode="io"
    ).passed


@integration
@pytest.mark.parametrize(
    "code",
    [
        "while True: pass",
        "x = bytearray(1024 * 1024 * 1024)",
        "import os\nwhile True: os.fork()",
        "open('/etc/escape', 'w').write('bad')",
        "import socket\nsocket.create_connection(('8.8.8.8', 53), timeout=0.5)",
        "import sys\nsys.stdout.write('x' * 1024 * 1024 * 1024)",
    ],
)
def test_resource_and_escape_attempts_are_contained(code: str) -> None:
    result = run(code, "assert True", Limits(timeout_seconds=1, output_bytes=4096))
    assert result.status in {"failed", "timeout", "output_limit", "resource_limit"}
    assert len(result.stdout) <= 4096


@integration
def test_host_environment_and_repository_are_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCHCODER_TEST_SECRET", "host-only")
    code = (
        "import os\nfrom pathlib import Path\n"
        "assert os.getenv('SCRATCHCODER_TEST_SECRET') is None\n"
        "assert not Path('/workspace').exists()\n"
        "assert not Path('/var/run/docker.sock').exists()"
    )
    assert run(code, "assert True").passed


@integration
def test_streaming_output_is_bounded() -> None:
    result = run(
        "while True: print('x' * 4096, flush=True)",
        "assert True",
        Limits(timeout_seconds=2, output_bytes=4096),
    )
    assert result.status == "output_limit"
    assert len(result.stdout) <= 4096
