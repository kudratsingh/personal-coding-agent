import os

import pytest

from scratchcoder.sandbox.runner import IMAGE, Limits, Runtime, command, run


def test_runtime_requires_immutable_image_and_retains_isolation() -> None:
    with pytest.raises(ValueError, match="immutable"):
        Runtime("python:latest", True)
    runtime = Runtime("sha256:" + "a" * 64, True)
    args = command("scratchcoder-test", Limits(input_bytes=2 * 1024 * 1024), runtime=runtime)
    assert runtime.image in args and IMAGE not in args
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--pull=never", "65534:65534"):
        assert flag in args
    assert "--mount" not in args and "-v" not in args
    for size in (0, True, 9 * 1024 * 1024):
        with pytest.raises(ValueError, match="input"):
            Limits(input_bytes=size)
    with pytest.raises(ValueError, match="oversized"):
        run("x" * 128001, "assert True")


integration = pytest.mark.skipif(
    not os.environ.get("SCRATCHCODER_NUMPY_IMAGE"),
    reason="requires an explicitly built disposable NumPy image",
)


def test_explicit_benchmark_memory_bound_does_not_change_default() -> None:
    assert Limits().memory_mb == 128
    args = command("scratchcoder-benchmark", Limits(memory_mb=4096))
    assert args[args.index("--memory") + 1] == "4096m"
    assert args[args.index("--memory-swap") + 1] == "4096m"
    with pytest.raises(ValueError, match="memory"):
        Limits(memory_mb=4097)


@integration
def test_numpy_runtime_correctness_and_bounded_large_test_script() -> None:
    runtime = Runtime(os.environ["SCRATCHCODER_NUMPY_IMAGE"], True)
    assert run(
        "import numpy as np\ndef add(a,b): return np.add(a,b)",
        "assert add(1,2)==3\n" + "# padding\n" * 15000,
        Limits(input_bytes=2 * 1024 * 1024),
        runtime=runtime,
    ).passed
    assert not run(
        "import numpy as np\ndef add(a,b): return 0", "assert add(1,2)==3", runtime=runtime
    ).passed
    assert not run("import numpy", "assert True", runtime=Runtime(runtime.image, False)).passed


@integration
@pytest.mark.parametrize(
    "code",
    [
        "while True: pass",
        "open('/etc/escape','w').write('bad')",
        "import socket; socket.create_connection(('8.8.8.8',53),timeout=0.1)",
    ],
)
def test_numpy_runtime_still_contains_abuse(code: str) -> None:
    result = run(
        code,
        "assert True",
        Limits(timeout_seconds=1),
        runtime=Runtime(os.environ["SCRATCHCODER_NUMPY_IMAGE"], True),
    )
    assert result.status in {"failed", "timeout", "resource_limit"}


@integration
def test_numpy_runtime_does_not_inherit_host_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCHCODER_HOST_SECRET", "host-only")
    assert run(
        "import os\nfrom pathlib import Path\n"
        "assert os.getenv('SCRATCHCODER_HOST_SECRET') is None\n"
        "assert os.getenv('OPENBLAS_NUM_THREADS') == '1'\n"
        "assert not Path('/var/run/docker.sock').exists()",
        "assert True",
        runtime=Runtime(os.environ["SCRATCHCODER_NUMPY_IMAGE"], True),
    ).passed
