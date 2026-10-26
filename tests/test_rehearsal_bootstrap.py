"""The Pod bootstrap's rehearsal mode, run against stub programs: order, flags and secrecy.

``python3`` and ``setsid`` are replaced by stubs that record every call, so the script runs
end to end without a GPU, a network or a Runpod key; the real interpreter still runs the
script's own inline checks (the armed receipt and the 0600 key file).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.rehearsal_ops import pod_environment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "pilot_pod_bootstrap.sh"
BASH = shutil.which("bash")
POD = "podfixture01"
SECRET = "pod-secret-value-never-printed"
PILOT_TESTS = [
    "tests/test_model.py",
    "tests/test_optimizer.py",
    "tests/test_bfloat16.py",
    "tests/test_training.py",
    "tests/test_training_control.py",
    "tests/test_gpu_pilot.py",
    "tests/test_pilot.py",
    "tests/test_pilot_data.py",
    "tests/test_pilot_preflight.py",
]
PYTHON_STUB = r"""#!/usr/bin/env bash
# Record the call, then imitate the bootstrap's python3 steps without a GPU or network.
printf '%s\n' "$*" >> "$STUB_LOG"
if [[ -n "${R2_POD_SECRET_ACCESS_KEY:-}" ]]; then
  echo "  (pod key in environment)" >> "$STUB_LOG"
fi
case "$1" in
  -c) exec "$REAL_PYTHON" "$@" ;;
  -)
    script=$(cat)
    if [[ "$script" == *nvidia_smi* ]]; then echo '{}' > "$2/environment.json"; exit 0; fi
    if [[ "$script" == *S_IMODE* ]]; then echo "  (key file check)" >> "$STUB_LOG"; fi
    printf '%s\n' "$script" | "$REAL_PYTHON" - "${@:2}"
    exit $? ;;
  */pod_self_destruct.py)
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --receipt ]]; then receipt="$2"; fi
      shift
    done
    printf '{"state": "armed", "pod_id": "%s"}\n' "$RUNPOD_POD_ID" > "$receipt"
    exit 0 ;;
  -m)
    if [[ "$2" == tools.pilot_preflight ]]; then mkdir -p "$4"; echo '{}' > "$4/report.json"; fi
    exit 0 ;;
esac
exit 0
"""

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required")


def setup(tmp_path: Path, *, mode: int = 0o600, mac_key: bool = False) -> dict[str, Path]:
    stubs = tmp_path / "stubs"
    stubs.mkdir(parents=True)
    (stubs / "python3").write_text(PYTHON_STUB)
    (stubs / "setsid").write_text('#!/bin/sh\nexec "$@"\n')
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    environment = {
        "R2_ENDPOINT": "https://account.r2.cloudflarestorage.com",
        "R2_BUCKET": "scratch-artifacts",
        "R2_POD_ACCESS_KEY_ID": "pod-key-id",
        "R2_POD_SECRET_ACCESS_KEY": SECRET,
    }
    text = pod_environment(environment)
    if mac_key:
        text += "R2_SECRET_ACCESS_KEY=mac-secret\n"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secrets = workspace / "r2-pod.env"
    secrets.write_text(text)
    secrets.chmod(mode)
    return {"stubs": stubs, "workspace": workspace, "env": secrets, "log": tmp_path / "calls.log"}


def rehearsal_arguments(paths: dict[str, Path], **change: str) -> list[str]:
    values = {
        "--mode": "rehearsal",
        "--run": "A",
        "--deadline-unix": "1800044820",
        "--workspace": str(paths["workspace"]),
        "--repo": str(ROOT),
        "--r2-env": str(paths["env"]),
        "--binding": str(paths["workspace"] / "binding.json"),
        "--binding-sha256": "ab" * 32,
        "--data-root": str(paths["workspace"] / "data"),
        "--initialization-key": "rehearsal-init/" + "cd" * 32 + "/rehearsal-init.pt",
        "--initialization-sha256": "cd" * 32,
        "--initialization-size": "541142016",
        **change,
    }
    return [item for key, value in values.items() if value for item in (key, value)]


def run(paths: dict[str, Path], arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("R2_", "RUNPOD_", "PIP_"))
    }
    environment.update(
        PATH=f"{paths['stubs']}{os.pathsep}{environment['PATH']}",
        RUNPOD_POD_ID=POD,
        RUNPOD_API_KEY="placeholder",
        STUB_LOG=str(paths["log"]),
        REAL_PYTHON=sys.executable,
    )
    assert BASH is not None
    return subprocess.run(
        [BASH, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )


def calls(paths: dict[str, Path]) -> list[str]:
    if not paths["log"].exists():
        return []
    return paths["log"].read_text().splitlines()


def index(lines: list[str], fragment: str) -> int:
    matches = [number for number, line in enumerate(lines) if fragment in line]
    assert matches, f"no call contains {fragment!r}"
    return matches[0]


def test_rehearsal_arms_the_timer_first_and_keeps_the_key_secret(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    result = run(paths, rehearsal_arguments(paths))
    assert result.returncode == 0, result.stderr
    lines = calls(paths)
    destruct = index(lines, "pod_self_destruct.py")
    assert destruct == 0, "the self-destruct timer is the first program the bootstrap starts"
    assert "--max-lifetime-seconds 45000" in lines[destruct]
    assert "--deadline-unix 1800044820" in lines[destruct]
    order = [
        destruct,
        index(lines, "(key file check)"),
        index(lines, "tools.rehearsal_ops round-trip"),
        index(lines, "tools.rehearsal_binding fetch"),
        index(lines, "tools.rehearsal_binding verify"),
        index(lines, "tools.rehearsal_ops fetch-object"),
        index(lines, "-m pytest"),
        index(lines, "tools.pilot_preflight"),
    ]
    assert order == sorted(order)
    for fragment in ("round-trip", "rehearsal_binding fetch", "fetch-object"):
        line = lines[index(lines, fragment)]
        assert "--key-prefix R2_POD_" in line
        assert lines[index(lines, fragment) + 1] == "  (pod key in environment)"
    assert f"--pod-id {POD}" in lines[index(lines, "round-trip")]
    assert "--sha256 " + "cd" * 32 in lines[index(lines, "fetch-object")]
    tests = lines[index(lines, "-m pytest")].split()
    assert tests[2 : 2 + len(PILOT_TESTS)] == PILOT_TESTS
    for name in ("test_onepass_loader.py", "test_r2_uploader.py", "test_rehearsal_ops.py"):
        assert f"tests/{name}" in tests
    assert "tests/test_rehearsal_bootstrap.py" in tests
    check_pod_test_list([name for name in tests if name.startswith("tests/")])
    # The worker requires the compile check; the receipt tool refuses a report without it.
    preflight = lines[index(lines, "tools.pilot_preflight")]
    assert preflight.endswith("--device cuda:0 --compile-check"), preflight
    assert SECRET not in result.stdout + result.stderr + paths["log"].read_text()
    assert "R2 environment loaded for run A (names only)" in result.stdout


NEEDS_DOCKER_OR_NETWORK = re.compile(
    r"(?i)docker|SCRATCHCODER_SANDBOX_TESTS|socket\.create_connection|urlopen\("
)


def check_pod_test_list(listed: list[str]) -> None:
    """The on-Pod list runs the worker's tests and nothing that needs Docker or the network."""
    assert "tests/test_rehearsal_run.py" in listed  # The worker's own tests.
    assert "tests/test_rehearsal_integration.py" in listed  # Worker records into the Mac side.
    assert len(set(listed)) == len(listed)
    for name in listed:
        path = ROOT / name
        assert path.is_file(), name  # A missing file would stop pytest on the Pod.
        if path.name != Path(__file__).name:  # This file names the patterns it looks for.
            assert not NEEDS_DOCKER_OR_NETWORK.search(path.read_text(encoding="utf-8")), name
    # Every rehearsal test file runs there, except the data build's (Mac-only, Parquet).
    expected = {
        f"tests/{path.name}"
        for path in (ROOT / "tests").glob("test_rehearsal_*.py")
        if path.name != "test_rehearsal_build.py"
    }
    assert expected <= set(listed) and "tests/test_rehearsal_build.py" not in listed


def test_a_readable_key_file_stops_after_the_timer_is_armed(tmp_path: Path) -> None:
    paths = setup(tmp_path, mode=0o644)
    result = run(paths, rehearsal_arguments(paths))
    assert result.returncode == 5 and "0600" in result.stderr
    lines = calls(paths)
    assert index(lines, "pod_self_destruct.py") == 0
    assert not [line for line in lines if "tools." in line]


def test_the_macs_key_on_the_pod_is_refused(tmp_path: Path) -> None:
    paths = setup(tmp_path, mac_key=True)
    result = run(paths, rehearsal_arguments(paths))
    assert result.returncode == 5 and "R2_SECRET_ACCESS_KEY is set on the Pod" in result.stderr
    assert "mac-secret" not in result.stdout + result.stderr
    assert not [line for line in calls(paths) if "tools." in line]


@pytest.mark.parametrize(
    "change",
    [
        {"--run": "C"},
        {"--binding-sha256": "short"},
        {"--initialization-size": "0"},
        {"--r2-env": ""},
        {"--mode": "other"},
    ],
)
def test_incomplete_rehearsal_arguments_stop_before_anything_runs(
    tmp_path: Path, change: dict[str, str]
) -> None:
    paths = setup(tmp_path)
    result = run(paths, rehearsal_arguments(paths, **change))
    assert result.returncode == 2 and not calls(paths)


def test_pilot_mode_is_unchanged(tmp_path: Path) -> None:
    paths = setup(tmp_path)
    arguments = ["--deadline-unix", "1800005220", "--workspace", str(paths["workspace"])]
    result = run(paths, [*arguments, "--repo", str(ROOT)])
    assert result.returncode == 0, result.stderr
    lines = calls(paths)
    destruct = lines[index(lines, "pod_self_destruct.py")]
    assert "--max-lifetime-seconds" not in destruct
    assert destruct.endswith(f"--stop-file {paths['workspace']}/TERMINATE_NOW")
    assert lines[index(lines, "-m pytest")].split()[2:] == [
        *PILOT_TESTS,
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    preflight = lines[index(lines, "tools.pilot_preflight")]
    assert preflight.endswith("--device cuda:0") and "--compile-check" not in preflight
    assert not [line for line in lines if "rehearsal" in line]
    assert "R2 environment" not in result.stdout
    refused = run(setup(tmp_path / "again"), [*arguments, "--run", "A"])
    assert refused.returncode == 2 and "need --mode rehearsal" in refused.stderr


def test_script_keeps_the_live_pilot_fixes_and_never_traces_secrets() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "set -a; . /etc/rp_environment; set +a" in script
    assert "export PIP_BREAK_SYSTEM_PACKAGES=1" in script
    assert "set -x" not in script
    for line in script.splitlines():
        if line.lstrip().startswith("echo"):
            assert "${!" not in line, line  # never prints a variable's value by name
    lines = script.splitlines()
    arm = next(i for i, line in enumerate(lines) if "pod_self_destruct.py" in line)
    source = next(i for i, line in enumerate(lines) if '. "$R2_ENV"' in line)
    first_tool = next(i for i, line in enumerate(lines) if "-m tools." in line)
    assert arm < source < first_tool


def test_pod_receipts_named_by_the_bootstrap_are_the_ones_the_runbook_pulls() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    runbook = (ROOT / "REHEARSAL_RUNBOOK.md").read_text(encoding="utf-8")
    for name in (
        "self-destruct.json",
        "r2-round-trip.json",
        "data-fetch.json",
        "data-verification.json",
        "initialization-fetch.json",
        "preflight-a40/report.json",
    ):
        assert f"$WORKSPACE/{name}" in script, name
        assert name in runbook, name
    assert "{root}/environment.json" in script and "environment.json" in runbook
