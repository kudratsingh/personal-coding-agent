"""The Pod bootstrap's post-training mode, run against the rehearsal test's stub programs.

The stub ``python3`` records every call; the real interpreter still runs the script's own
inline checks (the live timer, the code revision and the 0600 key file)."""

import re
from pathlib import Path

import pytest
from test_rehearsal_bootstrap import (
    BASH,
    NEEDS_DOCKER_OR_NETWORK,
    PILOT_TESTS,
    POD,
    PYTHON_STUB,
    ROOT,
    SECRET,
    calls,
    index,
    run,
    setup,
    stop_stub_timers,  # noqa: F401 - the autouse fixture that stops the stub timers.
)

STUB = PYTHON_STUB.replace(
    '    if [[ "$2 $3" == "tools.rehearsal_ops round-trip" ]]; then',
    '    if [[ "$2 $3" == "tools.rehearsal_ops round-trip"'
    ' || "$2 $3" == "tools.posttrain_ops round-trip" ]]; then',
)
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required")


def posttrain_setup(tmp_path: Path) -> dict[str, Path]:
    paths = setup(tmp_path)
    (paths["stubs"] / "python3").write_text(STUB)
    return paths


def arguments(paths: dict[str, Path], **change: str) -> list[str]:
    values = {
        "--mode": "posttrain",
        "--run": "1",
        "--deadline-unix": "1800017820",
        "--lifetime-seconds": "17820",
        "--idle-seconds": "900",
        "--workspace": str(paths["workspace"]),
        "--repo": str(ROOT),
        "--revision": str(paths["revision"]),
        "--archive": str(paths["archive"]),
        "--r2-env": str(paths["env"]),
        "--binding": str(paths["workspace"] / "binding.json"),
        "--binding-sha256": "ab" * 32,
        "--data-root": str(paths["workspace"] / "inputs"),
        "--heartbeat": "/workspace/posttrain-run/heartbeat.json",
        **change,
    }
    return [item for key, value in values.items() if value for item in (key, value)]


def test_posttrain_arms_the_derived_lifetime_and_idle_rule_first(tmp_path: Path) -> None:
    paths = posttrain_setup(tmp_path)
    result = run(paths, arguments(paths))
    assert result.returncode == 0, result.stderr
    lines = calls(paths)
    destruct = index(lines, "pod_self_destruct.py")
    assert destruct == 0, "the self-destruct timer is the first program the bootstrap starts"
    timer = lines[destruct]
    assert "--max-lifetime-seconds 17820" in timer and "60000" not in timer
    assert "--idle-seconds 900" in timer and "--deadline-unix 1800017820" in timer
    assert "--activity /workspace/posttrain-run/heartbeat.json" in timer
    assert f"--activity {paths['workspace']}/setup-progress.json" in timer
    order = [
        destruct,
        index(lines, "--source bootstrap:armed"),
        index(lines, f"- {paths['revision']} {paths['archive']}"),
        index(lines, "(key file check)"),
        index(lines, "tools.posttrain_ops round-trip"),
        index(lines, "tools.posttrain_binding fetch"),
        index(lines, "tools.posttrain_binding verify"),
        index(lines, "-m pytest"),
        index(lines, "tools.pilot_preflight"),
        index(lines, "--source bootstrap:preflight"),
    ]
    assert order == sorted(order)
    for fragment in ("posttrain_ops round-trip", "posttrain_binding fetch"):
        line = lines[index(lines, fragment)]
        assert "--key-prefix R2_POD_" in line
        assert lines[index(lines, fragment) + 1] == "  (pod key in environment)"
    assert f"--pod-id {POD}" in lines[index(lines, "round-trip")]
    stamps = [
        line.split("--source bootstrap:")[1] for line in lines if "--source bootstrap:" in line
    ]
    assert stamps == [
        "armed",
        "revision",
        "r2-env",
        "environment",
        "install",
        "round-trip",
        "fetch",
        "verify",
        "tests",
        "preflight",
    ]
    assert not [line for line in lines if "rehearsal_binding" in line or "fetch-object" in line]
    preflight = lines[index(lines, "tools.pilot_preflight")]
    assert preflight.endswith("--device cuda:0 --compile-check"), preflight
    tests = lines[index(lines, "-m pytest")].split()
    assert tests[2 : 2 + len(PILOT_TESTS)] == PILOT_TESTS
    listed = [name for name in tests if name.startswith("tests/")]
    # Every post-training test runs there, except the runbook's: a Mac document's test, which
    # names the Mac's container runtime.
    expected = {f"tests/{path.name}" for path in (ROOT / "tests").glob("test_posttrain_*.py")}
    assert expected - {"tests/test_posttrain_runbook.py"} <= set(listed)
    assert "tests/test_posttrain_runbook.py" not in listed
    for name in listed:
        path = ROOT / name
        assert path.is_file(), name  # A missing file would stop pytest on the Pod.
        if path.name != Path(__file__).name:  # This file names the patterns it looks for.
            assert not NEEDS_DOCKER_OR_NETWORK.search(path.read_text(encoding="utf-8")), name
    assert SECRET not in result.stdout + result.stderr + paths["log"].read_text()
    assert "R2 environment loaded for run 1 (names only)" in result.stdout
    assert "run 1 inputs are pinned and verified" in result.stdout


@pytest.mark.parametrize(
    "change",
    [
        {"--lifetime-seconds": ""},
        {"--lifetime-seconds": "60001"},
        {"--idle-seconds": ""},
        {"--heartbeat": ""},
        {"--run": "A"},
        {"--binding-sha256": "short"},
        {"--initialization-key": "rehearsal-init/x.pt"},
    ],
)
def test_incomplete_posttrain_arguments_stop_before_anything_runs(
    tmp_path: Path, change: dict[str, str]
) -> None:
    paths = posttrain_setup(tmp_path)
    result = run(paths, arguments(paths, **change))
    assert result.returncode == 2 and not calls(paths), result.stderr


def test_post_training_arguments_are_refused_in_the_other_modes(tmp_path: Path) -> None:
    paths = posttrain_setup(tmp_path)
    rehearsal = run(paths, [*arguments(paths, **{"--mode": "rehearsal"})])
    assert rehearsal.returncode == 2 and not calls(paths)
    assert re.search("need --mode posttrain", rehearsal.stderr)
    pilot = run(
        paths,
        [
            "--deadline-unix",
            "1800005220",
            "--workspace",
            str(paths["workspace"]),
            "--idle-seconds",
            "900",
        ],
    )
    assert pilot.returncode == 2 and "need --mode rehearsal" in pilot.stderr
