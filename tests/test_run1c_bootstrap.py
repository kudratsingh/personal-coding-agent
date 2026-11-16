"""The Pod bootstrap for run 1c: its timer lifetime, its R2 prefix and its CPU test list.

Run against the rehearsal bootstrap test's stub programs (``test_posttrain_bootstrap``); the
lists of runs 1, 1b and 2 are unchanged. A Mac-side test: it names the patterns the Pod's
list must not need, so it does not run there itself.
"""

from pathlib import Path

import pytest
from test_posttrain_bootstrap import arguments, posttrain_setup
from test_rehearsal_bootstrap import (
    BASH,
    NEEDS_DOCKER_OR_NETWORK,
    ROOT,
    calls,
    index,
    run,
    stop_stub_timers,  # noqa: F401 - the autouse fixture that stops the stub timers.
)

pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required")
RUN1C_TESTS = (
    "tests/test_run1c_posttrain.py",
    "tests/test_run1c_integration.py",
    "tests/test_executed_specifications.py",
)
MAC_ONLY = {
    "tests/test_posttrain_runbook.py",
    "tests/test_posttrain_run1b_runbook.py",
    "tests/test_run1c_runbook.py",
    "tests/test_run1c_bootstrap.py",
}


def listed_tests(lines: list[str]) -> list[str]:
    return [name for name in lines[index(lines, "-m pytest")].split() if name.startswith("tests/")]


def test_run_1c_arms_its_lifetime_and_runs_its_own_tests(tmp_path: Path) -> None:
    paths = posttrain_setup(tmp_path)
    change = {"--run": "1c", "--deadline-unix": "1800003420", "--lifetime-seconds": "3420"}
    result = run(paths, arguments(paths, **change))
    assert result.returncode == 0, result.stderr
    lines = calls(paths)
    timer = lines[index(lines, "pod_self_destruct.py")]
    assert index(lines, "pod_self_destruct.py") == 0
    assert "--max-lifetime-seconds 3420" in timer and "--deadline-unix 1800003420" in timer
    assert "--run 1c" in lines[index(lines, "tools.posttrain_ops round-trip")]
    listed = listed_tests(lines)
    for name in RUN1C_TESTS:
        assert name in listed, name
    # Every post-training test runs there but run 2's and the Mac's own.
    expected = {
        f"tests/{path.name}"
        for path in (ROOT / "tests").glob("test_posttrain_*.py")
        if not path.name.startswith("test_posttrain_run2")
    }
    assert expected - MAC_ONLY <= set(listed) and not MAC_ONLY & set(listed)
    assert len(set(listed)) == len(listed)
    for name in listed:
        path = ROOT / name
        assert path.is_file(), name
        if path.name != "test_posttrain_bootstrap.py":  # It names the patterns it looks for.
            assert not NEEDS_DOCKER_OR_NETWORK.search(path.read_text(encoding="utf-8")), name
    assert "R2 environment loaded for run 1c (names only)" in result.stdout
    assert "run 1c inputs are pinned and verified" in result.stdout


@pytest.mark.parametrize(("run_name", "lifetime"), [("1", "17820"), ("1b", "8820"), ("2", "34620")])
def test_the_other_runs_lists_do_not_change(tmp_path: Path, run_name: str, lifetime: str) -> None:
    paths = posttrain_setup(tmp_path)
    deadline = str(1_800_000_000 + int(lifetime))
    change = {"--run": run_name, "--deadline-unix": deadline, "--lifetime-seconds": lifetime}
    result = run(paths, arguments(paths, **change))
    assert result.returncode == 0, result.stderr
    listed = listed_tests(calls(paths))
    assert not [name for name in listed if "run1c" in name]
    assert "tests/test_executed_specifications.py" not in listed
