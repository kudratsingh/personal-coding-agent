"""Run 2's CPU smoke with the Mac scoring in the Docker sandbox.

Run with ``SCRATCHCODER_SANDBOX_TESTS=1`` on a machine with the pinned sandbox image present
locally (never pulled). The Mac's bridge server runs the real ``LocalRewardSource`` and the
evaluation runner's ``SandboxDevExecutor``: every scripted program runs against the fixture
tasks' real hidden tests, the right ones pass, and no ``scratchcoder`` container is left.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.posttrain_smoke2 import smoke

integration = pytest.mark.skipif(
    os.environ.get("SCRATCHCODER_SANDBOX_TESTS") != "1",
    reason="requires explicitly enabled disposable Docker worker",
)


def scratchcoder_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=scratchcoder", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result.stdout.split()


@integration
def test_run_two_completes_with_rewards_and_verdicts_from_the_sandbox(tmp_path: Path) -> None:
    summary = smoke(tmp_path / "smoke", docker=True)
    assert summary["status"] == "completed", summary
    assert set(summary["stages"].values()) == {"completed"}
    assert summary["objects_outside_prefix"] == []
    report = json.loads((tmp_path / "smoke" / "run" / "report.json").read_text())
    # The greedy dev programs are the references: the sandbox passes every one of them.
    for check in report["rl"]["checks"]:
        assert check["greedy_pass_at_1"] == 1.0, check
    result = json.loads((tmp_path / "smoke" / "run" / "rl" / "result.json").read_text())
    updates = result["monitoring"]
    statuses = {status for row in updates for status in row["statuses"]}
    assert statuses <= {"completed", "failed"} and "completed" in statuses
    assert any(0 < row["mean_reward"] < 1 for row in updates)  # Right and wrong programs.
    assert scratchcoder_containers() == []
