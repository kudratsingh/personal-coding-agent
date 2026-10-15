"""Guard against reporting success for the unimplemented pipeline."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("entrypoint", [[sys.executable, "-m", "scratchcoder"], ["scratchcoder"]])
def test_smoke_runs_complete_synthetic_pipeline(entrypoint: list[str], tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs/smoke.yaml"
    result = subprocess.run(
        [*entrypoint, "smoke", "--config", str(config), "--output", str(tmp_path / "smoke")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Synthetic smoke passed" in result.stdout
    assert (tmp_path / "smoke/stage1.pt").exists()
    assert (tmp_path / "smoke/stage2.pt").exists()
    assert (tmp_path / "smoke/summary.json").exists()


def test_unknown_command_cannot_report_success() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scratchcoder", "not-a-command"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
