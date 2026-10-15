"""Guard against reporting success for the unimplemented pipeline."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("entrypoint", [[sys.executable, "-m", "scratchcoder"], ["scratchcoder"]])
def test_smoke_fails_explicitly(entrypoint: list[str], tmp_path: Path) -> None:
    result = subprocess.run(
        [*entrypoint, "smoke"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert not result.stdout
    assert "Smoke pipeline is incomplete" in result.stderr
    assert "pretraining stage 1" in result.stderr
    assert "pretraining stage 2" in result.stderr
    assert "No training or evaluation was run" in result.stderr
    assert not list(tmp_path.iterdir())


def test_unknown_command_cannot_report_success() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scratchcoder", "train"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
