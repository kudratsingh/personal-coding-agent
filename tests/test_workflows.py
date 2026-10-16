from pathlib import Path
from typing import Any

import pytest
import yaml

from tools import check_workflows
from tools.check_workflows import validate_workflow


def test_current_literal_and_matrix_subset() -> None:
    validate_workflow(yaml.safe_load("jobs:\n  test:\n    runs-on: ubuntu-latest\n"))
    validate_workflow({"jobs": {"test": {"runs-on": "macos-latest"}}})
    validate_workflow(
        {
            "jobs": {
                "test": {
                    "runs-on": "${{ matrix.os }}",
                    "strategy": {"matrix": {"os": ["ubuntu-latest", "macos-latest"]}},
                }
            }
        }
    )


@pytest.mark.parametrize(
    "runner",
    [
        None,
        "ubuntu-latest-8-cores",
        "self-hosted",
        ["self-hosted", "linux"],
        {"group": "larger-runners"},
        "${{ inputs.runner }}",
        "${{ matrix.os || 'ubuntu-latest' }}",
    ],
)
def test_unknown_runner_declarations_fail(runner: Any) -> None:
    with pytest.raises(ValueError, match="owner review"):
        validate_workflow({"jobs": {"test": {"runs-on": runner}}})


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        "${{ fromJSON(inputs.matrix) }}",
        {},
        {"os": []},
        {"os": "ubuntu-latest"},
        {"os": ["ubuntu-latest", "custom-paid"]},
        {"os": [True]},
        {"os": ["ubuntu-latest"], "include": [{"os": "custom-paid"}]},
        {"os": ["ubuntu-latest"], "exclude": []},
    ],
)
def test_unreviewed_matrix_shapes_fail(matrix: Any) -> None:
    with pytest.raises(ValueError):
        validate_workflow(
            {
                "jobs": {
                    "test": {
                        "runs-on": "${{ matrix.os }}",
                        "strategy": {"matrix": matrix},
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"jobs": []},
        {"jobs": {}},
        {"jobs": {"test": {"uses": "owner/repo/.github/workflows/other.yml@main"}}},
        {"jobs": {"test": "not-a-job"}},
    ],
)
def test_invalid_or_unresolved_workflows_fail(value: Any) -> None:
    with pytest.raises(ValueError):
        validate_workflow(value)


def test_cli_checks_all_files_without_source_excerpts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / ".github/workflows"
    directory.mkdir(parents=True)
    monkeypatch.setattr(check_workflows, "ROOT", tmp_path)
    assert check_workflows.main() == 1
    (directory / "valid.yaml").write_text("jobs:\n  test:\n    runs-on: ubuntu-latest\n")
    assert check_workflows.main() == 0
    (directory / "invalid.yml").write_text("jobs: [fixture-private-text\n")
    assert check_workflows.main() == 1
    assert "fixture-private-text" not in capsys.readouterr().out
    (directory / "invalid.yml").unlink()
    (directory / "linked.yml").symlink_to("valid.yaml")
    assert check_workflows.main() == 1


def test_symlink_directory_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / "private").mkdir()
    (tmp_path / "private/secret.yml").write_bytes(b"\xff")
    (tmp_path / ".github/workflows").symlink_to("../private", target_is_directory=True)
    monkeypatch.setattr(check_workflows, "ROOT", tmp_path)
    assert check_workflows.main() == 1
