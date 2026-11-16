"""Case IDs are the identity of every agent result: the load rule (well formed, distinct even
ignoring case) and the accounting rule (a scorecard or comparison never merges two cases).
Host-only: nothing executes workspace or model code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_fixtures import FIXED_DOUBLE, ScriptedExecutor, write_case, write_registry
from test_agent_experiments import card
from test_agent_loop import Scripted, fake_loader, receipt, run, tool_args

from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent import scorecard
from scratchcoder.data.chat import PYTHON_END, PYTHON_START


@pytest.mark.parametrize("case_id", ["a--b", "X" * 33, "x-", "-x", "x.y", "", "K01 "])
def test_malformed_case_ids_are_refused_at_load(tmp_path: Path, case_id: str) -> None:
    write_case(tmp_path, id=case_id)
    with pytest.raises(agent_cases.CaseError, match="case id"):
        agent_cases.load_case(next(tmp_path.iterdir()))


@pytest.mark.parametrize("case_id", ["K01", "X" * 32, "a_b-c", "7", "x_"])
def test_well_formed_case_ids_load(tmp_path: Path, case_id: str) -> None:
    write_case(tmp_path, id=case_id)
    assert agent_cases.load_case(next(tmp_path.iterdir())).id == case_id


def registry_with_ids(root: Path, first: str, second: str) -> None:
    """Two cases in distinct directories whose case.json IDs are ``first`` and ``second``,
    with a registry that lists them as they are (written without the registry's checks)."""
    write_case(root, "A01", id=first)
    write_case(root, "B01", id=second)
    entries = []
    for directory in sorted(root.iterdir()):
        alone = root.parent / f"{root.name}-{directory.name}"
        alone.mkdir()
        directory.rename(alone / directory.name)
        entries += agent_cases.build_registry(alone)["cases"]
        (alone / directory.name).rename(directory)
        alone.rmdir()
    registry = {"schema": agent_cases.REGISTRY_SCHEMA, "cases": entries}
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [("X01", "x01", "equal ignoring case"), ("X01", "X01", "repeated")],
)
def test_colliding_case_ids_are_refused_at_registry_load(
    tmp_path: Path, first: str, second: str, message: str
) -> None:
    root = tmp_path / "cases"
    root.mkdir()
    registry_with_ids(root, first, second)
    for only in (None, {first}):
        with pytest.raises(agent_cases.CaseError, match=message):
            agent_cases.load_cases(root, only=only)
    (root / "registry.json").unlink()
    with pytest.raises(agent_cases.CaseError, match=message):
        agent_cases.build_registry(root)


def test_the_tool_refuses_colliding_cases_before_anything_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import agent_dev_run

    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    (tmp_path / "cases").mkdir()
    registry_with_ids(tmp_path / "cases", "X01", "x01")
    out, path = tmp_path / "out", tmp_path / "receipt.json"
    argv = tool_args(tmp_path, out, path)
    receipt(path, argv)
    executor = ScriptedExecutor()
    loader = fake_loader([[PYTHON_START, *FIXED_DOUBLE.encode(), PYTHON_END]])
    assert agent_dev_run.main(argv, execute=executor, model_loader=loader) == 2
    assert not executor.calls and not out.exists()


def test_a_scorecard_never_merges_two_cases_or_counts_an_instance_twice(tmp_path: Path) -> None:
    write_case(tmp_path)
    write_registry(tmp_path)
    case = agent_cases.load_cases(tmp_path)[0]
    first, _, _ = run(case, Scripted([None, None, None]))
    second, _, _ = run(case, Scripted([None, None, None]))
    with pytest.raises(agent_cases.CaseError, match="more than once"):
        scorecard.build([first, second], {})
    second.case_id = "x01"
    first.case_id = "X01"
    with pytest.raises(agent_cases.CaseError, match="equal ignoring case"):
        scorecard.build([first, second], {})
    second.case_id = "Y01"
    built = scorecard.build([first, second], {"cases": ["X01", "Y01"]})
    assert set(built["groups"]["scripted/repair-3"]["per_case"]) == {"X01", "Y01"}
    with pytest.raises(agent_cases.CaseError, match="identity cases"):
        scorecard.build([first, second], {"cases": ["X01", "x01"]})


def test_compare_refuses_rows_that_could_merge(tmp_path: Path) -> None:
    from tools import agent_dev_run

    with pytest.raises(agent_cases.CaseError, match="share a name"):
        scorecard.comparison([("A", card()), ("A", card())])
    with pytest.raises(agent_cases.CaseError, match="equal ignoring case"):
        scorecard.comparison([("A", card()), ("B", card(cases=["X01", "x01"]))])
    clashing = card()
    clashing["instances"] = [{"case_id": "X01"}, {"case_id": "x01"}]
    with pytest.raises(agent_cases.CaseError, match="run B"):
        scorecard.comparison([("A", card()), ("B", clashing)])
    for name, body in (("A", card()), ("B", clashing)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "scorecard.json").write_text(json.dumps(body), "utf-8")
    md, js = tmp_path / "c.md", tmp_path / "c.json"
    command = ["compare", "--runs", str(tmp_path / "A"), str(tmp_path / "B")]
    command += ["--out-md", str(md), "--out-json", str(js), "--allow-noncomparable"]
    assert agent_dev_run.main(command) == 2 and not md.exists() and not js.exists()
