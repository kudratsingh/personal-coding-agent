"""The static grammar compatibility tool: inputs, refusals and counts (text only)."""

import json
from pathlib import Path
from typing import Any

from tools import reward_grammar_compat


def _task(task_id: str, hidden: str, split: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "prompt": "p",
        "public_tests": "assert True",
        "hidden_tests": hidden,
        "source": "synthetic",
        "license": "project-authored",
        "written_by": "human",
        "split": split,
        "mode": "function",
    }


def _files(root: Path) -> list[str]:
    dev = root / "dev.jsonl"
    dev.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _task("d1", "assert f(1) == 2", "dev"),
                _task("d2", "assert isinstance(f(1), int) is True", "dev"),
                _task("d3", "assert f(1) > 2", "dev"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    train = root / "train.jsonl"
    train.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _task("t1", "assert f(1) == 2", "train"),
                _task("t2", "assert Box().empty() is True", "train"),
                _task("t3", "assert abs(-1) == 1", "train"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    candidates = root / "rl_candidates.jsonl"
    candidates.write_text('{"id": "t1"}\n{"id": "t2"}\n{"id": "t3"}\n', encoding="utf-8")
    screening = root / "screening.json"
    screening.write_text(json.dumps({"eligible": ["t1", "t2", "t3"]}), encoding="utf-8")
    exclusions = root / "exclusions.json"
    exclusions.write_text(
        json.dumps({"candidates": [{"id": "t3", "status": "excluded_confirmed_same_family"}]}),
        encoding="utf-8",
    )
    verifier = root / "verifier.json"
    verifier.write_text(json.dumps({"verifier_quarantine_clear": []}), encoding="utf-8")
    return [
        "--dev", str(dev), "--train", str(train), "--candidates", str(candidates),
        "--screening", str(screening), "--exclusions", str(exclusions),
        "--verifier-filter", str(verifier), "--commit", "test",
    ]  # fmt: skip


def test_the_comparison_counts_every_newly_unsupported_task(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert reward_grammar_compat.main([*_files(tmp_path), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))["sets"]
    assert summary["dev"]["newly_unsupported_ids"] == ["d2"]
    assert summary["dev"]["v1_parseable"] == 2 and summary["dev"]["v2_parseable"] == 1
    assert summary["rl_candidates"]["newly_unsupported_by_reason"] == {
        "candidate_independent": 1,
        "guessable_wrapper": 1,
    }
    assert summary["remainder"]["tasks"] == 2  # t3 is excluded by the exclusions record
    assert summary["remainder"]["refused_check_forms"] == {
        "guessable_wrapper: is True on method .empty()": 1
    }
    record = json.loads((out / "record.json").read_text(encoding="utf-8"))
    assert record["executed_code"] is False and len(record["inputs"]) == 6
    # Nothing is ever overwritten.
    assert reward_grammar_compat.main([*_files(tmp_path), "--out", str(out)]) == 3


def test_a_sealed_input_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    args = _files(tmp_path)
    sealed = tmp_path / "test.jsonl"
    sealed.write_text("not read", encoding="utf-8")
    args[args.index("--dev") + 1] = str(sealed)
    assert reward_grammar_compat.main([*args, "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
