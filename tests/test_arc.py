import hashlib
import importlib
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data import arc
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE
from tools import arc_check


def test_arc_manifest_is_pinned_evaluation_only() -> None:
    splits = arc.arc_splits()
    assert {name: spec.task_count for name, spec in splits.items()} == {
        "train": 2251,
        "validation": 570,
        "test": 2376,
    }
    for spec in splits.values():
        spec.validate()
        assert "/ARC-Easy/" in spec.manifest.url
        assert arc.REVISION in spec.manifest.url
        assert spec.manifest.purpose == "eval"
        assert spec.manifest.license == "CC-BY-SA-4.0"
        with pytest.raises(ValueError):
            replace(spec, manifest=replace(spec.manifest, purpose="train")).validate()
        with pytest.raises(ValueError):
            replace(spec, split="ARC-Challenge").validate()


requires_arrow = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None, reason="requires the locked evaluation extra"
)


def _row(identifier: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "question": "Authored fixture: which number is even?",
        "choices": {"label": ["2", "1"], "text": ["4", "3"]},
        "answerKey": "2",
    }


def _write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, arc.ArcSplit]:
    arrow = importlib.import_module("pyarrow")
    parquet = importlib.import_module("pyarrow.parquet")
    specs = {}
    for name, original in arc.arc_splits().items():
        rows = (changes or {}).get(name, [_row(f"fixture/{name}")])
        path = tmp_path / original.filename
        parquet.write_table(arrow.Table.from_pylist(rows), path)
        specs[name] = replace(
            original,
            task_count=len(rows),
            manifest=replace(
                original.manifest,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )
    monkeypatch.setattr(arc, "arc_splits", lambda: specs)
    return specs


@requires_arrow
def test_arc_complete_split_mapping_order_and_answer_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, monkeypatch)
    tasks, metadata = arc.read_arc_easy(tmp_path)
    assert len(tasks) == 1 and tasks[0].id == "fixture/test" and tasks[0].split == "test"
    assert tasks[0].labels == ("2", "1") and tasks[0].choices == ("4", "3")
    assert tasks[0].public_prompt() == replace(tasks[0], answer="1").public_prompt()
    assert metadata["source_split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert metadata["cross_split_id_overlap"] == 0
    assert metadata["official_scoring_protocol_equivalent"] is False


@requires_arrow
@pytest.mark.parametrize("failure", ["size", "hash", "count", "overlap", "duplicate"])
def test_arc_rejects_incomplete_corrupt_or_overlapping_splits(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changes = {}
    if failure == "overlap":
        changes["train"] = [_row("fixture/test")]
    if failure == "duplicate":
        changes["test"] = [_row("duplicate"), _row("duplicate")]
    specs = _write(tmp_path, monkeypatch, changes)
    spec = specs["test"]
    if failure in {"size", "hash"}:
        path = tmp_path / spec.filename
        raw = path.read_bytes()
        path.write_bytes(raw + b"x" if failure == "size" else bytes([raw[0] ^ 1]) + raw[1:])
    if failure == "count":
        specs["test"] = replace(spec, task_count=2)
    with pytest.raises(ValueError):
        arc.read_arc_easy(tmp_path)


@requires_arrow
@pytest.mark.parametrize("failure", ["answer", "labels", "null", "columns", "choices"])
def test_arc_rejects_malformed_fields(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row("fixture/test")
    if failure == "answer":
        row["answerKey"] = "Z"
    elif failure == "labels":
        row["choices"]["label"] = ["2", "2"]
    elif failure == "null":
        row["question"] = None
    elif failure == "columns":
        row["unreviewed"] = "extra"
    else:
        row["choices"]["unreviewed"] = ["extra"]
    _write(tmp_path, monkeypatch, {"test": [row]})
    with pytest.raises(ValueError):
        arc.read_arc_easy(tmp_path)


@requires_arrow
def test_arc_command_uses_real_own_checkpoint_and_safe_chunked_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, monkeypatch)
    tokenizer = ByteBPE.train(["authored fixture only"], vocab_size=272)
    tokenizer.save(tmp_path / "tokenizer.json")
    config = ModelConfig(vocab_size=272, context=256, width=16, layers=1, heads=2)
    model = GPT(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(config)},
            "tokenizer_hash": tokenizer.sha256,
        },
        checkpoint,
    )
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arc_check",
            "--data-directory",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--output",
            str(output),
        ],
    )
    assert arc_check.main() == 0
    lines = capsys.readouterr().out.splitlines()
    header = json.loads(
        next(line.removeprefix("ARC_REPORT=") for line in lines if line.startswith("ARC_REPORT="))
    )
    rows = [
        json.loads(line.removeprefix("ARC_TASK=")) for line in lines if line.startswith("ARC_TASK=")
    ]
    report = json.loads((output / "report.json").read_text())
    assert dict(header, per_task=rows) == report
    assert report["task_count"] == 1
    assert report["benchmark"]["source_split"] == "test"
    assert report["configuration"]["scoring"] == "sum_log_probability_of_space_prefixed_label"
    assert "Authored fixture:" not in json.dumps(report)
    assert "answerKey" not in json.dumps(report)
    with pytest.raises(SystemExit):
        arc_check.main()
