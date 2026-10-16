import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scratchcoder.common.config import ModelConfig, RunConfig, TrainConfig
from scratchcoder.common.records import RunRecord, atomic_json


def test_configuration_roundtrip(tmp_path: Path) -> None:
    config = RunConfig.load(Path("configs/nano.yaml"))
    path = tmp_path / "config.yaml"
    config.save(path)
    assert RunConfig.load(path) == config


def test_record_roundtrip_and_schema(tmp_path: Path) -> None:
    record = RunRecord.start("test-run", RunConfig(), "data-hash", "tokenizer-hash", "parent")
    path = record.save(tmp_path)
    record.finish(success=True)
    record.save(tmp_path)
    raw = json.loads(path.read_text())
    schema = json.loads(Path("schemas/run-record.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(raw)
    assert raw["status"] == "completed"
    assert raw["parent_run_id"] == "parent"
    assert raw["ended_at"] is not None
    assert raw["seed"] == raw["config"]["train"]["seed"]


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", "a/b"])
def test_run_id_cannot_escape_root(run_id: str) -> None:
    with pytest.raises(ValueError):
        RunRecord.start(run_id, RunConfig(), "", "")


def test_atomic_json_preserves_previous_record_on_error(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    atomic_json(path, {"valid": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"valid": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ModelConfig(width=17, heads=4)
    with pytest.raises(ValueError):
        TrainConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError):
        RunConfig(data_mix={"web": 0.4})
    path = tmp_path / "bad.yaml"
    path.write_text("unexpected: 1\n")
    with pytest.raises(ValueError):
        RunConfig.load(path)


@pytest.mark.parametrize(
    "text",
    [
        "train:\n  seed: 1\n  seed: 2\n",
        "data_mix: {web: true}\n",
        "train: {learning_rate: true}\n",
        "train: {beta1: 1.0}\n",
        "parent_checkpoint: ''\n",
    ],
)
def test_ambiguous_or_mistyped_configuration_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        RunConfig.load(path)


def test_mutated_record_and_symlink_escape_are_rejected(tmp_path: Path) -> None:
    record = RunRecord.start("safe", RunConfig(), "fixture", "fixture")
    record.run_id = "../escape"
    with pytest.raises(ValueError):
        record.save(tmp_path / "root")
    record.run_id = "safe"
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "safe").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        record.save(root)
    assert not list(outside.iterdir())


def test_v1_records_keep_original_shape_and_new_records_are_v2(tmp_path: Path) -> None:
    record = RunRecord.start("compatibility", RunConfig(), "fixture", "fixture")
    path = record.save(tmp_path)
    current = json.loads(Path("schemas/run-record.schema.json").read_text())
    legacy = json.loads(Path("tests/fixtures/run-record-v1.schema.json").read_text())
    new = json.loads(path.read_text())
    assert new["schema_version"] == 2 and new["resume_checkpoint"] is None
    jsonschema.validate(new, current)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(new, legacy)
    record.schema_version = 1
    record.save(tmp_path)
    old = json.loads(path.read_text())
    assert "resume_checkpoint" not in old
    jsonschema.validate(old, legacy)
    jsonschema.validate(old, current)
    RunRecord(**old).save(tmp_path)
    assert json.loads(path.read_text()) == old


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "bad"},
        {"sha256": None},
        {"source_run_id": "../escape"},
        {"source_run_id": None},
        {"step": True},
        {"step": -1},
        {"step": 1.5},
        {"extra": "not-allowed"},
    ],
)
def test_resume_metadata_is_strict_before_record_io(tmp_path: Path, change: dict[str, Any]) -> None:
    record = RunRecord.start("resume-meta", RunConfig(), "fixture", "fixture")
    record.resume_checkpoint = {"sha256": "a" * 64, "source_run_id": "source", "step": 3}
    record.resume_checkpoint.update(change)
    with pytest.raises(ValueError):
        record.save(tmp_path)
    assert not list(tmp_path.iterdir())


def test_v2_resume_metadata_schema_and_v1_field_rejection(tmp_path: Path) -> None:
    record = RunRecord.start("resumed", RunConfig(), "fixture", "fixture", "stage-parent")
    record.resume_checkpoint = {"sha256": "a" * 64, "source_run_id": "same-stage", "step": 3}
    path = record.save(tmp_path)
    value = json.loads(path.read_text())
    schema = json.loads(Path("schemas/run-record.schema.json").read_text())
    jsonschema.validate(value, schema)
    assert value["parent_run_id"] == "stage-parent"
    record.schema_version = 1
    with pytest.raises(ValueError, match="schema 2"):
        record.save(tmp_path)
    value["schema_version"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
    value["schema_version"] = 2
    del value["resume_checkpoint"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
