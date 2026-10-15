import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from scratchcoder.common.logging import EventLogger, WandbTracker, debug_logger


def test_ten_steps_written_locally_and_to_tracker(tmp_path: Path) -> None:
    tracker = Mock()
    path = tmp_path / "events.jsonl"
    with EventLogger(path, "test", tracker) as logger:
        for step in range(10):
            logger.metrics({"loss": 1.0 / (step + 1)}, step=step)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["fields"]["step"] for row in rows] == list(range(10))
    assert tracker.log.call_count == 10
    tracker.finish.assert_called_once()
    assert path.stat().st_mode & 0o077 == 0


def test_tracker_failure_preserves_local_metrics(tmp_path: Path) -> None:
    tracker = Mock()
    tracker.log.side_effect = RuntimeError("do not copy sensitive service errors")
    path = tmp_path / "events.jsonl"
    with EventLogger(path, "test", tracker) as logger:
        logger.metrics({"loss": 2.0}, step=1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["metrics", "tracker_error"]
    assert "sensitive service" not in path.read_text()


def test_offline_default_and_nested_redaction(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with EventLogger(path, "test") as logger:
        logger.event("debug", nested={"api_key": "never-output", "message": "Bearer secret-token"})
        with pytest.raises(ValueError):
            logger.metrics({"loss": float("nan")}, step=0)
    contents = path.read_text()
    assert "never-output" not in contents
    assert "secret-token" not in contents
    assert "REDACTED" in contents


def test_debug_traceback_and_no_root_changes(tmp_path: Path) -> None:
    root_handlers = list(logging.getLogger().handlers)
    path = tmp_path / "debug.jsonl"
    logger = debug_logger(path)
    try:
        raise ValueError("Bearer private-token")
    except ValueError:
        logger.exception("operation failed")
    for handler in logger.handlers:
        handler.close()
    row = json.loads(path.read_text())
    assert row["level"] == "ERROR"
    assert "ValueError" in row["traceback"]
    assert "private-token" not in path.read_text()
    assert logging.getLogger().handlers == root_handlers


def test_hosted_adapter_can_be_tested_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    module = Mock()
    monkeypatch.setattr("scratchcoder.common.logging.importlib.import_module", lambda _: module)
    tracker = WandbTracker(project="test", run_id="id")
    tracker.log({"loss": 1.0}, step=0)
    tracker.finish()
    module.init.assert_called_once_with(project="test", id="id", mode="online")
    module.init.return_value.log.assert_called_once_with({"loss": 1.0}, step=0)


def test_tracker_finish_failure_is_local_and_permissions_are_tightened(tmp_path: Path) -> None:
    tracker = Mock()
    tracker.finish.side_effect = RuntimeError("sensitive-provider-message")
    path = tmp_path / "existing.jsonl"
    path.touch(mode=0o644)
    with EventLogger(path, "fixture", tracker) as logger:
        logger.metrics({"loss": 1.0}, step=0)
    text = path.read_text()
    assert "sensitive-provider-message" not in text
    assert json.loads(text.splitlines()[-1])["fields"]["phase"] == "finish"
    assert path.stat().st_mode & 0o077 == 0
