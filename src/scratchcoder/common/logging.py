"""Always-local structured events and opt-in experiment tracking."""

import importlib
import json
import logging
import math
import os
import re
import traceback
from pathlib import Path
from typing import Protocol

from scratchcoder.common.records import utc_now

_SENSITIVE = re.compile(r"password|secret|api.?key|access.?token|authorization|credential", re.I)
_TOKENS = re.compile(r"(?:hf_|gh[pousr]_|sk-|rpa_)[A-Za-z0-9_-]{8,}|Bearer\s+\S+", re.I)


def sanitize(value: object) -> object:
    """Redact known credential forms; callers must still avoid logging secrets."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SENSITIVE.search(str(key)) else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return _TOKENS.sub("[REDACTED]", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("log fields must be finite JSON values")


class Tracker(Protocol):
    def log(self, metrics: dict[str, float], *, step: int) -> None: ...
    def finish(self) -> None: ...


class WandbTracker:
    """Explicit opt-in only: constructing this adapter can contact W&B."""

    def __init__(self, *, project: str, run_id: str) -> None:
        wandb = importlib.import_module("wandb")
        self._run = wandb.init(project=project, id=run_id, mode="online")

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._run.finish()


class EventLogger:
    """Append JSONL before forwarding metrics; never depend on the network."""

    def __init__(self, path: Path, run_id: str, tracker: Tracker | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        self._stream = os.fdopen(descriptor, "a", encoding="utf-8")
        self.run_id = run_id
        self.tracker = tracker

    def event(self, name: str, **fields: object) -> None:
        record = sanitize(
            {"time": utc_now(), "run_id": self.run_id, "event": name, "fields": fields}
        )
        self._stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        self._stream.flush()

    def metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if step < 0 or any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("metrics require a nonnegative step and finite values")
        if any(_SENSITIVE.search(key) for key in metrics):
            raise ValueError("credential-like metric names are forbidden")
        self.event("metrics", step=step, metrics=metrics)
        if self.tracker is not None:
            try:
                self.tracker.log(metrics, step=step)
            except Exception as error:
                self.event("tracker_error", error_type=type(error).__name__)

    def close(self) -> None:
        if self._stream.closed:
            return
        try:
            if self.tracker is not None:
                try:
                    self.tracker.finish()
                except Exception as error:
                    self.event("tracker_error", phase="finish", error_type=type(error).__name__)
        finally:
            self._stream.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DebugFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "traceback": "".join(traceback.format_exception(*record.exc_info))
            if record.exc_info
            else None,
        }
        return json.dumps(sanitize(payload), allow_nan=False, sort_keys=True)


def debug_logger(path: Path, *, name: str = "scratchcoder") -> logging.Logger:
    """Create an isolated JSONL debug logger without modifying root handlers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    logger = logging.Logger(name, level=logging.DEBUG)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(DebugFormatter())
    logger.addHandler(handler)
    return logger
