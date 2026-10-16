"""Atomic, provenance-rich records that live outside source control."""

import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from scratchcoder.common.config import RunConfig

RUN_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"


def validate_resume_details(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"sha256", "source_run_id", "step"}
        or not isinstance(value["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        or not isinstance(value["source_run_id"], str)
        or not re.fullmatch(RUN_ID, value["source_run_id"])
        or type(value["step"]) is not int
        or value["step"] < 0
    ):
        raise ValueError("invalid resume checkpoint identity")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_state() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, True


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a JSON file atomically; never leave half-written provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


@dataclass
class RunRecord:
    run_id: str
    config: dict[str, Any]
    git_commit: str | None
    dirty: bool
    seed: int
    data_manifest_hash: str
    tokenizer_hash: str
    parent_run_id: str | None
    hardware: dict[str, str]
    started_at: str
    ended_at: str | None = None
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    artifact_uri: str | None = None
    status: str = "running"
    schema_version: int = 2
    resume_checkpoint: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.run_id, str) or not re.fullmatch(RUN_ID, self.run_id):
            raise ValueError("run_id must be a safe path component")
        if self.status not in {"running", "completed", "failed"} or (
            type(self.schema_version) is not int or self.schema_version not in {1, 2}
        ):
            raise ValueError("invalid record status or schema version")
        if self.resume_checkpoint is not None:
            if self.schema_version != 2:
                raise ValueError("resume checkpoint metadata requires run-record schema 2")
            validate_resume_details(self.resume_checkpoint)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not 0 <= self.estimated_cost_usd < float("inf")
            or self.actual_cost_usd is not None
            and not 0 <= self.actual_cost_usd < float("inf")
        ):
            raise ValueError("record costs must be finite and nonnegative")

    @classmethod
    def start(
        cls,
        run_id: str,
        config: RunConfig,
        data_manifest_hash: str,
        tokenizer_hash: str,
        parent_run_id: str | None = None,
    ) -> "RunRecord":
        if not isinstance(run_id, str) or not re.fullmatch(RUN_ID, run_id):
            raise ValueError("run_id must be a safe path component")
        commit, dirty = git_state()
        return cls(
            run_id=run_id,
            config=asdict(config),
            git_commit=commit,
            dirty=dirty,
            seed=config.train.seed,
            data_manifest_hash=data_manifest_hash,
            tokenizer_hash=tokenizer_hash,
            parent_run_id=parent_run_id,
            hardware={"system": platform.system(), "machine": platform.machine()},
            started_at=utc_now(),
        )

    def save(self, root: Path) -> Path:
        self.validate()
        root = root.resolve()
        path = root / self.run_id / "record.json"
        if not path.resolve().is_relative_to(root):
            raise ValueError("run record path escapes its root through a symlink")
        value = asdict(self)
        if self.schema_version == 1:
            value.pop("resume_checkpoint")  # Preserve the original strict v1 shape.
        atomic_json(path, value)
        return path

    def finish(self, *, success: bool, actual_cost_usd: float = 0.0) -> None:
        if not 0 <= actual_cost_usd < float("inf"):
            raise ValueError("cost must be finite and nonnegative")
        self.ended_at = utc_now()
        self.actual_cost_usd = actual_cost_usd
        self.status = "completed" if success else "failed"
