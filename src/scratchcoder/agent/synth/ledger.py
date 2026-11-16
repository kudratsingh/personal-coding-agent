"""The sandbox-call ledger of agent-data preparation: one persisted counter and a hard cap.

Every execution of every data-preparation run goes through :meth:`Ledger.execute`, which
appends one JSON line to the ledger file **before** the call starts (a crash mid-call still
counts the call) and refuses to start a call that would take the total over the cap. Host
retries are executions too, so each attempt is counted. Calls made before the ledger existed
(the evaluation-case validation runs) are recorded once as ``prior`` entries with their
source files and hashes (:func:`prior_from_records`).

The total is the sum of ``calls`` over every entry, so the cap applies across all runs that
share the file.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scratchcoder.sandbox.runner import Result

LEDGER_SCHEMA = "agent-sft-calls-ledger-v1"
DEFAULT_CAP = 10_000
Execute = Callable[[str, str, str], Result]


class CallCapReached(RuntimeError):
    """Starting another sandbox call would exceed the ledger's cap."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Ledger:
    path: Path
    cap: int = DEFAULT_CAP
    entries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path, cap: int = DEFAULT_CAP) -> Ledger:
        if cap < 0:
            raise ValueError("the call cap must be non-negative")
        entries: list[dict[str, Any]] = []
        if path.exists():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("schema") != LEDGER_SCHEMA or type(entry.get("calls")) is not int:
                    raise ValueError(f"{path}: line {number} is not a ledger entry")
                if entry["calls"] < 0:
                    raise ValueError(f"{path}: line {number} has a negative call count")
                entries.append(entry)
        return cls(path, cap, entries)

    @property
    def total(self) -> int:
        return sum(int(entry["calls"]) for entry in self.entries)

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.total)

    def _append(self, entry: Mapping[str, Any]) -> None:
        row = {"schema": LEDGER_SCHEMA, "seq": len(self.entries) + 1, "at": _now(), **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.entries.append(row)

    def sources(self) -> set[str]:
        return {str(entry.get("source")) for entry in self.entries if entry.get("kind") == "prior"}

    def add_prior(self, source: str, calls: int, detail: Mapping[str, Any]) -> bool:
        """Record calls made before the ledger existed; once per source (False if present)."""
        if source in self.sources():
            return False
        if calls < 0:
            raise ValueError("prior calls must be non-negative")
        self._append({"kind": "prior", "source": source, "calls": calls, **detail})
        return True

    def require(self, calls: int) -> None:
        """Refuse unless ``calls`` more calls fit under the cap."""
        if self.total + calls > self.cap:
            raise CallCapReached(
                f"{calls} more sandbox calls would exceed the cap of {self.cap} ({self.total} used)"
            )

    def execute(self, inner: Execute, *, run: str, context: Callable[[], str]) -> Execute:
        """``inner`` wrapped so that every call is counted first and refused over the cap.
        ``context`` names what the call is for (the instance and step), read at call time."""

        def counted(program: str, tests: str, mode: str) -> Result:
            self.require(1)
            self._append(
                {
                    "kind": "call",
                    "calls": 1,
                    "run": run,
                    "context": context(),
                    "mode": mode,
                    "program_sha256": _sha(program),
                    "tests_sha256": _sha(tests),
                }
            )
            return inner(program, tests, mode)

        return counted

    def summary(self) -> dict[str, Any]:
        prior = sum(int(e["calls"]) for e in self.entries if e.get("kind") == "prior")
        by_run: dict[str, int] = {}
        for entry in self.entries:
            if entry.get("kind") == "call":
                key = str(entry.get("run"))
                by_run[key] = by_run.get(key, 0) + int(entry["calls"])
        return {
            "path": str(self.path),
            "cap": self.cap,
            "total": self.total,
            "prior": prior,
            "by_run": by_run,
            "remaining": self.remaining,
        }


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_record(path: Path) -> int:
    """Sandbox executions recorded in one run record, counted from its own fields:

    - ``validation.json`` (``scratchcoder.agent.validation``): each call plus its host
      retries;
    - ``scorecard.json``: each instance's ``sandbox_calls`` plus ``host_failures``;
    - ``controls.json``: its ``sandbox_calls`` plus each row's ``host_failures``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "validation.json":
        return sum(
            1 + int(call.get("host_failures", 0)) for case in raw["cases"] for call in case["calls"]
        )
    if path.name == "scorecard.json":
        return sum(
            int(row["sandbox_calls"]) + int(row.get("host_failures", 0)) for row in raw["instances"]
        )
    if path.name == "controls.json":
        return int(raw["sandbox_calls"]) + sum(
            int(row.get("host_failures", 0)) for row in raw["rows"]
        )
    raise ValueError(f"not a known run record: {path.name}")


RECORD_NAMES = ("validation.json", "scorecard.json", "controls.json")


def prior_from_records(ledger: Ledger, directories: Iterable[Path]) -> list[dict[str, Any]]:
    """Add one prior entry per run record found in ``directories`` (idempotent)."""
    added = []
    for directory in directories:
        for name in RECORD_NAMES:
            path = directory / name
            if not path.is_file():
                continue
            calls = count_record(path)
            source = f"{directory.name}/{name}"
            if ledger.add_prior(source, calls, {"sha256": _file_sha(path), "path": str(path)}):
                added.append({"source": source, "calls": calls})
    return added
