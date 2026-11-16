"""The sandbox-call ledger of agent-data preparation: one persisted counter and a hard cap.

Every execution of every data-preparation run goes through :meth:`Ledger.execute`, which
appends one JSON line to the ledger file **before** the call starts (a crash mid-call still
counts the call) and refuses to start a call that would take the total over the cap. Host
retries are executions too, so each attempt is counted. Calls made before the ledger existed
(the evaluation-case validation runs) are recorded once as ``prior`` entries with their
source files and hashes (:func:`prior_from_records`).

The total is the sum of ``calls`` over every entry, so the cap applies across all runs that
share the file. The cap is :data:`DEFAULT_CAP`, the approved hard cap: a lower cap may be
given for one run, a higher one is refused. A run that executes holds an exclusive lock on
the ledger (:meth:`Ledger.locked`) for its whole duration and reads the entries only after
taking it, so two concurrent runs cannot each spend the remainder from their own totals; it
also refuses a ledger file that does not exist or has no prior entries
(:meth:`Ledger.require_seeded`), so a mistyped or fresh path cannot restart the count at
zero.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
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


class LedgerError(ValueError):
    """The ledger cannot be used as given (unseeded, over the cap, or held by another run)."""


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
        if cap > DEFAULT_CAP:
            raise LedgerError(f"the call cap may not exceed the approved {DEFAULT_CAP}")
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

    @classmethod
    @contextlib.contextmanager
    def locked(cls, path: Path, cap: int = DEFAULT_CAP) -> Iterator[Ledger]:
        """The ledger, opened after taking an exclusive lock that is held until the block
        ends; refused at once when another run holds it."""
        if cap > DEFAULT_CAP:
            raise LedgerError(f"the call cap may not exceed the approved {DEFAULT_CAP}")
        lock_path = path.with_name(path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise LedgerError(f"{path} is in use by another run") from None
            try:
                yield cls.open(path, cap)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def require_seeded(self) -> None:
        """Refuse a ledger file that does not exist or holds no prior entries: an executing
        run must add to the shared count, never start a new one."""
        if not self.path.is_file():
            raise LedgerError(f"{self.path} does not exist; seed it with the ledger command")
        if not any(entry.get("kind") == "prior" for entry in self.entries):
            raise LedgerError(f"{self.path} has no prior entries; seed it first")

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

    def _priors(self) -> list[dict[str, Any]]:
        return [entry for entry in self.entries if entry.get("kind") == "prior"]

    def sources(self) -> set[str]:
        return {str(entry.get("source")) for entry in self._priors()}

    def known(self, path: str, sha256: str) -> str:
        """Why a run record is already counted ("" if it is not): the same resolved path,
        or the same bytes under another path (a copy of a counted record)."""
        for entry in self._priors():
            if entry.get("path") is not None and str(entry["path"]) == path:
                return "path"
            if entry.get("sha256") is not None and entry["sha256"] == sha256:
                return f"copy of {entry['path']}"
        return ""

    def add_prior(self, source: str, calls: int, detail: Mapping[str, Any]) -> bool:
        """Record calls made before the ledger existed. A record is counted once: refused
        (False) when ``detail`` names a path or SHA-256 already counted, or, without them,
        when ``source`` is already present."""
        path, sha = detail.get("path"), detail.get("sha256")
        if path is not None and sha is not None:
            if self.known(str(path), str(sha)):
                return False
        elif source in self.sources():
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


def prior_from_records(
    ledger: Ledger, directories: Iterable[Path], *, skipped: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Add one prior entry per run record found in ``directories`` (idempotent). A record is
    keyed by its resolved path and its SHA-256: one already counted under either is skipped
    and, when ``skipped`` is given, listed there with the reason."""
    added = []
    for directory in directories:
        for name in RECORD_NAMES:
            path = directory / name
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            sha = _file_sha(path)
            reason = ledger.known(resolved, sha) or ledger.known(str(path), sha)
            source = f"{directory.name}/{name}"
            if reason:
                if skipped is not None:
                    skipped.append({"path": resolved, "reason": reason})
                continue
            calls = count_record(path)
            if ledger.add_prior(source, calls, {"sha256": sha, "path": resolved}):
                added.append({"source": source, "path": resolved, "calls": calls})
    return added
