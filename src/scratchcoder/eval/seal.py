"""Sealing the pool test split: stored completions, one logged read per checkpoint.

The test split is scored only after every endpoint is frozen. Until then its completions
are stored and their SHA-256 recorded in a *seal ledger*, an append-only JSON Lines file in
which every event names the SHA-256 of the line before it (a hash chain), so an edited or
removed line is detected on the next read. Events:

- ``sealed``: a pool-test completions file was stored (its SHA-256, checkpoint, protocol).
- ``opened``: the one logged read of a sealed file began (its results ID and the unseal
  record's SHA-256). It is written before anything executes.
- ``resumed``: the same read continued after an interruption (same results ID).
- ``completed``: every attempt of the read was resolved (the results file's SHA-256).
- ``baseline``: the test split's constant-answer baseline was computed inside a read.

Execution of a test completions file needs an *unseal record*, a JSON file the lead writes
once all four endpoints (``base``, ``sft``, ``rl``, ``more-sft``) are frozen::

    {"kind": "unseal", "schema_version": 1, "frozen_at": "...",
     "endpoints": {"base": {"checkpoint_sha256": "...", "completions_sha256": ["..."]},
                   "sft": {...}, "rl": {...}, "more-sft": {...}}}

Execution is refused unless the file is sealed, every completions file the record lists
is sealed, the file is listed under the endpoint whose checkpoint generated it, and no
other read of it was ever opened. A second execution of the same completions is refused.
"""

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from scratchcoder.common.records import utc_now
from scratchcoder.eval.completions import SHA256, Completions, read_completions

ENDPOINTS = ("base", "sft", "rl", "more-sft")
UNSEAL_KIND = "unseal"
LEDGER_FORMAT = "scratchcoder-seal-ledger"
GENESIS = "0" * 64
EVENTS = ("sealed", "opened", "resumed", "completed", "baseline")


class SealError(PermissionError):
    """A sealed test split may not be read this way."""


@dataclass(frozen=True)
class Endpoint:
    name: str
    checkpoint_sha256: str
    completions_sha256: tuple[str, ...]


@dataclass(frozen=True)
class UnsealRecord:
    endpoints: tuple[Endpoint, ...]
    frozen_at: str
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "UnsealRecord":
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except ValueError as error:
            raise SealError("the unseal record is not JSON") from error
        if (
            not isinstance(value, dict)
            or value.get("kind") != UNSEAL_KIND
            or value.get("schema_version") != 1
            or set(value) != {"kind", "schema_version", "frozen_at", "endpoints"}
        ):
            raise SealError("not an unseal record (kind, schema_version, frozen_at, endpoints)")
        if not isinstance(value["frozen_at"], str) or not value["frozen_at"]:
            raise SealError("the unseal record needs frozen_at")
        endpoints = value["endpoints"]
        if not isinstance(endpoints, dict) or set(endpoints) != set(ENDPOINTS):
            raise SealError(f"the unseal record must name exactly the endpoints {ENDPOINTS}")
        parsed = []
        for name in ENDPOINTS:
            item = endpoints[name]
            if not isinstance(item, dict) or set(item) != {
                "checkpoint_sha256",
                "completions_sha256",
            }:
                raise SealError(f"endpoint {name} needs checkpoint_sha256 and completions_sha256")
            checkpoint, listed = item["checkpoint_sha256"], item["completions_sha256"]
            if not isinstance(checkpoint, str) or not SHA256.fullmatch(checkpoint):
                raise SealError(f"endpoint {name}: invalid checkpoint SHA-256")
            if (
                not isinstance(listed, list)
                or not listed
                or any(
                    not isinstance(digest, str) or not SHA256.fullmatch(digest) for digest in listed
                )
            ):
                raise SealError(f"endpoint {name}: completions_sha256 must list SHA-256 values")
            parsed.append(Endpoint(name, checkpoint, tuple(listed)))
        if len({endpoint.checkpoint_sha256 for endpoint in parsed}) != len(parsed):
            raise SealError("each endpoint must be a different checkpoint")
        listed_all = [digest for endpoint in parsed for digest in endpoint.completions_sha256]
        if len(set(listed_all)) != len(listed_all):
            raise SealError("a completions file is listed twice")
        return cls(tuple(parsed), value["frozen_at"], hashlib.sha256(raw).hexdigest())

    def endpoint_for(self, completions_sha256: str) -> Endpoint | None:
        for endpoint in self.endpoints:
            if completions_sha256 in endpoint.completions_sha256:
                return endpoint
        return None


def _parse(data: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous = GENESIS
    if data and not data.endswith(b"\n"):
        raise SealError("the seal ledger ends with a torn line")
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        try:
            event = json.loads(line)
        except ValueError as error:
            raise SealError(f"seal ledger line {number} is not JSON") from error
        if (
            not isinstance(event, dict)
            or event.get("format") != LEDGER_FORMAT
            or event.get("event") not in EVENTS
        ):
            raise SealError(f"seal ledger line {number} is not a ledger event")
        if event.get("previous_sha256") != previous:
            raise SealError(f"seal ledger chain broken at line {number}")
        previous = hashlib.sha256(line).hexdigest()
        events.append(event)
    return events


class SealLedger:
    """The append-only, hash-chained seal ledger at ``path`` (created on first use)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _locked(self) -> Iterator[tuple[list[dict[str, Any]], IO[bytes]]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                yield _parse(handle.read()), handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._locked() as (events, _):
            return events

    @staticmethod
    def _append(
        events: list[dict[str, Any]], handle: IO[bytes], event: str, **fields: Any
    ) -> dict[str, Any]:
        handle.seek(0)
        data = handle.read()
        previous = (
            hashlib.sha256(data.splitlines(keepends=True)[-1]).hexdigest() if data else GENESIS
        )
        record = {
            "format": LEDGER_FORMAT,
            "event": event,
            "at": utc_now(),
            "previous_sha256": previous,
            **fields,
        }
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        events.append(record)
        return record

    def seal(self, completions: Completions) -> dict[str, Any]:
        """Record a pool-test completions file; sealing the same file again is a no-op."""
        header = completions.header
        if header.suite != "pool" or header.split != "test":
            raise SealError("only pool test-split completions are sealed")
        with self._locked() as (events, handle):
            for event in events:
                if event["event"] == "sealed" and event["completions_sha256"] == completions.sha256:
                    return event
            return self._append(
                events,
                handle,
                "sealed",
                completions_sha256=completions.sha256,
                checkpoint_sha256=header.checkpoint_sha256,
                task_file_sha256=header.task_file_sha256,
                protocol=header.protocol,
                protocol_sha256=header.protocol_sha256,
                samples=header.sampling.samples,
                temperature=header.sampling.temperature,
                rows=header.rows,
                path=str(completions.path) if completions.path else None,
            )

    def open_read(
        self, completions: Completions, unseal: UnsealRecord, *, results_id: str, results_path: Path
    ) -> dict[str, Any]:
        """Log the one read of a sealed test completions file, or its resumption.

        Raises :class:`SealError` (and logs nothing) if the read is not allowed.
        """
        digest = completions.sha256
        with self._locked() as (events, handle):
            sealed = {event["completions_sha256"] for event in events if event["event"] == "sealed"}
            if digest not in sealed:
                raise SealError("these test completions were never sealed in this ledger")
            missing = [
                listed
                for endpoint in unseal.endpoints
                for listed in endpoint.completions_sha256
                if listed not in sealed
            ]
            if missing:
                raise SealError(
                    f"{len(missing)} completions listed in the unseal record are not sealed"
                )
            endpoint = unseal.endpoint_for(digest)
            if endpoint is None:
                raise SealError("the unseal record does not list these completions")
            if endpoint.checkpoint_sha256 != completions.header.checkpoint_sha256:
                raise SealError(
                    "the unseal record lists these completions under another checkpoint"
                )
            reads = [
                event
                for event in events
                if event["event"] in ("opened", "completed")
                and event["completions_sha256"] == digest
            ]
            fields = {
                "completions_sha256": digest,
                "checkpoint_sha256": endpoint.checkpoint_sha256,
                "endpoint": endpoint.name,
                "unseal_sha256": unseal.sha256,
                "results_id": results_id,
                "results_path": str(results_path),
            }
            if not reads:
                return self._append(events, handle, "opened", **fields)
            opened = [event for event in reads if event["event"] == "opened"]
            if any(event["event"] == "completed" for event in reads):
                raise SealError(
                    "these test completions were already executed; a second read is refused"
                )
            if len(opened) != 1 or opened[0]["results_id"] != results_id:
                raise SealError("these test completions were already opened by another execution")
            return self._append(events, handle, "resumed", **fields)

    def complete_read(
        self, completions_sha256: str, *, results_id: str, results_sha256: str
    ) -> None:
        with self._locked() as (events, handle):
            self._append(
                events,
                handle,
                "completed",
                completions_sha256=completions_sha256,
                results_id=results_id,
                results_sha256=results_sha256,
            )

    def baseline_event(self, task_file_sha256: str) -> dict[str, Any] | None:
        for event in self.events():
            if event["event"] == "baseline" and event["task_file_sha256"] == task_file_sha256:
                return event
        return None

    def record_baseline(
        self, *, task_file_sha256: str, path: Path, sha256: str, results_id: str
    ) -> dict[str, Any]:
        with self._locked() as (events, handle):
            return self._append(
                events,
                handle,
                "baseline",
                task_file_sha256=task_file_sha256,
                path=str(path),
                sha256=sha256,
                results_id=results_id,
            )


def seal_completions(completions_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Validate a pool-test completions file and record its SHA-256 in the ledger."""
    return SealLedger(ledger_path).seal(read_completions(completions_path))
