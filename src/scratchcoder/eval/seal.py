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

A second read cannot be had by regenerating. One unseal record governs a ledger: once any
read has opened, no file can be sealed and every later read must present the same record.
The record may not list two files of one checkpoint, protocol and decoding kind (greedy or
sampled), because the second would be a second look at the same endpoint.

A hash chain alone cannot show that its *last* lines were removed, so a *head record*,
``<ledger>.head``, beside the ledger names the event count and the hash of the last event
after every append. A ledger shorter than its head record, a head record whose hash is not
in the chain, a missing head record beside a non-empty ledger, and a head record with no
ledger are all refused. The head record is written after the ledger line is synced, so a
crash between the two leaves a ledger longer than its head record, which is accepted and
repaired by the next append. The two files travel together; this guards against accidents
and casual edits, not against someone who rewrites both.
"""

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Any

from scratchcoder.common.records import utc_now
from scratchcoder.eval.completions import SHA256, Completions, read_completions

ENDPOINTS = ("base", "sft", "rl", "more-sft")
UNSEAL_KIND = "unseal"
LEDGER_FORMAT = "scratchcoder-seal-ledger"
GENESIS = "0" * 64
EVENTS = ("sealed", "opened", "resumed", "completed", "baseline")
HEAD_FORMAT = "scratchcoder-seal-ledger-head"
HEAD_SUFFIX = ".head"


def decoding_kind(sealed: dict[str, Any]) -> str:
    """``greedy`` or ``sampled``, from a ``sealed`` event: the kind of decoding a file holds."""
    return "greedy" if sealed["temperature"] == 0 else "sampled"


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


def _parse(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """The ledger's events and the SHA-256 of each line, or :class:`SealError` if the chain
    is broken."""
    events: list[dict[str, Any]] = []
    hashes: list[str] = []
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
        hashes.append(previous)
        events.append(event)
    return events, hashes


class SealLedger:
    """The append-only, hash-chained seal ledger at ``path`` (created on first use)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def head_path(self) -> Path:
        return self.path.with_name(self.path.name + HEAD_SUFFIX)

    def _verify_head(self, hashes: list[str]) -> None:
        """Refuse a ledger that lost its last events (or all of them, or its head record)."""
        if not self.head_path.exists():
            if hashes:
                raise SealError(
                    f"the seal ledger has no head record ({self.head_path.name}); "
                    "keep the two files together"
                )
            return
        try:
            head = json.loads(self.head_path.read_text(encoding="utf-8"))
            count, digest = head["events"], head["head_sha256"]
            valid = (
                head["format"] == HEAD_FORMAT
                and type(count) is int
                and count >= 1
                and isinstance(digest, str)
                and bool(SHA256.fullmatch(digest))
            )
        except (ValueError, KeyError, TypeError) as error:
            raise SealError("the seal ledger's head record is unreadable") from error
        if not valid:
            raise SealError("the seal ledger's head record is malformed")
        if len(hashes) < count:
            raise SealError(
                f"the seal ledger has {len(hashes)} events but its head record names {count}: "
                "events were removed"
            )
        if hashes[count - 1] != digest:
            raise SealError("the seal ledger does not match its head record")

    def _write_head(self, count: int, digest: str) -> None:
        record = {"format": HEAD_FORMAT, "events": count, "head_sha256": digest}
        with NamedTemporaryFile(
            mode="w", dir=self.head_path.parent, encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, self.head_path)

    @contextmanager
    def _locked(self) -> Iterator[tuple[list[dict[str, Any]], IO[bytes]]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                events, hashes = _parse(handle.read())
                self._verify_head(hashes)
                yield events, handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            if self.head_path.exists():
                raise SealError("the seal ledger is missing but its head record exists")
            return []
        with self._locked() as (events, _):
            return events

    def head(self) -> dict[str, Any]:
        """The ledger's current event count and last-event hash (what the head record holds)."""
        if not self.path.exists():
            return {"events": 0, "head_sha256": GENESIS}
        with self._locked() as (events, handle):
            handle.seek(0)
            lines = handle.read().splitlines(keepends=True)
            return {
                "events": len(events),
                "head_sha256": hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS,
            }

    def _append(
        self, events: list[dict[str, Any]], handle: IO[bytes], event: str, **fields: Any
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
        line = json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n"
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        events.append(record)
        self._write_head(len(events), hashlib.sha256(line).hexdigest())
        return record

    def seal(self, completions: Completions) -> dict[str, Any]:
        """Record a pool-test completions file; sealing the same file again is a no-op.

        Once any read has opened, no new file can be sealed: the unseal record that opened it
        fixed the set of files, and a regenerated file would be a second look.
        """
        header = completions.header
        if header.suite != "pool" or header.split != "test":
            raise SealError("only pool test-split completions are sealed")
        with self._locked() as (events, handle):
            for event in events:
                if event["event"] == "sealed" and event["completions_sha256"] == completions.sha256:
                    return event
            if any(event["event"] == "opened" for event in events):
                raise SealError(
                    "a read of the test split has already opened; no new completions can be sealed"
                )
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

    @staticmethod
    def _sealed(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            event["completions_sha256"]: event for event in events if event["event"] == "sealed"
        }

    @classmethod
    def _check_record(
        cls, events: list[dict[str, Any]], unseal: UnsealRecord
    ) -> dict[str, dict[str, Any]]:
        """Checks that depend on the unseal record and the sealed files, not on one read."""
        sealed = cls._sealed(events)
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
        if any(
            event["unseal_sha256"] != unseal.sha256
            for event in events
            if event["event"] == "opened"
        ):
            raise SealError(
                "a read of this ledger opened under another unseal record; "
                "every read must use the one record"
            )
        for endpoint in unseal.endpoints:
            kinds: dict[tuple[str, str], int] = {}
            for listed in endpoint.completions_sha256:
                key = (sealed[listed]["protocol_sha256"], decoding_kind(sealed[listed]))
                kinds[key] = kinds.get(key, 0) + 1
            if any(count > 1 for count in kinds.values()):
                raise SealError(
                    f"endpoint {endpoint.name}: the unseal record lists two completions files "
                    "of one protocol and decoding kind; only one may be read"
                )
        return sealed

    @classmethod
    def _check_read(
        cls, events: list[dict[str, Any]], completions: Completions, unseal: UnsealRecord
    ) -> tuple[Endpoint, list[dict[str, Any]]]:
        digest = completions.sha256
        if digest not in cls._sealed(events):
            raise SealError("these test completions were never sealed in this ledger")
        cls._check_record(events, unseal)
        endpoint = unseal.endpoint_for(digest)
        if endpoint is None:
            raise SealError("the unseal record does not list these completions")
        if endpoint.checkpoint_sha256 != completions.header.checkpoint_sha256:
            raise SealError("the unseal record lists these completions under another checkpoint")
        reads = [
            event
            for event in events
            if event["event"] in ("opened", "completed") and event["completions_sha256"] == digest
        ]
        return endpoint, reads

    def check_read(self, completions: Completions, unseal: UnsealRecord) -> None:
        """Raise :class:`SealError` if :meth:`open_read` would refuse, logging nothing.

        The executor calls this before it opens the task file, so a refused read never
        touches the sealed split. A resumption still passes (only a second, different read
        or a completed one is refused by :meth:`open_read`).
        """
        with self._locked() as (events, _):
            _, reads = self._check_read(events, completions, unseal)
            if any(event["event"] == "completed" for event in reads):
                raise SealError(
                    "these test completions were already executed; a second read is refused"
                )

    def open_read(
        self, completions: Completions, unseal: UnsealRecord, *, results_id: str, results_path: Path
    ) -> dict[str, Any]:
        """Log the one read of a sealed test completions file, or its resumption.

        Raises :class:`SealError` (and logs nothing) if the read is not allowed.
        """
        digest = completions.sha256
        with self._locked() as (events, handle):
            endpoint, reads = self._check_read(events, completions, unseal)
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

    def finished_unseal(self, unseal: UnsealRecord) -> dict[str, Any]:
        """The sealed event's task file and the last results ID, once every file the unseal
        record lists has a completed read. Raises :class:`SealError` before that.

        The test split's baseline retry is allowed only through this gate."""
        with self._locked() as (events, _):
            sealed = self._check_record(events, unseal)
            done = {
                event["completions_sha256"] for event in events if event["event"] == "completed"
            }
            listed = [
                digest for endpoint in unseal.endpoints for digest in endpoint.completions_sha256
            ]
            pending = [digest for digest in listed if digest not in done]
            if pending:
                raise SealError(
                    f"{len(pending)} of {len(listed)} listed test completions have no completed "
                    "read; the baseline is computed after every read finished"
                )
            task_files = {sealed[digest]["task_file_sha256"] for digest in listed}
            if len(task_files) != 1:
                raise SealError("the listed test completions name different task files")
            last = [event for event in events if event["event"] == "completed"][-1]
            return {"task_file_sha256": task_files.pop(), "results_id": last["results_id"]}


def seal_completions(completions_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Validate a pool-test completions file and record its SHA-256 in the ledger."""
    return SealLedger(ledger_path).seal(read_completions(completions_path))
