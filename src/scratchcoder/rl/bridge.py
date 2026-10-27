"""The RL reward bridge, Pod side: rewards and dev verdicts from the Mac, through files.

In run 2 the Pod's RL loop needs rewards and dev-check verdicts that only the Mac can
compute: the hidden tests and the Docker sandbox are there. The Mac reaches the Pod over
SSH; the Pod cannot reach the Mac. So the Pod writes each request as a file into an
*outbox* directory and waits for the matching reply file in an *inbox* directory; the Mac
side (``scratchcoder.rl.bridge_mac``) polls the outbox over SSH, scores the request and
pushes the reply. Both sides write a file by atomic rename (a hidden temporary file,
fsync, rename, fsync of the directory), so a reader never sees half a file under its final
name.

Layout under the bridge root::

    outbox/<request id>.json      requests waiting for the Mac (the Pod writes)
    inbox/<request id>.json       replies (the Mac writes)
    done/                         answered requests and their replies (the Pod moves them)
    withdrawn/                    requests the Pod gave up on (timeout) and stray replies
    rejected/                     replies the Pod refused (wrong ID, kind, index, count,
                                  SHA-256 or schema, or oversized)

A request ID is ``<session>-<seq>-<kind>-<index>``: twelve hex digits naming this
client, a six-digit sequence number that increases with every request, the kind
(``reward`` or ``dev``) and the six-digit update or check index. The request carries the
kind, the index, the attempt count, the SHA-256 of the canonical attempts payload and the
SHA-256 of the Pod's task file. The reply echoes all of them and carries, per attempt,
only what ``AttemptReward.to_json`` holds (reward, status, audit flags, seconds,
unfinished count) or a pass flag for a dev program. Hidden tests and expected values never
reach the Pod.

``BridgeRewardSource`` implements ``RewardSource``; ``BridgeDevExecutor`` is the
``DevExecutor`` that ``PoolDevEvaluator`` calls, and ``BridgeDevEvaluator`` labels
each dev request with its check index. A host failure the Mac could not resolve comes back
as a ``host_failure`` reply and raises ``RewardHostFailure``; the Pod never scores it. No
reply within the caller's timeout raises ``RewardTimeout`` and withdraws the request. A
refusal by the Mac raises ``BridgeRefused``; a reply that fails validation is moved to
``rejected/`` (the Mac pushes its stored reply again), and too many of them raise
``BridgeReplyError``. See ``docs/walkthroughs/rl-bridge.md``.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data.chat import TextDecoder, TextEncoder
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.dev import DevCheck, PoolDevEvaluator
from scratchcoder.rl.rewards import (
    NO_PROGRAM,
    Attempt,
    AttemptReward,
    RewardHostFailure,
    RewardTimeout,
)

REQUEST_FORMAT = "rl-bridge-request-v1"
REPLY_FORMAT = "rl-bridge-reply-v1"
REWARD = "reward"
DEV = "dev"
KINDS = (REWARD, DEV)
SCORED = "scored"
HOST_FAILURE = "host_failure"
TIMEOUT = "timeout"
REFUSED = "refused"
OUTCOMES = (SCORED, HOST_FAILURE, TIMEOUT, REFUSED)
REFUSALS = (
    "oversized",
    "corrupt",
    "schema",
    "payload_sha256",
    "task_file",
    "unknown_task",
    "stale",
    "conflict",
)
"""Why the Mac refused a request: too large, not JSON, wrong shape, a payload whose SHA-256
does not match, a task-file SHA-256 other than the configured one, a task ID outside the
phase's allowed list, a request older than one already answered, or a request ID reused
with another payload."""
AWAITING_REWARDS = "awaiting_rewards"
"""The heartbeat status while the Pod waits on the Mac (rewards or dev verdicts)."""

MAX_REQUEST_BYTES = 64 * 1024 * 1024
"""A dev check sends at most 200 x 9 programs of at most 512 tokens: a few MB in practice."""
MAX_REPLY_BYTES = 4 * 1024 * 1024
"""A reply holds at most 4,096 rows of a reward (with its audit flags) or a pass flag."""
MAX_ITEMS = 4096
MAX_DETAIL = 500
MAX_SEQ = 999_999
MAX_INDEX = 999_999

OUTBOX = "outbox"
INBOX = "inbox"
DONE = "done"
WITHDRAWN = "withdrawn"
REJECTED = "rejected"
BOXES = (OUTBOX, INBOX, DONE, WITHDRAWN, REJECTED)
LOCK_NAME = "client.lock"

REQUEST_ID = re.compile(
    r"(?P<session>[0-9a-f]{12})-(?P<seq>[0-9]{6})-(?P<kind>reward|dev)-(?P<index>[0-9]{6})\Z"
)
FILE_NAME = re.compile(r"[0-9a-f]{12}-[0-9]{6}-(?:reward|dev)-[0-9]{6}\.json\Z")
SESSION = re.compile(r"[0-9a-f]{12}\Z")
SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")

REQUEST_KEYS = frozenset(
    {
        "format",
        "request_id",
        "seq",
        "kind",
        "index",
        "count",
        "payload_sha256",
        "task_file_sha256",
        "created_unix",
        "items",
    }
)
REPLY_KEYS = frozenset(
    {
        "format",
        "request_id",
        "kind",
        "index",
        "count",
        "payload_sha256",
        "task_file_sha256",
        "outcome",
        "reason",
        "detail",
        "results",
        "seconds",
        "answered_unix",
    }
)
DEV_ITEM_KEYS = frozenset({"task_id", "code"})


class BridgeError(RuntimeError):
    """The bridge cannot carry this request; not a reward stop, so the run fails loudly."""


class BridgeRefused(BridgeError):
    """The Mac refused the request (``reason`` is one of ``REFUSALS``)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"the Mac refused the request ({reason}): {detail}")
        self.reason = reason


class BridgeReplyError(BridgeError):
    """Replies to one request kept failing validation."""


class BridgeSchemaError(ValueError):
    """A request or reply that breaks the schema; ``reason`` is a ``REFUSALS`` code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# ----------------------------------------------------------------------------- encoding


def canonical_json(value: object) -> bytes:
    """Sorted keys, no spaces, ASCII escapes, no NaN: one byte string per value."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def payload_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of the canonical JSON of the attempts list, as both sides compute it."""
    return hashlib.sha256(canonical_json(list(items))).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = item
    return value


def _load(raw: bytes, max_bytes: int, what: str) -> dict[str, Any]:
    if len(raw) > max_bytes:
        raise BridgeSchemaError("oversized", f"the {what} has {len(raw)} bytes, over {max_bytes}")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise BridgeSchemaError("corrupt", f"the {what} is not valid JSON") from error
    if not isinstance(value, dict):
        raise BridgeSchemaError("schema", f"the {what} is not an object")
    return value


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise BridgeSchemaError("schema", f"{name} must be an integer in [{low}, {high}]")
    return value


def _seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeSchemaError("schema", f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise BridgeSchemaError("schema", f"{name} must be finite and nonnegative")
    return number


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        raise BridgeSchemaError("schema", f"{name} must be 64 lowercase hex digits")
    return value


def require_sha256(value: object, name: str = "task_file_sha256") -> str:
    """``value`` if it is a lowercase hex SHA-256, else ``ValueError``."""
    if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hex digits")
    return value


@dataclass(frozen=True)
class RequestName:
    """The parts of a request ID; the file is ``<request id>.json`` in both boxes."""

    session: str
    seq: int
    kind: str
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.session, str) or not SESSION.fullmatch(self.session):
            raise ValueError("a bridge session is 12 lowercase hex digits")
        if type(self.seq) is not int or not 1 <= self.seq <= MAX_SEQ:
            raise ValueError(f"a request sequence number lies in [1, {MAX_SEQ}]")
        if self.kind not in KINDS:
            raise ValueError(f"a request kind is one of {KINDS}")
        if type(self.index) is not int or not 0 <= self.index <= MAX_INDEX:
            raise ValueError(f"a request index lies in [0, {MAX_INDEX}]")

    @property
    def request_id(self) -> str:
        return f"{self.session}-{self.seq:06d}-{self.kind}-{self.index:06d}"

    @property
    def file_name(self) -> str:
        return f"{self.request_id}.json"


def parse_request_id(value: object) -> RequestName:
    """The parts of a request ID (or of its file name, ``.json`` included)."""
    if isinstance(value, str) and value.endswith(".json"):
        value = value[: -len(".json")]
    match = REQUEST_ID.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise BridgeSchemaError("schema", "not a bridge request ID")
    return RequestName(match["session"], int(match["seq"]), match["kind"], int(match["index"]))


@dataclass(frozen=True)
class DevItem:
    """One dev program to run against a dev task's hidden tests on the Mac."""

    task_id: str
    code: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("a dev item needs a nonempty task ID")
        if not isinstance(self.code, str):
            raise ValueError("a dev item's code is text")

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "code": self.code}

    @classmethod
    def from_json(cls, value: object) -> DevItem:
        if not isinstance(value, Mapping) or set(value) != DEV_ITEM_KEYS:
            raise ValueError(f"a dev item has exactly the keys {sorted(DEV_ITEM_KEYS)}")
        return cls(value["task_id"], value["code"])


Item = Attempt | DevItem


@dataclass(frozen=True)
class BridgeRequest:
    """A validated request: ``items`` are ``Attempt`` values (reward) or ``DevItem`` (dev)."""

    name: RequestName
    count: int
    payload_sha256: str
    task_file_sha256: str
    created_unix: float
    items: tuple[Item, ...]

    @property
    def request_id(self) -> str:
        return self.name.request_id

    @property
    def kind(self) -> str:
        return self.name.kind

    @property
    def index(self) -> int:
        return self.name.index

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(item.task_id for item in self.items)

    @classmethod
    def build(
        cls,
        name: RequestName,
        items: Sequence[Item],
        *,
        task_file_sha256: str,
        created_unix: float,
    ) -> BridgeRequest:
        expected = Attempt if name.kind == REWARD else DevItem
        if not items or len(items) > MAX_ITEMS:
            raise ValueError(f"a request carries 1 to {MAX_ITEMS} items")
        if any(not isinstance(item, expected) for item in items):
            raise ValueError(f"a {name.kind} request carries {expected.__name__} items")
        rows = [item.to_json() for item in items]
        return cls(
            name,
            len(items),
            payload_sha256(rows),
            require_sha256(task_file_sha256),
            float(created_unix),
            tuple(items),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": REQUEST_FORMAT,
            "request_id": self.request_id,
            "seq": self.name.seq,
            "kind": self.kind,
            "index": self.index,
            "count": self.count,
            "payload_sha256": self.payload_sha256,
            "task_file_sha256": self.task_file_sha256,
            "created_unix": self.created_unix,
            "items": [item.to_json() for item in self.items],
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.to_json())


def decode_request(
    raw: bytes, *, max_bytes: int = MAX_REQUEST_BYTES, expected_id: str | None = None
) -> BridgeRequest:
    """Parse and check a request: exact keys, consistent ID, count and payload SHA-256.

    Raises ``BridgeSchemaError`` whose ``reason`` is ``oversized``, ``corrupt``, ``schema``
    or ``payload_sha256``.
    """
    value = _load(raw, max_bytes, "request")
    if set(value) != REQUEST_KEYS:
        raise BridgeSchemaError("schema", f"a request has exactly the keys {sorted(REQUEST_KEYS)}")
    if value["format"] != REQUEST_FORMAT:
        raise BridgeSchemaError("schema", f"the request format is not {REQUEST_FORMAT}")
    name = parse_request_id(value["request_id"])
    if expected_id is not None and name.request_id != expected_id:
        raise BridgeSchemaError("schema", "the request ID differs from its file name")
    if (
        value["kind"] != name.kind
        or _integer(value["seq"], "seq", 1, MAX_SEQ) != name.seq
        or _integer(value["index"], "index", 0, MAX_INDEX) != name.index
    ):
        raise BridgeSchemaError("schema", "the request's kind, seq or index differs from its ID")
    count = _integer(value["count"], "count", 1, MAX_ITEMS)
    rows = value["items"]
    if not isinstance(rows, list) or len(rows) != count:
        raise BridgeSchemaError("schema", "the request's count differs from its items")
    digest = _sha(value["payload_sha256"], "payload_sha256")
    task_file = _sha(value["task_file_sha256"], "task_file_sha256")
    created = _seconds(value["created_unix"], "created_unix")
    try:
        items: tuple[Item, ...] = tuple(
            Attempt.from_json(row) if name.kind == REWARD else DevItem.from_json(row)
            for row in rows
        )
    except (ValueError, TypeError) as error:
        raise BridgeSchemaError("schema", f"a request item is malformed: {error}") from error
    if payload_sha256(rows) != digest:
        raise BridgeSchemaError("payload_sha256", "the payload SHA-256 does not match the items")
    return BridgeRequest(name, count, digest, task_file, created, items)


@dataclass(frozen=True)
class BridgeReply:
    """A reply. ``rewards`` (reward kind) or ``passed`` (dev kind) is set only when
    ``outcome`` is ``scored``; a refusal names its ``reason`` and may leave ``count`` and
    the two SHA-256 values unknown (``None``) when the request could not be read."""

    request_id: str
    kind: str
    index: int
    count: int | None
    payload_sha256: str | None
    task_file_sha256: str | None
    outcome: str
    reason: str | None = None
    detail: str = ""
    rewards: tuple[AttemptReward, ...] = ()
    passed: tuple[bool, ...] = ()
    seconds: float = 0.0
    answered_unix: float = 0.0

    def __post_init__(self) -> None:
        name = parse_request_id(self.request_id)
        if type(self.index) is not int or (name.kind, name.index) != (self.kind, self.index):
            raise ValueError("a reply's kind and index must match its request ID")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"a reply outcome is one of {OUTCOMES}")
        if (self.reason is not None) != (self.outcome == REFUSED) or (
            self.reason is not None and self.reason not in REFUSALS
        ):
            raise ValueError("a refusal, and only a refusal, names a reason from REFUSALS")
        if not isinstance(self.detail, str) or len(self.detail) > MAX_DETAIL:
            raise ValueError(f"a reply detail is text of at most {MAX_DETAIL} characters")
        if self.outcome != REFUSED and (
            self.count is None or self.payload_sha256 is None or self.task_file_sha256 is None
        ):
            raise ValueError("only a refusal may leave the count or a SHA-256 unknown")
        rows = len(self.rewards) if self.kind == REWARD else len(self.passed)
        other = len(self.passed) if self.kind == REWARD else len(self.rewards)
        if (
            other
            or (self.outcome == SCORED) != (rows > 0)
            or (self.outcome == SCORED and rows != self.count)
        ):
            raise ValueError("a scored reply carries one result per item; any other carries none")
        for name_, value in (("seconds", self.seconds), ("answered_unix", self.answered_unix)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name_} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name_} must be finite and nonnegative")

    @property
    def results(self) -> list[Any]:
        if self.kind == REWARD:
            return [reward.to_json() for reward in self.rewards]
        return list(self.passed)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": REPLY_FORMAT,
            "request_id": self.request_id,
            "kind": self.kind,
            "index": self.index,
            "count": self.count,
            "payload_sha256": self.payload_sha256,
            "task_file_sha256": self.task_file_sha256,
            "outcome": self.outcome,
            "reason": self.reason,
            "detail": self.detail,
            "results": self.results,
            "seconds": self.seconds,
            "answered_unix": self.answered_unix,
        }

    def to_bytes(self) -> bytes:
        # NaN is allowed on purpose: a non-finite reward must reach the RL loop, which
        # stops on it (reason non_finite) instead of training on it.
        return json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")


def scored_reply(
    request: BridgeRequest,
    *,
    rewards: Sequence[AttemptReward] = (),
    passed: Sequence[bool] = (),
    seconds: float,
    answered_unix: float,
) -> BridgeReply:
    """The reply to a scored request: ``rewards`` for the reward kind, ``passed`` for dev."""
    return BridgeReply(
        request.request_id,
        request.kind,
        request.index,
        request.count,
        request.payload_sha256,
        request.task_file_sha256,
        SCORED,
        rewards=tuple(rewards),
        passed=tuple(passed),
        seconds=seconds,
        answered_unix=answered_unix,
    )


def failure_reply(
    request: BridgeRequest, outcome: str, detail: str, *, seconds: float, answered_unix: float
) -> BridgeReply:
    if outcome not in (HOST_FAILURE, TIMEOUT):
        raise ValueError("a failure reply is a host failure or a timeout")
    return BridgeReply(
        request.request_id,
        request.kind,
        request.index,
        request.count,
        request.payload_sha256,
        request.task_file_sha256,
        outcome,
        detail=detail[:MAX_DETAIL],
        seconds=seconds,
        answered_unix=answered_unix,
    )


def refusal_reply(
    name: RequestName,
    reason: str,
    detail: str,
    *,
    request: BridgeRequest | None,
    answered_unix: float,
) -> BridgeReply:
    return BridgeReply(
        name.request_id,
        name.kind,
        name.index,
        None if request is None else request.count,
        None if request is None else request.payload_sha256,
        None if request is None else request.task_file_sha256,
        REFUSED,
        reason=reason,
        detail=detail[:MAX_DETAIL],
        answered_unix=answered_unix,
    )


def decode_reply(raw: bytes, *, max_bytes: int = MAX_REPLY_BYTES) -> BridgeReply:
    """Parse a reply's structure (exact keys, types, result rows); ``BridgeSchemaError``."""
    value = _load(raw, max_bytes, "reply")
    if set(value) != REPLY_KEYS:
        raise BridgeSchemaError("schema", f"a reply has exactly the keys {sorted(REPLY_KEYS)}")
    if value["format"] != REPLY_FORMAT:
        raise BridgeSchemaError("schema", f"the reply format is not {REPLY_FORMAT}")
    name = parse_request_id(value["request_id"])
    count = value["count"]
    if count is not None:
        count = _integer(count, "count", 1, MAX_ITEMS)
    digests = [
        None if value[key] is None else _sha(value[key], key)
        for key in ("payload_sha256", "task_file_sha256")
    ]
    rows = value["results"]
    if not isinstance(rows, list) or len(rows) > MAX_ITEMS:
        raise BridgeSchemaError("schema", "reply results must be a bounded list")
    try:
        rewards: tuple[AttemptReward, ...] = ()
        passed: tuple[bool, ...] = ()
        if name.kind == REWARD:
            rewards = tuple(AttemptReward.from_json(row) for row in rows)
        else:
            if any(type(row) is not bool for row in rows):
                raise ValueError("dev results are booleans")
            passed = tuple(rows)
        return BridgeReply(
            value["request_id"],
            value["kind"],
            value["index"],
            count,
            digests[0],
            digests[1],
            value["outcome"],
            value["reason"],
            value["detail"],
            rewards,
            passed,
            value["seconds"],
            value["answered_unix"],
        )
    except (ValueError, TypeError) as error:
        raise BridgeSchemaError("schema", f"the reply is malformed: {error}") from error


def verify_reply(reply: BridgeReply, request: BridgeRequest) -> None:
    """Refuse a reply that does not answer ``request``: ``BridgeSchemaError``.

    The request ID, kind and index must match; the count and both SHA-256 values must match
    (a refusal may leave them unknown); and a scored reward reply must score every attempt
    without a program as 0 with ``no_program``.
    """
    if (reply.request_id, reply.kind, reply.index) != (
        request.request_id,
        request.kind,
        request.index,
    ):
        raise BridgeSchemaError("schema", "the reply answers another request, kind or index")
    for field, expected in (
        ("count", request.count),
        ("payload_sha256", request.payload_sha256),
        ("task_file_sha256", request.task_file_sha256),
    ):
        found = getattr(reply, field)
        if found != expected and not (reply.outcome == REFUSED and found is None):
            raise BridgeSchemaError("schema", f"the reply's {field} does not match the request")
    if reply.outcome == SCORED and reply.kind == REWARD:
        for item, reward in zip(request.items, reply.rewards, strict=True):
            assert isinstance(item, Attempt)
            if item.code is None and (reward.reward != 0.0 or reward.status != NO_PROGRAM):
                raise BridgeSchemaError(
                    "schema", "an attempt without a program must score 0 with no_program"
                )


# --------------------------------------------------------------------------- file moves


def fsync_directory(path: Path) -> None:
    """Make a rename in ``path`` durable (best effort where the platform cannot)."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` by atomic rename: hidden temporary file, fsync, rename."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    fsync_directory(path.parent)


def read_bounded(path: Path, max_bytes: int) -> bytes | None:
    """The bytes of a regular file of at most ``max_bytes``; ``None`` if missing; raises
    ``BridgeSchemaError("oversized")`` if larger or not a regular file."""
    # O_NONBLOCK: opening a FIFO planted under the name must not block the reader.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise BridgeSchemaError("oversized", f"{path.name} is a symlink") from error
        raise
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise BridgeSchemaError("oversized", f"{path.name} is not a bounded regular file")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise BridgeSchemaError("oversized", f"{path.name} grew past {max_bytes} bytes")
    return data


def _move_aside(source: Path, directory: Path, name: str) -> Path:
    """Rename ``source`` into ``directory`` as ``name`` (with a suffix if taken)."""
    target = directory / name
    counter = 1
    while target.exists():
        target = directory / f"{name}.{counter}"
        counter += 1
    os.replace(source, target)
    return target


# ----------------------------------------------------------------------------- Pod side


@dataclass(frozen=True)
class BridgeWait:
    """What the Pod is waiting for: a heartbeat shows it as ``awaiting_rewards``."""

    request_id: str
    kind: str
    index: int
    count: int
    started_unix: float
    age_seconds: float

    def to_json(self) -> dict[str, Any]:
        return {
            "status": AWAITING_REWARDS,
            "request_id": self.request_id,
            "kind": self.kind,
            "index": self.index,
            "count": self.count,
            "started_unix": self.started_unix,
            "age_seconds": self.age_seconds,
        }


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of seconds")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class BridgeClient:
    """The Pod's end of the bridge: one request at a time, through ``root``.

    Construction creates the boxes, takes an exclusive lock on ``root/client.lock`` (one
    client per root) and moves anything left in the outbox or inbox by an earlier client
    into ``withdrawn/``, so the Mac never scores a request nobody waits for. ``session``
    (12 hex digits) names this client in every request ID; a random one is drawn when it is
    not given.

    ``exchange`` writes one request, waits for its reply, checks it and returns it when it
    is ``scored``. ``poll_seconds`` is the wait between looks at the inbox. ``clock`` (a
    monotonic clock), ``wall`` (Unix time) and ``sleep`` are injectable for tests.
    """

    def __init__(
        self,
        root: Path,
        *,
        session: str | None = None,
        poll_seconds: float = 0.5,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_reply_bytes: int = MAX_REPLY_BYTES,
        max_rejected: int = 3,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.poll_seconds = _positive_seconds(poll_seconds, "poll_seconds")
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_reply_bytes", max_reply_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(max_rejected) is not int or max_rejected < 0:
            raise ValueError("max_rejected must be a nonnegative integer")
        self.session = session if session is not None else secrets.token_hex(6)
        if not isinstance(self.session, str) or not SESSION.fullmatch(self.session):
            raise ValueError("a bridge session is 12 lowercase hex digits")
        self.root = root
        self.max_request_bytes = max_request_bytes
        self.max_reply_bytes = max_reply_bytes
        self.max_rejected = max_rejected
        self._clock, self._wall, self._sleep = clock, wall, sleep
        root.mkdir(parents=True, exist_ok=True)
        for box in BOXES:
            (root / box).mkdir(exist_ok=True)
        self._lock = os.open(root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._lock)
            raise BridgeError(f"another bridge client holds {root}") from error
        self._closed = False
        self._seq = 0
        self._busy = threading.Lock()
        self._state = threading.Lock()
        self._waiting: tuple[BridgeRequest, float, float] | None = None
        try:
            self.withdrawn_at_start = self._withdraw_all()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)

    def __enter__(self) -> BridgeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def requests_sent(self) -> int:
        return self._seq

    def waiting(self) -> BridgeWait | None:
        """The request the Pod is waiting on, or ``None``; safe to call from any thread."""
        with self._state:
            if self._waiting is None:
                return None
            request, started, started_unix = self._waiting
        return BridgeWait(
            request.request_id,
            request.kind,
            request.index,
            request.count,
            started_unix,
            max(0.0, self._clock() - started),
        )

    def heartbeat(self) -> dict[str, Any] | None:
        """``{"status": "awaiting_rewards", ...}`` while waiting, else ``None``."""
        wait = self.waiting()
        return None if wait is None else wait.to_json()

    def _withdraw_all(self) -> int:
        moved = 0
        for box in (OUTBOX, INBOX):
            for path in sorted((self.root / box).iterdir()):
                if path.is_file() or path.is_symlink():
                    suffix = ".reply" if box == INBOX else ""
                    _move_aside(path, self.root / WITHDRAWN, f"leftover{suffix}-{path.name}")
                    moved += 1
        return moved

    def _sweep_inbox(self) -> None:
        """Move replies to requests this client no longer waits on out of the inbox."""
        for path in sorted((self.root / INBOX).iterdir()):
            if FILE_NAME.fullmatch(path.name) and (path.is_file() or path.is_symlink()):
                _move_aside(path, self.root / WITHDRAWN, f"stray-{path.name}")

    def _take_reply(self, request: BridgeRequest) -> BridgeReply | str | None:
        """The checked reply, ``None`` if none yet, or a rejection reason (file moved aside)."""
        path = self.root / INBOX / request.name.file_name
        try:
            raw = read_bounded(path, self.max_reply_bytes)
            if raw is None:
                return None
            reply = decode_reply(raw, max_bytes=self.max_reply_bytes)
            verify_reply(reply, request)
        except BridgeSchemaError as error:
            if path.exists() or path.is_symlink():
                _move_aside(path, self.root / REJECTED, path.name)
            return f"{error.reason}: {error}"
        return reply

    def exchange(
        self,
        kind: str,
        index: int,
        items: Sequence[Item],
        *,
        task_file_sha256: str,
        timeout: float,
    ) -> BridgeReply:
        """Send one request and return its ``scored`` reply.

        Raises ``RewardTimeout`` when no acceptable reply arrives within ``timeout``
        seconds (the request is withdrawn), ``RewardHostFailure`` for a ``host_failure``
        reply, ``RewardTimeout`` for a Mac-side ``timeout`` reply, ``BridgeRefused`` for a
        refusal and ``BridgeReplyError`` after more than ``max_rejected`` rejected replies.
        """
        if self._closed:
            raise BridgeError("the bridge client is closed")
        if not self._busy.acquire(blocking=False):
            raise BridgeError("the bridge carries one request at a time")
        try:
            return self._exchange(kind, index, items, task_file_sha256, timeout)
        finally:
            self._busy.release()

    def _exchange(
        self,
        kind: str,
        index: int,
        items: Sequence[Item],
        task_file_sha256: str,
        timeout: float,
    ) -> BridgeReply:
        limit = _positive_seconds(timeout, "timeout")
        if self._seq >= MAX_SEQ:
            raise BridgeError("this bridge session has used every sequence number")
        name = RequestName(self.session, self._seq + 1, kind, index)
        request = BridgeRequest.build(
            name, items, task_file_sha256=task_file_sha256, created_unix=self._wall()
        )
        raw = request.to_bytes()
        if len(raw) > self.max_request_bytes:
            raise BridgeError(f"request {name.request_id} has {len(raw)} bytes, over the bound")
        self._seq += 1
        self._sweep_inbox()
        started = self._clock()
        with self._state:
            self._waiting = (request, started, self._wall())
        outbox = self.root / OUTBOX / name.file_name
        rejections: list[str] = []
        try:
            atomic_write(outbox, raw)
            while True:
                found = self._take_reply(request)
                if isinstance(found, BridgeReply):
                    _move_aside(outbox, self.root / DONE, f"{name.request_id}.request.json")
                    _move_aside(
                        self.root / INBOX / name.file_name,
                        self.root / DONE,
                        f"{name.request_id}.reply.json",
                    )
                    return self._resolve(found)
                if found is not None:
                    rejections.append(found)
                    if len(rejections) > self.max_rejected:
                        self._withdraw(outbox)
                        raise BridgeReplyError(
                            f"{len(rejections)} replies to {name.request_id} were rejected; "
                            f"last: {rejections[-1]}"
                        )
                waited = self._clock() - started
                if waited >= limit:
                    self._withdraw(outbox)
                    raise RewardTimeout(
                        f"no reply to {kind} request {name.request_id} (index {index}, "
                        f"{request.count} items) within {limit:g} s"
                    )
                self._sleep(min(self.poll_seconds, limit - waited))
        except BaseException:
            # Whatever ends the wait (a timeout, a refusal, or an error raised by the injected
            # sleep, such as the worker giving up on a lost bridge), nobody waits for the
            # request any more: withdraw it, so the Mac never scores it. A request already
            # answered was moved to done/, and withdrawing a missing file is a no-op.
            self._withdraw(outbox)
            raise
        finally:
            with self._state:
                self._waiting = None

    def _withdraw(self, outbox: Path) -> None:
        with contextlib.suppress(FileNotFoundError):
            _move_aside(outbox, self.root / WITHDRAWN, outbox.name)

    @staticmethod
    def _resolve(reply: BridgeReply) -> BridgeReply:
        where = f"{reply.kind} request {reply.request_id}"
        if reply.outcome == HOST_FAILURE:
            raise RewardHostFailure(
                f"the Mac could not resolve a host failure ({where}): {reply.detail}"
            )
        if reply.outcome == TIMEOUT:
            raise RewardTimeout(f"the Mac timed out scoring ({where}): {reply.detail}")
        if reply.outcome == REFUSED:
            assert reply.reason is not None
            raise BridgeRefused(reply.reason, f"{where}: {reply.detail}")
        return reply


class BridgeRewardSource:
    """A ``RewardSource`` whose rewards come from the Mac through ``client``.

    Every attempt travels, programs and ``None`` alike, so the reply has one reward per
    attempt in order; the Mac scores a ``None`` program as 0 with ``no_program`` without
    running it, and the reply is refused otherwise. ``task_file_sha256`` is the SHA-256 of
    the Pod's RL task file, which the Mac checks against its configuration. ``timeout``
    (seconds, required) bounds each ``rewards`` call; the caller derives it from the reward
    budgets, the Mac's workers and the bridge's polling and SSH allowance (``docs/rl.md``).
    """

    def __init__(self, client: BridgeClient, *, task_file_sha256: str, timeout: float) -> None:
        self.client = client
        self.task_file_sha256 = require_sha256(task_file_sha256)
        self.timeout = _positive_seconds(timeout, "timeout")

    def rewards(self, update: int, attempts: Sequence[Attempt]) -> list[AttemptReward]:
        if type(update) is not int or update < 0:
            raise ValueError("update must be a nonnegative integer")
        attempts = list(attempts)
        if not attempts:
            return []
        reply = self.client.exchange(
            REWARD,
            update,
            attempts,
            task_file_sha256=self.task_file_sha256,
            timeout=self.timeout,
        )
        return list(reply.rewards)


class BridgeDevExecutor:
    """The ``DevExecutor`` that sends dev programs to the Mac: one pass flag per pair.

    Only each pair's task ID and program travel. The request's index is the check index
    set by ``labelled`` (``BridgeDevEvaluator`` sets it for every check); without a label it
    is this executor's own call count, starting at 0.
    """

    def __init__(self, client: BridgeClient, *, task_file_sha256: str, timeout: float) -> None:
        self.client = client
        self.task_file_sha256 = require_sha256(task_file_sha256)
        self.timeout = _positive_seconds(timeout, "timeout")
        self.calls = 0
        self._label: int | None = None

    @contextlib.contextmanager
    def labelled(self, check: int) -> Iterator[None]:
        if type(check) is not int or not 0 <= check <= MAX_INDEX:
            raise ValueError(f"a check index lies in [0, {MAX_INDEX}]")
        previous, self._label = self._label, check
        try:
            yield
        finally:
            self._label = previous

    def __call__(self, pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        pairs = list(pairs)
        if not pairs:
            return []
        index = self.calls if self._label is None else self._label
        self.calls += 1
        reply = self.client.exchange(
            DEV,
            index,
            [DevItem(task.id, code) for task, code in pairs],
            task_file_sha256=self.task_file_sha256,
            timeout=self.timeout,
        )
        return list(reply.passed)


class BridgeDevEvaluator:
    """The RL ``PoolDevEvaluator`` with its executions sent through the bridge, each dev
    request labelled with the check index the RL loop or the control asked for."""

    def __init__(
        self,
        tasks: Sequence[Task],
        encoder: TextEncoder,
        decoder: TextDecoder,
        executor: BridgeDevExecutor,
        **options: Any,
    ) -> None:
        self.executor = executor
        self.pool = PoolDevEvaluator(tasks, encoder, decoder, executor, **options)

    def evaluate(self, model: GPT, check: int) -> DevCheck:
        with self.executor.labelled(check):
            return self.pool.evaluate(model, check)
