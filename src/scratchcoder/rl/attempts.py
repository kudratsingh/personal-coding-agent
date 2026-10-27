"""The screening attempts file: what a Pod generated for RL screening, bound to its tasks.

``scratchcoder rl-screen`` reads JSON lines, one ``Attempt.to_json`` per attempt. A file
written by the post-training Pod worker starts with one header line::

    {"kind": "header", "format": "scratchcoder-screening-attempts", "schema_version": 1, ...}

The header names the SHA-256 of the task file the attempts were generated from (on a Pod,
the candidates export: the train candidates with their hidden tests removed), the candidate
file's SHA-256, the checkpoint and tokenizer, the sampling settings and seed base, and the
attempt count. ``rl-screen`` checks it against the full train task file it is given: the
candidates export is derived again from that file and the candidate list, and its SHA-256
compared (``scratchcoder.eval.export.task_file_identity``); the export's own record is never
read. A file without a header (one ``Attempt`` per line) is still read as before unless the
caller requires a header.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from scratchcoder.rl.rewards import Attempt

FORMAT = "scratchcoder-screening-attempts"
SCHEMA_VERSION = 1
SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
SCREENING_PROTOCOL: Mapping[str, int | float] = {
    "samples": 8,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
    "seed_base": 2**40,
}
"""The screening protocol's sampling: ``rl.rollout.screening_rollouts``' defaults (8 samples
at temperature 1.0 and top-p 1.0, at most 256 new tokens, ``SCREENING_SEED``). A header must
carry exactly these; ``tests/test_pod_task_export.py`` keeps them equal to the rollout's
constants, which live in a module that imports torch, which ``rl-screen`` does not need."""


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= SHA256_HEX_DIGITS:
        raise ValueError(f"{name} must be a lower-case hex SHA-256")
    return value


@dataclass(frozen=True)
class ScreeningHeader:
    task_file_sha256: str
    candidates_sha256: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    tasks: int
    samples: int
    attempts: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed_base: int
    precision: str
    device: str
    batch_size: int
    generation_seconds: float
    created_at: str
    git_commit: str | None = None
    dirty: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_file_sha256",
            "candidates_sha256",
            "checkpoint_sha256",
            "tokenizer_sha256",
        ):
            _sha(getattr(self, name), name)
        for name in ("tasks", "samples", "attempts", "max_new_tokens", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.attempts != self.tasks * self.samples:
            raise ValueError("attempts must be tasks x samples")
        if type(self.seed_base) is not int or self.seed_base < 0:
            raise ValueError("seed_base must be a nonnegative integer")
        for name in ("temperature", "top_p", "generation_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        for name in ("precision", "device", "created_at"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "header",
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ScreeningHeader:
        if raw.get("kind") != "header" or raw.get("format") != FORMAT:
            raise ValueError("not a screening attempts header")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported screening attempts schema version")
        fields = {k: v for k, v in raw.items() if k not in ("kind", "format", "schema_version")}
        if set(fields) != set(cls.__dataclass_fields__):
            raise ValueError("screening attempts header fields differ from the schema")
        return cls(**fields)


def check_protocol(
    header: ScreeningHeader, protocol: Mapping[str, int | float] = SCREENING_PROTOCOL
) -> None:
    """The header's sampling is the screening protocol's: samples, temperature, top-p, the
    new-token cap and the seed base. Screening scores attempts drawn under that protocol."""
    for name, expected in protocol.items():
        found = getattr(header, name)
        if found != expected:
            raise ValueError(
                f"the screening header's {name} is {found!r}, not the protocol's {expected!r}"
            )


def check_grid(
    header: ScreeningHeader, attempts: Sequence[Attempt], task_ids: Sequence[str]
) -> None:
    """Every candidate has samples ``0 .. samples - 1`` exactly once, and nothing else."""
    if len(attempts) != header.attempts or len(task_ids) != header.tasks:
        raise ValueError("the attempt or task count differs from the screening header")
    expected = {(task, sample) for task in task_ids for sample in range(header.samples)}
    found = [(attempt.task_id, attempt.sample) for attempt in attempts]
    if len(set(found)) != len(found) or set(found) != expected:
        raise ValueError("the attempts are not every candidate's samples exactly once")


def write_attempts(path: Path, header: ScreeningHeader, attempts: Sequence[Attempt]) -> str:
    """Write a new attempts file (header first) atomically; return its SHA-256."""
    if len(attempts) != header.attempts:
        raise ValueError("the attempt count differs from the screening header")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists; an attempts file is never replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            for record in (header.to_json(), *(attempt.to_json() for attempt in attempts)):
                line = json.dumps(record, sort_keys=True, allow_nan=False).encode() + b"\n"
                digest.update(line)
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def read_attempts_file(path: Path) -> tuple[ScreeningHeader | None, list[Attempt]]:
    """The header (None for a header-less file) and the attempts, in file order."""
    header: ScreeningHeader | None = None
    attempts: list[Attempt] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if number == 1 and isinstance(raw, dict) and raw.get("kind") == "header":
            header = ScreeningHeader.from_json(raw)
            continue
        attempts.append(Attempt.from_json(raw))
    return header, attempts
