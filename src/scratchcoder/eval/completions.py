"""The completions file: what a checkpoint generated for a suite, joined to execution by hash.

Generation runs on a GPU and execution runs elsewhere, so the two are separate steps. The
file between them is JSON Lines: one header record, then one record per attempt::

    {"kind": "header", "format": "scratchcoder-completions", "schema_version": 1, ...}
    {"kind": "attempt", "task_id": "...", "sample": 0, "seed": ..., "tokens": [...], ...}

The file's SHA-256 is its identity: execution results, seal ledgers and benchmark
artifacts name it. Writing is atomic (a temporary file in the same directory, then a
rename), and an existing file is never replaced. Reading validates the schema, the
attempt grid (every task has samples ``0 .. n-1`` with the harness seeds) and the protocol
record, and :func:`validate_against_tasks` checks the header against the task file.

The header names the suite and split, the protocol and its record's SHA-256, the
checkpoint SHA-256, the tokenizer identity, the task file SHA-256 and task IDs, the
sampling settings and seed rule, precision, device, batch size, generation seconds, git
commit and dirty flag, and the creation time. An attempt records its task ID, sample
index, seed, prompt length and prompt hash, the generated token IDs and their count, the
decoded completion text, the protocol's extracted program (``null`` when there is none)
and the stop reason (``stop``, ``length`` or ``context_limit``).
"""

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import PRECISIONS, SAMPLER_VERSION
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.eval.protocols import (
    BENCHMARK_SPLIT,
    POOL,
    POOL_SPLITS,
    SUITES,
    canonical_json,
    load_suite_tasks,
    protocol_for,
    record_sha256,
)
from scratchcoder.eval.tasks import Task

FORMAT = "scratchcoder-completions"
SCHEMA_VERSION = 1
STOP_REASONS = ("stop", "length", "context_limit")
SEED_RULE = (
    "scratchcoder.eval.harness.attempt_seed(seed, task_id, sample) = "
    "(seed + int(sha256(task_id)[:16], 16) + sample) mod (2**63 - 1)"
)
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_LINE_BYTES = 64 * 1024 * 1024


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lower-case hex SHA-256")
    return value


def task_ids_sha256(task_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted task IDs, one per line."""
    return hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest()


def prompt_sha256(ids: Sequence[int]) -> str:
    """SHA-256 of a prompt's token IDs as compact JSON, e.g. ``[256,257]``."""
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Sampling:
    """Decoding settings: greedy (one sample, temperature 0) or ``samples`` seeded draws."""

    samples: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.samples) is not int or self.samples < 1:
            raise ValueError("samples must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2^63)")
        for value in (self.temperature, self.top_p):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError("temperature and top_p must be numbers")
        if not 0 <= self.temperature < math.inf or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature or top_p")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or null")
        if self.temperature == 0 and self.samples != 1:
            raise ValueError("greedy decoding has one deterministic attempt; sample for repeats")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    def seeds(self, task_id: str) -> list[int]:
        return [attempt_seed(self.seed, task_id, index) for index in range(self.samples)]


@dataclass(frozen=True)
class CompletionsHeader:
    suite: str
    split: str
    protocol: str
    protocol_sha256: str
    protocol_record: Mapping[str, Any]
    checkpoint_sha256: str
    tokenizer_sha256: str
    tokenizer_vocab_size: int
    model_context: int
    task_file_sha256: str
    task_count: int
    task_ids_sha256: str
    sampling: Sampling
    max_new_tokens: int
    precision: str
    device: str
    rows: int
    created_at: str
    excluded_tasks: tuple[Mapping[str, str], ...] = ()
    sampler_version: str | None = None
    seed_rule: str = SEED_RULE
    device_name: str = ""
    batch_size: int = 1
    generation_seconds: float = 0.0
    git_commit: str | None = None
    dirty: bool | None = None

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError("unknown suite")
        if self.split not in (POOL_SPLITS if self.suite == POOL else (BENCHMARK_SPLIT,)):
            raise ValueError("invalid split for the suite")
        protocol = protocol_for(self.protocol, self.suite)
        for name in ("protocol_sha256", "checkpoint_sha256", "tokenizer_sha256"):
            _sha(getattr(self, name), name)
        _sha(self.task_file_sha256, "task_file_sha256")
        _sha(self.task_ids_sha256, "task_ids_sha256")
        if record_sha256(self.protocol_record) != self.protocol_sha256:
            raise ValueError("the protocol record does not hash to protocol_sha256")
        if dict(self.protocol_record) != protocol.record():
            raise ValueError("the protocol record differs from this code's record for the name")
        if not isinstance(self.sampling, Sampling):
            raise ValueError("sampling must be a Sampling")
        if self.max_new_tokens != protocol.cap:
            raise ValueError("max_new_tokens must be the protocol's cap")
        if self.sampler_version != (None if self.sampling.greedy else SAMPLER_VERSION):
            raise ValueError("sampled files name the sampler; greedy files name none")
        if self.precision not in PRECISIONS:
            raise ValueError("precision must be float32 or bfloat16")
        counts = (
            self.tokenizer_vocab_size,
            self.model_context,
            self.task_count,
            self.rows,
            self.batch_size,
        )
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("vocabulary, context, task, row and batch counts must be positive")
        if self.rows != self.task_count * self.sampling.samples:
            raise ValueError("rows must be task_count x samples")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device is required")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("created_at is required")
        if self.seed_rule != SEED_RULE:
            raise ValueError("unknown seed rule")
        if not math.isfinite(self.generation_seconds) or self.generation_seconds < 0:
            raise ValueError("generation_seconds must be finite and nonnegative")
        for item in self.excluded_tasks:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"task_id", "reason"}
                or not all(isinstance(value, str) and value for value in item.values())
            ):
                raise ValueError("an excluded task names task_id and reason")

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["protocol_record"] = dict(self.protocol_record)
        value["excluded_tasks"] = [dict(item) for item in self.excluded_tasks]
        return {"kind": "header", "format": FORMAT, "schema_version": SCHEMA_VERSION, **value}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "CompletionsHeader":
        if raw.get("kind") != "header" or raw.get("format") != FORMAT:
            raise ValueError("not a completions header")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported completions schema version")
        fields = {key: value for key, value in raw.items() if key not in ("kind", "format")}
        fields.pop("schema_version")
        names = set(cls.__dataclass_fields__)
        if set(fields) != names:
            raise ValueError("completions header fields differ from the schema")
        sampling = fields["sampling"]
        if not isinstance(sampling, dict) or set(sampling) != set(Sampling.__dataclass_fields__):
            raise ValueError("invalid sampling settings")
        fields["sampling"] = Sampling(**sampling)
        if not isinstance(fields["excluded_tasks"], list):
            raise ValueError("excluded_tasks must be a list")
        fields["excluded_tasks"] = tuple(fields["excluded_tasks"])
        if not isinstance(fields["protocol_record"], dict):
            raise ValueError("protocol_record must be an object")
        return cls(**fields)


@dataclass(frozen=True)
class CompletionRow:
    task_id: str
    sample: int
    seed: int
    prompt_tokens: int
    prompt_sha256: str
    tokens: tuple[int, ...]
    completion: str
    program: str | None
    stop_reason: str
    budget: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id is required")
        for name in ("sample", "seed", "prompt_tokens", "budget"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        _sha(self.prompt_sha256, "prompt_sha256")
        if not isinstance(self.tokens, tuple) or any(
            type(token) is not int or token < 0 for token in self.tokens
        ):
            raise ValueError("tokens must be nonnegative token IDs")
        if not isinstance(self.completion, str):
            raise ValueError("completion must be text")
        if self.program is not None and not isinstance(self.program, str):
            raise ValueError("program must be text or null")
        if self.stop_reason not in STOP_REASONS:
            raise ValueError("stop_reason must be stop, length or context_limit")
        if len(self.tokens) > self.budget:
            raise ValueError("an attempt cannot generate more tokens than its budget")
        if self.stop_reason == "context_limit" and (
            self.tokens or self.budget or self.program is not None
        ):
            raise ValueError("a context-limit attempt generates nothing and has no program")
        if self.stop_reason == "length" and len(self.tokens) != self.budget:
            raise ValueError("a length stop spends the whole budget")

    @property
    def generated_tokens(self) -> int:
        return len(self.tokens)

    @property
    def key(self) -> tuple[str, int]:
        return self.task_id, self.sample

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["tokens"] = list(self.tokens)
        return {"kind": "attempt", "generated_tokens": len(self.tokens), **value}

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "CompletionRow":
        if raw.get("kind") != "attempt":
            raise ValueError("not a completions attempt record")
        fields = {key: value for key, value in raw.items() if key not in ("kind",)}
        generated = fields.pop("generated_tokens", None)
        if set(fields) != set(cls.__dataclass_fields__) or not isinstance(fields["tokens"], list):
            raise ValueError("completions attempt fields differ from the schema")
        fields["tokens"] = tuple(fields["tokens"])
        row = cls(**fields)
        if generated != row.generated_tokens:
            raise ValueError("generated_tokens differs from the token count")
        return row


@dataclass(frozen=True)
class Completions:
    header: CompletionsHeader
    rows: tuple[CompletionRow, ...]
    sha256: str
    path: Path | None = None
    _index: dict[tuple[str, int], CompletionRow] = field(default_factory=dict, compare=False)

    def row(self, task_id: str, sample: int) -> CompletionRow:
        if not self._index:
            self._index.update((row.key, row) for row in self.rows)
        return self._index[(task_id, sample)]

    @property
    def task_ids(self) -> list[str]:
        return sorted({row.task_id for row in self.rows})


def _check_grid(header: CompletionsHeader, rows: Sequence[CompletionRow]) -> None:
    if len(rows) != header.rows:
        raise ValueError("the attempt count differs from the header")
    keys = [row.key for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate attempt in the completions file")
    task_ids = {row.task_id for row in rows}
    if len(task_ids) != header.task_count or task_ids_sha256(task_ids) != header.task_ids_sha256:
        raise ValueError("the attempts' task IDs differ from the header")
    for row in rows:
        if not 0 <= row.sample < header.sampling.samples:
            raise ValueError("sample index outside the header's sample count")
        if row.seed != attempt_seed(header.sampling.seed, row.task_id, row.sample):
            raise ValueError("an attempt's seed differs from the recorded seed rule")
        if row.budget > header.max_new_tokens:
            raise ValueError("an attempt's budget exceeds the protocol cap")
        if row.stop_reason != "context_limit" and (
            row.prompt_tokens + row.budget > header.model_context
        ):
            raise ValueError("an attempt's prompt and budget exceed the model context")


def write_completions(path: Path, header: CompletionsHeader, rows: Sequence[CompletionRow]) -> str:
    """Write a new completions file atomically and return its SHA-256.

    Rows are written in (task ID, sample) order. An existing path is never replaced.
    """
    ordered = sorted(rows, key=lambda row: row.key)
    _check_grid(header, ordered)
    if path.exists():
        raise FileExistsError(f"{path} already exists; completions files are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as stream:
            for record in (header.to_json(), *(row.to_json() for row in ordered)):
                line = canonical_json(record) + b"\n"
                digest.update(line)
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"{path} already exists; completions files are immutable")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest.hexdigest()


def read_completions(path: Path, *, sha256: str | None = None) -> Completions:
    """Read and validate a completions file; with ``sha256``, refuse any other bytes."""
    digest = hashlib.sha256()
    header: CompletionsHeader | None = None
    rows: list[CompletionRow] = []
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                raise ValueError(f"line {number} is oversized or not newline-terminated")
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("a record must be an object")
                if number == 1:
                    header = CompletionsHeader.from_json(raw)
                else:
                    rows.append(CompletionRow.from_json(raw))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"invalid completions record on line {number}: {error}") from error
    if header is None:
        raise ValueError("empty completions file")
    _check_grid(header, rows)
    if [row.key for row in rows] != sorted(row.key for row in rows):
        raise ValueError("attempts are not in (task ID, sample) order")
    identity = digest.hexdigest()
    if sha256 is not None and identity != sha256:
        raise ValueError("completions SHA-256 differs from the expected value")
    return Completions(header, tuple(rows), identity, path)


def validate_against_tasks(completions: Completions, tasks_path: Path) -> list[Task]:
    """Check the header against the task file and return the suite's scored tasks.

    The task file's SHA-256 must be the header's, its scored task IDs must be exactly the
    file's attempts' task IDs, and the exclusions must agree.
    """
    header = completions.header
    if file_sha256(tasks_path) != header.task_file_sha256:
        raise ValueError("the task file's SHA-256 differs from the completions header")
    tasks, exclusions = load_suite_tasks(header.suite, header.split, tasks_path)
    if [task.id for task in tasks] != completions.task_ids:
        raise ValueError("the task file's tasks differ from the completions' tasks")
    if exclusions != [dict(item) for item in header.excluded_tasks]:
        raise ValueError("the suite's exclusions differ from the completions header")
    return tasks
