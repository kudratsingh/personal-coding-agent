"""The remaining stages of a bounded RL-task data pilot, as one driver with hard stops.

A pilot's candidates arrive here after intake and the sealed-test integrity scan
(``tools.pilot_intake``, ``tools.holdout_integrity_scan``). This tool takes them through
reference verification, admission, selection, CPU screening and a report, in one run
directory, with every limit enforced in code. It reuses the reviewed pieces and loosens none
of them.

Subcommands (every path is explicit; nothing is globbed or walked):

``init --out RUN ...``
    Create the run directory (it must not exist) and pin every input by SHA-256: the
    pilot's clear candidates, the surviving RL remainder, the scan's ``summary.json``, the
    intake ``record.json``, the pool spec (its ``verification`` policy), the SFT checkpoint
    with its ``record.json`` (the checkpoint must hash to the record's selected SHA-256) and
    the tokenizer. An input beside a ``SHA256SUMS`` must match its line there. Runs nothing.
    Every run is entered in ``stages-registry.jsonl`` beside it, keyed by the pilot input's
    SHA-256. A second run on the same inputs needs ``--supersedes PRIOR`` naming the latest
    one, which must have stopped on a sandbox host failure; it then starts with PRIOR's
    sandbox calls, generation time and wall time, so no cap is reset by a rerun.
``verify --run RUN --receipt R`` (executes)
    Reference verification: each reference against its public and its hidden tests, once
    per distinct call, through ``tools.task_pool.verify`` (its resumable cache, its retry
    rules for host failures, stalls and resource limits). Calls run through the evaluation
    path of record, ``scratchcoder.eval.harness.isolated_execution``, whose deadline rule
    must equal the pool's (``tools.task_pool.limits_for`` under the spec's verification
    policy) for every call, or nothing runs. Cap: 20,000 sandbox calls.
``admit --run RUN --receipt R`` (executes)
    Admission (``scratchcoder.data.admission``) of the verified candidates: the static
    gate, three fresh-process reference runs scheduled breadth-first and the ``None`` and
    empty-output controls, through ``tools.admission_check``'s planning and recording. An
    admitted task must then be parseable under the reward grammar and its best constant
    answer must earn less than the keep rule's threshold, as for the RL remainder. Cap:
    25,000 calls.
``select --run RUN``
    The first 1,000 admitted candidates in ``rl-candidate-v1`` hash order
    (``scratchcoder.rl.screening.hash_order``), with their positions. Runs nothing.
``generate --run RUN --format-gate GATE``
    CPU screening generation: needs a passing ``scratchcoder sft-format-gate`` record in the
    current gate format, measured on the CPU after ``init``, of the same checkpoint and
    tokenizer; the record's path and SHA-256 are pinned on first use. The existing seeded
    sampler (``scratchcoder.rl.rollout``) draws the
    screening protocol's 8 samples per candidate (temperature 1.0, top-p 1.0, at most 256 new
    tokens, seed base 2^40, chat-v1 prompt with the public test) on ``cpu`` in ``float32`` at
    batch 64, in chunks of 64 candidates, at torch's default thread count. Every chunk's time
    is logged durably, so the 4 h cap counts interrupted invocations too. The first chunk is
    timed: if it projects past 4 h, generation stops; no chunk starts unless the longest one
    measured still fits. The attempts header records the
    device, precision, batch size, the thread count actually in use, the torch, Python and
    NumPy versions, the candidate order's SHA-256 and the seeds' SHA-256. Executes nothing.
``score --run RUN --receipt R`` (executes)
    Rewards through the local reward channel and the unchanged keep rule
    (``scratchcoder.rl.command.screen_attempts``, as ``scratchcoder rl-screen
    --require-header``), under reward grammar v2 only. Cap: 9,000 reward calls.
``report --run RUN``
    The measured outcome and one recommendation by the pilot's fixed decision rule. It
    also runs after a stop, and then draws no decision.

Every ``(executes)`` subcommand refuses to start (exit 6) without ``--receipt``: a passing
execution-window receipt written by ``tools.mac_window_check exec`` for this very command
line in the last 30 minutes (``tools.admission_check``'s rule). The receipt's image ID is
bound to the run on first use; a later stage under another image or evaluator identity is
refused. A run-wide ledger (``budget.jsonl``) counts every sandbox try, retries included,
against the stage's cap and 60,000 for the whole run; a call that would pass either cap is
not made.

Stopping criteria are hard stops with a recorded reason (``stop.json``, exit 4); after one,
only ``report`` runs: a containment failure (a supervisor that could not establish a required
protection: no queued call starts after it); a sandbox host failure that outlasted its
retries; any cap reached (calls, the run budget, 4 h of generation, or 12 h of wall time from
``init``, checked at every stage, call and chunk); under 20% of candidates verified; more than
5% of admission runs unresolved after retries; a failing format gate; a reward grammar or
facts identity other than v2; a generation projection over 4 h. A stopping stage refuses
further calls and waits for those in flight before it records the stop. An interrupted
stage (no stop recorded) resumes from its cache. Every stage writes its outputs as new files and a
``SHA256SUMS``; the report adds one for the whole run.

Exit codes: 0 done; 2 a sealed input refused (nothing opened); 3 the output exists already;
4 a stopping criterion fired; 5 another process holds the run; 6 refused (a precondition
failed; nothing ran).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data import admission
from scratchcoder.data.verifier_scan import scan_task
from scratchcoder.eval.export import task_lines
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.rl.attempts import (
    SCREENING_PROTOCOL,
    ScreeningHeader,
    read_attempts_file,
    write_attempts,
)
from scratchcoder.rl.rewards import NO_PROGRAM, Attempt
from scratchcoder.rl.screening import (
    DEFAULT_THRESHOLD,
    FLOOR,
    MAX_KEPT,
    ORDER_SALT,
    baseline_of,
    hash_order,
)
from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import IMAGE, Limits, Result, Runtime
from tools import admission_check, task_pool
from tools.rl_reward_signal import wilson
from tools.verifier_scan import SealedPathError, guard

FORMAT = "pilot-stages-v1"
STAGES = ("verify", "admit", "select", "generate", "score", "report")
PREDECESSOR = {"verify": None, "admit": "verify", "select": "admit"}
PREDECESSOR |= {"generate": "select", "score": "generate", "report": None}
EXECUTING = frozenset({"verify", "admit", "score"})
"""The stages that run code in the sandbox; each needs a window receipt."""

CALL_CAPS = {"verify": 20_000, "admit": 25_000, "score": 9_000}
"""Sandbox tries per stage, retries included (the pilot plan's limits)."""
RUN_CALL_CAP = 60_000
"""Sandbox tries for the whole run."""
CANDIDATE_CAP = MAX_KEPT
GENERATION_CAP_SECONDS = 4 * 3600.0
RUN_WALL_CAP_SECONDS = 12 * 3600.0
"""End-to-end wall time from ``init`` (less any carried from a superseded run)."""
DRAIN_SECONDS = 900.0
"""How long a stopping stage waits for calls already in flight before recording the stop."""
GENERATION_LOG = "generation.jsonl"
"""Under ``generate/``: one row per invocation start and end and per completed chunk, so the
4 h generation cap counts every invocation, interrupted ones too."""
GATE_PIN = "format-gate-pin.json"
REGISTRY_NAME = "stages-registry.jsonl"
"""Beside the run directories: every run, keyed by the pilot input's SHA-256."""
SUPERSEDABLE = frozenset({"sandbox_failure"})
"""Stops after which a new run on the same inputs may replace the stopped one: only a host
failure that outlasted its retries (the pilot plan reruns only for host failures)."""
MIN_VERIFIED_SHARE = 0.20
MAX_UNRESOLVED_SHARE = 0.05

DEVICE = "cpu"
PRECISION = "float32"
BATCH_SIZE = 64
CHUNK_TASKS = 64
SAMPLES = int(SCREENING_PROTOCOL["samples"])
TEMPERATURE = float(SCREENING_PROTOCOL["temperature"])
TOP_P = float(SCREENING_PROTOCOL["top_p"])
MAX_NEW_TOKENS = int(SCREENING_PROTOCOL["max_new_tokens"])
SEED_BASE = int(SCREENING_PROTOCOL["seed_base"])
ADMIT_WORKERS = 4
"""``tools.admission_check``'s default."""
REWARD_WORKERS = 8
REWARD_TIMEOUT_SECONDS = 5600.0
"""Per batch of 256 attempts, as run 1b's screening (``--timeout 5600 --workers 8``)."""
EXPECTED_GRAMMAR = "reward-grammar-v2"
EXPECTED_FACTS = "rl-task-facts-v2"

POOL_FLOOR = FLOOR
MARGIN = Fraction(115, 100)
YIELD_THRESHOLD = 0.15
ALL_ZERO_THRESHOLD = 0.60
FORMATTING_THRESHOLD = 0.50
EXECUTION_THRESHOLD = 0.10
LOW_CONFIDENCE_SCREENED = 300
FURTHER_SLICE_START = 30_000
FURTHER_SLICE_MAX_ROWS = 20_000
SHARD_UNUSED_ROWS = 70_000
"""Rows 30,000 to 99,999 of the pinned shard."""
INTAKE_DEDUP_RULES = ("dev_contamination", "near_duplicate", "dev_paraphrase", "dev_family")
INTAKE_DEDUP_LIMIT = 0.75

EXIT_SEALED, EXIT_EXISTS, EXIT_STOP, EXIT_LOCKED, EXIT_REFUSED = 2, 3, 4, 5, 6
INPUTS = (
    "pilot",
    "remainder",
    "scan_summary",
    "intake_record",
    "spec",
    "checkpoint",
    "checkpoint_record",
    "tokenizer",
)
NUMERICS = (
    "CPU float32 screening and run 1b's CUDA bfloat16 screening are different numerical "
    "executions of the same intended protocol; distributional equivalence has not been shown."
)
HEURISTICS = (
    "The thresholds (15% yield, 60% all-zero, 50% formatting, 10% execution) are planning "
    "heuristics: they guide diagnosis and never permit ignoring failures below them."
)
STOP_CRITERIA = {
    "containment_failure": "the sandbox supervisor could not establish a required protection",
    "sandbox_failure": "a sandbox host failure outlasted its retries",
    "cap_reached": "a sandbox call cap, the run budget or the run's 12 h wall time was reached",
    "generation_time_cap": "screening generation reached its 4 h wall-time cap",
    "under_verified": "under 20% of the candidates verified",
    "admission_unresolved": "more than 5% of admission runs unresolved after retries",
    "format_gate_failed": "the CPU format gate failed",
    "grammar_identity": "screening rewards under a grammar or facts identity other than v2",
    "time_projection": "the first chunk projects generation past 4 h",
}


class Refused(Exception):
    """A precondition failed. Nothing ran and no stop is recorded."""


class Stop(Exception):
    """A stopping criterion fired. The stage records ``result`` and the run stops."""

    def __init__(self, reason: str, detail: str, result: Mapping[str, Any] | None = None):
        super().__init__(f"{reason}: {detail}")
        if reason not in STOP_CRITERIA:
            raise ValueError(f"unknown stopping criterion {reason!r}")
        self.reason = reason
        self.detail = detail
        self.result = dict(result or {})


class CapReached(Exception):
    """A sandbox call would pass its stage's cap or the run's budget; it was not made."""


class ContainmentFailure(Exception):
    """The sandbox supervisor could not establish a required protection."""


UNPROTECTED = "supervisor_unprotected"


def tripwire(
    execute: Callable[[str, str, str], Result], budget: Budget
) -> Callable[[str, str, str], Result]:
    """``execute``, stopping the run at the first call whose supervisor could not establish
    a required protection. The path of record would retry it as a host failure (no
    candidate code ran); the pilot treats it as a containment failure and stops. The budget
    is tripped first, so no queued call starts afterwards."""

    def checked(code: str, tests: str, mode: str) -> Result:
        result = execute(code, tests, mode)
        if result.status == "infrastructure_error" and result.detail == UNPROTECTED:
            budget.trip()
            raise ContainmentFailure("the sandbox supervisor reported a missing protection")
        return result

    return checked


def _log(message: str) -> None:
    print(f"pilot-stages: {message}", file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, data: bytes) -> str:
    """Write a new file atomically (never replacing one); return its SHA-256."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists; outputs are never replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, allow_nan=False).encode() + b"\n" for row in rows
    )


def _sums(directory: Path, *, recursive: bool) -> bytes:
    """``SHA256SUMS`` lines for the files under ``directory`` (locks and temporaries left
    out), as relative paths in sorted order."""
    files = directory.rglob("*") if recursive else directory.iterdir()
    lines = []
    for path in sorted(files):
        name = path.relative_to(directory).as_posix()
        if (
            not path.is_file()
            or path.name in {"SHA256SUMS", ".lock"}
            or path.name.endswith((".lock", ".tmp"))
        ):
            continue
        lines.append(f"{_sha256(path)}  {name}\n")
    return "".join(lines).encode()


# --------------------------------------------------------------------------------------
# The run directory and its ledger.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs.json"

    @property
    def budget(self) -> Path:
        return self.root / "budget.jsonl"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime.json"

    @property
    def stop(self) -> Path:
        return self.root / "stop.json"

    def stage(self, name: str) -> Path:
        return self.root / name

    def result_path(self, name: str) -> Path:
        return self.root / name / "result.json"

    def result(self, name: str) -> dict[str, Any] | None:
        path = self.result_path(name)
        return _json(path) if path.exists() else None

    def output(self, stage: str, name: str) -> Path:
        """A completed stage's output, checked against the SHA-256 its result recorded."""
        result = self.result(stage)
        if result is None or result.get("status") != "completed":
            raise Refused(f"stage {stage} has not completed")
        path = self.root / stage / name
        if _sha256(path) != result["outputs"][name]:
            raise Refused(f"{stage}/{name} differs from the SHA-256 its stage recorded")
        return path


def ledger_counts(path: Path) -> Counter[str]:
    """Sandbox tries recorded per stage. A line is one try, or ``calls`` tries carried
    from a superseded run. A torn last line still counts (towards ``unknown``), so an
    interrupted write never frees budget."""
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    for line in raw[:cut].splitlines():
        row = json.loads(line)
        calls = row.get("calls", 1)
        if type(calls) is not int or calls < 0:
            raise ValueError(f"{path}: malformed ledger row")
        counts[str(row["stage"])] += calls
    if cut != len(raw):
        counts["unknown"] += 1
    return counts


class Budget:
    """The run-wide sandbox ledger. ``wrap`` returns a function that appends one line per
    call before making it, and raises instead of making a call that would pass the stage's
    cap, the run's cap or the run's wall-time deadline (:class:`CapReached`), or any call
    after a containment failure (:class:`ContainmentFailure`) or once the stage is stopping.
    It counts the calls in flight so a stopping stage can wait for them."""

    def __init__(
        self,
        path: Path,
        stage: str,
        *,
        cap: int,
        run_cap: int = RUN_CALL_CAP,
        deadline: datetime | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.path = path
        self.stage = stage
        self.cap = cap
        self.run_cap = run_cap
        self.deadline = deadline
        self.now = now
        counts = ledger_counts(path)
        self.stage_used = counts[stage]
        self.total = sum(counts.values())
        self.made = 0
        self.active = 0
        self.tripped = False
        self.closed = False
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)

    def require(self, planned: int) -> None:
        """Refuse a plan that cannot fit: ``planned`` is the fewest tries it needs."""
        if self.stage_used + planned > self.cap or self.total + planned > self.run_cap:
            raise CapReached(
                f"{self.stage} needs at least {planned} sandbox calls; it has used "
                f"{self.stage_used} of {self.cap}, and the run {self.total} of {self.run_cap}"
            )

    def charge(self) -> None:
        with self._lock:
            if self.tripped:
                raise ContainmentFailure("a containment failure was reported; no further calls")
            if self.closed:
                raise CapReached(f"{self.stage} is stopping; no further calls")
            if self.deadline is not None and self.now() >= self.deadline:
                raise CapReached(f"run wall time: the 12 h deadline {self.deadline} has passed")
            if self.stage_used + 1 > self.cap or self.total + 1 > self.run_cap:
                raise CapReached(
                    f"{self.stage} reached its cap ({self.stage_used} of {self.cap} calls; "
                    f"the run {self.total} of {self.run_cap})"
                )
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(descriptor, (json.dumps({"stage": self.stage}) + "\n").encode())
            finally:
                os.close(descriptor)
            self.stage_used += 1
            self.total += 1
            self.made += 1
            self.active += 1

    def release(self) -> None:
        with self._lock:
            self.active -= 1
            self._idle.notify_all()

    def trip(self) -> None:
        with self._lock:
            self.tripped = True

    def close_and_drain(self, timeout: float = DRAIN_SECONDS) -> int:
        """Refuse every further call, then wait up to ``timeout`` seconds for the calls in
        flight to end; return how many are still running."""
        end = time.monotonic() + timeout
        with self._lock:
            self.closed = True
            while self.active > 0 and (left := end - time.monotonic()) > 0:
                self._idle.wait(left)
            return self.active

    def wrap(self, function: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any) -> Any:
            self.charge()
            try:
                return function(*args)
            finally:
                self.release()

        return counted

    def record(self) -> dict[str, Any]:
        return {
            "calls_in_flight": self.active,
            "tripped": self.tripped,
            "calls_this_invocation": self.made,
            "stage_calls": self.stage_used,
            "stage_cap": self.cap,
            "run_calls": self.total,
            "run_cap": self.run_cap,
        }


# --------------------------------------------------------------------------------------
# init: pin the inputs.
# --------------------------------------------------------------------------------------


def _listed_sum(path: Path) -> str | None:
    """The SHA-256 a ``SHA256SUMS`` beside ``path`` lists for it, if one does."""
    sums = path.parent / "SHA256SUMS"
    if not sums.is_file():
        return None
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if name.strip() == path.name:
            return digest.strip()
    return None


def _train_tasks(path: Path, label: str) -> list[Task]:
    tasks = list(load_tasks(path))
    bad = [task.id for task in tasks if task.split != "train"]
    if bad:
        raise Refused(f"{label}: {len(bad)} tasks are not train-split tasks")
    return tasks


def init(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() or out.is_symlink():
        print(f"refused: {out} already exists", file=sys.stderr)
        return EXIT_EXISTS
    paths = {name: Path(getattr(args, name)) for name in INPUTS}
    registry = out.parent / REGISTRY_NAME
    supersedes: Path | None = args.supersedes
    try:
        guard([*paths.values(), out, registry, *([supersedes] if supersedes else [])])
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_SEALED
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / (REGISTRY_NAME + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            record = _pin_inputs(paths)
            carried = _carried(registry, record["files"]["pilot"]["sha256"], supersedes)
        except Refused as error:
            print(f"refused: {error}", file=sys.stderr)
            return EXIT_REFUSED
        if carried is not None:
            record["carried"] = carried
        out.mkdir()
        run = RunDir(out)
        _write_new(run.inputs, _json_bytes(record))
        seed = [] if carried is None else carried["ledger_rows"]
        _write_new(run.budget, _jsonl_bytes(seed))
        entry = {
            "pilot_sha256": record["files"]["pilot"]["sha256"],
            "run": str(out.resolve()),
            "created_at": record["created_at"],
            "supersedes": None if carried is None else carried["from"],
        }
        with registry.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
    _log(f"initialized {out}")
    return 0


def _registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def generation_seconds_used(run: RunDir) -> float:
    """Screening generation time this run has used, with any carried from the run it
    superseded. An invocation that ended without a terminal record is charged its longest
    measured chunk as well, for the chunk it was in when it was interrupted."""
    carried = float(_json(run.inputs).get("carried", {}).get("generation_seconds", 0.0))
    path = run.stage("generate") / GENERATION_LOG
    if not path.exists():
        return carried
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    chunks = [float(row["seconds"]) for row in rows if row["event"] == "chunk"]
    longest = max(chunks, default=0.0)
    open_invocations = 0
    for row in rows:
        if row["event"] == "start":
            open_invocations += 1
        elif row["event"] in {"end", "stopped"}:
            open_invocations = max(0, open_invocations - 1)
    return carried + sum(chunks) + longest * open_invocations


def _created(run: RunDir) -> datetime:
    return datetime.fromisoformat(_json(run.inputs)["created_at"])


def run_deadline(run: RunDir) -> datetime:
    """The run's end-to-end wall-time deadline: 12 h from init, less carried wall time."""
    carried = float(_json(run.inputs).get("carried", {}).get("wall_seconds", 0.0))
    return _created(run) + timedelta(seconds=RUN_WALL_CAP_SECONDS - carried)


def _carried(registry: Path, pilot_sha256: str, supersedes: Path | None) -> dict[str, Any] | None:
    """What a new run on these inputs carries from the run it supersedes, or ``None`` for a
    first run. A second run needs ``--supersedes`` naming the latest run on the same inputs,
    and that run must have stopped on a host failure."""
    prior = [entry for entry in _registry(registry) if entry["pilot_sha256"] == pilot_sha256]
    if not prior:
        if supersedes is not None:
            raise Refused("--supersedes names no earlier run on these inputs")
        return None
    latest = prior[-1]["run"]
    if supersedes is None:
        raise Refused(
            f"a run on these inputs exists ({latest}); a new one needs --supersedes naming it "
            "after a host failure"
        )
    if str(supersedes.resolve()) != latest:
        raise Refused(f"--supersedes must name the latest run on these inputs ({latest})")
    run = RunDir(Path(latest))
    if not run.stop.is_file() or _json(run.stop)["reason"] not in SUPERSEDABLE:
        raise Refused("only a run stopped by a sandbox host failure can be superseded")
    stop = _json(run.stop)
    counts = ledger_counts(run.budget)
    inputs = _json(run.inputs)
    wall = float(inputs.get("carried", {}).get("wall_seconds", 0.0))
    wall += (datetime.fromisoformat(stop["stopped_at"]) - _created(run)).total_seconds()
    return {
        "from": latest,
        "stop_reason": stop["reason"],
        "calls": dict(sorted(counts.items())),
        "generation_seconds": generation_seconds_used(run),
        "wall_seconds": wall,
        "ledger_rows": [
            {"stage": stage, "calls": calls, "carried_from": latest}
            for stage, calls in sorted(counts.items())
        ],
    }


def _pin_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise Refused(f"--{name.replace('_', '-')} {path} is not a file")
        digest = _sha256(path)
        listed = _listed_sum(path)
        if listed is not None and listed != digest:
            raise Refused(f"{path} differs from its line in the SHA256SUMS beside it")
        files[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "sha256sums_checked": listed is not None,
        }
    pilot = _train_tasks(paths["pilot"], "pilot")
    remainder = _train_tasks(paths["remainder"], "remainder")
    overlap = {task.id for task in pilot} & {task.id for task in remainder}
    if overlap:
        raise Refused(f"{len(overlap)} task IDs are both pilot candidates and remainder tasks")
    scan = _json(paths["scan_summary"])
    if scan.get("format") != "holdout-integrity-scan-v1":
        raise Refused("the scan summary is not a holdout-integrity-scan-v1 summary")
    subjects = scan["counts"]["subjects"]
    for label, tasks in (("pilot", pilot), ("remainder", remainder)):
        if subjects[label]["clear"] != len(tasks):
            raise Refused(f"the scan's clear {label} count differs from the {label} file")
    intake = _json(paths["intake_record"])
    if intake.get("kind") != "pilot-intake" or intake.get("sealed_files_opened") is not False:
        raise Refused("the intake record is not a pilot-intake record that opened nothing sealed")
    if intake["counts"]["kept"] != subjects["pilot"]["total"]:
        raise Refused("the intake's kept count differs from the scan's pilot subjects")
    spec = _json(paths["spec"])
    verification = spec["verification"]
    for key in ("timeout_seconds", "io_seconds_per_case", "memory_mb", "workers"):
        if key not in verification:
            raise Refused(f"the spec's verification policy has no {key}")
    selected = _json(paths["checkpoint_record"])["sft"]["selected"]["sha256"]
    if files["checkpoint"]["sha256"] != selected:
        raise Refused("the checkpoint does not hash to its record's selected SHA-256")
    commit, dirty = git_state()
    return {
        "format": FORMAT,
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "files": files,
        "counts": {"pilot": len(pilot), "remainder": len(remainder)},
        "limits": {
            "call_caps": CALL_CAPS,
            "run_call_cap": RUN_CALL_CAP,
            "candidate_cap": CANDIDATE_CAP,
            "generation_cap_seconds": GENERATION_CAP_SECONDS,
            "min_verified_share": MIN_VERIFIED_SHARE,
            "max_unresolved_share": MAX_UNRESOLVED_SHARE,
        },
    }


def load_inputs(run: RunDir, names: Sequence[str]) -> dict[str, Path]:
    """The pinned inputs ``names``, each guarded and checked against its pinned SHA-256."""
    files = _json(run.inputs)["files"]
    paths = {name: Path(files[name]["path"]) for name in names}
    guard(list(paths.values()))
    for name, path in paths.items():
        if _sha256(path) != files[name]["sha256"]:
            raise Refused(f"input {name} changed since init")
    return paths


# --------------------------------------------------------------------------------------
# Stage plumbing: locks, receipts, results, stops.
# --------------------------------------------------------------------------------------


@dataclass
class Context:
    """What a stage body gets: the run, its ledger and (for an executing stage) the image."""

    run: RunDir
    stage: str
    image_id: str | None = None
    budget: Budget | None = None
    options: dict[str, Any] = field(default_factory=dict)


def _bind_runtime(run: RunDir, image_id: str, stage: str) -> admission.Identities:
    ids = admission_check.identities(image_id)
    record = {
        "image": ids.image,
        "image_id": ids.image_id,
        "evaluator_sha256": ids.evaluator_sha256,
    }
    if run.runtime.exists():
        bound = _json(run.runtime)
        if {key: bound.get(key) for key in record} != record:
            raise Refused(
                "this stage's image or evaluator identity differs from the one the run is bound to"
            )
    else:
        atomic_json(run.runtime, {**record, "bound_by": stage, "bound_at": utc_now()})
    return ids


def _finish(run: RunDir, stage: str, result: dict[str, Any]) -> None:
    commit, dirty = git_state()
    result = {"format": FORMAT, "stage": stage, "git_commit": commit, "dirty": dirty, **result}
    result.setdefault("finished_at", utc_now())
    _write_new(run.result_path(stage), _json_bytes(result))
    _write_new(run.stage(stage) / "SHA256SUMS", _sums(run.stage(stage), recursive=False))


def _record_stop(run: RunDir, stage: str, stop: Stop, budget: Budget | None) -> None:
    ledger = None if budget is None else budget.record()
    record = {
        "format": FORMAT,
        "stage": stage,
        "reason": stop.reason,
        "criterion": STOP_CRITERIA[stop.reason],
        "detail": stop.detail,
        "stopped_at": utc_now(),
        "budget": ledger,
    }
    run.stage(stage).mkdir(parents=True, exist_ok=True)
    _finish(run, stage, {**stop.result, "status": "stopped", "stop": record})
    _write_new(run.stop, _json_bytes(record))
    _log(f"STOPPED at {stage}: {record['criterion']} ({stop.detail})")


Body = Callable[[Context], dict[str, Any]]


def run_stage(
    stage: str,
    root: Path,
    body: Body,
    *,
    argv: Sequence[str],
    receipt: Path | None = None,
    extra_inputs: Sequence[Path] = (),
    options: Mapping[str, Any] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    run = RunDir(root)
    try:
        guard([root, *([receipt] if receipt is not None else []), *extra_inputs])
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_SEALED
    if not run.inputs.is_file():
        print(f"refused: {root} is not an initialized pilot-stages run", file=sys.stderr)
        return EXIT_REFUSED
    lock = (root / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("refused: another process holds this run", file=sys.stderr)
        return EXIT_LOCKED
    try:
        return _run_locked(stage, run, body, argv, receipt, options or {}, now)
    finally:
        lock.close()


def _run_locked(
    stage: str,
    run: RunDir,
    body: Body,
    argv: Sequence[str],
    receipt: Path | None,
    options: Mapping[str, Any],
    now: Callable[[], datetime],
) -> int:
    if run.result_path(stage).exists():
        print(f"refused: stage {stage} already has a result", file=sys.stderr)
        return EXIT_EXISTS
    if stage != "report" and run.stop.exists():
        print(f"refused: the run stopped ({_json(run.stop)['reason']})", file=sys.stderr)
        return EXIT_REFUSED
    if stage == "report" and not run.stop.exists():
        result = run.result("score")
        if result is None or result.get("status") != "completed":
            print("refused: report needs a completed score stage or a stop", file=sys.stderr)
            return EXIT_REFUSED
    before = PREDECESSOR[stage]
    if before is not None:
        result = run.result(before)
        if result is None or result.get("status") != "completed":
            print(f"refused: stage {before} has not completed", file=sys.stderr)
            return EXIT_REFUSED
    clock_now: Callable[[], datetime] = options.get("now") or now
    context = Context(run, stage, options=dict(options))
    context.options["now"] = clock_now
    try:
        if stage != "report" and clock_now() >= run_deadline(run):
            raise Stop(
                "cap_reached", f"run wall time: the 12 h deadline {run_deadline(run)} has passed"
            )
        if stage in EXECUTING:
            if receipt is None:
                raise Refused(f"{stage} executes code and needs --receipt")
            try:
                context.image_id = admission_check._receipt_image(receipt, now(), list(argv))
            except (ValueError, KeyError, OSError) as error:
                raise Refused(f"window receipt: {error}") from error
            _bind_runtime(run, context.image_id, stage)
            context.budget = Budget(
                run.budget,
                stage,
                cap=CALL_CAPS[stage],
                run_cap=RUN_CALL_CAP,
                deadline=run_deadline(run),
                now=clock_now,
            )
        run.stage(stage).mkdir(parents=True, exist_ok=True)
        try:
            try:
                result = body(context)
            except BaseException:
                # Nothing more starts, and calls still in flight (reward containers too)
                # end before the stop is recorded and the run's lock is released.
                if context.budget is not None:
                    left = context.budget.close_and_drain()
                    if left:
                        _log(f"{left} calls were still running after {DRAIN_SECONDS:.0f} s")
                raise
        except CapReached as error:
            raise Stop("cap_reached", str(error), _partial(context)) from error
        except ContainmentFailure as error:
            raise Stop("containment_failure", str(error), _partial(context)) from error
    except Refused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_SEALED
    except Stop as stop:
        if context.budget is not None:
            stop.result.setdefault("budget", context.budget.record())
        _record_stop(run, stage, stop, context.budget)
        return EXIT_STOP
    if context.budget is not None:
        result["budget"] = context.budget.record()
    _finish(run, stage, {**result, "status": "completed"})
    _log(f"{stage} completed")
    return 0


def _partial(context: Context) -> dict[str, Any]:
    return {} if context.budget is None else {"budget": context.budget.record()}


# --------------------------------------------------------------------------------------
# verify.
# --------------------------------------------------------------------------------------


def _text_bytes(text: str) -> int:
    return len(text.encode("utf-8", errors="surrogatepass"))


def _unverifiable(task: Task) -> str | None:
    """Why a task's reference cannot be verified at all, before any call."""
    from dataclasses import replace

    if task.solution is None:
        return "no_reference"
    if not admission.runnable(task) or not admission.runnable(
        replace(task, hidden_tests=task.public_tests)
    ):
        return "not_runnable"
    limit = Limits().input_bytes
    if any(
        _text_bytes(task.solution) + _text_bytes(tests) > limit
        for tests in (task.public_tests, task.hidden_tests)
    ):
        return "oversized_for_sandbox"
    return None


def verify_stage(context: Context) -> dict[str, Any]:
    from scratchcoder.eval import harness

    run, budget, image_id = context.run, context.budget, context.image_id
    assert budget is not None and image_id is not None
    files = load_inputs(run, ("pilot", "spec"))
    verification = _json(files["spec"])["verification"]
    tasks = list(load_tasks(files["pilot"]))
    reasons: dict[str, str] = {}
    calls: list[tuple[str, str, str]] = []
    for task in tasks:
        refused = _unverifiable(task)
        if refused is not None:
            reasons[task.id] = refused
            continue
        assert task.solution is not None
        for tests in (task.public_tests, task.hidden_tests):
            if task_pool.limits_for(verification, tests, task.mode) != harness.execution_limits(
                tests, task.mode
            ):
                raise Refused(
                    "the evaluation path's limits differ from the pool's deadline rule under "
                    "the spec's verification policy"
                )
            calls.append((task.solution, tests, task.mode))
    directory = run.stage("verify")
    cache = directory / f"results-{task_pool.policy_hash(verification)}.jsonl"
    policy_path = cache.with_suffix(".policy.json")
    policy = {
        **task_pool.runtime_policy({"image": IMAGE, **asdict(Runtime())}),
        "image_id": image_id,
        "execute": "scratchcoder.eval.harness.isolated_execution",
        "evaluator_sha256": admission_check.identities(image_id).evaluator_sha256,
    }
    if policy_path.exists():
        if _json(policy_path) != policy:
            raise Refused("the verdict cache was produced under another runtime or evaluator")
    else:
        atomic_json(policy_path, policy)
    keys = {task_pool.call_key(*call) for call in calls}
    cached = {row["key"] for row in task_pool._read_cache(cache)}
    budget.require(len(keys - cached))
    execute = context.options.get("execute") or harness.isolated_execution

    def deadline(tests: str, mode: str) -> float:
        return task_pool.limits_for(verification, tests, mode).timeout_seconds

    try:
        results, counters = task_pool.verify(
            calls,
            cache,
            execute=budget.wrap(tripwire(execute, budget)),
            deadline=deadline,
            workers=int(verification["workers"]),
            progress=_log,
        )
    except RuntimeError as error:
        if str(error).startswith("sandbox unavailable or failing"):
            raise Stop("sandbox_failure", str(error), _partial(context)) from error
        raise
    verified: list[Task] = []
    outcomes: Counter[str] = Counter()
    for task in tasks:
        if task.id in reasons:
            continue
        assert task.solution is not None
        for part, tests in (("public", task.public_tests), ("hidden", task.hidden_tests)):
            outcome: Result = results[task_pool.call_key(task.solution, tests, task.mode)]
            outcomes[f"{part}:{failure_category(outcome)}"] += 1
            if not outcome.passed and task.id not in reasons:
                reasons[task.id] = f"{part}:{failure_category(outcome)}"
        if task.id not in reasons:
            verified.append(task)
    digest = _write_new(directory / "verified.jsonl", task_lines(verified))
    share = len(verified) / len(tasks) if tasks else None
    result = {
        "tasks": len(tasks),
        "verified": len(verified),
        "verified_share": share,
        "dropped_by_reason": dict(sorted(Counter(reasons.values()).items())),
        "verdicts_by_part": dict(sorted(outcomes.items())),
        "distinct_calls": len(keys),
        "counters": counters,
        "policy": {
            "verification": verification,
            "policy_hash": task_pool.policy_hash(verification),
            "image_id": image_id,
        },
        "outputs": {"verified.jsonl": digest},
    }
    if share is not None and share < MIN_VERIFIED_SHARE:
        raise Stop(
            "under_verified",
            f"{len(verified)} of {len(tasks)} candidates verified",
            {**result, **_partial(context)},
        )
    return result


# --------------------------------------------------------------------------------------
# admit.
# --------------------------------------------------------------------------------------


def admit_stage(context: Context) -> dict[str, Any]:
    from scratchcoder.eval import harness
    from scratchcoder.eval.parallel import HostUnavailableError
    from scratchcoder.sandbox.reward import REWARD_GRAMMAR_VERSION, unsupported_reason

    run, budget, image_id = context.run, context.budget, context.image_id
    assert budget is not None and image_id is not None
    tasks = list(load_tasks(run.output("verify", "verified.jsonl")))
    by_id = {task.id: task for task in tasks}
    ids = admission_check.identities(image_id)
    cache = run.stage("admit") / "cache.jsonl"
    admission_check._repair_tail(cache)
    runs = admission_check.pending_runs({"pilot": by_id}, cache, ids)
    budget.require(len(runs))
    execute = context.options.get("execute") or harness.isolated_execution
    _log(f"admission: {len(runs)} runs pending")
    try:
        admission_check.record_runs(
            runs,
            cache,
            ids,
            run_id=f"pilot-stages:{run.root.name}",
            workers=ADMIT_WORKERS,
            call=budget.wrap(tripwire(execute, budget)),
        )
    except HostUnavailableError as error:
        raise Stop("sandbox_failure", str(error), _partial(context)) from error
    rows = admission.read_cache(cache)
    done = admission.usable(rows, by_id, ids)
    total = sum(len(admission.plan(task)) for task in tasks)
    unresolved = total - len(done)
    share = unresolved / total if total else 0.0
    decisions = [
        admission.decide(by_id[task_id], scan_task(asdict(by_id[task_id])), done)
        for task_id in sorted(by_id)
    ]
    admitted: list[Task] = []
    grammar: Counter[str] = Counter()
    unparseable: Counter[str] = Counter()
    decision_rows = []
    for decision in decisions:
        row = {**decision.to_row(), "reward_grammar": REWARD_GRAMMAR_VERSION}
        if decision.status == "admitted":
            task = by_id[decision.id]
            baseline = baseline_of(task)
            # The best constant answer itself can be a hidden expected value: not written.
            row.update(parseable=baseline.parseable, baseline_reward=baseline.reward)
            if not baseline.parseable:
                refused = unsupported_reason(task)
                cause = None if refused is None else refused.reason
                row["dropped"] = "not_parseable"
                row["unsupported"] = cause
                grammar["not_parseable"] += 1
                unparseable[str(cause)] += 1
            elif baseline.reward is not None and baseline.reward >= DEFAULT_THRESHOLD:
                row["dropped"] = "constant_baseline"
                grammar["constant_baseline"] += 1
            else:
                admitted.append(task)
        decision_rows.append(row)
    order = {task.id: position for position, task in enumerate(tasks)}
    admitted.sort(key=lambda task: order[task.id])
    directory = run.stage("admit")
    outputs = {
        "decisions-pilot.jsonl": _write_new(
            directory / "decisions-pilot.jsonl", _jsonl_bytes(decision_rows)
        ),
        "admitted.jsonl": _write_new(directory / "admitted.jsonl", task_lines(admitted)),
    }
    result = {
        "tasks": len(tasks),
        "admission": admission.ADMISSION_VERSION,
        "identities": asdict(ids),
        "runs_total": total,
        "runs_satisfied": len(done),
        "runs_unresolved": unresolved,
        "unresolved_share": share,
        "runs_this_invocation": len(runs),
        "cache_rows": len(rows),
        "cache_rows_unresolved": sum(not row.get("resolved") for row in rows),
        "decisions": admission.union_counts(decisions),
        "reward_grammar": REWARD_GRAMMAR_VERSION,
        "threshold": DEFAULT_THRESHOLD,
        "after_admission_dropped": dict(sorted(grammar.items())),
        "not_parseable_by_cause": dict(sorted(unparseable.items())),
        "admitted": len(admitted),
        "outputs": outputs,
    }
    if share > MAX_UNRESOLVED_SHARE:
        raise Stop(
            "admission_unresolved",
            f"{unresolved} of {total} runs unresolved",
            {**result, **_partial(context)},
        )
    return result


# --------------------------------------------------------------------------------------
# select.
# --------------------------------------------------------------------------------------


def order_sha256(ids: Sequence[str]) -> str:
    """The SHA-256 of the candidate IDs in screening order, one per line."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def select_stage(context: Context) -> dict[str, Any]:
    run = context.run
    admitted = list(load_tasks(run.output("admit", "admitted.jsonl")))
    ordered = sorted(admitted, key=lambda task: hash_order(task.id))
    selected = ordered[:CANDIDATE_CAP]
    directory = run.stage("select")
    ids = [task.id for task in selected]
    outputs = {
        "candidates.jsonl": _write_new(
            directory / "candidates.jsonl",
            _jsonl_bytes({"id": task.id, "source": task.source} for task in selected),
        ),
        "tasks.jsonl": _write_new(directory / "tasks.jsonl", task_lines(selected)),
        "order.jsonl": _write_new(
            directory / "order.jsonl",
            _jsonl_bytes(
                {"position": position, "id": task.id, "order_key": hash_order(task.id).hex()}
                for position, task in enumerate(selected)
            ),
        ),
    }
    return {
        "admitted": len(admitted),
        "selected": len(selected),
        "reserve": len(admitted) - len(selected),
        "cap": CANDIDATE_CAP,
        "order": f"sha256('{ORDER_SALT}|' + task id)",
        "candidate_order_sha256": order_sha256(ids),
        "outputs": outputs,
    }


# --------------------------------------------------------------------------------------
# generate.
# --------------------------------------------------------------------------------------


class Generator(Protocol):
    """Draws the screening samples of a chunk of candidates, task-major."""

    tokenizer_sha256: str

    def __call__(self, tasks: Sequence[Task]) -> list[Attempt]: ...

    def environment(self) -> dict[str, Any]: ...


class CheckpointGenerator:
    """The SFT checkpoint on the CPU through ``scratchcoder.rl.rollout.screening_rollouts``.

    ``load_checkpoint`` pins torch to one thread; the approved backend is torch's default
    thread count, so the count from before loading is restored, and the count in use at
    every chunk is recorded."""

    def __init__(self, checkpoint: Path, tokenizer: Path) -> None:
        import torch

        from scratchcoder.eval.checkpoint import load_checkpoint
        from scratchcoder.tokenizer.bulk import FastEncoder

        self.default_threads = torch.get_num_threads()
        self.model, self.tokenizer = load_checkpoint(checkpoint, tokenizer, device=DEVICE)
        torch.set_num_threads(self.default_threads)
        self.encoder = FastEncoder(self.tokenizer)
        self.tokenizer_sha256 = self.tokenizer.sha256
        self.threads: list[int] = []

    def __call__(self, tasks: Sequence[Task]) -> list[Attempt]:
        import torch

        from scratchcoder.rl.rollout import screening_rollouts

        self.threads.append(torch.get_num_threads())
        with torch.inference_mode():
            rollouts = screening_rollouts(
                self.model,
                list(tasks),
                self.encoder,
                self.tokenizer,
                batch_size=BATCH_SIZE,
                precision=PRECISION,
                samples=SAMPLES,
                seed=SEED_BASE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        return [rollout.attempt for rollout in rollouts]

    def environment(self) -> dict[str, Any]:
        import numpy
        import torch

        return {
            "torch_threads": self.threads[0] if self.threads else torch.get_num_threads(),
            "torch_threads_per_chunk_distinct": sorted(set(self.threads)),
            "torch_threads_default": self.default_threads,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        }


def seeds_sha256(ids: Sequence[str]) -> str:
    """The SHA-256 of every screening seed, task-major, one decimal per line."""
    from scratchcoder.eval.harness import attempt_seed

    seeds = [str(attempt_seed(SEED_BASE, task_id, s)) for task_id in ids for s in range(SAMPLES)]
    return hashlib.sha256("\n".join(seeds).encode()).hexdigest()


def _check_gate(
    run: RunDir, gate_path: Path, checkpoint_sha256: str, tokenizer_sha256: str
) -> dict[str, Any]:
    """The CPU format gate of this checkpoint, measured after this run was initialized, in
    the current gate format. Its path and SHA-256 are pinned on first use; a later
    invocation must present the same file."""
    from scratchcoder.eval import format_gate

    digest = _sha256(gate_path)
    pin_path = run.stage("generate") / GATE_PIN
    pin = {"path": str(gate_path.resolve()), "sha256": digest}
    if pin_path.exists() and _json(pin_path) != pin:
        raise Refused("this run is pinned to another format-gate record")
    gate = _json(gate_path)
    if not isinstance(gate, dict) or "passed" not in gate:
        raise Refused("the format-gate file is not a gate record")
    if gate.get("format") != format_gate.FORMAT or gate.get("version") != format_gate.VERSION:
        raise Refused("the format-gate record is not in the current gate format and version")
    if gate.get("device") != DEVICE:
        raise Refused(f"the format gate ran on {gate.get('device')!r}, not on the CPU")
    inputs = gate.get("inputs", {})
    if inputs.get("checkpoint_sha256") != checkpoint_sha256:
        raise Refused("the format gate measured another checkpoint")
    if inputs.get("tokenizer_sha256") != tokenizer_sha256:
        raise Refused("the format gate used another tokenizer")
    try:
        created = datetime.fromisoformat(str(gate.get("created_at")))
    except ValueError as error:
        raise Refused("the format-gate record has no valid created_at") from error
    if created <= _created(run):
        raise Refused("the format gate was measured before this run was initialized")
    if not pin_path.exists():
        atomic_json(pin_path, pin)
    return {
        **pin,
        "passed": gate["passed"] is True,
        "failed_checks": gate.get("failed_checks"),
        "checkpoint_sha256": checkpoint_sha256,
        "device": gate.get("device"),
        "torch_threads": gate.get("torch_threads"),
        "created_at": gate.get("created_at"),
        "format": gate.get("format"),
        "version": gate.get("version"),
    }


def _append_log(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**row, "at": utc_now()}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _longest_chunk(path: Path) -> float:
    if not path.exists():
        return 0.0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return max((float(row["seconds"]) for row in rows if row["event"] == "chunk"), default=0.0)


def generate_stage(context: Context) -> dict[str, Any]:
    run = context.run
    files = load_inputs(run, ("checkpoint", "checkpoint_record", "tokenizer"))
    pinned = _json(run.inputs)["files"]
    checkpoint_sha256 = pinned["checkpoint"]["sha256"]
    if _json(files["checkpoint_record"])["sft"]["selected"]["sha256"] != checkpoint_sha256:
        raise Refused("the checkpoint does not hash to its record's selected SHA-256")
    gate = _check_gate(
        run, Path(context.options["format_gate"]), checkpoint_sha256, pinned["tokenizer"]["sha256"]
    )
    if not gate["passed"]:
        raise Stop("format_gate_failed", f"failed checks: {gate['failed_checks']}", {"gate": gate})
    tasks_path = run.output("select", "tasks.jsonl")
    candidates_path = run.output("select", "candidates.jsonl")
    tasks = list(load_tasks(tasks_path))
    selected = run.result("select")
    assert selected is not None
    ids = [task.id for task in tasks]
    if order_sha256(ids) != selected["candidate_order_sha256"]:
        raise Refused("the selected tasks are not in the recorded order")
    if not tasks:
        return {"tasks": 0, "skipped": "no candidates were selected", "gate": gate, "outputs": {}}
    log = run.stage("generate") / GENERATION_LOG
    prior = generation_seconds_used(run)
    now: Callable[[], datetime] = context.options["now"]
    deadline = run_deadline(run)
    if prior >= GENERATION_CAP_SECONDS:
        raise Stop(
            "generation_time_cap",
            f"{prior / 3600:.2f} h of generation already used",
            {"prior_generation_seconds": prior, "gate": gate},
        )
    factory = context.options.get("generator")
    generator: Generator = (
        factory(files) if factory else CheckpointGenerator(files["checkpoint"], files["tokenizer"])
    )
    clock: Callable[[], float] = context.options.get("clock") or time.perf_counter
    loadavg_before = list(os.getloadavg())
    attempts: list[Attempt] = []
    chunk_seconds: list[float] = []
    first: dict[str, Any] = {}
    longest = _longest_chunk(log)
    _append_log(log, {"event": "start", "prior_generation_seconds": prior})
    started = clock()

    def stop(reason: str, detail: str, **extra: Any) -> Stop:
        _append_log(log, {"event": "stopped", "reason": reason})
        result = {
            "tasks": len(tasks),
            "chunks_done": len(chunk_seconds),
            "chunk_seconds": chunk_seconds,
            "elapsed_seconds": clock() - started,
            "prior_generation_seconds": prior,
            "gate": gate,
            **first,
            **extra,
        }
        return Stop(reason, detail, result)

    for start in range(0, len(tasks), CHUNK_TASKS):
        chunk = tasks[start : start + CHUNK_TASKS]
        elapsed = clock() - started
        # Before every chunk: the remaining budget must hold the longest chunk measured.
        if prior + elapsed + longest > GENERATION_CAP_SECONDS:
            raise stop(
                "generation_time_cap",
                f"{(prior + elapsed) / 3600:.2f} h used; the next chunk could take {longest:.0f} s",
            )
        if now() + timedelta(seconds=longest) >= deadline:
            raise stop("cap_reached", f"run wall time: the next chunk would pass {deadline}")
        chunk_started = clock()
        drawn = generator(chunk)
        seconds = clock() - chunk_started
        expected = [(task.id, sample) for task in chunk for sample in range(SAMPLES)]
        if [(attempt.task_id, attempt.sample) for attempt in drawn] != expected:
            raise RuntimeError("the generator did not return each candidate's samples in order")
        attempts += drawn
        chunk_seconds.append(seconds)
        longest = max(longest, seconds)
        _append_log(log, {"event": "chunk", "index": len(chunk_seconds) - 1, "seconds": seconds})
        elapsed = clock() - started
        if start == 0:
            first = {
                "first_chunk_tasks": len(chunk),
                "first_chunk_seconds": seconds,
                "projected_generation_seconds": prior + seconds * len(tasks) / len(chunk),
            }
            _log(
                f"first chunk: {seconds:.1f} s for {len(chunk)} candidates; projected "
                f"{first['projected_generation_seconds'] / 3600:.2f} h"
            )
            if first["projected_generation_seconds"] > GENERATION_CAP_SECONDS:
                raise stop(
                    "time_projection",
                    f"{first['projected_generation_seconds'] / 3600:.2f} h projected",
                )
        if prior + elapsed > GENERATION_CAP_SECONDS:  # The backstop.
            raise stop("generation_time_cap", f"{(prior + elapsed) / 3600:.2f} h used")
    elapsed = clock() - started
    _append_log(log, {"event": "end", "seconds": elapsed})
    commit, dirty = git_state()
    environment = {
        **generator.environment(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "chunk_tasks": CHUNK_TASKS,
        "candidate_order": f"sha256('{ORDER_SALT}|' + task id)",
        "candidate_order_sha256": order_sha256(ids),
        "seed_rule": "attempt_seed(seed_base, task id, sample)",
        "seeds_sha256": seeds_sha256(ids),
        "generation_cap_seconds": GENERATION_CAP_SECONDS,
        "prior_generation_seconds": prior,
        "loadavg_before": loadavg_before,
        "loadavg_after": list(os.getloadavg()),
        "numerics": NUMERICS,
        **first,
    }
    header = ScreeningHeader(
        task_file_sha256=_sha256(tasks_path),
        candidates_sha256=_sha256(candidates_path),
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=generator.tokenizer_sha256,
        tasks=len(tasks),
        samples=SAMPLES,
        attempts=len(attempts),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        seed_base=SEED_BASE,
        precision=PRECISION,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        generation_seconds=elapsed,
        created_at=utc_now(),
        git_commit=commit,
        dirty=dirty,
        environment=environment,
    )
    digest = write_attempts(run.stage("generate") / "attempts.jsonl", header, attempts)
    tokens = [attempt.completion_tokens for attempt in attempts]
    return {
        "tasks": len(tasks),
        "attempts": len(attempts),
        "with_program": sum(attempt.code is not None for attempt in attempts),
        "generated_tokens": sum(tokens),
        "generation_seconds": elapsed,
        "prior_generation_seconds": prior,
        "chunk_seconds": chunk_seconds,
        **first,
        "header": header.to_json(),
        "gate": gate,
        "outputs": {"attempts.jsonl": digest},
    }


# --------------------------------------------------------------------------------------
# score.
# --------------------------------------------------------------------------------------


def score_stage(context: Context) -> dict[str, Any]:
    from scratchcoder.rl.command import screen_attempts
    from scratchcoder.rl.facts import FACTS_FORMAT
    from scratchcoder.rl.rewards import RewardHostFailure, RewardTimeout
    from scratchcoder.sandbox import reward

    run, budget = context.run, context.budget
    assert budget is not None
    if reward.REWARD_GRAMMAR_VERSION != EXPECTED_GRAMMAR or FACTS_FORMAT != EXPECTED_FACTS:
        raise Stop(
            "grammar_identity",
            f"grammar {reward.REWARD_GRAMMAR_VERSION}, facts {FACTS_FORMAT}",
            _partial(context),
        )
    generated = run.result("generate")
    assert generated is not None
    if generated["tasks"] == 0:
        return {"tasks": 0, "skipped": "no candidates were screened", "outputs": {}}
    attempts_path = run.output("generate", "attempts.jsonl")
    tasks_path = run.output("select", "tasks.jsonl")
    candidates_path = run.output("select", "candidates.jsonl")
    _, attempts = read_attempts_file(attempts_path)
    budget.require(sum(attempt.code is not None for attempt in attempts))
    score = context.options.get("score")

    def channel(task: Task, code: str) -> reward.RewardResult:
        return (score or reward.score_attempt)(task, code)

    output = run.stage("score") / "screening.json"
    try:
        summary = screen_attempts(
            tasks_path=tasks_path,
            candidates_path=candidates_path,
            attempts_path=attempts_path,
            output=output,
            timeout=REWARD_TIMEOUT_SECONDS,
            workers=REWARD_WORKERS,
            threshold=DEFAULT_THRESHOLD,
            cap=MAX_KEPT,
            floor=FLOOR,
            require_header=True,
            score=budget.wrap(channel),
        )
    except (RewardHostFailure, RewardTimeout) as error:
        detail = f"{type(error).__name__}: {error}"
        raise Stop("sandbox_failure", detail, _partial(context)) from error
    record = _json(output)
    if record["rule"].get("reward_grammar") != EXPECTED_GRAMMAR:
        raise Stop("grammar_identity", f"record grammar {record['rule'].get('reward_grammar')}")
    return {
        "tasks": generated["tasks"],
        "summary": summary,
        "reward_grammar": record["rule"]["reward_grammar"],
        "facts_format": FACTS_FORMAT,
        "rule": record["rule"],
        "outputs": {"screening.json": _sha256(output)},
    }


# --------------------------------------------------------------------------------------
# report: the measured outcome and one recommendation by the fixed decision rule.
# --------------------------------------------------------------------------------------


def _interval(successes: int, trials: int) -> dict[str, Any]:
    bounds = wilson(successes, trials)
    return {
        "successes": successes,
        "trials": trials,
        "share": successes / trials if trials else None,
        "wilson95": None if bounds is None else list(bounds),
    }


SAMPLE_CLASSES = ("formatting", "execution", "genuine")


def sample_class(row: Mapping[str, Any]) -> str:
    """A sample's failure class: ``formatting`` (no closed program), ``execution`` (a
    resource kill, a lost report, or a check or case that never finished in time) or
    ``genuine`` (it ran, finished and failed, including an oversized or undecodable
    program, which is the candidate's own)."""
    from scratchcoder.sandbox.reward import MALFORMED_REPORT, RESOURCE_KILL

    if row["status"] == NO_PROGRAM:
        return "formatting"
    if row["status"] in {RESOURCE_KILL, MALFORMED_REPORT} or int(row.get("unfinished", 0)) > 0:
        return "execution"
    return "genuine"


def screening_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """N, E, y, z, f, x, g and pass@8 from a screening record (``rl-screening-v1``)."""
    tasks = record["tasks"]
    samples = {(row["task_id"], row["sample"]): row for row in record["rewards"]}
    eligible = list(record["eligible"])
    reached = [task for task in tasks if task["reason"] in (None, "equal_rewards", "over_cap")]
    all_zero = [task for task in reached if all(value == 0.0 for value in task["rewards"])]
    all_one = [task for task in reached if all(value == 1.0 for value in task["rewards"])]
    other_equal = [
        task
        for task in reached
        if task["reason"] == "equal_rewards" and task not in all_zero and task not in all_one
    ]
    classes: Counter[str] = Counter()
    for task in all_zero:
        for sample in range(len(task["rewards"])):
            classes[sample_class(samples[(task["id"], sample)])] += 1
    failing: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for row in record["rewards"]:
        statuses[str(row["status"])] += 1
        if row["reward"] < 1.0:
            failing[sample_class(row)] += 1
    in_zero = sum(classes.values())
    passing = [task for task in tasks if task["rewards"] and max(task["rewards"]) == 1.0]
    return {
        "N": len(tasks),
        "E": len(eligible),
        "eligible_ids": eligible,
        "y": _interval(len(eligible), len(tasks)),
        "reached_equal_rewards_check": len(reached),
        "z": _interval(len(all_zero), len(reached)),
        "all_one_groups": len(all_one),
        "other_equal_groups": len(other_equal),
        "all_zero_sample_classes": {
            name: _interval(classes[name], in_zero) for name in SAMPLE_CLASSES
        },
        "failing_sample_classes_all_groups": dict(sorted(failing.items())),
        "sample_statuses": dict(sorted(statuses.items())),
        "pass_at_8": _interval(len(passing), len(tasks)),
        "mean_reward": (
            sum(sum(task["rewards"]) for task in tasks)
            / sum(len(task["rewards"]) for task in tasks)
            if tasks
            else None
        ),
        "by_reason": dict(record["counts"]["by_reason"]),
        "note": "f, x and g count samples, which cluster within tasks; their intervals are "
        "optimistic.",
    }


def _ceil(value: Fraction | float) -> int:
    return math.ceil(value)


def recommend(
    *,
    stopped: Mapping[str, Any] | None,
    pool: int,
    screened: int,
    eligible: int,
    y_lo: float | None,
    z: float | None,
    f: float | None,
    x: float | None,
    reserve: int,
    rate: Fraction | None,
    funnel: Sequence[tuple[str, int, int]],
    drops: Mapping[str, int],
) -> dict[str, Any]:
    """One recommendation by the pilot's fixed decision rule, its rules applied in order.

    ``rate`` is r, admitted candidates per intake row; ``funnel`` is each step's name with
    the candidates entering and leaving it; ``drops`` counts the drop reasons for rule 3."""
    if stopped is not None:
        return {
            "rule": 0,
            "decision": "report_only",
            "text": f"A stopping criterion fired at {stopped['stage']} "
            f"({stopped['criterion']}). The outcome is reported; no decision is drawn.",
        }
    if pool >= POOL_FLOOR:
        return {
            "rule": 1,
            "decision": "stop_intake",
            "text": f"The combined pool is {pool}, at or above {POOL_FLOOR}: stop intake and "
            "go to the RL-readiness assessment. Reaching the floor authorizes nothing.",
        }
    if screened == 0 or eligible == 0:
        step = next((name for name, entering, left in funnel if entering and not left), None)
        return {
            "rule": 2,
            "decision": "no_proposal_investigate_step",
            "step": step or "screening",
            "text": f"No eligible candidates (N = {screened}, E = {eligible}). No intake "
            f"proposal: investigate the step that removed them ({step or 'screening'}).",
        }
    y = Fraction(eligible, screened)
    if float(y) < YIELD_THRESHOLD:
        if z is not None and z >= ALL_ZERO_THRESHOLD:
            if f is not None and f >= FORMATTING_THRESHOLD:
                kind, text = (
                    "formatting",
                    (
                        "a formatting problem: investigate SFT format and data, using the format "
                        "gate and per-source pass@8"
                    ),
                )
            elif x is not None and x >= EXECUTION_THRESHOLD:
                kind, text = (
                    "execution",
                    (
                        "an execution problem: investigate the evaluator or the tests; "
                        "infrastructure failures are never scored"
                    ),
                )
            else:
                kind, text = (
                    "difficulty",
                    (
                        "difficulty or capability: look at pass@8 by source and prompt length, "
                        "then consider an easier slice or better SFT"
                    ),
                )
        else:
            meaning = {
                "constant_baseline": "weak tests",
                "equal_rewards_all_one": "tasks too easy",
                "not_parseable": "grammar coverage",
                "equal_rewards_all_zero": "the model fails outright on these tasks",
                "equal_rewards_other": "equal partial rewards: weakly discriminating tests",
            }
            ranked = sorted(meaning, key=lambda name: (-drops.get(name, 0), name))
            kind = ranked[0]
            text = f"the dominant drop reason is {kind} ({drops.get(kind, 0)}): {meaning[kind]}"
        return {
            "rule": 3,
            "decision": "no_proposal_diagnose",
            "diagnosis": kind,
            "text": f"y = {float(y):.1%} is under 15%: no intake proposal. Diagnose before any "
            f"more data: {text}.",
        }
    need = POOL_FLOOR - pool
    k = _ceil(need * MARGIN / y)
    k_lo = _ceil(need * MARGIN / Fraction(y_lo)) if y_lo else None
    low = screened < LOW_CONFIDENCE_SCREENED
    sizing: dict[str, Any] = {
        "needed_pool": need,
        "K": k,
        "K_at_y_lo": k_lo,
        "reserve_R": reserve,
        "r_admitted_per_row": None if rate is None else float(rate),
        "low_confidence": low,
    }
    suffix = (
        " Low confidence: fewer than 300 screened, so the interval is wide; K at the lower "
        "bound is shown."
        if low
        else ""
    )

    def rows_for(candidates: int | None) -> int | None:
        if candidates is None or rate is None or rate <= 0:
            return None
        return max(0, _ceil(Fraction(candidates - reserve) / rate))

    def fits(candidates: int | None) -> bool:
        if candidates is None:
            return False
        if candidates <= reserve:
            return True
        rows = rows_for(candidates)
        return (
            rows is not None
            and rows <= FURTHER_SLICE_MAX_ROWS
            and rows <= SHARD_UNUSED_ROWS
            and candidates <= CANDIDATE_CAP
        )

    if k <= reserve:
        return {
            "rule": 4,
            "decision": "screen_reserve",
            **sizing,
            "text": f"Propose screening K = {k} more admitted candidates from this slice's "
            f"reserve of {reserve} on the same backend (no new intake); K at y_lo is {k_lo}. "
            "The proposal goes to the owner and is not run." + suffix,
        }
    rows, rows_lo = rows_for(k), rows_for(k_lo)
    sizing.update(rows_needed=rows, rows_needed_at_y_lo=rows_lo)
    if fits(k) and fits(k_lo):
        return {
            "rule": 4,
            "decision": "further_slice",
            **sizing,
            "start_row": FURTHER_SLICE_START,
            "text": f"Propose one further slice of {rows} rows from row {FURTHER_SLICE_START} "
            f"of the same pinned shard, screening K = {k} (K at y_lo {k_lo}, {rows_lo} rows). "
            "The proposal goes to the owner and is not run." + suffix,
        }
    return {
        "rule": 4,
        "decision": "no_proposal_owner_choice",
        **sizing,
        "text": f"K = {k} (K at y_lo {k_lo}) needs more than one pilot-sized step "
        f"({rows} rows; {rows_lo} at y_lo): no proposal. The owner chooses between accepting "
        "the cost and changing the experiment design; the floor is unchanged." + suffix,
    }


def _intake_dedup(intake: Mapping[str, Any]) -> dict[str, Any]:
    counts = intake["counts"]
    order = intake["rules"]["order"]
    remaining = counts["rows"]
    shares: dict[str, Any] = {}
    for reason in order:
        if reason in INTAKE_DEDUP_RULES:
            share = counts[reason] / remaining if remaining else 0.0
            shares[reason] = {"dropped": counts[reason], "reaching": remaining, "share": share}
        remaining -= counts[reason]
    return {
        "rules": shares,
        "over_limit": [name for name, item in shares.items() if item["share"] > INTAKE_DEDUP_LIMIT],
    }


def _criteria(
    run: RunDir, stop: Mapping[str, Any] | None, intake: Mapping[str, Any]
) -> list[dict[str, Any]]:
    verify = run.result("verify")
    admit = run.result("admit")
    generate = run.result("generate")
    score = run.result("score")
    fired = None if stop is None else stop["reason"]

    def status(reason: str, reached: bool) -> str:
        return "fired" if fired == reason else ("checked" if reached else "not_reached")

    dedup = _intake_dedup(intake)
    caps = fired in {"cap_reached", "generation_time_cap"}
    return [
        {
            "criterion": STOP_CRITERIA["containment_failure"],
            "status": status("containment_failure", verify is not None),
        },
        {
            "criterion": STOP_CRITERIA["sandbox_failure"],
            "status": status("sandbox_failure", verify is not None),
        },
        {
            "criterion": "the run's 12 h end-to-end wall time",
            "status": "fired"
            if stop is not None and "run wall time" in str(stop.get("detail"))
            else "checked",
            "deadline": str(run_deadline(run)),
        },
        {
            "criterion": "sealed access outside the approved scan",
            "status": "checked",
            "detail": "every input passes the sealed-file guard before it is opened; a refusal "
            "opens nothing",
        },
        {
            "criterion": "any cap reached (calls, run budget, 4 h of generation, 12 h of run)",
            "status": "fired" if caps else "checked",
            "detail": None if stop is None or not caps else stop["detail"],
        },
        {
            "criterion": STOP_CRITERIA["under_verified"],
            "status": status("under_verified", verify is not None),
            "value": None if verify is None else verify.get("verified_share"),
        },
        {
            "criterion": "intake dedup or family drop over 75% (applied at intake; the owner's "
            "ruling on that stop applies and it is not re-applied here)",
            "status": "fired_at_intake" if dedup["over_limit"] else "checked_at_intake",
            "value": dedup,
        },
        {
            "criterion": STOP_CRITERIA["admission_unresolved"],
            "status": status("admission_unresolved", admit is not None),
            "value": None if admit is None else admit.get("unresolved_share"),
        },
        {
            "criterion": STOP_CRITERIA["format_gate_failed"],
            "status": status("format_gate_failed", generate is not None),
            "value": None if generate is None else generate.get("gate", {}).get("passed"),
        },
        {
            "criterion": STOP_CRITERIA["grammar_identity"],
            "status": status("grammar_identity", score is not None),
            "value": None if score is None else score.get("reward_grammar"),
        },
        {
            "criterion": STOP_CRITERIA["time_projection"],
            "status": status("time_projection", generate is not None),
            "value": None if generate is None else generate.get("projected_generation_seconds"),
        },
    ]


def report_stage(context: Context) -> dict[str, Any]:
    run = context.run
    pinned = _json(run.inputs)
    files = load_inputs(run, ("remainder", "scan_summary", "intake_record"))
    scan = _json(files["scan_summary"])["counts"]["subjects"]
    intake = _json(files["intake_record"])
    remainder_ids = {task.id for task in load_tasks(files["remainder"])}
    stop = _json(run.stop) if run.stop.exists() else None
    results = {stage: run.result(stage) for stage in STAGES if stage != "report"}
    verify, admit, select = results["verify"], results["admit"], results["select"]
    metrics: dict[str, Any] | None = None
    score = results["score"]
    if score is not None and score.get("status") == "completed" and score.get("tasks"):
        metrics = screening_metrics(_json(run.output("score", "screening.json")))
    eligible_ids = set(metrics["eligible_ids"]) if metrics else set()
    pool = len(remainder_ids | eligible_ids)
    rows_read = int(intake["counts"]["rows"])
    verified = None if verify is None else verify.get("verified")
    admitted_wo10 = None if admit is None else admit.get("decisions", {}).get("admitted")
    admitted = None if admit is None else admit.get("admitted")
    selected = None if select is None else select.get("selected")
    screened = metrics["N"] if metrics else (0 if score and score.get("tasks") == 0 else None)
    eligible = metrics["E"] if metrics else (0 if screened == 0 else None)
    reserve = (admitted or 0) - (screened or 0) if admitted is not None else 0
    rate = Fraction(admitted, rows_read) if admitted is not None and rows_read else None
    funnel = [
        ("intake", rows_read, int(intake["counts"]["kept"])),
        ("integrity_scan", int(scan["pilot"]["total"]), int(scan["pilot"]["clear"])),
        ("verification", pinned["counts"]["pilot"], verified),
        ("admission", verified, admitted_wo10),
        ("reward_grammar_and_constant_baseline", admitted_wo10, admitted),
        ("selection", admitted, selected),
        ("screening", screened, eligible),
    ]
    drops: dict[str, int] = {}
    if admit is not None:
        drops.update(admit.get("after_admission_dropped", {}))
    if metrics is not None:
        for name in ("not_parseable", "constant_baseline"):
            drops[name] = drops.get(name, 0) + metrics["by_reason"].get(name, 0)
        drops["equal_rewards_all_zero"] = metrics["z"]["successes"]
        drops["equal_rewards_all_one"] = metrics["all_one_groups"]
        drops["equal_rewards_other"] = metrics["other_equal_groups"]
    classes = metrics["all_zero_sample_classes"] if metrics else {}
    interval = metrics["y"]["wilson95"] if metrics else None
    decision = recommend(
        stopped=stop,
        pool=pool,
        screened=screened or 0,
        eligible=eligible or 0,
        y_lo=None if interval is None else interval[0],
        z=metrics["z"]["share"] if metrics else None,
        f=classes.get("formatting", {}).get("share"),
        x=classes.get("execution", {}).get("share"),
        reserve=reserve,
        rate=rate,
        funnel=[(name, entering or 0, left or 0) for name, entering, left in funnel],
        drops=drops,
    )
    report = {
        "format": "pilot-report-v1",
        "created_at": utc_now(),
        "stopped": stop,
        "P_combined_pool": pool,
        "floor": POOL_FLOOR,
        "shortfall_to_floor": max(0, POOL_FLOOR - pool),
        "remainder": {
            "before_scan": scan["remainder"]["total"],
            "excluded": scan["remainder"]["excluded"],
            "flagged": scan["remainder"]["flagged"],
            "after_scan": len(remainder_ids),
            "treatment": "flagged and excluded are both removed",
        },
        "pilot_scan": {
            "subjects": scan["pilot"]["total"],
            "excluded": scan["pilot"]["excluded"],
            "flagged": scan["pilot"]["flagged"],
            "clear": scan["pilot"]["clear"],
            "treatment": "flagged and excluded are both removed",
        },
        "funnel": [{"step": n, "entering": a, "leaving": b} for n, a, b in funnel],
        "drops": {
            "intake": {k: v for k, v in intake["counts"].items() if k not in {"rows", "kept"}},
            "verification": None if verify is None else verify.get("dropped_by_reason"),
            "admission": None if admit is None else admit.get("decisions"),
            "after_admission": None if admit is None else admit.get("after_admission_dropped"),
            "not_parseable_by_cause": None
            if admit is None
            else admit.get("not_parseable_by_cause"),
            "screening": None if metrics is None else metrics["by_reason"],
        },
        "R_reserve": reserve,
        "r_admitted_per_intake_row": None if rate is None else float(rate),
        "screening": None
        if metrics is None
        else {key: value for key, value in metrics.items() if key != "eligible_ids"},
        "generation": _generation_summary(results["generate"]),
        "budget": dict(ledger_counts(run.budget)),
        "stopping_criteria": _criteria(run, stop, intake),
        "decision": decision,
        "numerics": NUMERICS,
        "heuristics": HEURISTICS,
    }
    directory = run.stage("report")
    outputs = {
        "report.json": _write_new(directory / "report.json", _json_bytes(report)),
        "report.md": _write_new(directory / "report.md", render(report).encode()),
    }
    if metrics is not None:
        record = _json(run.output("score", "screening.json"))
        outputs["per-task.jsonl"] = _write_new(
            directory / "per-task.jsonl",
            _jsonl_bytes(
                {
                    "id": task["id"],
                    "pass_at_8": bool(task["rewards"]) and max(task["rewards"]) == 1.0,
                    "best_reward": max(task["rewards"]) if task["rewards"] else None,
                    "mean_reward": sum(task["rewards"]) / len(task["rewards"])
                    if task["rewards"]
                    else None,
                    "reason": task["reason"],
                }
                for task in record["tasks"]
            ),
        )
    return {"decision": decision, "P": pool, "outputs": outputs}


def _generation_summary(generate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if generate is None:
        return None
    header = generate.get("header") or {}
    environment = header.get("environment") or {}
    return {
        "status": generate.get("status"),
        "device": header.get("device"),
        "precision": header.get("precision"),
        "batch_size": header.get("batch_size"),
        "torch_threads": environment.get("torch_threads"),
        "generation_seconds": generate.get("generation_seconds", generate.get("elapsed_seconds")),
        "first_chunk_seconds": generate.get("first_chunk_seconds"),
        "projected_generation_seconds": generate.get("projected_generation_seconds"),
        "gate_passed": (generate.get("gate") or {}).get("passed"),
    }


def _pct(item: Mapping[str, Any] | None) -> str:
    if not item or item.get("share") is None:
        return "n/a"
    low, high = item["wilson95"]
    return (
        f"{item['share']:.1%} ({item['successes']}/{item['trials']}; "
        f"Wilson 95% {low:.1%}-{high:.1%})"
    )


def render(report: Mapping[str, Any]) -> str:
    """The report as Markdown, for the owner."""
    lines = ["# Pilot screening report", ""]
    decision = report["decision"]
    lines += [f"**Recommendation (rule {decision['rule']}).** {decision['text']}", ""]
    if report["stopped"]:
        stop = report["stopped"]
        lines += [f"**Stopped** at `{stop['stage']}`: {stop['criterion']} ({stop['detail']}).", ""]
    lines += [
        f"- P (combined validated pool) = {report['P_combined_pool']}; floor {report['floor']}; "
        f"shortfall {report['shortfall_to_floor']}.",
        f"- Remainder: {report['remainder']['before_scan']} before the scan, "
        f"{report['remainder']['excluded']} excluded and {report['remainder']['flagged']} flagged "
        f"(both removed), {report['remainder']['after_scan']} after.",
        f"- Pilot scan: {report['pilot_scan']['subjects']} subjects, "
        f"{report['pilot_scan']['excluded']} excluded, {report['pilot_scan']['flagged']} flagged, "
        f"{report['pilot_scan']['clear']} clear.",
        f"- Reserve R = {report['R_reserve']}; r = {report['r_admitted_per_intake_row']} admitted "
        "per intake row.",
        "",
        "## Funnel",
        "",
        "| Step | Entering | Leaving |",
        "| --- | ---: | ---: |",
    ]
    lines += [f"| {s['step']} | {s['entering']} | {s['leaving']} |" for s in report["funnel"]]
    screening = report["screening"]
    if screening:
        classes = screening["all_zero_sample_classes"]
        lines += [
            "",
            "## Screening",
            "",
            f"- N = {screening['N']}, E = {screening['E']}, y = {_pct(screening['y'])}.",
            f"- z (all-zero among {screening['reached_equal_rewards_check']} reaching the "
            f"equal-rewards check) = {_pct(screening['z'])}; all-one groups "
            f"{screening['all_one_groups']}; other equal groups {screening['other_equal_groups']}.",
            f"- In all-zero groups: formatting f = {_pct(classes['formatting'])}; execution "
            f"x = {_pct(classes['execution'])}; genuine g = {_pct(classes['genuine'])}. "
            + screening["note"],
            f"- Failing samples in all groups by class: "
            f"{screening['failing_sample_classes_all_groups']}.",
            f"- pass@8 = {_pct(screening['pass_at_8'])}; mean reward {screening['mean_reward']}.",
        ]
    lines += ["", "## Drops by step", ""]
    for step, value in report["drops"].items():
        lines.append(f"- {step}: `{json.dumps(value, sort_keys=True)}`")
    lines += ["", "## Stopping criteria", ""]
    for item in report["stopping_criteria"]:
        lines.append(f"- {item['criterion']}: **{item['status']}**")
    lines += ["", "## Generation", "", f"`{json.dumps(report['generation'], sort_keys=True)}`"]
    lines += ["", f"Sandbox calls by stage: `{json.dumps(report['budget'], sort_keys=True)}`."]
    lines += ["", report["numerics"], "", report["heuristics"], ""]
    return "\n".join(lines)


def _write_run_sums(run: RunDir) -> None:
    _write_new(run.root / "SHA256SUMS", _sums(run.root, recursive=True))


# --------------------------------------------------------------------------------------
# The command line.
# --------------------------------------------------------------------------------------


BODIES: dict[str, Body] = {
    "verify": verify_stage,
    "admit": admit_stage,
    "select": select_stage,
    "generate": generate_stage,
    "score": score_stage,
    "report": report_stage,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("init")
    start.add_argument("--out", type=Path, required=True, help="the new run directory")
    start.add_argument(
        "--supersedes", type=Path, help="the latest run on these inputs, stopped by a host failure"
    )
    for name in INPUTS:
        start.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    for stage in STAGES:
        command = commands.add_parser(stage)
        command.add_argument("--run", type=Path, required=True)
        if stage in EXECUTING:
            command.add_argument("--receipt", type=Path, help="the execution-window receipt")
    commands.choices["generate"].add_argument("--format-gate", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, options: Mapping[str, Any] | None = None) -> int:
    """``options`` injects fakes for tests (``execute``, ``score``, ``generator``, ``clock``)."""
    words = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(words)
    if args.command == "init":
        return init(args)
    extra: list[Path] = []
    settings = dict(options or {})
    if args.command == "generate":
        extra.append(args.format_gate)
        settings["format_gate"] = args.format_gate
    status = run_stage(
        args.command,
        args.run,
        BODIES[args.command],
        argv=words,
        receipt=getattr(args, "receipt", None),
        extra_inputs=extra,
        options=settings,
    )
    if args.command == "report" and status == 0:
        _write_run_sums(RunDir(args.run))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
