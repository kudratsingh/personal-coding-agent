"""Holdout integrity scan: one approved, logged, static read of the sealed pool test split.

It answers one question: which *subject* tasks (tasks that may be trained on) share a task
family with a sealed test task (:mod:`scratchcoder.data.integrity_scan`). *Bridge* tasks
(for example the pool's train and dev splits) join families but are not judged. The scan
generates, executes and scores nothing. Its outputs name subjects only, with a status each:
no test prompt, reference, test, test ID or per-test value is written, printed or logged.

``request``
    Writes ``approval-request.json`` in ``--out`` with:

    - ``request``: the scan's code identity (SHA-256 of every module it runs) and the git
      commit, the rules version, the sealed file's path with its SHA-256 and task count
      pinned by the pool build report (``--pool-report``), every subject and bridge file's
      SHA-256 and task count, the seal ledger with its current head, and the output
      directory;
    - ``request_sha256``: the SHA-256 of the request's canonical JSON;
    - an empty ``approval`` block.

    The sealed file is never opened, only ``os.lstat``-ed. Every other input is refused if
    it is, or is the same file as, the sealed file.
``run --approval FILE``
    FILE is the request document with its ``approval`` block filled by the owner:
    ``approved_by``, ``approved_at``, ``statement`` and ``request_sha256``, which quotes
    the request's hash.

    The run refuses (exit 2, nothing logged, nothing sealed opened) unless all of these
    hold:

    - the quoted hash equals the canonical hash of the request in the file;
    - the code, git commit (clean tree), rules, subjects, bridges, pinned report, sealed
      path (a regular file named ``test.jsonl``, not a link) and output names still match;
    - the ledger is the same ledger, at most extended;
    - that request was never scanned.

    Then it appends ``integrity_scan_opened`` (keyed by the request hash, with the approval
    block) to the seal ledger, and only then opens the sealed file once. It checks that the
    opened file is the file it checked, and its SHA-256 and task count. It scans, writes the
    outputs under temporary names and appends ``integrity_scan_closed``: ``completed`` with
    the output hashes and counts, or ``failed`` with an error class name only. Only after a
    completed close are the outputs renamed to ``exclusions.json``, ``summary.json`` and
    ``record.json``. A run whose close is not ``completed`` leaves no outputs.

Exit codes:

- 0: done.
- 2: refused before the sealed file was opened; nothing was logged.
- 3: an output exists.
- 4: the sealed file differs from what was checked or pinned: another file, SHA-256 or task
  count (logged as failed).
- 5: any other failure after the ledger event (logged as failed when the ledger allows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import scratchcoder.common.records as records_module
import scratchcoder.data.adapters as adapters_module
import scratchcoder.data.clusters as clusters_module
import scratchcoder.data.filters as filters_module
import scratchcoder.data.integrity_scan as scan_module
import scratchcoder.data.pool as pool_module
import scratchcoder.eval.completions as completions_module
import scratchcoder.eval.seal as seal_module
import scratchcoder.eval.tasks as tasks_module
import tools.verifier_scan as verifier_scan_module
from scratchcoder.common.records import git_state
from scratchcoder.data.clusters import RULES_VERSION
from scratchcoder.data.integrity_scan import (
    SCAN_FORMAT,
    SealedContentError,
    item_of,
    parse_sealed_tasks,
    scan,
)
from scratchcoder.eval.seal import SealError, SealLedger
from scratchcoder.eval.tasks import Task, load_tasks
from tools.verifier_scan import SEALED_NAMES, SealedPathError, guard

REQUEST_KIND = "holdout-integrity-scan-request"
SCHEMA_VERSION = 1
SEALED_NAME = "test.jsonl"
REQUEST_NAME = "approval-request.json"
OUTPUTS = ("exclusions.json", "summary.json", "record.json")
APPROVAL_FIELDS = ("approved_by", "approved_at", "statement", "request_sha256")
SUBJECT_SPLITS = ("train",)
BRIDGE_SPLITS = ("train", "dev")
MAX_SMALL_INPUT_BYTES = 1 << 20
"""An approval document or pool report larger than this is refused unread (both are a few
kilobytes; the sealed split is megabytes), so neither can stand in for the sealed file."""
PERFORMS = {"generation": False, "execution": False, "scoring": False}
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
PINNED_SEALED_FILES = ("test", "holdout", "rl_only")
"""Pool-report entries whose files are sealed: no input may have one of their hashes."""


class Refused(Exception):
    """Refused before the sealed file was opened."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(body: Any) -> str:
    """SHA-256 of ``body`` as canonical JSON (sorted keys, compact, ASCII)."""
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha_bytes(text.encode())


def code_identity() -> dict[str, str]:
    """SHA-256 of every module whose code decides what the scan reads, finds or writes."""
    modules = {
        "tools/holdout_integrity_scan.py": Path(__file__),
        "tools/verifier_scan.py": Path(str(verifier_scan_module.__file__)),
        "scratchcoder/common/records.py": Path(str(records_module.__file__)),
        "scratchcoder/data/integrity_scan.py": Path(str(scan_module.__file__)),
        "scratchcoder/data/clusters.py": Path(str(clusters_module.__file__)),
        "scratchcoder/data/filters.py": Path(str(filters_module.__file__)),
        "scratchcoder/eval/completions.py": Path(str(completions_module.__file__)),
        "scratchcoder/data/pool.py": Path(str(pool_module.__file__)),
        "scratchcoder/data/adapters.py": Path(str(adapters_module.__file__)),
        "scratchcoder/eval/tasks.py": Path(str(tasks_module.__file__)),
        "scratchcoder/eval/seal.py": Path(str(seal_module.__file__)),
    }
    return {name: _sha_bytes(path.read_bytes()) for name, path in sorted(modules.items())}


def _sealed_stat(path: Path) -> os.stat_result:
    """The sealed path's own metadata (never followed, never opened)."""
    if path.name != SEALED_NAME:
        raise Refused(f"the sealed file must be named {SEALED_NAME}")
    try:
        info = os.lstat(path)
    except OSError as error:
        raise Refused(f"the sealed file is not there: {type(error).__name__}") from None
    if not stat.S_ISREG(info.st_mode):
        raise Refused("the sealed path is not a regular file (a link or something else)")
    return info


@dataclass(frozen=True)
class SealedSet:
    """The sealed test file's checked metadata, the inodes of every sealed-named file beside
    it (test, holdout union, RL-only) and the pinned SHA-256 values of the sealed files."""

    info: os.stat_result
    inodes: frozenset[tuple[int, int]]
    hashes: frozenset[str]


def _sealed_set(path: Path, hashes: frozenset[str] = frozenset()) -> SealedSet:
    info = _sealed_stat(path)
    inodes = {(info.st_dev, info.st_ino)}
    for name in sorted(SEALED_NAMES):
        try:
            sibling = os.stat(path.parent / name)
        except OSError:
            continue
        inodes.add((sibling.st_dev, sibling.st_ino))
    return SealedSet(info, frozenset(inodes), hashes)


def _pinned(report: bytes) -> tuple[str, int, frozenset[str]]:
    """The test file's pinned SHA-256 and task count, and every sealed file's pinned hash."""
    try:
        files = json.loads(report)["files"]
        test = files["test"]
        digest, tasks = test["file_sha256"], test["tasks"]
        hashes = {files[name]["file_sha256"] for name in PINNED_SEALED_FILES if name in files}
    except (ValueError, KeyError, TypeError):
        raise Refused("the pool report does not pin the test file") from None
    if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
        raise Refused("the pool report's test SHA-256 is malformed")
    if type(tasks) is not int or tasks < 1:
        raise Refused("the pool report's test task count must be a positive integer")
    if any(not isinstance(value, str) or not SHA256_HEX.fullmatch(value) for value in hashes):
        raise Refused("the pool report's sealed-file hashes are malformed")
    return digest, tasks, frozenset(hashes)


def _not_sealed(path: Path, sealed: SealedSet, what: str) -> None:
    """Refuse ``path`` before reading it if it is sealed by name or is any sealed file."""
    guard([path])
    try:
        info = os.stat(path)
    except OSError as error:
        raise Refused(f"{what} is not there: {type(error).__name__}") from None
    if (info.st_dev, info.st_ino) in sealed.inodes:
        raise Refused(f"{what} is a sealed file")


def _read_small(path: Path, sealed: SealedSet, what: str) -> bytes:
    _not_sealed(path, sealed, what)
    if os.stat(path).st_size > MAX_SMALL_INPUT_BYTES:
        raise Refused(f"{what} is larger than {MAX_SMALL_INPUT_BYTES} bytes")
    return path.read_bytes()


def _task_file(
    label: str, path: Path, sealed: SealedSet, splits: Sequence[str]
) -> tuple[dict[str, Any], list[Task]]:
    """A subject or bridge file's identity and tasks; refused if it is or copies the sealed
    file."""
    _not_sealed(path, sealed, f"set {label}")
    data = path.read_bytes()
    digest = _sha_bytes(data)
    if digest in sealed.hashes:
        raise Refused(f"set {label} is a copy of a sealed file")
    try:
        tasks = list(load_tasks(path))
    except ValueError:
        raise Refused(f"set {label} is not a task file") from None
    if _sha_bytes(path.read_bytes()) != digest:
        raise Refused(f"set {label} changed while it was read")
    if not tasks:
        raise Refused(f"set {label} is empty")
    if any(task.split not in splits for task in tasks):
        raise Refused(f"set {label} must hold only {'/'.join(splits)} tasks")
    entry = {"label": label, "path": str(path.resolve()), "sha256": digest, "tasks": len(tasks)}
    return entry, tasks


def _labelled(values: Sequence[str]) -> list[tuple[str, Path]]:
    pairs = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise Refused(f"a set is LABEL=PATH, not {value!r}")
        pairs.append((label, Path(path)))
    return pairs


def _task_hash(task: Task) -> str:
    return canonical_sha256(asdict(task))


def _merge(
    subjects: dict[str, list[Task]], bridges: dict[str, list[Task]]
) -> tuple[dict[str, list[Task]], int]:
    """Bridges without the tasks that are also subjects, and how many were dropped.

    Subject IDs must be unique across subject sets, and bridge IDs across bridge sets. A
    task that is both a subject and a bridge (an RL candidate is a pool train task) is
    judged as the subject; the same ID with different content is refused."""
    judged: dict[str, str] = {}
    for tasks in subjects.values():
        for task in tasks:
            if task.id in judged:
                raise Refused("subject IDs must be unique across all subject sets")
            judged[task.id] = _task_hash(task)
    seen: set[str] = set()
    kept: dict[str, list[Task]] = {}
    dropped = 0
    for label, tasks in bridges.items():
        kept[label] = []
        for task in tasks:
            if task.id in seen:
                raise Refused("bridge IDs must be unique across all bridge sets")
            seen.add(task.id)
            if task.id in judged:
                if judged[task.id] != _task_hash(task):
                    raise Refused("a subject and a bridge share an ID but differ in content")
                dropped += 1
            else:
                kept[label].append(task)
    return kept, dropped


def build_request(
    *,
    sealed: Path,
    pool_report: Path,
    subjects: Sequence[tuple[str, Path]],
    bridges: Sequence[tuple[str, Path]],
    ledger: Path,
    out: Path,
) -> dict[str, Any]:
    report_bytes = _read_small(pool_report, _sealed_set(sealed), "the pool report")
    pinned_sha256, pinned_tasks, hashes = _pinned(report_bytes)
    sealed_info = _sealed_set(sealed, hashes)
    labels = [label for label, _ in [*subjects, *bridges]]
    if not subjects:
        raise Refused("at least one subject set is required")
    if len(set(labels)) != len(labels):
        raise Refused("set labels must be unique")
    made = [_task_file(lb, p, sealed_info, SUBJECT_SPLITS) for lb, p in subjects]
    linked = [_task_file(lb, p, sealed_info, BRIDGE_SPLITS) for lb, p in bridges]
    _merge(
        {entry["label"]: tasks for entry, tasks in made},
        {entry["label"]: tasks for entry, tasks in linked},
    )
    seal = SealLedger(ledger)
    _not_sealed(ledger, sealed_info, "the seal ledger")
    _not_sealed(seal.head_path, sealed_info, "the seal ledger's head record")
    if not ledger.exists() or not seal.head_path.exists():
        raise Refused("the seal ledger and its head record must exist")
    commit, dirty = git_state()
    if dirty or commit is None:
        raise Refused("the working tree must be a clean git checkout")
    return {
        "kind": REQUEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "scan_format": SCAN_FORMAT,
        "rules_version": RULES_VERSION,
        "code": code_identity(),
        "git_commit": commit,
        "sealed": {
            "path": str(sealed.resolve()),
            "sha256": pinned_sha256,
            "tasks": pinned_tasks,
            "pinned_by": {"path": str(pool_report.resolve()), "sha256": _sha_bytes(report_bytes)},
        },
        "subjects": [entry for entry, _ in made],
        "bridges": [entry for entry, _ in linked],
        "ledger": {"path": str(ledger.resolve()), "head": seal.head()},
        "out": str(out.resolve()),
        "outputs": list(OUTPUTS),
        "performs": dict(PERFORMS),
    }


def request(args: argparse.Namespace) -> int:
    out: Path = args.out
    if (out / REQUEST_NAME).exists() or any((out / name).exists() for name in OUTPUTS):
        print("refused: the request or an output already exists", file=sys.stderr)
        return 3
    body = build_request(
        sealed=args.sealed,
        pool_report=args.pool_report,
        subjects=_labelled(args.subject),
        bridges=_labelled(args.bridge),
        ledger=args.ledger,
        out=out,
    )
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "request": body,
        "request_sha256": canonical_sha256(body),
        "approval": dict.fromkeys(APPROVAL_FIELDS),
    }
    with (out / REQUEST_NAME).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out / REQUEST_NAME} (request {document['request_sha256']});")
    print("a run needs the owner's approval block, quoting that request SHA-256")
    return 0


def _checked(
    document: Any,
) -> tuple[dict[str, Any], str, dict[str, str], dict[str, list[Task]], dict[str, list[Task]]]:
    """The approved request, its hash, the approval and the subject and (merged) bridge
    sets, all re-verified."""
    if not isinstance(document, dict) or set(document) != {
        "request",
        "request_sha256",
        "approval",
    }:
        raise Refused("not an approval document")
    body, approval = document["request"], document["approval"]
    if not isinstance(body, dict) or not isinstance(approval, dict):
        raise Refused("not an approval document")
    if set(approval) != set(APPROVAL_FIELDS):
        raise Refused("the approval block's fields differ from the expected fields")
    for name in APPROVAL_FIELDS:
        value = approval[name]
        if not isinstance(value, str) or not value.strip():
            raise Refused(f"the approval block's {name} is empty: the scan is not approved")
    request_sha256 = canonical_sha256(body)
    if approval["request_sha256"] != request_sha256 or document["request_sha256"] != (
        request_sha256
    ):
        raise Refused("the approval does not quote this request's SHA-256")
    try:
        if body["kind"] != REQUEST_KIND or body["schema_version"] != SCHEMA_VERSION:
            raise Refused("not a holdout integrity-scan request of this schema")
        if body["code"] != code_identity():
            raise Refused("the scan's code differs from the approved code")
        commit, dirty = git_state()
        if dirty or commit != body["git_commit"]:
            raise Refused("the checkout is not the approved, clean git commit")
        if body["scan_format"] != SCAN_FORMAT or body["rules_version"] != RULES_VERSION:
            raise Refused("the scan's rules differ from the approved rules")
        if body["outputs"] != list(OUTPUTS) or body["performs"] != PERFORMS:
            raise Refused("the request's outputs or scope differ from this tool's")
        sealed = Path(body["sealed"]["path"])
        if str(sealed.resolve()) != body["sealed"]["path"]:
            raise Refused("the sealed path no longer resolves to the approved path")
        pinned_by = body["sealed"]["pinned_by"]
        report = _read_small(Path(pinned_by["path"]), _sealed_set(sealed), "the pool report")
        if _sha_bytes(report) != pinned_by["sha256"]:
            raise Refused("the pool report that pins the sealed file changed")
        pinned_sha256, pinned_tasks, hashes = _pinned(report)
        if (pinned_sha256, pinned_tasks) != (body["sealed"]["sha256"], body["sealed"]["tasks"]):
            raise Refused("the request's pinned values differ from the pool report")
        sealed_info = _sealed_set(sealed, hashes)
        sets: list[dict[str, list[Task]]] = [{}, {}]
        for which, key, splits in ((0, "subjects", SUBJECT_SPLITS), (1, "bridges", BRIDGE_SPLITS)):
            for entry in body[key]:
                current, tasks = _task_file(
                    entry["label"], Path(entry["path"]), sealed_info, splits
                )
                if current != entry:
                    raise Refused(f"set {entry['label']} differs from the approved file")
                sets[which][entry["label"]] = tasks
        if not sets[0]:
            raise Refused("at least one subject set is required")
        sets[1], _ = _merge(sets[0], sets[1])
        ledger = Path(body["ledger"]["path"])
        _not_sealed(ledger, sealed_info, "the seal ledger")
        _not_sealed(SealLedger(ledger).head_path, sealed_info, "the seal ledger's head record")
        head = body["ledger"]["head"]
        if not SealLedger(ledger).extends(head["events"], head["head_sha256"]):
            raise Refused("the seal ledger is not the approved ledger (or lost events)")
        wording = {key: approval[key] for key in APPROVAL_FIELDS if key != "request_sha256"}
        for event in SealLedger(ledger).events():
            if event.get("request_sha256") == request_sha256:
                raise Refused("this request was already scanned; a new scan needs a new request")
            earlier = event.get("approval")
            if event["event"] == "integrity_scan_opened" and isinstance(earlier, dict):
                if {key: earlier.get(key) for key in wording} == wording:
                    raise Refused("this approval wording was already used for another scan")
    except (KeyError, TypeError) as error:
        raise Refused(f"the request is malformed ({type(error).__name__})") from None
    out = Path(body["out"])
    if any((out / name).exists() for name in OUTPUTS):
        raise FileExistsError("an output already exists")
    return body, request_sha256, approval, sets[0], sets[1]


def _write_partial(path: Path, value: dict[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_bytes(data)


def _partial(out: Path, name: str) -> Path:
    return out / f".{name}.partial"


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class SealedFileChanged(SealedContentError):
    """The file opened is not the sealed file whose metadata was checked."""


def _read_sealed(path: Path, checked: os.stat_result) -> bytes:
    """The sealed file's bytes, opened without following links; refused unless the opened
    file is the regular file whose metadata was checked."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise SealedFileChanged("the sealed file could not be opened as checked") from None
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        # Device and inode alone are not enough: a filesystem may reuse a freed inode number
        # for a replacement file. Its change and modification times still differ.
        same = _identity(info) == _identity(checked)
        if not stat.S_ISREG(info.st_mode) or not same:
            raise SealedFileChanged("the opened sealed file is not the file that was checked")
        return stream.read()


@contextmanager
def _terminate_as_exit() -> Iterator[None]:
    """SIGTERM raises ``SystemExit`` for the duration, so the failure close still runs."""

    def handler(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _open(
    ledger: SealLedger, request_sha256: str, body: dict[str, Any], approval: dict[str, str]
) -> None:
    ledger.open_integrity_scan(
        request_sha256,
        approval=approval,
        tool="tools/holdout_integrity_scan.py",
        code=body["code"],
        git_commit=body["git_commit"],
        rules_version=body["rules_version"],
        scan_format=body["scan_format"],
        sealed_path=body["sealed"]["path"],
        sealed_sha256_pinned=body["sealed"]["sha256"],
        sets=[
            {key: entry[key] for key in ("label", "sha256", "tasks")} | {"role": role}
            for role in ("subjects", "bridges")
            for entry in body[role]
        ],
    )


def _has_event(ledger: SealLedger, request_sha256: str, kind: str) -> bool:
    """Whether the ledger holds ``kind`` for this request (False if it cannot be read)."""
    try:
        return any(
            event["event"] == kind and event.get("request_sha256") == request_sha256
            for event in ledger.events()
        )
    except BaseException:  # noqa: BLE001
        return False


def _close_failed(ledger: SealLedger, request_sha256: str, error: str) -> bool:
    try:
        ledger.close_integrity_scan(request_sha256, "failed", error=error)
        return True
    except BaseException as close_error:  # noqa: BLE001 - never let a traceback print
        print(f"failed: the failure could not be logged ({type(close_error).__name__})")
        return False


def run(args: argparse.Namespace) -> int:
    approval_path: Path = args.approval
    if approval_path.name == SEALED_NAME:
        raise Refused(f"the approval document cannot be named {SEALED_NAME}")
    guard([approval_path])
    info = os.lstat(approval_path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        # The sealed path is only known after this file is read, so a hard link to the sealed
        # file under another name and directory is refused by its link count.
        raise Refused("the approval document must be a regular file with one link")
    if info.st_size > MAX_SMALL_INPUT_BYTES:
        raise Refused(f"the approval document is larger than {MAX_SMALL_INPUT_BYTES} bytes")
    try:
        document = json.loads(approval_path.read_bytes())
    except ValueError:
        raise Refused("the approval document is not JSON") from None
    try:
        body, request_sha256, approval, subjects, bridges = _checked(document)
    except FileExistsError:
        print("refused: an output already exists", file=sys.stderr)
        return 3
    sealed_path = Path(body["sealed"]["path"])
    checked = _sealed_stat(sealed_path)
    ledger = SealLedger(Path(body["ledger"]["path"]))
    out = Path(body["out"])
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        _partial(out, name).unlink(missing_ok=True)
    state = {"completed": False}
    with _terminate_as_exit():
        try:
            _open(ledger, request_sha256, body, approval)
        except (SealError, Refused):
            raise
        except BaseException as error:  # noqa: BLE001 - e.g. the head record write failed
            if not _has_event(ledger, request_sha256, "integrity_scan_opened"):
                raise
            _close_failed(ledger, request_sha256, type(error).__name__)
            print(f"failed: {type(error).__name__} (the sealed file was not opened)")
            return 5
        try:
            return _scan_and_close(
                body, request_sha256, approval, subjects, bridges, checked, state
            )
        except BaseException as error:  # noqa: BLE001 - every failure is logged, never quoted
            if state["completed"] or _has_event(ledger, request_sha256, "integrity_scan_closed"):
                # The ledger already records a completed scan with these outputs' hashes:
                # keep whatever is on disk (final or .partial names), never delete it.
                print(
                    f"failed after the completed close ({type(error).__name__}): rename the "
                    ".partial outputs whose hashes the ledger's close event records",
                    file=sys.stderr,
                )
                return 5
            for name in OUTPUTS:
                try:
                    _partial(out, name).unlink(missing_ok=True)
                except BaseException:  # noqa: BLE001, S110 - the failure close must still run
                    pass
            _close_failed(ledger, request_sha256, type(error).__name__)
            print(f"failed: {type(error).__name__}", file=sys.stderr)
            return 5


def _scan_and_close(
    body: dict[str, Any],
    request_sha256: str,
    approval: dict[str, str],
    subjects: dict[str, list[Task]],
    bridges: dict[str, list[Task]],
    checked: os.stat_result,
    state: dict[str, bool],
) -> int:
    ledger = SealLedger(Path(body["ledger"]["path"]))
    out = Path(body["out"])
    try:
        data = _read_sealed(Path(body["sealed"]["path"]), checked)
    except SealedFileChanged:
        _close_failed(ledger, request_sha256, "sealed_file_changed")
        print("failed: the sealed file differs from the file that was checked")
        return 4
    digest = _sha_bytes(data)
    if digest != body["sealed"]["sha256"]:
        del data
        _close_failed(ledger, request_sha256, "sealed_sha256_mismatch")
        print("failed: the sealed file's SHA-256 differs from the pinned value")
        return 4
    test = parse_sealed_tasks(data)
    del data
    if len(test) != body["sealed"]["tasks"]:
        _close_failed(ledger, request_sha256, "sealed_task_count")
        print("failed: the sealed file's task count differs from the pinned value")
        return 4
    results, counts = scan(
        [item_of(task) for task in test],
        {label: [item_of(task) for task in tasks] for label, tasks in subjects.items()},
        bridges={label: [item_of(task) for task in tasks] for label, tasks in bridges.items()},
    )
    del test
    exclusions = {
        "format": SCAN_FORMAT,
        "request_sha256": request_sha256,
        "valid_only_with": "a completed integrity_scan_closed ledger event with these hashes",
        "excluded": {
            label: sorted(r.id for r in results if r.label == label and r.status == "excluded")
            for label in subjects
        },
        "flagged": {
            label: sorted(r.id for r in results if r.label == label and r.status == "flagged")
            for label in subjects
        },
        "clear": {
            label: sorted(r.id for r in results if r.label == label and r.status == "clear")
            for label in subjects
        },
    }
    summary = {
        "format": SCAN_FORMAT,
        "request_sha256": request_sha256,
        "rules_version": body["rules_version"],
        "counts": counts,
    }
    hashes = {
        "exclusions.json": _write_partial(_partial(out, "exclusions.json"), exclusions),
        "summary.json": _write_partial(_partial(out, "summary.json"), summary),
    }
    hashes["record.json"] = _write_partial(
        _partial(out, "record.json"),
        {
            "format": SCAN_FORMAT,
            "request_sha256": request_sha256,
            "approval": approval,
            "request": body,
            "sealed_sha256_read": digest,
            "outputs": dict(hashes),
            "performed": dict(PERFORMS),
        },
    )
    # From the completed close on, nothing may delete the outputs: ignore SIGTERM (the
    # handler is restored on leaving _terminate_as_exit).
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ledger.close_integrity_scan(request_sha256, "completed", outputs=hashes, counts=counts)
    state["completed"] = True
    for name in OUTPUTS:
        os.replace(_partial(out, name), out / name)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("request", help="write the approval request; opens nothing sealed")
    make.add_argument("--sealed", type=Path, required=True)
    make.add_argument("--pool-report", type=Path, required=True)
    make.add_argument("--subject", action="append", default=[], help="LABEL=PATH, repeatable")
    make.add_argument("--bridge", action="append", default=[], help="LABEL=PATH, repeatable")
    make.add_argument("--ledger", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.set_defaults(handler=request)
    go = commands.add_parser("run", help="the approved, logged scan")
    go.add_argument("--approval", type=Path, required=True)
    go.set_defaults(handler=run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (Refused, SealedPathError, SealError, SealedContentError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"refused: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
