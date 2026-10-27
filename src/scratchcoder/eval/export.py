"""Pod task exports: task files with the hidden tests removed, bound to their source by hash.

Hidden tests never leave the Mac. A rented Pod generates completions and screening
attempts, which needs each task's prompt and public test (and, for training tasks, the
reference solution), never its hidden tests. ``pod-task-export`` writes such a file:

- ``pool``: every task of a pool split file, in file order;
- ``candidates``: the RL candidate tasks of the train split file, in candidate-file order;
- ``arc``: the ARC-Easy test questions as choice-task JSON lines, in ID order (ARC holds no
  hidden tests; the export only spares the Pod the Parquet reader).

In a pool or candidates export every task keeps all its fields except ``hidden_tests``,
which becomes a placeholder for its mode (``placeholder``) naming only the SHA-256 of the
withheld tests (salted with ``HIDDEN_SALT``). The placeholder keeps the file a valid task
file, so generation reads it unchanged; it fails any program run against it, so an export
can never be scored as if it held the tests; and its hash binds the export to the exact
hidden tests it replaced, so a full file with other hidden tests derives another export.
Each line is ``json.dumps(asdict(task), sort_keys=True)``, the serialization of
``save_tasks``, so the export of one source is always the same bytes. A record next to the
export names both files' SHA-256 and the rule; it is a convenience only.

Execution never trusts that record. ``eval-execute`` and ``rl-screen`` are given the full
task file: when a completions or attempts header names a task file SHA-256 that is not the
full file's, they derive the export from the full file again (``task_file_identity``) and
compare its SHA-256 with the header's. A task file that is itself an export is refused,
since its placeholder tests would score every attempt as failed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from scratchcoder.eval.tasks import Task, load_tasks

SCHEMA_VERSION = 1
RECORD_KIND = "pod-task-export"
KINDS = ("pool", "candidates", "arc")
HIDDEN_SALT = "pod-export-v1|"
WITHHELD = "hidden tests are withheld from this Pod export; sha256 "
_FUNCTION = re.compile(r"raise AssertionError\('" + re.escape(WITHHELD) + r"[0-9a-f]{64}'\)")
_OUTPUT = re.compile(re.escape(WITHHELD) + r"[0-9a-f]{64}")
RULE = (
    "every task of the source in order (pool) or the candidate tasks in candidate-file order "
    "(candidates), each with hidden_tests replaced by placeholder(mode, hidden_tests) and "
    "every other field kept, one json.dumps(asdict(task), sort_keys=True) line per task"
)
ARC_RULE = "the ARC-Easy test questions in ID order, one json.dumps(asdict(task), sort_keys=True)"


class ExportError(ValueError):
    """A task file and a header disagree, or a Pod export was given as the full task file."""


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def placeholder(mode: str, hidden_tests: str) -> str:
    """The ``hidden_tests`` of an exported task: valid for the task schema, failing any program
    the runner could execute against it, and naming ``sha256(HIDDEN_SALT + hidden_tests)``."""
    digest = hashlib.sha256((HIDDEN_SALT + hidden_tests).encode()).hexdigest()
    if mode == "function":
        return f"raise AssertionError('{WITHHELD}{digest}')"
    if mode == "io":
        return json.dumps([{"input": "", "output": WITHHELD + digest}], sort_keys=True)
    raise ValueError(f"unknown task mode {mode!r}")


def exported(task: Task) -> Task:
    """The task as a Pod sees it: ``hidden_tests`` replaced by its placeholder."""
    return replace(task, hidden_tests=placeholder(task.mode, task.hidden_tests))


def is_exported(task: Task) -> bool:
    """Whether the task's hidden tests are an export placeholder rather than tests."""
    if task.mode == "function":
        return _FUNCTION.fullmatch(task.hidden_tests) is not None
    try:
        cases = json.loads(task.hidden_tests)
    except ValueError:
        return False
    return (
        isinstance(cases, list)
        and len(cases) == 1
        and isinstance(cases[0], dict)
        and cases[0].get("input") == ""
        and _OUTPUT.fullmatch(str(cases[0].get("output"))) is not None
    )


def task_lines(tasks: Iterable[Task]) -> bytes:
    return b"".join(json.dumps(asdict(task), sort_keys=True).encode() + b"\n" for task in tasks)


def read_candidate_list(path: Path) -> list[tuple[str, str]]:
    """``(id, source)`` per line of an RL candidate file; IDs must be unique."""
    listed: list[tuple[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"id", "source"}:
            raise ValueError(f"candidate line {number} needs exactly id and source")
        listed.append((row["id"], row["source"]))
    if len({identifier for identifier, _ in listed}) != len(listed):
        raise ValueError("candidate IDs must be unique")
    return listed


def candidate_tasks(tasks_path: Path, candidates_path: Path) -> list[Task]:
    """The candidate tasks, in candidate-file order, each checked against its listed source."""
    listed = read_candidate_list(candidates_path)
    wanted = {identifier for identifier, _ in listed}
    found = {task.id: task for task in load_tasks(tasks_path) if task.id in wanted}
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{len(missing)} candidates are not in the task file, e.g. {missing[0]}")
    tasks = [found[identifier] for identifier, _ in listed]
    for task, (_, source) in zip(tasks, listed, strict=True):
        if task.split != "train":
            raise ValueError(f"candidate {task.id} is not a train task")
        if task.source != source:
            raise ValueError(f"candidate {task.id} is listed under another source")
    return tasks


def source_tasks(tasks_path: Path, candidates_path: Path | None = None) -> list[Task]:
    """The tasks an export of ``tasks_path`` holds, before their hidden tests are replaced."""
    if candidates_path is None:
        return list(load_tasks(tasks_path))
    return candidate_tasks(tasks_path, candidates_path)


def export_bytes(tasks_path: Path, candidates_path: Path | None = None) -> bytes:
    """The export's exact bytes; a source that is already an export is refused."""
    tasks = source_tasks(tasks_path, candidates_path)
    if not tasks:
        raise ValueError("an export needs at least one task")
    if any(is_exported(task) for task in tasks):
        raise ExportError(f"{tasks_path} is already a Pod export; export the full task file")
    return task_lines(exported(task) for task in tasks)


def export_sha256(tasks_path: Path, candidates_path: Path | None = None) -> str:
    return hashlib.sha256(export_bytes(tasks_path, candidates_path)).hexdigest()


def refuse_exported(tasks: Sequence[Task], path: Path) -> None:
    """Refuse to execute against a Pod export: its placeholder tests fail every program."""
    if any(is_exported(task) for task in tasks):
        raise ExportError(
            f"{path} is a Pod export with placeholder hidden tests; give the full task file"
        )


def task_file_identity(
    tasks_path: Path, expected_sha256: str, *, candidates_path: Path | None = None
) -> str:
    """How ``tasks_path`` matches a header's task file SHA-256: ``"file"`` or ``"export"``.

    ``"file"``: the file itself hashes to it. ``"export"``: the Pod export derived again from
    the file (with ``candidates_path`` for a candidates export) hashes to it; no record is
    read. Anything else raises ``ExportError``.
    """
    if file_sha256(tasks_path) == expected_sha256:
        return "file"
    try:
        derived = export_sha256(tasks_path, candidates_path)
    except ExportError:
        raise
    except ValueError as error:
        raise ExportError(f"cannot derive the Pod export of {tasks_path}: {error}") from error
    if derived != expected_sha256:
        raise ExportError(
            "the task file's SHA-256 differs from the header's, and so does the SHA-256 of "
            "the Pod export derived from it"
        )
    return "export"


def _write_new(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists; an export is never replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if anything took the name meanwhile.
    finally:
        temporary.unlink(missing_ok=True)


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def write_export(
    kind: str,
    *,
    output: Path,
    record: Path,
    tasks_path: Path | None = None,
    candidates_path: Path | None = None,
    arc_directory: Path | None = None,
) -> dict[str, Any]:
    """Write one export and its record; return the record. Both files must be new.

    The record holds no time: the same inputs always give the same two files.
    """
    if kind not in KINDS:
        raise ValueError(f"the export kind must be one of {', '.join(KINDS)}")
    if record.exists() or record.is_symlink():
        raise FileExistsError(f"{record} already exists; an export record is never replaced")
    value: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "kind": RECORD_KIND, "export": kind}
    if kind == "arc":
        if arc_directory is None or tasks_path is not None or candidates_path is not None:
            raise ValueError("an ARC export takes the ARC-Easy data directory only")
        from scratchcoder.data.arc import read_arc_easy

        questions, metadata = read_arc_easy(arc_directory)
        ordered = sorted(questions, key=lambda question: question.id)
        data = b"".join(
            json.dumps(asdict(question), sort_keys=True).encode() + b"\n" for question in ordered
        )
        _write_new(output, data)
        value.update(
            rule=ARC_RULE,
            source={
                "directory": str(arc_directory),
                "manifests": metadata["data_manifests"],
                "source_split_counts": metadata["source_split_counts"],
            },
            questions=len(ordered),
        )
    else:
        if tasks_path is None or arc_directory is not None:
            raise ValueError("a pool or candidates export takes a task file")
        if (kind == "candidates") != (candidates_path is not None):
            raise ValueError("a candidates export, and only it, takes the candidate file")
        tasks = source_tasks(tasks_path, candidates_path)
        data = export_bytes(tasks_path, candidates_path)
        _write_new(output, data)
        value.update(
            rule=RULE,
            hidden_salt=HIDDEN_SALT,
            source=_identity(tasks_path),
            candidates=None if candidates_path is None else _identity(candidates_path),
            tasks=len(tasks),
            by_split=dict(sorted(Counter(task.split for task in tasks).items())),
            by_mode=dict(sorted(Counter(task.mode for task in tasks).items())),
            by_source=dict(sorted(Counter(task.source for task in tasks).items())),
        )
    value["output"] = _identity(output)
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    _write_new(record, data)
    return value
