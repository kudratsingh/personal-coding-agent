"""Diagnostic static scan of task files for verifier-quality risks (text only, never runs code).

Takes explicit input JSONL paths (no globs, no directory walking) and writes, under ``--out``:
``scan-<label>.jsonl`` (one row per task with at least one rule hit), ``summary.json`` (counts
per rule and sub-tag per input, the rule-set version, the tool commit passed with ``--commit``
and the SHA-256 of every input) and ``record.json`` (every file the tool opened).

``--id-join IDS POOL`` scans an IDs-only JSONL (rows with ``id`` or ``task_id``, such as the RL
candidate list) with each task's text joined from the explicitly given pool file.

Sealed-file guard. Before anything is opened, every path (and its symlink-resolved target) is
checked, and the run is refused with a non-zero exit if a basename is ``test.jsonl``,
``holdout.jsonl`` or ``rl_only.jsonl``, matches ``eval-*-test-*``, contains ``holdout`` or
``rl_only``/``rl-only``, or has ``test``/``tests`` as a whole word in its name (so
``dev_test.jsonl`` and ``test-split.jsonl`` are refused while ``latest.jsonl`` is allowed), or
if any parent directory name contains ``holdout``. ``dev.jsonl``, ``train.jsonl`` and
``rl_candidates.jsonl`` are allowed. The rule is deliberately conservative: a refused file that
is actually fine is renamed or copied by its owner; the reverse mistake is not recoverable.
A path that is a hard link to a sealed-named sibling in the same directory (same device and
inode, compared with ``os.stat`` only) is refused too.

Exit codes: 0 done, 2 sealed input refused, 3 an output file already exists (nothing is
overwritten and nothing is written). Paths in ``summary.json`` and ``record.json`` are absolute.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json
from scratchcoder.data.verifier_scan import RULESET_VERSION, TaskScan, scan_task, summarize

SEALED_NAMES = frozenset({"test.jsonl", "holdout.jsonl", "rl_only.jsonl"})
_TEST_WORD = re.compile(r"(?<![a-z0-9])tests?(?![a-z0-9])")
_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


class SealedPathError(ValueError):
    """An input path names or resolves to a sealed split."""


def _name_reason(raw: str) -> str | None:
    name = raw.lower()
    if name in SEALED_NAMES:
        return f"sealed split file name: {raw}"
    if fnmatch.fnmatch(name, "eval-*-test-*"):
        return f"sealed evaluation file name: {raw}"
    if "holdout" in name or "rl_only" in name or "rl-only" in name:
        return f"sealed split file name: {raw}"
    if _TEST_WORD.search(name):
        return f"test split file name: {raw}"
    return None


def _hard_link_reason(path: Path) -> str | None:
    """Whether ``path`` is the same file (device and inode) as a sealed-named sibling.
    Lists the directory and calls ``os.stat`` only; never opens any file."""
    try:
        own = os.stat(path)
        siblings = sorted(os.listdir(path.parent))
    except OSError:
        return None
    for sibling in siblings:
        if _name_reason(sibling) is None:
            continue
        try:
            other = os.stat(path.parent / sibling)
        except OSError:
            continue
        if (other.st_dev, other.st_ino) == (own.st_dev, own.st_ino):
            return f"same file as sealed-named sibling: {sibling}"
    return None


def sealed_reason(path: Path) -> str | None:
    """Why ``path`` is refused, or ``None`` if it may be opened. Never opens the file."""
    resolved = path.resolve()
    for candidate in (path, resolved):
        reason = _name_reason(candidate.name)
        if reason is not None:
            return reason
    linked = _hard_link_reason(resolved)
    if linked is not None:
        return linked
    for part in resolved.parent.parts:
        if "holdout" in part.lower():
            return f"path is inside a holdout directory: {part}"
    return None


def guard(paths: Sequence[Path]) -> None:
    """Refuse the whole run if any path is sealed; call before opening anything."""
    for path in paths:
        reason = sealed_reason(path)
        if reason is not None:
            raise SealedPathError(f"refusing {path}: {reason}")


class _Reader:
    """Reads input files once each, hashing the bytes and recording every file opened."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.digests: dict[str, str] = {}

    def rows(self, path: Path) -> list[dict[str, Any]]:
        self.opened.append(str(path.resolve()))
        data = path.read_bytes()
        self.digests[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {number} is not a JSON object")
            rows.append(row)
        return rows


def _label(path: Path) -> str:
    return _LABEL.sub("_", path.stem)


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("id", row.get("task_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("an IDs-only row needs a text id or task_id")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="task JSONL files (explicit paths)")
    parser.add_argument(
        "--id-join",
        nargs=2,
        action="append",
        default=[],
        metavar=("IDS", "POOL"),
        type=Path,
        help="IDs-only JSONL and the pool JSONL its texts are joined from",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--commit", default=None, help="tool git commit to record")
    args = parser.parse_args(argv)
    if not args.inputs and not args.id_join:
        parser.error("give at least one input")

    plain: list[Path] = list(args.inputs)
    joins: list[tuple[Path, Path]] = [(ids, pool) for ids, pool in args.id_join]
    everything = plain + [path for pair in joins for path in pair]
    try:
        guard(everything)
    except SealedPathError as error:
        print(str(error), file=sys.stderr)
        return 2
    labels = [_label(path) for path in plain] + [_label(ids) for ids, _ in joins]
    if len(set(labels)) != len(labels):
        parser.error("input file names must give distinct labels")

    out_dir: Path = args.out.resolve()
    planned = [out_dir / f"scan-{label}.jsonl" for label in labels]
    planned += [out_dir / "summary.json", out_dir / "record.json"]
    existing = [target.name for target in planned if target.exists()]
    if existing:
        print(
            f"refusing to overwrite existing outputs in {out_dir}: {', '.join(existing)}",
            file=sys.stderr,
        )
        return 3

    reader = _Reader()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    outputs: list[str] = []

    def emit(label: str, scans: list[TaskScan], extra: dict[str, Any]) -> None:
        target = out_dir / f"scan-{label}.jsonl"
        with target.open("x", encoding="utf-8") as stream:
            for scan in scans:
                if scan.hits:
                    stream.write(json.dumps(scan.to_row(), sort_keys=True) + "\n")
        outputs.append(target.name)
        summaries[label] = {**summarize(scans), **extra}

    for path in plain:
        rows = reader.rows(path)
        emit(_label(path), [scan_task(row) for row in rows], {})
        inputs.append({"kind": "tasks", "label": _label(path), "path": str(path.resolve())})
    for ids_path, pool_path in joins:
        ids = [_row_id(row) for row in reader.rows(ids_path)]
        pool: dict[str, dict[str, Any]] = {}
        for row in reader.rows(pool_path):
            pool.setdefault(_row_id(row), row)
        missing = sorted(task_id for task_id in ids if task_id not in pool)
        scans = [scan_task(pool[task_id]) for task_id in ids if task_id in pool]
        emit(_label(ids_path), scans, {"ids": len(ids), "unjoined_ids": missing})
        inputs.append(
            {
                "kind": "id_join",
                "label": _label(ids_path),
                "path": str(ids_path.resolve()),
                "pool_path": str(pool_path.resolve()),
            }
        )
    for entry in inputs:
        entry["sha256"] = reader.digests[entry["path"]]
        if "pool_path" in entry:
            entry["pool_sha256"] = reader.digests[entry["pool_path"]]
    atomic_json(
        out_dir / "summary.json",
        {
            "inputs": inputs,
            "ruleset_version": RULESET_VERSION,
            "summaries": summaries,
            "tool_commit": args.commit,
        },
    )
    outputs.append("summary.json")
    atomic_json(
        out_dir / "record.json",
        {"files_opened": reader.opened, "outputs": sorted(outputs + ["record.json"])},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
