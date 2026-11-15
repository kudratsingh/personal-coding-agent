"""Holdout integrity scan: one approved, logged, static read of the sealed pool test split.

It answers one question: which *subject* tasks (tasks that may be trained on) share a task
family with a sealed test task (:mod:`scratchcoder.data.integrity_scan`). It generates,
executes and scores nothing, and its outputs name subjects only: no test prompt, reference,
test, test ID or per-test value is written, printed or logged.

``request``
    Writes ``approval-request.json`` in ``--out``: the scan's code identity (SHA-256 of
    every module it runs), the rules version, the sealed file's path and its pinned SHA-256
    and task count (from the pool build report, ``--pool-report``), every subject file's
    SHA-256 and task count, the ledger and the output directory. The sealed file is not
    opened (``os.lstat`` only). The request carries an empty ``approval`` block.
``run --approval FILE``
    FILE is the request with its ``approval`` block filled by the owner (``approved_by``,
    ``approved_at``, ``statement``). The run refuses unless the code, subjects, sealed path,
    ledger and outputs still match the request, the sealed path is a regular file named
    ``test.jsonl`` (not a link), the outputs do not exist yet and the approval has never
    been used. Then it appends ``integrity_scan_opened`` to the seal ledger, and only then
    opens the sealed file, once, checks its SHA-256 and task count, scans, writes
    ``exclusions.json``, ``summary.json`` and ``record.json`` and appends
    ``integrity_scan_closed`` (``completed`` with the outputs' SHA-256 values and the
    aggregates, or ``failed`` with an error class name only).

Exit codes: 0 done; 2 refused before the sealed file was opened (nothing logged); 3 an
output exists; 4 the sealed file's SHA-256 or task count differs from the pinned values
(logged as failed); 5 any other failure after the ledger event (logged as failed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import scratchcoder.data.adapters as adapters_module
import scratchcoder.data.clusters as clusters_module
import scratchcoder.data.integrity_scan as scan_module
import scratchcoder.data.pool as pool_module
import scratchcoder.eval.seal as seal_module
import scratchcoder.eval.tasks as tasks_module
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
from scratchcoder.eval.tasks import load_tasks
from tools.verifier_scan import SealedPathError, guard

REQUEST_KIND = "holdout-integrity-scan-request"
SCHEMA_VERSION = 1
SEALED_NAME = "test.jsonl"
REQUEST_NAME = "approval-request.json"
OUTPUTS = ("exclusions.json", "summary.json", "record.json")
APPROVAL_FIELDS = ("approved_by", "approved_at", "statement")
SUBJECT_SPLIT = "train"


class Refused(Exception):
    """Refused before the sealed file was opened."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def code_identity() -> dict[str, str]:
    """SHA-256 of every module whose code decides what the scan reads, finds or writes."""
    modules = {
        "tools/holdout_integrity_scan.py": Path(__file__),
        "scratchcoder/data/integrity_scan.py": Path(scan_module.__file__),
        "scratchcoder/data/clusters.py": Path(clusters_module.__file__),
        "scratchcoder/data/pool.py": Path(pool_module.__file__),
        "scratchcoder/data/adapters.py": Path(adapters_module.__file__),
        "scratchcoder/eval/tasks.py": Path(tasks_module.__file__),
        "scratchcoder/eval/seal.py": Path(seal_module.__file__),
    }
    return {name: _sha_file(path) for name, path in sorted(modules.items())}


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


def _subject(label: str, path: Path) -> tuple[dict[str, Any], list[Any]]:
    guard([path])
    data = path.read_bytes()
    tasks = list(load_tasks(path))
    if _sha_bytes(data) != _sha_file(path):
        raise Refused(f"subject {label} changed while it was read")
    if any(task.split != SUBJECT_SPLIT for task in tasks):
        raise Refused(f"subject {label} must hold {SUBJECT_SPLIT}-split tasks only")
    entry = {"label": label, "path": str(path.resolve()), "sha256": _sha_bytes(data)}
    return entry | {"tasks": len(tasks)}, tasks


def _labelled(values: Sequence[str]) -> list[tuple[str, Path]]:
    pairs = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise Refused(f"a subject is LABEL=PATH, not {value!r}")
        pairs.append((label, Path(path)))
    if len({label for label, _ in pairs}) != len(pairs):
        raise Refused("subject labels must be unique")
    return pairs


def build_request(
    *,
    sealed: Path,
    pool_report: Path,
    subjects: Sequence[tuple[str, Path]],
    ledger: Path,
    out: Path,
) -> dict[str, Any]:
    guard([pool_report])
    _sealed_stat(sealed)
    report = json.loads(pool_report.read_text(encoding="utf-8"))
    pinned = report["files"]["test"]
    entries = [_subject(label, path)[0] for label, path in subjects]
    if not entries:
        raise Refused("at least one subject set is required")
    commit, dirty = git_state()
    return {
        "kind": REQUEST_KIND,
        "schema_version": SCHEMA_VERSION,
        "scan_format": SCAN_FORMAT,
        "rules_version": RULES_VERSION,
        "code": code_identity(),
        "git": {"commit": commit, "dirty": dirty},
        "sealed": {
            "path": str(sealed.resolve()),
            "sha256": pinned["file_sha256"],
            "tasks": pinned["tasks"],
            "pinned_by": {"path": str(pool_report.resolve()), "sha256": _sha_file(pool_report)},
        },
        "subjects": entries,
        "ledger": str(ledger.resolve()),
        "out": str(out.resolve()),
        "outputs": list(OUTPUTS),
        "performs": {"generation": False, "execution": False, "scoring": False},
    }


def request(args: argparse.Namespace) -> int:
    out: Path = args.out
    if (out / REQUEST_NAME).exists():
        print("refused: the request already exists", file=sys.stderr)
        return 3
    body = build_request(
        sealed=args.sealed,
        pool_report=args.pool_report,
        subjects=_labelled(args.subject),
        ledger=args.ledger,
        out=out,
    )
    out.mkdir(parents=True, exist_ok=True)
    document = {"request": body, "approval": dict.fromkeys(APPROVAL_FIELDS)}
    with (out / REQUEST_NAME).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out / REQUEST_NAME}; it needs the owner's approval block before a run")
    return 0


def _checked(document: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, list[Any]]]]:
    """The approved request, re-verified against the current code and inputs."""
    body, approval = document.get("request"), document.get("approval")
    if not isinstance(body, dict) or not isinstance(approval, dict):
        raise Refused("not an approval document")
    if body.get("kind") != REQUEST_KIND or body.get("schema_version") != SCHEMA_VERSION:
        raise Refused("not a holdout integrity-scan request of this schema")
    for name in APPROVAL_FIELDS:
        value = approval.get(name)
        if not isinstance(value, str) or not value.strip():
            raise Refused(f"the approval block's {name} is empty: the scan is not approved")
    if set(approval) != set(APPROVAL_FIELDS):
        raise Refused("the approval block has unexpected fields")
    if body["code"] != code_identity():
        raise Refused("the scan's code differs from the approved code")
    if body["scan_format"] != SCAN_FORMAT or body["rules_version"] != RULES_VERSION:
        raise Refused("the scan's rules differ from the approved rules")
    if body["outputs"] != list(OUTPUTS) or body["performs"] != {
        "generation": False,
        "execution": False,
        "scoring": False,
    }:
        raise Refused("the request's outputs or scope differ from this tool's")
    sealed = Path(body["sealed"]["path"])
    if str(sealed.resolve()) != body["sealed"]["path"]:
        raise Refused("the sealed path no longer resolves to the approved path")
    _sealed_stat(sealed)
    pinned_by = body["sealed"]["pinned_by"]
    if _sha_file(Path(pinned_by["path"])) != pinned_by["sha256"]:
        raise Refused("the pool report that pins the sealed file changed")
    subjects = []
    for entry in body["subjects"]:
        current, tasks = _subject(entry["label"], Path(entry["path"]))
        if current != entry:
            raise Refused(f"subject {entry['label']} differs from the approved file")
        subjects.append((entry["label"], tasks))
    out = Path(body["out"])
    if any((out / name).exists() for name in OUTPUTS):
        raise FileExistsError("an output already exists")
    return body, subjects


def _write(path: Path, value: dict[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_bytes(data)


def _read_sealed(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def run(args: argparse.Namespace) -> int:
    approval_bytes = args.approval.read_bytes()
    approval_sha256 = _sha_bytes(approval_bytes)
    try:
        body, subjects = _checked(json.loads(approval_bytes))
    except FileExistsError:
        print("refused: an output already exists", file=sys.stderr)
        return 3
    ledger = SealLedger(Path(body["ledger"]))
    out = Path(body["out"])
    out.mkdir(parents=True, exist_ok=True)
    opened = ledger.open_integrity_scan(
        approval_sha256,
        tool="tools/holdout_integrity_scan.py",
        code=body["code"],
        rules_version=body["rules_version"],
        scan_format=body["scan_format"],
        sealed_path=body["sealed"]["path"],
        sealed_sha256_pinned=body["sealed"]["sha256"],
        subjects=[
            {key: entry[key] for key in ("label", "sha256", "tasks")} for entry in body["subjects"]
        ],
    )
    try:
        data = _read_sealed(Path(body["sealed"]["path"]))
        digest = _sha_bytes(data)
        if digest != body["sealed"]["sha256"]:
            del data
            ledger.close_integrity_scan(approval_sha256, "failed", error="sealed_sha256_mismatch")
            print("failed: the sealed file's SHA-256 differs from the pinned value")
            return 4
        test = parse_sealed_tasks(data)
        del data
        if len(test) != body["sealed"]["tasks"]:
            ledger.close_integrity_scan(approval_sha256, "failed", error="sealed_task_count")
            print("failed: the sealed file's task count differs from the pinned value")
            return 4
        results, counts = scan(
            [item_of(task) for task in test],
            {label: [item_of(task) for task in tasks] for label, tasks in subjects},
        )
        del test
        labels = [label for label, _ in subjects]
        exclusions = {
            "format": SCAN_FORMAT,
            "approval_sha256": approval_sha256,
            "excluded": {
                label: sorted(r.id for r in results if r.label == label and r.status == "excluded")
                for label in labels
            },
            "flagged": {
                label: sorted(r.id for r in results if r.label == label and r.status == "flagged")
                for label in labels
            },
            "subjects": [result.to_json() for result in results],
        }
        summary = {
            "format": SCAN_FORMAT,
            "approval_sha256": approval_sha256,
            "rules_version": body["rules_version"],
            "counts": counts,
        }
        hashes = {
            "exclusions.json": _write(out / "exclusions.json", exclusions),
            "summary.json": _write(out / "summary.json", summary),
        }
        hashes["record.json"] = _write(
            out / "record.json",
            {
                "format": SCAN_FORMAT,
                "approval_sha256": approval_sha256,
                "approval": json.loads(approval_bytes)["approval"],
                "code": body["code"],
                "git_at_request": body["git"],
                "sealed": {"path": body["sealed"]["path"], "sha256": digest},
                "subjects": body["subjects"],
                "ledger": body["ledger"],
                "ledger_opened_at": opened["at"],
                "outputs": dict(hashes),
                "performed": {"generation": False, "execution": False, "scoring": False},
            },
        )
        ledger.close_integrity_scan(approval_sha256, "completed", outputs=hashes, counts=counts)
    except BaseException as error:  # noqa: BLE001 - every failure is logged, never quoted
        try:
            ledger.close_integrity_scan(approval_sha256, "failed", error=type(error).__name__)
        except SealError:
            pass
        print(f"failed: {type(error).__name__}", file=sys.stderr)
        return 5
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("request", help="write the approval request; opens nothing sealed")
    make.add_argument("--sealed", type=Path, required=True)
    make.add_argument("--pool-report", type=Path, required=True)
    make.add_argument("--subject", action="append", default=[], help="LABEL=PATH, repeatable")
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


if __name__ == "__main__":
    raise SystemExit(main())
