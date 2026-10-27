"""The Pod's RL task files, written on the Mac: withheld tasks, the dev subset and the facts.

``scratchcoder rl-pod-export`` reads the pool's train split, the RL screening record (its
kept list) and the pool's dev split, and writes four new files into a new directory:

- ``rl-tasks.jsonl``: every kept task, in the record's order, as the Pod may hold it
  (``withheld_task``: the hidden tests replaced by the placeholder, the reference kept for
  the more-SFT control);
- ``rl-dev.jsonl``: the fixed dev subset (``dev_subset``: the 200 dev tasks with the
  smallest ``sha256("rl-dev-subset-v1|" + ID)``, in that order), withheld, no references;
- ``rl-facts.json``: ``facts_bytes(task_facts(kept + subset))``, each task's parseability
  and best constant answer's reward computed from the real hidden tests (the answer itself
  is left out);
- ``rl-pod-export.json``: a record naming the SHA-256 and size of the sources and of the
  exact bytes written.

The output is deterministic: the same inputs give the same four files byte for byte (no
time is recorded). Run 2's specification takes the three files' SHA-256s at the owner's
gate, the binding pins them, and the worker refuses any other bytes; ``rl-bridge`` is given
the two task files' SHA-256s (``--pod-reward-sha256``, ``--pod-dev-sha256``) and refuses a
request naming another. The sealed test split and the holdouts are refused by name, through
a link too (``bridge_command._refuse_sealed``), before anything is opened. See
``docs/walkthroughs/rl-bridge.md``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from scratchcoder.eval.export import file_sha256, task_lines
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.rl.bridge_command import _refuse_sealed, read_kept
from scratchcoder.rl.dev import DEV_SUBSET_SALT, DEV_SUBSET_SIZE, dev_subset
from scratchcoder.rl.facts import facts_bytes, task_facts
from scratchcoder.rl.withheld import is_withheld, withheld_task

SCHEMA_VERSION = 1
RECORD_KIND = "rl-pod-export"
RL_TASKS = "rl-tasks.jsonl"
RL_DEV = "rl-dev.jsonl"
RL_FACTS = "rl-facts.json"
RECORD = "rl-pod-export.json"
FILES = {"rl-tasks": RL_TASKS, "rl-dev": RL_DEV, "rl-facts": RL_FACTS}
"""Each Pod file's input name in run 2's specification, and its file name."""
RULE = (
    "rl-tasks: withheld_task of each kept task in screening-record order; rl-dev: withheld_task "
    "of dev_subset(pool dev split) in subset order; one json.dumps(asdict(task), "
    "sort_keys=True) line per task; rl-facts: facts_bytes(task_facts(kept + subset))"
)


def _write_new(path: Path, data: bytes) -> None:
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


def kept_tasks(train_tasks: Path, kept: Path) -> list[Task]:
    """The full records of the kept screening list, in the record's order.

    Every kept task must be a train task with a reference solution (the control's target)
    whose hidden tests are real (never a placeholder)."""
    identifiers = read_kept(kept)
    wanted = set(identifiers)
    found = {task.id: task for task in load_tasks(_refuse_sealed(train_tasks)) if task.id in wanted}
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{len(missing)} kept tasks are not in {train_tasks}, e.g. {missing[0]}")
    tasks = [found[identifier] for identifier in identifiers]
    for label, bad in (
        ("not train-split tasks", [task.id for task in tasks if task.split != "train"]),
        ("without a reference solution", [task.id for task in tasks if task.solution is None]),
        ("already withheld", [task.id for task in tasks if is_withheld(task)]),
    ):
        if bad:
            raise ValueError(f"{len(bad)} kept tasks are {label}, e.g. {bad[0]}")
    return tasks


def subset_tasks(dev_tasks: Path, *, size: int = DEV_SUBSET_SIZE) -> list[Task]:
    """The fixed dev subset of the pool's dev split, with its real hidden tests."""
    tasks = dev_subset(list(load_tasks(_refuse_sealed(dev_tasks))), size)
    withheld = [task.id for task in tasks if is_withheld(task)]
    if withheld:
        raise ValueError(f"the dev file withholds its hidden tests (e.g. {withheld[0]})")
    return tasks


def pod_files(
    kept: Sequence[Task], subset: Sequence[Task]
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    """The three Pod files' bytes and the facts' counts, from the full tasks."""
    facts = task_facts([*kept, *subset])
    unparseable = sorted(identifier for identifier, fact in facts.items() if not fact.parseable)
    rl_unparseable = [task.id for task in kept if not facts[task.id].parseable]
    if rl_unparseable:
        raise ValueError(
            f"{len(rl_unparseable)} kept tasks have hidden tests the reward channel cannot "
            f"parse (e.g. {rl_unparseable[0]}); screening keeps only parseable tasks"
        )
    counts = {"tasks": len(facts), "unparseable": len(unparseable)}
    return (
        task_lines(withheld_task(task) for task in kept),
        task_lines(withheld_task(task) for task in subset),
        facts_bytes(facts),
        counts,
    )


def write_rl_pod_export(
    *,
    train_tasks: Path,
    kept: Path,
    dev_tasks: Path,
    output: Path,
    dev_size: int = DEV_SUBSET_SIZE,
) -> dict[str, Any]:
    """Write the Pod's RL task file, dev-subset file and facts file, and their record.

    ``output`` must be a new directory. ``dev_size`` is the protocol's 200; only the CPU
    smoke and the tests, with a handful of fixture tasks, give another size.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists; an RL Pod export is never replaced")
    tasks = kept_tasks(train_tasks, kept)  # A "kept" record holds at least the floor.
    subset = subset_tasks(dev_tasks, size=dev_size)
    rl_bytes, dev_bytes, facts, counts = pod_files(tasks, subset)
    output.mkdir(parents=True)
    paths = {name: output / file for name, file in FILES.items()}
    for name, data in (("rl-tasks", rl_bytes), ("rl-dev", dev_bytes), ("rl-facts", facts)):
        _write_new(paths[name], data)
    files: dict[str, Any] = {}
    for name, path in paths.items():
        files[name] = {"name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
    files["rl-tasks"].update(
        tasks=len(tasks),
        by_source=dict(sorted(Counter(task.source for task in tasks).items())),
        by_mode=dict(sorted(Counter(task.mode for task in tasks).items())),
    )
    files["rl-dev"].update(
        tasks=len(subset),
        by_source=dict(sorted(Counter(task.source for task in subset).items())),
        by_mode=dict(sorted(Counter(task.mode for task in subset).items())),
    )
    files["rl-facts"].update(counts)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECORD_KIND,
        "rule": RULE,
        "dev_subset": {"size": dev_size, "salt": DEV_SUBSET_SALT},
        "sources": {
            "train_tasks": _identity(train_tasks),
            "kept": _identity(kept),
            "dev_tasks": _identity(dev_tasks),
        },
        "files": files,
    }
    _write_new(output / RECORD, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
    return record


def read_record(path: Path) -> dict[str, Any]:
    """An ``rl-pod-export.json`` record, checked for its kind and its three files."""
    value = json.loads(path.read_text(encoding="utf-8"))
    files = value.get("files") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("kind") != RECORD_KIND
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(files, dict)
        or set(files) != set(FILES)
    ):
        raise ValueError(f"{path} is not an rl-pod-export record")
    for name, row in files.items():
        if not isinstance(row, dict) or row.get("name") != FILES[name]:
            raise ValueError(f"{path}: the {name} entry is malformed")
    return value
