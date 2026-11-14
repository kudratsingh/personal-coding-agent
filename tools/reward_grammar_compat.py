"""Static compatibility comparison of the reward channel's check grammar, v1 against v2.

Text only: it parses hidden tests with :mod:`ast` and never runs any code. For each task set
it reports, task by task, whether the reward channel could score the task under the previous
grammar (``reward-grammar-v1``) and under the current one, and why not; then every task that
v2 newly excludes, and every check shape v2 refuses or keeps, with counts.

Task sets (every path is explicit; nothing is globbed or walked):

- ``dev``: every task of ``--dev`` (the pool's dev split).
- ``rl_candidates``: the IDs of ``--candidates`` (an IDs-only JSONL), joined to ``--train``.
- ``remainder``: the screening record's ``eligible`` IDs (``--screening``), minus the
  candidates the exclusions record (``--exclusions``) marks ``excluded_*``, minus the
  ``verifier_quarantine_clear`` IDs of ``--verifier-filter``: the provisional RL remainder.

Paths go through :func:`tools.verifier_scan.guard` first, so a sealed split is refused before
anything is opened (exit 2). Outputs are written under ``--out``, which must not hold them
already (exit 3): ``compat-<set>.jsonl`` (one row per task), ``summary.json`` and
``record.json`` (every input's SHA-256, the grammar versions and the tool commit).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.sandbox.reward import (
    GRAMMAR_V1,
    GRAMMAR_V2_REASONS,
    REWARD_GRAMMAR_VERSION,
    Unsupported,
    check_forms,
    explain_function_checks,
    explain_task,
)
from tools.verifier_scan import SealedPathError, guard


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome(parsed: object) -> dict[str, Any]:
    if isinstance(parsed, Unsupported):
        return {"parseable": False, "reason": parsed.reason, "index": parsed.index}
    return {"parseable": True, "reason": None, "index": None}


def compare_task(task: Task) -> dict[str, Any]:
    """One task's v1 and v2 outcomes, and every check shape v2 refuses or keeps."""
    current = explain_task(task)
    if task.mode == "io":
        previous: object = current  # I/O parsing did not change.
        forms: list[tuple[int, str, str | None]] = []
    else:
        previous = explain_function_checks(task.hidden_tests, grammar=GRAMMAR_V1)
        forms = check_forms(task.hidden_tests)
    v1, v2 = _outcome(previous), _outcome(current)
    newly = v1["parseable"] and not v2["parseable"]
    return {
        "id": task.id,
        "source": task.source,
        "mode": task.mode,
        "v1": v1,
        "v2": v2,
        "newly_unsupported": newly,
        "refused_checks": [
            {"index": index, "reason": reason, "form": form}
            for index, reason, form in forms
            if reason in GRAMMAR_V2_REASONS
        ],
        "kept_checks": [form for _, reason, form in forms if reason == "supported"],
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    newly = [row for row in rows if row["newly_unsupported"]]
    # A newly excluded task may break several v2 rules; count every refused check shape.
    refused_forms: Counter[str] = Counter()
    refused_tasks_by_form: Counter[str] = Counter()
    for row in newly:
        shapes = [f"{item['reason']}: {item['form']}" for item in row["refused_checks"]]
        refused_forms.update(shapes)
        refused_tasks_by_form.update(set(shapes))
    # Shapes v2 refuses in tasks that were already unparseable for another reason.
    already: Counter[str] = Counter()
    for row in rows:
        if not row["v1"]["parseable"]:
            already.update(f"{item['reason']}: {item['form']}" for item in row["refused_checks"])
    kept_wrappers: Counter[str] = Counter()
    for row in rows:
        if row["v2"]["parseable"]:
            kept_wrappers.update(form for form in row["kept_checks"] if " on call " not in form)
    return {
        "tasks": len(rows),
        "v1_parseable": sum(row["v1"]["parseable"] for row in rows),
        "v2_parseable": sum(row["v2"]["parseable"] for row in rows),
        "newly_unsupported": len(newly),
        "newly_supported": sum(
            row["v2"]["parseable"] and not row["v1"]["parseable"] for row in rows
        ),
        "newly_unsupported_by_reason": dict(
            sorted(Counter(row["v2"]["reason"] for row in newly).items())
        ),
        "newly_unsupported_by_source": dict(
            sorted(Counter(row["source"] for row in newly).items())
        ),
        "newly_unsupported_ids": sorted(row["id"] for row in newly),
        "refused_check_forms": dict(refused_forms.most_common()),
        "refused_check_forms_task_counts": dict(refused_tasks_by_form.most_common()),
        "refused_forms_in_already_unparseable_tasks": dict(already.most_common()),
        "kept_wrapper_check_forms_in_parseable_tasks": dict(kept_wrappers.most_common()),
        "v2_unsupported_by_reason": dict(
            sorted(
                Counter(row["v2"]["reason"] for row in rows if not row["v2"]["parseable"]).items()
            )
        ),
    }


def _ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ids.append(row.get("id") or row["task_id"])
    return ids


def remainder_ids(screening: Path, exclusions: Path, verifier_filter: Path) -> list[str]:
    eligible = json.loads(screening.read_text(encoding="utf-8"))["eligible"]
    excluded = {
        item["id"]
        for item in json.loads(exclusions.read_text(encoding="utf-8"))["candidates"]
        if str(item.get("status", "")).startswith("excluded")
    }
    quarantined = set(
        json.loads(verifier_filter.read_text(encoding="utf-8"))["verifier_quarantine_clear"]
    )
    return sorted(set(eligible) - excluded - quarantined)


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("dev", "train", "candidates", "screening", "exclusions", "verifier-filter"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="the tool's git commit, recorded")
    args = parser.parse_args(argv)
    inputs = [
        args.dev,
        args.train,
        args.candidates,
        args.screening,
        args.exclusions,
        args.verifier_filter,
    ]
    try:
        guard(inputs)
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    names = ["compat-dev.jsonl", "compat-rl_candidates.jsonl", "compat-remainder.jsonl"]
    names += ["summary.json", "record.json"]
    if any((args.out / name).exists() for name in names):
        print("refused: an output already exists", file=sys.stderr)
        return 3
    args.out.mkdir(parents=True, exist_ok=True)
    dev = list(load_tasks(args.dev))
    train = {task.id: task for task in load_tasks(args.train)}
    candidates = [train[task_id] for task_id in _ids(args.candidates)]
    remainder = [
        train[task_id]
        for task_id in remainder_ids(args.screening, args.exclusions, args.verifier_filter)
    ]
    sets = {"dev": dev, "rl_candidates": candidates, "remainder": remainder}
    summary: dict[str, Any] = {
        "format": "reward-grammar-compat-v1",
        "grammar_previous": GRAMMAR_V1,
        "grammar_current": REWARD_GRAMMAR_VERSION,
        "sets": {},
    }
    for label, tasks in sets.items():
        rows = [compare_task(task) for task in tasks]
        _write_rows(args.out / f"compat-{label}.jsonl", rows)
        summary["sets"][label] = summarize(rows)
    atomic_json(args.out / "summary.json", summary)
    atomic_json(
        args.out / "record.json",
        {
            "tool": "tools/reward_grammar_compat.py",
            "commit": args.commit,
            "executed_code": False,
            "inputs": {str(path.resolve()): _sha256(path) for path in inputs},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
