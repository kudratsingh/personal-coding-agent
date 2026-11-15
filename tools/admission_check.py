"""Admission checks for the development set and the RL remainder (``scratchcoder.data.admission``).

Subcommands, all with explicit paths (nothing is globbed or walked; sealed inputs are refused
before anything is opened, exit 2):

- ``plan``: count the runs each set needs and how many the cache already satisfies. Runs
  nothing.
- ``execute``: run the missing reference runs and negative controls through the evaluation
  path of record, appending every outcome (unresolved ones too) to ``--cache`` as it arrives.
  It refuses to start without ``--receipt``, a passing execution-window receipt written by
  ``tools.mac_window_check exec`` for this very command, and takes the runtime image ID from
  it. ``--limit`` bounds the number of calls in one invocation.
- ``decide``: combine the cache with the static gate and the reviewed exclusions, and write
  ``decisions-<set>.jsonl`` and ``summary.json`` under ``--out``. Runs nothing.

Sets: ``dev`` is every task of ``--dev`` whose adjudication record (``--dev-adjudication``)
is not ``quarantined``; ``remainder`` is the screening record's ``eligible`` IDs minus the
candidates the exclusions record marks ``excluded_*``, joined to ``--train``. Reviewed
exclusions come from ``--dev-quarantine`` (``quarantine_static_clear``, read-confirmed, with
their reasons, and ``retained_existing``) and ``--verifier-filter``
(``verifier_quarantine_clear``); its provisional lists stay warnings.
"""

import argparse
import fcntl
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scratchcoder.common.records import atomic_json
from scratchcoder.data import admission
from scratchcoder.data.verifier_scan import RULESET_VERSION, scan_task
from scratchcoder.eval.tasks import Task, load_tasks
from tools.verifier_scan import SealedPathError, guard

if TYPE_CHECKING:
    from scratchcoder.eval import parallel

RECEIPT_MAX_AGE = timedelta(minutes=30)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sets(args: argparse.Namespace) -> dict[str, dict[str, Task]]:
    adjudicated = {
        row["id"]: row["status"]
        for row in (
            json.loads(line)
            for line in args.dev_adjudication.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    dev = {
        task.id: task for task in load_tasks(args.dev) if adjudicated.get(task.id) != "quarantined"
    }
    train = {task.id: task for task in load_tasks(args.train)}
    eligible = _json(args.screening)["eligible"]
    excluded = {
        item["id"]
        for item in _json(args.exclusions)["candidates"]
        if str(item.get("status", "")).startswith("excluded")
    }
    remainder = {task_id: train[task_id] for task_id in sorted(set(eligible) - excluded)}
    return {"dev": dev, "remainder": remainder}


def reviewed(args: argparse.Namespace) -> dict[str, dict[str, list[str]]]:
    """Reviewed exclusions and warnings per task ID, with their recorded reasons."""
    exclusions: dict[str, list[str]] = {}
    warnings: dict[str, list[str]] = {}
    quarantine = _json(args.dev_quarantine)
    for item in quarantine["quarantine_static_clear"]:
        exclusions.setdefault(item["id"], []).extend(
            f"reviewed:{reason}" for reason in item["reasons"]
        )
    for text in quarantine.get("retained_existing", {}).values():
        exclusions.setdefault(text.split()[0], []).append("reviewed:owner_ruling")
    for item in quarantine.get("provisional_heuristic", []):
        warnings.setdefault(item["id"], []).extend(
            f"provisional:{reason}" for reason in item["reasons"]
        )
    verifier = _json(args.verifier_filter)
    for task_id in verifier["verifier_quarantine_clear"]:
        exclusions.setdefault(task_id, []).append("reviewed:reference_dependent_tests")
    for task_id in verifier.get("verifier_provisional", []):
        warnings.setdefault(task_id, []).append("provisional:reviewed_heuristic")
    return {"exclusions": exclusions, "warnings": warnings}


def identities(image_id: str) -> admission.Identities:
    from scratchcoder.eval import harness, parallel
    from scratchcoder.eval.execution_policy import execution_policy, policy_sha256
    from scratchcoder.sandbox import runner

    policy = execution_policy(
        harness.isolated_execution,
        harness.HOST_RETRIES,
        unexecuted_statuses=(),
        input_limit=parallel.INPUT_LIMIT,
    )
    return admission.Identities(policy_sha256(policy), runner.IMAGE, image_id)


SECRET_VARIABLES = ("SSH_AUTH_SOCK",)


def _receipt_image(path: Path, now: datetime, argv: Sequence[str]) -> str:
    """The runtime image ID from a passing ``exec`` window receipt written for this very
    command in the last 30 minutes; anything else refuses execution."""
    from scratchcoder.sandbox import runner

    receipt = _json(path)
    checked = datetime.fromisoformat(receipt["checked_at"])
    if (
        receipt.get("kind") != "mac-execution-window-check"
        or receipt.get("status") != "passed"
        or not now - RECEIPT_MAX_AGE <= checked <= now
    ):
        raise ValueError("execute needs a passing execution-window receipt from this window")
    command = receipt.get("command")
    if not isinstance(command, list) or command[-len(argv) :] != list(argv):
        raise ValueError("the window receipt was written for another command")
    image = receipt["checks"]["image"]
    if not image.get("passed") or image.get("image") != runner.IMAGE:
        raise ValueError("the window receipt's image check did not pass for the pinned image")
    if image.get("found_id") != image.get("expected_id"):
        raise ValueError("the window receipt's image ID does not match its expected ID")
    if any(os.environ.get(name) for name in SECRET_VARIABLES):
        raise ValueError("execute refuses to run with an SSH agent socket in its environment")
    return str(image["found_id"])


def pending_runs(
    sets: dict[str, dict[str, Task]], cache: Path, ids: admission.Identities
) -> list[tuple[admission.Run, str]]:
    tasks = {task_id: task for group in sets.values() for task_id, task in group.items()}
    done = admission.usable(admission.read_cache(cache), tasks, ids)
    runs: list[tuple[admission.Run, str]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        digest = admission.task_sha256(task)
        runs.extend((run, digest) for run in admission.plan(task) if run.key not in done)
    # Spread each task's runs over the whole pass: every first reference run, then every
    # second, then every third, then the controls. One burst of host load then cannot reach
    # all three reference runs of a task, and --limit proceeds breadth-first.
    order = {kind: position for position, kind in enumerate(admission.RUN_ORDER)}
    runs.sort(key=lambda item: (order[item[0].kind], item[0].index, item[0].task_id))
    return runs


def _repair_tail(path: Path) -> None:
    """Cut a torn last line (an interrupted write) so the next row starts on its own line."""
    if not path.exists():
        return
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    if cut != len(raw):
        with path.open("r+b") as stream:
            stream.truncate(cut)


def execute(args: argparse.Namespace, sets: dict[str, dict[str, Task]]) -> int:
    from scratchcoder.eval import harness

    image_id = _receipt_image(args.receipt, datetime.now(UTC), args.argv)
    ids = identities(image_id)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.cache.parent / (args.cache.name + ".lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("refused: another execute holds the cache", file=sys.stderr)
        return 5
    _repair_tail(args.cache)
    runs = pending_runs(sets, args.cache, ids)[: args.limit]
    print(f"executing {len(runs)} calls", file=sys.stderr)
    record_runs(
        runs,
        args.cache,
        ids,
        run_id=args.run_id,
        workers=args.workers,
        call=harness.isolated_execution,
    )
    return 0


def record_runs(
    runs: Sequence[tuple[admission.Run, str]],
    cache: Path,
    ids: admission.Identities,
    *,
    run_id: str,
    workers: int,
    call: "parallel.Execute",
) -> None:
    """Execute ``runs`` through ``call`` and append every outcome, unresolved ones too, to
    ``cache`` as it arrives. ``call`` is the evaluation path of record
    (``harness.isolated_execution``), or a wrapper that only counts or bounds its calls;
    the caller holds the cache lock and has repaired a torn tail."""
    from scratchcoder.eval import parallel

    by_key = {run.key: (run, digest) for run, digest in runs}
    with cache.open("a", encoding="utf-8") as stream:

        def record(outcome: parallel.Outcome) -> None:
            run, digest = by_key[outcome.key]  # type: ignore[index]
            row = admission.cache_row(
                run,
                digest,
                ids,
                status=outcome.status,
                resolved=outcome.resolved,
                host_failures=outcome.host_failures,
                elapsed_seconds=outcome.elapsed_seconds,
            )
            row["run_id"] = run_id
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

        parallel.run_calls(
            (parallel.Call(run.key, run.code, run.tests, run.mode) for run, _ in runs),
            call,
            on_outcome=record,
            workers=workers,
            max_consecutive_unresolved=20,
        )


def decide(args: argparse.Namespace, sets: dict[str, dict[str, Task]]) -> int:
    names = [f"decisions-{label}.jsonl" for label in sets] + ["summary.json"]
    if any((args.out / name).exists() for name in names):
        print("refused: an output already exists", file=sys.stderr)
        return 3
    image_id = args.image_id
    ids = identities(image_id)
    tasks = {task_id: task for group in sets.values() for task_id, task in group.items()}
    rows = admission.read_cache(args.cache)
    done = admission.usable(rows, tasks, ids)
    notes = reviewed(args)
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "format": "admission-summary-v1",
        "admission": admission.ADMISSION_VERSION,
        "scan_ruleset": RULESET_VERSION,
        "identities": {
            "evaluator_sha256": ids.evaluator_sha256,
            "image": ids.image,
            "image_id": ids.image_id,
        },
        "cache_rows": len(rows),
        "cache_rows_with_this_image_id": sum(row.get("image_id") == image_id for row in rows),
        "cache_rows_unresolved": sum(not row.get("resolved") for row in rows),
        "runs_satisfied": len(done),
        "sets": {},
    }
    for label, group in sets.items():
        decisions = []
        for task_id in sorted(group):
            task = group[task_id]
            decisions.append(
                admission.decide(
                    task,
                    scan_task(asdict(task)),
                    done,
                    notes["exclusions"].get(task_id, ()),
                    notes["warnings"].get(task_id, ()),
                )
            )
        with (args.out / f"decisions-{label}.jsonl").open("x", encoding="utf-8") as stream:
            for decision in decisions:
                stream.write(json.dumps(decision.to_row(), sort_keys=True) + "\n")
        summary["sets"][label] = admission.union_counts(decisions)
    if "dev" in sets:
        none_only = set(_json(args.dev_quarantine)["quarantine_recommended_none_only"]["ids"])
        dev_rows = [
            json.loads(line) for line in (args.out / "decisions-dev.jsonl").read_text().splitlines()
        ]
        gated = {row["id"] for row in dev_rows if "none_only_checks" in row["defects"]}
        summary["sets"]["dev"]["phase_b_none_only_reproduced"] = {
            "recorded": len(none_only),
            "gated": len(none_only & gated),
            "missing": sorted(none_only - gated),
        }
    if "remainder" in sets:
        from scratchcoder.sandbox.reward import REWARD_GRAMMAR_VERSION, unsupported_reason

        admitted = [
            json.loads(line)["id"]
            for line in (args.out / "decisions-remainder.jsonl").read_text().splitlines()
            if json.loads(line)["status"] == "admitted"
        ]
        unparseable = {
            task_id: refused.reason
            for task_id in admitted
            if (refused := unsupported_reason(sets["remainder"][task_id])) is not None
        }
        summary["sets"]["remainder"]["reward_grammar"] = REWARD_GRAMMAR_VERSION
        summary["sets"]["remainder"]["admitted_but_not_reward_parseable"] = unparseable
        summary["sets"]["remainder"]["rl_eligible_after_admission"] = len(admitted) - len(
            unparseable
        )
    atomic_json(args.out / "summary.json", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("plan", "execute", "decide"))
    for name in (
        "dev",
        "dev-adjudication",
        "train",
        "screening",
        "exclusions",
        "dev-quarantine",
        "verifier-filter",
        "cache",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, help="execute: the window receipt")
    parser.add_argument("--image-id", help="plan/decide: the runtime image ID of the cache rows")
    parser.add_argument("--run-id", default="", help="execute: a label stored on every row")
    parser.add_argument("--limit", type=int, default=20000, help="execute: at most this many calls")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="decide: a new output directory")
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    inputs = [
        args.dev,
        args.dev_adjudication,
        args.train,
        args.screening,
        args.exclusions,
        args.dev_quarantine,
        args.verifier_filter,
        args.cache,
        *([args.receipt] if args.receipt is not None else []),
    ]
    try:
        guard(inputs)
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    sets = load_sets(args)
    if args.command == "execute":
        if args.receipt is None:
            parser.error("execute needs --receipt")
        return execute(args, sets)
    if args.image_id is None:
        parser.error(f"{args.command} needs --image-id")
    if args.command == "plan":
        ids = identities(args.image_id)
        runs = pending_runs(sets, args.cache, ids)
        total = sum(len(admission.plan(task)) for group in sets.values() for task in group.values())
        print(
            json.dumps(
                {
                    "tasks": {label: len(group) for label, group in sets.items()},
                    "runs_total": total,
                    "runs_pending": len(runs),
                    "evaluator_sha256": ids.evaluator_sha256,
                }
            )
        )
        return 0
    if args.out is None:
        parser.error("decide needs --out")
    return decide(args, sets)


if __name__ == "__main__":
    raise SystemExit(main())
