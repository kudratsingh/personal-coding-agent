"""Behavioural equivalence of candidate task pairs, run only in the isolated sandbox.

For a candidate pair (A, B) from a task-cluster manifest (``tools.task_clusters``), each
reference runs against the partner's hidden and public tests, and each against its own
hidden tests as a control. Every call repeats (``--repeats``, default 3) in a fresh
sandbox container. Calls go through ``scratchcoder.sandbox.runner.run`` and nothing else.

``plan``
    Builds the calls and prints their counts. It executes nothing and needs no window.
``run``
    Refuses to start unless it was launched by ``tools.mac_window_check exec`` with
    ``--receipt``: the receipt must be a passing window check written moments earlier
    for this command with the expected image ID, and no withheld credential name may be
    set. Then it runs the planned calls and appends one record per pair to
    ``behavior.jsonl``. A call the sandbox could not decide (infrastructure error,
    unavailable backend, host stall) is retried and, if it persists, stops the run; it is
    never scored.
``merge``
    Copies the manifest to a new file with the recorded outcomes filled in. Adjudication
    statuses are left alone: a reviewer records them (``tools.task_clusters adjudicate``).

Outcomes, per pair (``classify``):

- ``equivalent``: both directions pass every partner test on every repeat, both own-test
  controls pass, and each partner's hidden tests hold at least ``MIN_CHECKS`` checks;
- ``not_equivalent``: both directions fail on every repeat while both controls pass;
- ``unclear``: anything else, with the reasons kept: one direction passes and the other
  fails, a reference passes public but not hidden tests (or the reverse), repeats
  disagree (nondeterminism), tests are weak, a control fails, the modes differ, or the
  partner's called name could not be mapped onto the reference.

Tasks carry no entry point, so the name a partner's tests call is inferred: the
module-level names the partner's reference defines and its tests use. When the reference
defines a different single module-level name of the same kind, one alias line
``partner_name = own_name`` is appended and recorded; otherwise the direction is not
comparable.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.clusters import BEHAVIOR_OUTCOMES, validate_manifest
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.sandbox.runner import IMAGE, Result, Runtime, run
from scratchcoder.sandbox.window import IMAGE_ID, dotenv_names, withheld
from tools.task_clusters import refuse_sealed
from tools.task_pool import limits_for

FORMAT = "task-behavior-v1"
MIN_CHECKS = 2
"""Fewer hidden checks than this make a passing direction too weak to confirm."""
REPEATS = 3
VERIFICATION = {"timeout_seconds": 2.0, "io_seconds_per_case": 0.25, "memory_mb": 128}
"""The pool verification's limits (``configs/task-pool-rehearsal.json``)."""
VERDICT_STATUSES = frozenset({"passed", "failed", "timeout", "output_limit", "resource_limit"})
RETRIES = 2
RETRY_DELAY_SECONDS = 2.0
MAX_RECEIPT_AGE_SECONDS = 600.0
COMMAND_MODULE = "tools.task_equivalence"


@dataclass(frozen=True)
class Call:
    direction: str
    """``a_on_b`` (A's reference, B's tests), ``b_on_a``, or the controls ``a_on_a``/``b_on_b``."""
    part: str
    code: str
    tests: str
    mode: str
    alias: str | None = None


@dataclass
class PairPlan:
    a: str
    b: str
    calls: list[Call] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    """Why some or all of the pair cannot be compared, decided before any execution."""
    checks: dict[str, int] = field(default_factory=dict)


def top_level(source: str) -> dict[str, str]:
    """Module-level definitions: name to ``function`` or ``class``."""
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = "function"
        elif isinstance(node, ast.ClassDef):
            found[node.name] = "class"
    return found


def used_names(tests: str) -> set[str]:
    return {node.id for node in ast.walk(ast.parse(tests)) if isinstance(node, ast.Name)}


def check_count(tests: str, mode: str) -> int:
    """Top-level ``assert`` statements, or I/O cases."""
    if mode == "io":
        return len(json.loads(tests))
    return sum(isinstance(node, ast.Assert) for node in ast.parse(tests).body)


def alias_line(code: str, partner_code: str, partner_tests: str) -> tuple[str | None, str | None]:
    """The alias needed to run ``code`` against ``partner_tests``: (line, problem).

    Returns ``(None, None)`` when no alias is needed and ``(None, reason)`` when the
    called name cannot be mapped.
    """
    try:
        own = top_level(code)
        partner = top_level(partner_code)
        called = used_names(partner_tests)
    except SyntaxError:
        return None, "unparsable_reference_or_tests"
    needed = {name: kind for name, kind in partner.items() if name in called}
    missing = {name: kind for name, kind in needed.items() if name not in own}
    if not missing:
        return None, None
    if len(missing) > 1:
        return None, "called_names_unmapped"
    ((name, kind),) = missing.items()
    spare = [
        own_name
        for own_name, own_kind in own.items()
        if own_kind == kind and own_name not in partner
    ]
    if len(spare) != 1:
        return None, "called_names_unmapped"
    return f"\n{name} = {spare[0]}\n", None


def plan_pair(a: Task, b: Task) -> PairPlan:
    """The calls that compare two tasks, or the reasons they cannot be compared."""
    plan = PairPlan(a.id, b.id)
    if a.solution is None or b.solution is None:
        plan.reasons.append("missing_reference")
        return plan
    if a.mode != b.mode:
        plan.reasons.append("mode_mismatch")
        return plan
    for own, partner, direction in ((a, b, "a_on_b"), (b, a, "b_on_a")):
        assert own.solution is not None and partner.solution is not None
        plan.checks[direction] = check_count(partner.hidden_tests, partner.mode)
        alias: str | None = None
        if own.mode == "function":
            needs = [partner.hidden_tests, partner.public_tests]
            lines = {alias_line(own.solution, partner.solution, tests) for tests in needs}
            problems = sorted({problem for _, problem in lines if problem})
            if problems:
                plan.reasons.extend(f"{direction}:{problem}" for problem in problems)
                continue
            found = {line for line, _ in lines if line}
            alias = found.pop() if len(found) == 1 else None
        for part, tests in (("hidden", partner.hidden_tests), ("public", partner.public_tests)):
            plan.calls.append(
                Call(direction, part, own.solution + (alias or ""), tests, own.mode, alias)
            )
    for own, direction in ((a, "a_on_a"), (b, "b_on_b")):
        assert own.solution is not None
        plan.calls.append(Call(direction, "hidden", own.solution, own.hidden_tests, own.mode))
    return plan


def direction_verdict(statuses: Mapping[str, Sequence[str]]) -> str:
    """``pass``, ``fail``, ``partial`` or ``nondeterministic`` for one direction.

    ``statuses`` maps each test part to its repeated sandbox statuses.
    """
    outcomes = {}
    for part, values in statuses.items():
        if not values or any(value not in VERDICT_STATUSES for value in values):
            raise ValueError("only decided sandbox statuses can be classified")
        passed = {value == "passed" for value in values}
        if len(passed) > 1:
            return "nondeterministic"
        outcomes[part] = passed.pop()
    if all(outcomes.values()):
        return "pass"
    if not any(outcomes.values()):
        return "fail"
    return "partial"


def classify(
    verdicts: Mapping[str, str], checks: Mapping[str, int], reasons: Sequence[str] = ()
) -> tuple[str, list[str]]:
    """Map direction verdicts to ``equivalent``, ``not_equivalent`` or ``unclear``."""
    why = list(reasons)
    if why or set(verdicts) != {"a_on_b", "b_on_a", "a_on_a", "b_on_b"}:
        return "unclear", why or ["incomplete"]
    if any(verdicts[name] != "pass" for name in ("a_on_a", "b_on_b")):
        why.append("reference_control_failed")
    if "nondeterministic" in verdicts.values():
        why.append("nondeterministic")
    forward, backward = verdicts["a_on_b"], verdicts["b_on_a"]
    if "partial" in (forward, backward):
        why.append("partial_agreement_public_vs_hidden")
    if why:
        return "unclear", why
    if forward == backward == "pass":
        weak = [name for name, count in checks.items() if count < MIN_CHECKS]
        if weak:
            return "unclear", [f"weak_tests:{name}" for name in sorted(weak)]
        return "equivalent", []
    if forward == backward == "fail":
        return "not_equivalent", []
    return "unclear", ["partial_agreement_one_direction"]


def decided(call: Callable[[], Result], retries: int = RETRIES) -> Result:
    """A decided result, retrying undecided ones; raise rather than score an undecided one."""
    for attempt in range(retries + 1):
        result = call()
        if result.status in VERDICT_STATUSES and not result.host_stalled:
            return result
        if attempt < retries:
            time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("the sandbox could not decide a call; nothing is scored")


def execute_pair(
    plan: PairPlan, execute: Callable[[Call], Result], *, repeats: int = REPEATS
) -> dict[str, Any]:
    """Run every planned call ``repeats`` times and classify the pair."""
    if repeats < 1:
        raise ValueError("at least one repeat")
    statuses: dict[str, dict[str, list[str]]] = {}
    aliases: dict[str, str] = {}
    for call in plan.calls:

        def once(call: Call = call) -> Result:
            return execute(call)

        values = [decided(once).status for _ in range(repeats)]
        statuses.setdefault(call.direction, {})[call.part] = values
        if call.alias:
            aliases[call.direction] = call.alias.strip()
    verdicts = {name: direction_verdict(parts) for name, parts in statuses.items()}
    outcome, reasons = classify(verdicts, plan.checks, plan.reasons)
    return {
        "a": plan.a,
        "b": plan.b,
        "outcome": outcome,
        "reasons": reasons,
        "directions": [
            {"direction": name, "verdict": verdicts[name], "statuses": statuses[name]}
            for name in sorted(statuses)
        ],
        "checks": plan.checks,
        "aliases": aliases,
        "repeats": repeats,
    }


def require_window(
    receipt_path: Path,
    *,
    image_id: str,
    environ: Mapping[str, str],
    dotenv: Sequence[str],
    now: datetime,
    max_age_seconds: float = MAX_RECEIPT_AGE_SECONDS,
) -> dict[str, Any]:
    """Refuse unless a fresh, passing window receipt for this command exists."""
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("--image-id must be sha256:<64 hex digits>")
    leaked = sorted(name for name in environ if withheld(name, dotenv))
    if leaked:
        raise PermissionError("withheld names are set; run through mac_window_check exec")
    record: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    if record.get("kind") != "mac-execution-window-check" or record.get("status") != "passed":
        raise PermissionError("the window receipt is not a passing window check")
    image = record.get("checks", {}).get("image", {})
    if image.get("found_id") != image_id or image.get("expected_id") != image_id:
        raise PermissionError("the window receipt names a different sandbox image")
    command = list(record.get("command") or [])
    position = command.index(COMMAND_MODULE) if COMMAND_MODULE in command else -1
    if position < 0 or command[position + 1 : position + 2] != ["run"]:
        raise PermissionError("the window receipt was not written for this command")
    if str(receipt_path) not in command:
        raise PermissionError("the window receipt names a different receipt path")
    checked = datetime.fromisoformat(record["checked_at"])
    age = (now - checked).total_seconds()
    if not 0 <= age <= max_age_seconds:
        raise PermissionError("the window receipt is stale; check the window again")
    return record


def _tasks(paths: Iterable[Path]) -> dict[str, Task]:
    found: dict[str, Task] = {}
    for path in paths:
        for task in load_tasks(refuse_sealed(path)):
            if task.split not in {"train", "dev"}:
                raise ValueError("only pool train and dev tasks are compared")
            found[task.id] = task
    return found


def pending_pairs(manifest: Mapping[str, Any], clusters: set[str] | None) -> list[tuple[str, str]]:
    return [
        (record["a"], record["b"])
        for cluster in manifest["clusters"]
        if clusters is None or cluster["cluster_id"] in clusters
        for record in cluster["behavior"]
        if record["outcome"] == "pending"
    ]


def _plans(args: argparse.Namespace) -> list[PairPlan]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    tasks = _tasks(args.tasks)
    pairs = pending_pairs(manifest, set(args.cluster) if args.cluster else None)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    return [plan_pair(tasks[a], tasks[b]) for a, b in pairs]


def plan_command(args: argparse.Namespace) -> int:
    plans = _plans(args)
    print(
        json.dumps(
            {
                "pairs": len(plans),
                "calls_per_repeat": sum(len(plan.calls) for plan in plans),
                "sandbox_calls": sum(len(plan.calls) for plan in plans) * args.repeats,
                "pairs_with_precomputed_reasons": sum(bool(plan.reasons) for plan in plans),
                "executed": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def run_command(args: argparse.Namespace) -> int:
    receipt = require_window(
        args.window_receipt,
        image_id=args.image_id,
        environ=os.environ,
        dotenv=dotenv_names(args.dotenv),
        now=datetime.now(UTC),
    )
    plans = _plans(args)
    output: Path = args.output
    done: set[tuple[str, str]] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("kind") == "pair":
                done.add((row["a"], row["b"]))
    runtime = Runtime()
    commit, dirty = git_state()
    with output.open("a", encoding="utf-8") as stream:
        header = {
            "kind": "header",
            "format": FORMAT,
            "started_at": utc_now(),
            "manifest_sha256": file_sha256(args.manifest),
            "window_receipt_sha256": file_sha256(args.window_receipt),
            "window_checked_at": receipt["checked_at"],
            "image": IMAGE,
            "image_id": args.image_id,
            "runtime": asdict(runtime),
            "verification": VERIFICATION,
            "repeats": args.repeats,
            "git_commit": commit,
            "dirty": dirty,
        }
        stream.write(json.dumps(header, sort_keys=True) + "\n")

        def execute(call: Call) -> Result:
            limits = limits_for(VERIFICATION, call.tests, call.mode)
            return run(call.code, call.tests, limits, mode=call.mode, runtime=runtime)

        for plan in plans:
            if (plan.a, plan.b) in done:
                continue
            row = {"kind": "pair", **execute_pair(plan, execute, repeats=args.repeats)}
            row["recorded_at"] = utc_now()
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
    return 0


def merge(manifest: dict[str, Any], rows: Iterable[Mapping[str, Any]], source: str) -> int:
    """Fill pending behaviour records from result rows; return how many were filled."""
    found = {(row["a"], row["b"]): row for row in rows if row.get("kind") == "pair"}
    filled = 0
    for cluster in manifest["clusters"]:
        for record in cluster["behavior"]:
            row = found.get((record["a"], record["b"]))
            if row is None or record["outcome"] != "pending":
                continue
            if row["outcome"] not in BEHAVIOR_OUTCOMES:
                raise ValueError("unknown outcome in the results")
            record.update(
                outcome=row["outcome"],
                directions=row["directions"],
                run={"source": source, "reasons": row["reasons"], "aliases": row["aliases"]},
            )
            filled += 1
    validate_manifest(manifest)
    return filled


def merge_command(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; a merged manifest is never overwritten")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines()]
    filled = merge(manifest, rows, file_sha256(args.results))
    manifest["kind"] = "behavior_recorded"
    atomic_json(args.output, manifest)
    print(json.dumps({"filled": filled}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument(
            "--tasks", type=Path, nargs="+", required=True, help="pool train and dev files"
        )
        command.add_argument("--cluster", action="append", help="only these cluster IDs")
        command.add_argument("--limit", type=int)
        command.add_argument("--repeats", type=int, default=REPEATS)
    run_parser = commands.choices["run"]
    run_parser.add_argument("--output", type=Path, required=True, help="behavior.jsonl")
    run_parser.add_argument("--window-receipt", type=Path, required=True)
    run_parser.add_argument("--image-id", required=True)
    run_parser.add_argument("--dotenv", type=Path, help="the Mac's .env (names only)")
    merged = commands.add_parser("merge")
    merged.add_argument("--manifest", type=Path, required=True)
    merged.add_argument("--results", type=Path, required=True)
    merged.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"plan": plan_command, "run": run_command, "merge": merge_command}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
