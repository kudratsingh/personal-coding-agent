"""Run 2's four endpoints on the Mac: which checkpoint each is, where its results come from.

After run 2 the Mac executes the dev completions, seals the test completions, scores the
benchmarks, writes the unseal record and builds the scorecard. Run 2's two arms may not have
their own files: an arm whose selected weights are the SFT weights (compared bit for bit on
the Pod) reuses run 1's SFT results, and an arm the run never evaluated (the control after a
stop of the Mac's side) has none (the owner's run 2 gate rulings). This tool reads the
verified reports once, records each endpoint's source in one plan file, and every later step
reads that plan, so a reused arm is never presented as an independent measurement:

``plan`` (Mac)
    From the launch specification and both verified reports: ``base`` and ``sft`` are run 1's;
    each arm is ``pod`` (its seven files and its ARC report in run 2's R2 prefix),
    ``reuses-run-1-sft`` or ``not-evaluated``, as run 2's ``selection.endpoints`` records. A
    reuse is checked three ways: run 2's record of the run 1 artifacts it stands on equals the
    specification's ``sft_results`` pins, those pins equal run 1's verified report, and the
    Mac's fetched copies under ``--run1-artifacts`` have the pinned size and SHA-256.
``list`` (Mac)
    Print one label per line from a plan (``LISTS``): run 2's dev and test completions and
    benchmark files that exist, for the runbook's loops.
``unseal`` (Mac, only after the owner froze all four endpoints)
    Write the unseal record: each executing endpoint's checkpoint and pool-test completions,
    a reused arm as ``{"reuses": "sft", ...}``, an absent arm as ``{"not_evaluated": ...}``
    (``scratchcoder.eval.seal``). The record is never rewritten.
``scorecard-inputs`` (Mac)
    Write the scorecard's inputs file (``scratchcoder.eval.scorecard.load_inputs``): the
    names, the reuses with their reasons, and every results file, benchmark artifact and ARC
    report, each checked to exist.

Nothing here contacts R2 or reads a sealed split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, utc_now
from scratchcoder.data.shards import file_sha256
from tools.posttrain_ops import REPORT_KIND, report_artifacts, run_prefix
from tools.posttrain_run import SFT_RESULT_FILES, check_sft_results, gate_nulls

PLAN_KIND = "posttrain-endpoints"
ARMS = ("rl", "more-sft")
POD, REUSES_SFT, NOT_EVALUATED = "pod", "reuses-run-1-sft", "not-evaluated"
POOL = {"dev": ("pool-dev-greedy", "pool-dev-n8"), "test": ("pool-test-greedy", "pool-test-n8")}
BENCHMARKS = tuple(name for name in SFT_RESULT_FILES if not name.startswith("pool-"))
LISTS = ("run2-pool-dev", "run2-pool-test", "run2-benchmarks")
"""What ``list`` prints: run 2's labels of each kind, for the arms evaluated on the Pod."""


def _report(path: Path, run: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != REPORT_KIND or value.get("run") != run:
        raise ValueError(f"{path} is not post-training run {run}'s report")
    return value


def _artifacts(report: Mapping[str, Any], run: str) -> dict[str, dict[str, Any]]:
    prefix = run_prefix(run, str(report.get("pod_id")))
    rows = report_artifacts(report, prefix=prefix)
    return {f"{row['label']}/{row['name']}": row for row in rows}


def _completions(
    rows: Mapping[str, Mapping[str, Any]], stage: str, names: Sequence[str]
) -> list[str]:
    found = []
    for name in names:
        row = rows.get(f"{stage}-{name}/completions.jsonl")
        if row is None:
            raise ValueError(f"the report lists no {stage}-{name} completions")
        found.append(str(row["sha256"]))
    return found


def check_reuse(
    entry: Mapping[str, Any],
    *,
    arm: str,
    specification: Mapping[str, Any],
    run1_rows: Mapping[str, Mapping[str, Any]],
    run1_artifacts: Path,
) -> list[dict[str, Any]]:
    """Check a reused arm's record against the pins and run 1's files; return what was checked.

    Run 2 recorded which run 1 artifact stands for each of the arm's own (``reuses.results``):
    each must be the specification's ``sft_results`` pin, that pin run 1's verified report's
    row, and the Mac's copy at ``run1_artifacts/<label>/<name>`` the pinned bytes."""
    pins = check_sft_results(specification)
    results = (entry.get("reuses") or {}).get("results")
    expected = {f"arc-post-{arm}/arc.json"} | {
        f"eval-{arm}-{name}/completions.jsonl" for name in SFT_RESULT_FILES
    }
    if not isinstance(results, Mapping) or set(results) != expected:
        raise ValueError(f"run 2's report does not map each of {arm}'s artifacts to run 1's")
    checked = []
    for own, source in sorted(results.items()):
        name = source.get("run_1")
        pin = pins.get(str(name))
        if pin is None or {key: source.get(key) for key in pin} != pin:
            raise ValueError(f"{arm}'s {own} names {name} with another pin than sft_results")
        row = run1_rows.get(str(name))
        if row is None or (row["key"], row["bytes"], row["sha256"]) != (
            pin["key"],
            pin["size_bytes"],
            pin["sha256"],
        ):
            raise ValueError(f"sft_results {name} is not the artifact run 1's report lists")
        local = run1_artifacts / str(name)
        if not local.is_file() or local.stat().st_size != pin["size_bytes"]:
            raise ValueError(f"the Mac's copy of {name} is missing or another size: {local}")
        if file_sha256(local) != pin["sha256"]:
            raise ValueError(f"the Mac's copy of {name} has another SHA-256: {local}")
        checked.append({"for": own, "run_1": name, "path": str(local), **pin})
    return checked


def plan(
    *,
    specification: Path,
    run1_report: Path,
    run2_report: Path,
    run1_artifacts: Path,
) -> dict[str, Any]:
    """The four endpoints and their sources (see the module docstring)."""
    raw = json.loads(specification.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("run") != "2" or gate_nulls(raw):
        raise ValueError(f"{specification} is not run 2's filled launch specification")
    run1, run2 = _report(run1_report, "1"), _report(run2_report, "2")
    if run2.get("specification") != raw:
        raise ValueError("run 2's report ran another specification than the one given")
    run1_rows, run2_rows = _artifacts(run1, "1"), _artifacts(run2, "2")
    tests = POOL["test"]
    endpoints: dict[str, dict[str, Any]] = {
        "base": {
            "source": "run 1",
            "checkpoint_sha256": run1["inputs"]["b-mixed-final"],
            "stage": "eval-base",
            "test_completions_sha256": _completions(run1_rows, "eval-base", tests),
        },
        "sft": {
            "source": "run 1",
            "checkpoint_sha256": run1["sft"]["selected"]["sha256"],
            "stage": "eval-sft",
            "test_completions_sha256": _completions(run1_rows, "eval-sft", tests),
        },
    }
    recorded = (run2.get("selection") or {}).get("endpoints") or {}
    for arm in ARMS:
        entry = recorded.get(arm)
        if entry is None:
            endpoints[arm] = {
                "source": NOT_EVALUATED,
                "reason": "run 2 selected no endpoint for this arm",
            }
            continue
        evaluation = entry.get("evaluation")
        missing = [
            name
            for name in (
                f"arc-post-{arm}/arc.json",
                *(f"eval-{arm}-{file}/completions.jsonl" for file in SFT_RESULT_FILES),
            )
            if name not in run2_rows
        ]
        if evaluation == POD and missing:
            # Selected for the Pod, but a later stop (the projected finish, the reserve, a
            # limit) came before its evaluation reached R2: nothing of it is executed here.
            endpoints[arm] = {
                "source": NOT_EVALUATED,
                "reason": f"selected {entry['selected']} for the Pod, but {len(missing)} of its "
                f"8 evaluation artifacts are not in run 2's report, e.g. {missing[0]}",
            }
        elif evaluation == POD:
            endpoints[arm] = {
                "source": POD,
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "selected": entry["selected"],
                "stage": f"eval-{arm}",
                "test_completions_sha256": _completions(run2_rows, f"eval-{arm}", tests),
                "dev_completions_sha256": _completions(run2_rows, f"eval-{arm}", POOL["dev"]),
            }
        elif evaluation == REUSES_SFT:
            endpoints[arm] = {
                "source": REUSES_SFT,
                "reuses": "sft",
                "selected": entry["selected"],
                "selected_checkpoint_sha256": entry["checkpoint_sha256"],
                "reason": entry["reason"],
                "independent": False,
                "checked": check_reuse(
                    entry,
                    arm=arm,
                    specification=raw,
                    run1_rows=run1_rows,
                    run1_artifacts=run1_artifacts,
                ),
            }
        elif evaluation == NOT_EVALUATED:
            endpoints[arm] = {"source": NOT_EVALUATED, "reason": entry["reason"]}
        else:
            raise ValueError(f"run 2's report records an unknown evaluation for {arm}")
    evaluated = [arm for arm in ARMS if endpoints[arm]["source"] == POD]
    return {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "created_at": utc_now(),
        "sources": {
            "specification_sha256": file_sha256(specification),
            "run1_report_sha256": file_sha256(run1_report),
            "run2_report_sha256": file_sha256(run2_report),
        },
        "run2_pod_id": run2["pod_id"],
        "endpoints": endpoints,
        "lists": {
            "run2-pool-dev": [f"eval-{arm}-{name}" for arm in evaluated for name in POOL["dev"]],
            "run2-pool-test": [f"eval-{arm}-{name}" for arm in evaluated for name in tests],
            "run2-benchmarks": [f"eval-{arm}-{name}" for arm in evaluated for name in BENCHMARKS],
        },
    }


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != PLAN_KIND:
        raise ValueError(f"{path} is not an endpoints plan (posttrain_endpoints plan)")
    return value


def unseal_record(plan_value: Mapping[str, Any]) -> dict[str, Any]:
    """The unseal record of a plan: one entry per endpoint (``scratchcoder.eval.seal``)."""
    endpoints: dict[str, Any] = {}
    for name, entry in plan_value["endpoints"].items():
        if entry["source"] == REUSES_SFT:
            endpoints[name] = {
                "reuses": entry["reuses"],
                "selected_checkpoint_sha256": entry["selected_checkpoint_sha256"],
            }
        elif entry["source"] == NOT_EVALUATED:
            endpoints[name] = {"not_evaluated": entry["reason"]}
        else:
            endpoints[name] = {
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "completions_sha256": list(entry["test_completions_sha256"]),
            }
    return {
        "kind": "unseal",
        "schema_version": 1,
        "frozen_at": utc_now(),
        "endpoints": endpoints,
    }


def _one_benchmark(directory: Path) -> str:
    found = sorted(directory.glob("benchmark-*.json"))
    if len(found) != 1:
        raise ValueError(f"{directory} holds {len(found)} benchmark artifacts, not one")
    return str(found[0])


def scorecard_inputs(
    plan_value: Mapping[str, Any], *, run1_shared: Path, shared: Path, artifacts: Path
) -> dict[str, Any]:
    """The scorecard's inputs for a plan, every path checked to exist, laid out as the two
    runbooks write them: run 1's base and SFT dev results, benchmark artifacts and ARC reports
    under ``run1_shared``; every endpoint's pool-test results under ``shared`` (the unseal runs
    only after run 2, so run 2's runbook writes base's and SFT's there too); each arm evaluated
    on the Pod's dev results and benchmark results under ``shared`` and its ARC report under
    ``artifacts``; a reused arm as a reuse of ``sft``, with no inputs of its own."""
    endpoints = plan_value["endpoints"]
    checkpoints, reuses = {}, {}
    results: list[str] = []
    benchmarks: list[str] = []
    arc: list[str] = []
    sources = {
        "base": (run1_shared, run1_shared / "artifacts" / "arc-base-b-mixed-final" / "arc.json"),
        "sft": (run1_shared, run1_shared / "artifacts" / "arc-sft-sft" / "arc.json"),
        **{arm: (shared, artifacts / f"arc-post-{arm}" / "arc.json") for arm in ARMS},
    }
    for name, entry in endpoints.items():
        if entry["source"] == REUSES_SFT:
            reuses[name] = {"endpoint": entry["reuses"], "reason": entry["reason"]}
            continue
        if entry["source"] == NOT_EVALUATED:
            continue
        checkpoints[name] = entry["checkpoint_sha256"]
        root, arc_report = sources[name]
        stage = entry["stage"]
        for split, directory in (("dev", root), ("test", shared)):
            results += [
                str(directory / "results" / f"{stage}-{file}.jsonl") for file in POOL[split]
            ]
        benchmarks += [
            _one_benchmark(root / "benchmarks-results" / f"{stage}-{file}") for file in BENCHMARKS
        ]
        arc.append(str(arc_report))
    missing = [path for path in [*results, *arc] if not Path(path).is_file()]
    if missing:
        raise ValueError(f"{len(missing)} scorecard inputs are missing, e.g. {missing[0]}")
    return {
        "kind": "scorecard-inputs",
        "schema_version": 1,
        "created_at": utc_now(),
        "sources": {"plan_run2_report_sha256": plan_value["sources"]["run2_report_sha256"]},
        "checkpoints": checkpoints,
        "reuses": reuses,
        "results": results,
        "benchmarks": benchmarks,
        "arc": arc,
    }


def _new(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists; it is never rewritten")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("plan", help="record each endpoint's source (Mac)")
    made.add_argument("--specification", type=Path, required=True, help="the launch specification")
    made.add_argument("--run1-report", type=Path, required=True)
    made.add_argument("--run2-report", type=Path, required=True)
    made.add_argument("--run1-artifacts", type=Path, required=True, help="run 1's fetched labels")
    made.add_argument("--output", type=Path, required=True)
    listed = commands.add_parser("list", help="print one label per line (for the runbook)")
    listed.add_argument("--plan", type=Path, required=True)
    listed.add_argument("--list", required=True, help=f"one of {', '.join(LISTS)}")
    unseal = commands.add_parser("unseal", help="write the unseal record (after the freeze)")
    unseal.add_argument("--plan", type=Path, required=True)
    unseal.add_argument("--output", type=Path, required=True)
    inputs = commands.add_parser("scorecard-inputs", help="write the scorecard's inputs file")
    inputs.add_argument("--plan", type=Path, required=True)
    inputs.add_argument("--run1-shared", type=Path, required=True)
    inputs.add_argument("--shared", type=Path, required=True)
    inputs.add_argument("--artifacts", type=Path, required=True, help="run 2's fetched labels")
    inputs.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            value = plan(
                specification=args.specification,
                run1_report=args.run1_report,
                run2_report=args.run2_report,
                run1_artifacts=args.run1_artifacts,
            )
            atomic_json(_new(args.output), value)
            print(
                json.dumps(
                    {
                        name: entry["source"]
                        + (f" ({entry['reuses']})" if entry.get("reuses") else "")
                        for name, entry in value["endpoints"].items()
                    }
                )
            )
            return 0
        found = load_plan(args.plan)
        if args.command == "list":
            if args.list not in LISTS:
                raise ValueError(f"--list is one of {list(LISTS)}")
            for label in found["lists"][args.list]:
                print(label)
            return 0
        if args.command == "unseal":
            record = unseal_record(found)
            target = _new(args.output)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, sort_keys=True)
            print(json.dumps({name: sorted(entry) for name, entry in record["endpoints"].items()}))
            return 0
        value = scorecard_inputs(
            found, run1_shared=args.run1_shared, shared=args.shared, artifacts=args.artifacts
        )
        atomic_json(_new(args.output), value)
        print(json.dumps({"checkpoints": sorted(value["checkpoints"]), "reuses": value["reuses"]}))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"posttrain-endpoints: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
