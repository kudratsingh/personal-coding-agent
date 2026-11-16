"""AGENT-SFT-1 synthetic agent data: generate, specify, validate in the sandbox, filter, mix.

Subcommands, all with explicit paths; every input path passes the sealed-file guard
(``tools.verifier_scan.guard``) and held-out agent-case locations are refused before
anything is opened:

- ``generate --out DIR --tokenizer PATH --families PATH``: build instances from the template
  catalog (``scratchcoder.agent.synth``), one per (template, bug kind) by default
  (``--per-bug``), check each statically (parse, sizes, the ``tools.agent_dev_run plan`` fit
  rule, target shown, case-ID rule, evaluation-family blocklist) and write the kept ones as
  harness v0 case directories with a registry. Runs nothing.
- ``specs --instances DIR --out PATH``: the natural-language specification of every template,
  the constants of every instance and the independent-oracle file format. Runs nothing.
- ``ledger --ledger PATH [--seed-from DIR ...]``: record earlier sandbox calls (run records
  of evaluation-case validation) as prior entries, once each, and print the totals.
- ``validate-and-build``: everything that executes. It refuses to start without
  ``--receipt``, a passing execution-window receipt written by ``tools.mac_window_check exec``
  for this very command in the last 30 minutes, and without a new ``--out`` directory; it
  refuses when the planned calls exceed what the ledger cap leaves. It also refuses unless
  ``--independent`` verifies against its ``SHA256SUMS`` and that file's SHA-256 equals
  ``--independent-sha256``; every oracle is read (never imported) and checked against the pin
  before the first call, and the pin is written into every validation record, control row and
  row-metadata line. Per instance it drives
  the harness loop with a scripted policy (``scratchcoder.agent.synth.rows``), at
  ``k + 4`` sandbox calls for ``k`` mutants, and writes rows, row metadata, per-instance
  validation records and traces; with ``--controls`` it runs the four negative controls once
  per template (four calls each). Every call is counted in the ledger first and refused over
  the cap.
- ``contamination``: the status-only agent-case contamination check
  (``scratchcoder.agent.synth.contamination``) on generated instances, built rows or a decoded
  replay index; writes the rejected row IDs only.
- ``mix``: agent rows plus replay rows at a target ratio of assistant-loss tokens.
- ``manifest``: families, templates, bug and mutation kinds, workflow types, counts, file
  hashes and ledger totals.

Case code, references, mutants and the independent oracle never run on the host: the host
parses them; execution goes only through ``scratchcoder.eval.harness.isolated_execution``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent import execution
from scratchcoder.agent.synth import (
    GENERATOR_VERSION,
    contamination,
    independent,
    instances,
    mutation,
    rows,
)
from scratchcoder.agent.synth import ledger as call_ledger
from scratchcoder.agent.synth import mix as mixing
from scratchcoder.agent.synth import specs as spec_export
from scratchcoder.agent.synth.catalog import catalog
from scratchcoder.agent.synth.template import TemplateError
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.chat import TextDecoder, TextEncoder
from scratchcoder.data.sft import read_rows
from tools.admission_check import _receipt_image
from tools.verifier_scan import SealedPathError, guard

DEFAULT_MUTANTS = 2
MUTATION_PREFERENCE = (
    "off_by_one",
    "wrong_return",
    "drop_branch",
    "wrong_operator",
    "swap_arguments",
    "negate_condition",
    "wrong_initial_value",
    "flip_comparison",
)
"""Order in which a template's applicable kinds are cycled (kinds more likely to change the
public tests' outcome first); every applicable kind is still used across the instances."""
Loader = Callable[[Path], tuple[TextEncoder, TextDecoder]]


def _tokenizer(path: Path) -> tuple[TextEncoder, TextDecoder]:
    from scratchcoder.tokenizer.bpe import ByteBPE
    from scratchcoder.tokenizer.bulk import FastEncoder

    tokenizer = ByteBPE.load(path)
    return FastEncoder(tokenizer), tokenizer


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse(paths: Sequence[Path | None]) -> None:
    present = [path for path in paths if path is not None]
    for path in present:
        if any(part.lower() in agent_cases.FORBIDDEN_PARTS for part in path.resolve().parts):
            raise SealedPathError(f"refusing {path}: held-out agent cases are sealed")
    guard(present)


def _ordered_kinds(kinds: Sequence[str]) -> list[str]:
    return [kind for kind in MUTATION_PREFERENCE if kind in kinds]


def generate(args: argparse.Namespace, loader: Loader = _tokenizer) -> int:
    out: Path = args.out
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    encoder, _ = loader(args.tokenizer)
    blocklist = instances.Blocklist.load(args.families)
    selected = [
        template
        for template in catalog().templates
        if not args.templates or template.id in args.templates
    ]
    out.mkdir(parents=True)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    per_template: Counter[str] = Counter()
    for template, bug, index in instances.plan_instances(selected, args.per_bug):
        start = per_template[template.id] * args.mutants
        try:
            probe = instances.build(template, bug, index, args.seed_base, encoder=encoder)
            kinds = _ordered_kinds(
                mutation.applicable(probe.reference_target, forbid=[probe.original_target])
            )
            chosen = instances.mutation_cycle(kinds, start, args.mutants)
            if len(chosen) < args.mutants:
                raise TemplateError(f"only {len(chosen)} applicable mutation kinds")
            instance = instances.build(
                template, bug, index, args.seed_base, mutation_kinds=chosen, encoder=encoder
            )
        except (TemplateError, mutation.MutationError) as error:
            dropped.append(
                {
                    "template_id": template.id,
                    "bug_kind": bug,
                    "index": index,
                    "problems": [str(error)],
                }
            )
            continue
        static = instances.static_check(instance, encoder, blocklist)
        if static["problems"]:
            dropped.append(
                {
                    "id": instance.id,
                    "template_id": template.id,
                    "bug_kind": bug,
                    "index": index,
                    "problems": static["problems"],
                }
            )
            continue
        written = instances.write(out, instance, static)
        per_template[template.id] += 1
        kept.append(
            {
                "id": instance.id,
                "template_id": template.id,
                "family": template.family.id,
                "workflow_type": template.workflow,
                "bug_kind": bug,
                "mutation_kinds": chosen,
                "applicable_mutation_kinds": kinds,
                "files": written,
            }
        )
    if not kept:
        print("refused: no instance passed the static checks", file=sys.stderr)
        return 1
    registry = agent_cases.build_registry(out)
    (out / agent_cases.REGISTRY_NAME).write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = agent_cases.load_cases(out)  # Hash-checked and statically validated again.
    commit, dirty = git_state()
    record = {
        "schema": "agent-synth-generate-v1",
        "generator_version": GENERATOR_VERSION,
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "seed_base": args.seed_base,
        "per_bug": args.per_bug,
        "mutants_per_instance": args.mutants,
        "tokenizer_sha256": _sha_file(args.tokenizer),
        "families_sha256": _sha_file(args.families),
        "registry_sha256": agent_cases.registry_sha256(out),
        "kept": kept,
        "dropped": dropped,
        "counts": {
            "kept": len(loaded),
            "dropped": len(dropped),
            "templates": len({row["template_id"] for row in kept}),
            "families": len({row["family"] for row in kept}),
            "by_workflow": dict(Counter(row["workflow_type"] for row in kept)),
        },
    }
    atomic_json(out / "generate.json", record)
    print(json.dumps(record["counts"]))
    return 0


def _synth_records(root: Path) -> list[dict[str, Any]]:
    found = []
    for directory in sorted(path for path in root.iterdir() if (path / "synth.json").is_file()):
        found.append(json.loads((directory / "synth.json").read_text(encoding="utf-8")))
    return found


def specs(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists():
        print(f"refused: {out} exists", file=sys.stderr)
        return 2
    agent_cases.load_cases(args.instances)  # The instances are the registered ones.
    document = spec_export.specs(catalog(), _synth_records(args.instances))
    document["instances_registry_sha256"] = agent_cases.registry_sha256(args.instances)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, document)
    print(
        json.dumps(
            {"specs": str(out), "sha256": _sha_file(out), "templates": len(document["templates"])}
        )
    )
    return 0


def ledger(args: argparse.Namespace) -> int:
    book = call_ledger.Ledger.open(args.ledger, args.max_calls)
    added = call_ledger.prior_from_records(book, args.seed_from or [])
    print(json.dumps({"added": added, **book.summary()}))
    return 0


def validate_and_build(
    args: argparse.Namespace,
    *,
    execute: execution.Execute,
    loader: Loader = _tokenizer,
) -> int:
    try:
        image_id = _receipt_image(args.receipt, datetime.now(UTC), args.argv)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    out: Path = args.out
    if out.exists():
        print(f"refused: {out} exists; every run writes a new directory", file=sys.stderr)
        return 2
    try:
        loaded = agent_cases.load_cases(args.instances)
    except (agent_cases.CaseError, OSError, ValueError, KeyError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    try:
        pinned = independent.pin(args.independent, expected=args.independent_sha256)
    except (independent.OracleError, OSError, UnicodeDecodeError) as error:
        print(f"refused: independent oracles: {error}", file=sys.stderr)
        return 2
    records = {record["id"]: record for record in _synth_records(args.instances)}
    excluded = contamination.rejected_ids(args.exclude)
    chosen: list[agent_cases.Case] = []
    per_template: Counter[str] = Counter()
    for case in loaded:
        record = records[case.id]
        if case.id in excluded or (args.templates and record["template_id"] not in args.templates):
            continue
        if args.per_template and per_template[record["template_id"]] >= args.per_template:
            continue
        per_template[record["template_id"]] += 1
        chosen.append(case)
        if args.limit and len(chosen) >= args.limit:
            break
    book = call_ledger.Ledger.open(args.ledger, args.max_calls)
    templates = {records[case.id]["template_id"] for case in chosen}
    # Every oracle is read and checked against the pin once, before any call; later edits to
    # the directory cannot reach this run.
    oracles: dict[str, independent.Oracle | independent.OracleError] = {}
    for template_id in sorted(templates):
        try:
            oracles[template_id] = independent.load(args.independent, template_id, pinned)
        except independent.OracleError as error:
            oracles[template_id] = error
    pin_sha = pinned.sums_sha256
    planned = sum(rows.calls_per_instance(len(records[c.id]["mutants"])) for c in chosen)
    planned += len(mutation.CONTROL_KINDS) * len(templates) if args.controls else 0
    if planned > book.remaining:
        print(
            f"refused: {planned} planned sandbox calls exceed the {book.remaining} the ledger "
            f"cap leaves ({book.total} of {book.cap} used)",
            file=sys.stderr,
        )
        return 2
    encoder, decoder = loader(args.tokenizer)
    out.mkdir(parents=True)
    (out / "traces").mkdir()
    for name in ("rows.jsonl", "rows-meta.jsonl", "instances.jsonl", "controls.jsonl"):
        (out / name).touch(exist_ok=False)
    started = utc_now()
    context = {"value": ""}
    counted = book.execute(execute, run=out.name, context=lambda: context["value"])
    controlled: set[str] = set()
    stopped = ""
    totals: Counter[str] = Counter()
    for case in chosen:
        record = records[case.id]
        template_id = record["template_id"]
        oracle = oracles[template_id]
        if isinstance(oracle, independent.OracleError):
            rows.append_jsonl(
                out / "instances.jsonl",
                [
                    {
                        "instance": case.id,
                        "accepted": False,
                        "reasons": [f"oracle: {oracle}"],
                        "calls": 0,
                        "independent_sums_sha256": pin_sha,
                    }
                ],
            )
            totals["no_oracle"] += 1
            continue
        stamp = {"independent_sums_sha256": pin_sha, "oracle_sha256": oracle.sha256}
        try:
            if args.controls and template_id not in controlled:
                context["value"] = f"{case.id}/controls"
                rows.append_jsonl(
                    out / "controls.jsonl",
                    [
                        {**row, **stamp}
                        for row in rows.run_controls(case, record, oracle, execute=counted)
                    ],
                )
                controlled.add(template_id)
            context["value"] = case.id
            with (out / "traces" / f"{case.id}.jsonl").open("x", encoding="utf-8") as stream:
                built = rows.build_instance(
                    case,
                    record,
                    oracle,
                    encoder,
                    decoder,
                    execute=counted,
                    trace_stream=stream,
                    diagnose=args.diagnose,
                )
        except call_ledger.CallCapReached as error:
            stopped = str(error)
            break
        rows.append_jsonl(out / "instances.jsonl", [{**built.record(), **stamp}])
        if built.accepted:
            rows.append_rows(out / "rows.jsonl", built.rows)
            rows.append_jsonl(out / "rows-meta.jsonl", [{**row, **stamp} for row in built.meta])
            totals["accepted"] += 1
            totals["rows"] += len(built.rows)
        else:
            totals["rejected"] += 1
        print(
            json.dumps(
                {
                    "instance": case.id,
                    "accepted": built.accepted,
                    "reasons": built.reasons,
                    "calls": built.calls,
                }
            ),
            flush=True,
        )
    controls = [
        json.loads(line)
        for line in (out / "controls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    commit, dirty = git_state()
    summary = {
        "schema": "agent-synth-build-v1",
        "generator_version": GENERATOR_VERSION,
        "started_at": started,
        "finished_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "runtime_image_id": image_id,
        "supervisor_sha256": _sha_file(
            Path(execution.__file__).parents[1] / "sandbox" / "program.py"
        ),
        "instances_registry_sha256": agent_cases.registry_sha256(args.instances),
        "tokenizer_sha256": _sha_file(args.tokenizer),
        "independent": {
            "directory": str(args.independent),
            **pinned.record(),
            "missing_templates": sorted(
                template
                for template, item in oracles.items()
                if not isinstance(item, independent.Oracle)
            ),
        },
        "planned_calls": planned,
        "selected_instances": len(chosen),
        "totals": dict(totals),
        "controls": {"rows": len(controls), "all_failed": all(row["ok"] for row in controls)},
        "stopped": stopped,
        "ledger": book.summary(),
    }
    atomic_json(out / "record.json", summary)
    print(
        json.dumps(
            {
                "record": str(out / "record.json"),
                "totals": dict(totals),
                "ledger_total": book.total,
                "stopped": stopped,
            }
        )
    )
    return 0 if not stopped else 3


def run_contamination(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"refused: {args.out} exists", file=sys.stderr)
        return 2
    index = contamination.build_index(
        args.cases, families=args.families, benchmarks=args.benchmarks or (), dev=args.dev
    )
    if args.instances is not None:
        found = contamination.instance_rows(args.instances)
        source = args.instances
    elif args.rows_meta is not None:
        found = contamination.meta_rows(args.rows_meta)
        source = args.rows_meta
    else:
        found = contamination.decoded_rows(args.decoded)
        source = args.decoded
    result = contamination.check(index, found)
    result["input"] = str(source)
    result["index"] = dict(contamination.describe(index))
    atomic_json(args.out, result)
    print(json.dumps({"checked": result["checked"], "rejected": len(result["rejected"])}))
    return 0


def run_mix(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"refused: {args.out} exists", file=sys.stderr)
        return 2
    agent_skip = contamination.rejected_ids(args.agent_rejected)
    replay_skip = contamination.rejected_ids(args.replay_rejected)
    meta = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in args.agent_meta.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    kinds = {}
    for line in args.replay_index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            kinds[str(row["id"])] = str(row["kind"])
    agent = [row for row in read_rows(args.agent_rows) if row.id not in agent_skip]
    replay = [
        row
        for row in read_rows(args.replay)
        if row.id not in replay_skip and kinds.get(row.id) in set(args.replay_kinds)
    ]
    mixed, report = mixing.mix(agent, meta, replay, ratio=args.ratio)
    args.out.mkdir(parents=True)
    from scratchcoder.data.sft import write_rows

    write_rows(args.out / "mix.jsonl", mixed)
    report.update(
        {
            "agent_rows_sha256": _sha_file(args.agent_rows),
            "replay_sha256": _sha_file(args.replay),
            "replay_kinds": list(args.replay_kinds),
            "agent_rejected": len(agent_skip),
            "replay_rejected": len(replay_skip),
            "mix_sha256": _sha_file(args.out / "mix.jsonl"),
        }
    )
    atomic_json(args.out / "mix-record.json", report)
    print(json.dumps({key: report[key] for key in ("agent", "replay", "achieved_ratio")}))
    return 0


def manifest(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"refused: {args.out} exists", file=sys.stderr)
        return 2
    cat = catalog()
    records = _synth_records(args.instances)
    builds = []
    for build in args.build or []:
        summary = json.loads((build / "record.json").read_text(encoding="utf-8"))
        meta = [
            json.loads(line)
            for line in (build / "rows-meta.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        builds.append(
            {
                "directory": str(build),
                "record_sha256": _sha_file(build / "record.json"),
                "rows_sha256": _sha_file(build / "rows.jsonl"),
                "totals": summary["totals"],
                "rows_by_kind": dict(Counter(row["kind"] for row in meta)),
                "loss_tokens": sum(int(row["loss_tokens"]) for row in meta),
                "distinct_templates": len({row["template_id"] for row in meta}),
                "distinct_families": len({row["family"] for row in meta}),
                "mutation_kinds": sorted(
                    {row["mutation_kind"] for row in meta if row["mutation_kind"]}
                ),
            }
        )
    document = {
        "schema": "agent-synth-manifest-v1",
        "generator_version": GENERATOR_VERSION,
        "created_at": utc_now(),
        "families": [{"id": family.id, "domain": family.domain} for family in cat.families],
        "templates": [
            {
                "id": template.id,
                "family": template.family.id,
                "workflow_type": template.workflow,
                "concept": template.concept,
                "bug_kinds": list(template.bugs),
            }
            for template in cat.templates
        ],
        "mutation_kinds": list(mutation.MUTATION_KINDS),
        "control_kinds": list(mutation.CONTROL_KINDS),
        "instances": {
            "registry_sha256": agent_cases.registry_sha256(args.instances),
            "count": len(records),
            "by_workflow": dict(Counter(r["workflow_type"] for r in records)),
            "by_family": dict(Counter(r["family"] for r in records)),
            "templates": len({r["template_id"] for r in records}),
            "mutation_kinds_used": dict(Counter(m["kind"] for r in records for m in r["mutants"])),
        },
        "builds": builds,
    }
    if args.specs is not None:
        document["specs_sha256"] = _sha_file(args.specs)
    if args.ledger is not None:
        document["ledger"] = call_ledger.Ledger.open(args.ledger, args.max_calls).summary()
    atomic_json(args.out, document)
    print(json.dumps({"manifest": str(args.out), "templates": len(document["templates"])}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    g = commands.add_parser("generate")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--tokenizer", type=Path, required=True)
    g.add_argument("--families", type=Path, required=True)
    g.add_argument("--per-bug", type=int, default=1)
    g.add_argument("--seed-base", type=int, default=0)
    g.add_argument("--mutants", type=int, default=DEFAULT_MUTANTS)
    g.add_argument("--templates", nargs="+")
    s = commands.add_parser("specs")
    s.add_argument("--instances", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    led = commands.add_parser("ledger")
    led.add_argument("--ledger", type=Path, required=True)
    led.add_argument("--seed-from", type=Path, nargs="+")
    led.add_argument("--max-calls", type=int, default=call_ledger.DEFAULT_CAP)
    v = commands.add_parser("validate-and-build")
    v.add_argument("--instances", type=Path, required=True)
    v.add_argument("--independent", type=Path, required=True)
    v.add_argument(
        "--independent-sha256",
        required=True,
        help="the SHA-256 of the independent directory's SHA256SUMS; the run refuses another",
    )
    v.add_argument("--tokenizer", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--receipt", type=Path, required=True)
    v.add_argument("--ledger", type=Path, required=True)
    v.add_argument("--max-calls", type=int, default=call_ledger.DEFAULT_CAP)
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--per-template", type=int, default=0)
    v.add_argument("--templates", nargs="+")
    v.add_argument("--exclude", type=Path, help="a contamination result whose IDs are skipped")
    v.add_argument("--controls", action="store_true", help="negative controls once per template")
    v.add_argument(
        "--diagnose",
        action="store_true",
        help="after a failed grading call, run its three parts separately (3 calls)",
    )
    c = commands.add_parser("contamination")
    c.add_argument("--cases", type=Path, nargs="+", required=True)
    c.add_argument("--families", type=Path, required=True)
    c.add_argument("--benchmarks", type=Path, nargs="*")
    c.add_argument("--dev", type=Path)
    group = c.add_mutually_exclusive_group(required=True)
    group.add_argument("--instances", type=Path)
    group.add_argument("--rows-meta", type=Path)
    group.add_argument("--decoded", type=Path)
    c.add_argument("--out", type=Path, required=True)
    m = commands.add_parser("mix")
    m.add_argument("--agent-rows", type=Path, required=True)
    m.add_argument("--agent-meta", type=Path, required=True)
    m.add_argument("--replay", type=Path, required=True)
    m.add_argument("--replay-index", type=Path, required=True)
    m.add_argument("--replay-kinds", nargs="+", default=["code"])
    m.add_argument("--agent-rejected", type=Path)
    m.add_argument("--replay-rejected", type=Path)
    m.add_argument("--ratio", type=float, default=1.0)
    m.add_argument("--out", type=Path, required=True)
    f = commands.add_parser("manifest")
    f.add_argument("--instances", type=Path, required=True)
    f.add_argument("--build", type=Path, nargs="*")
    f.add_argument("--specs", type=Path)
    f.add_argument("--ledger", type=Path)
    f.add_argument("--max-calls", type=int, default=call_ledger.DEFAULT_CAP)
    f.add_argument("--out", type=Path, required=True)
    return parser


PATH_ARGS = (
    "out",
    "tokenizer",
    "families",
    "instances",
    "ledger",
    "independent",
    "receipt",
    "exclude",
    "dev",
    "rows_meta",
    "decoded",
    "agent_rows",
    "agent_meta",
    "replay",
    "replay_index",
    "agent_rejected",
    "replay_rejected",
    "specs",
)
PATH_LIST_ARGS = ("seed_from", "cases", "benchmarks", "build")


def main(
    argv: Sequence[str] | None = None,
    *,
    execute: execution.Execute | None = None,
    loader: Loader | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    paths: list[Path | None] = [getattr(args, name, None) for name in PATH_ARGS]
    for name in PATH_LIST_ARGS:
        paths.extend(getattr(args, name, None) or [])
    if getattr(args, "per_bug", 1) < 1 or getattr(args, "mutants", 1) < 1:
        parser.error("--per-bug and --mutants must be at least 1")
    try:
        _refuse(paths)
    except SealedPathError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    load = _tokenizer if loader is None else loader
    if args.command == "generate":
        return generate(args, load)
    if args.command == "specs":
        return specs(args)
    if args.command == "ledger":
        return ledger(args)
    if args.command == "contamination":
        return run_contamination(args)
    if args.command == "mix":
        return run_mix(args)
    if args.command == "manifest":
        return manifest(args)
    from scratchcoder.eval.harness import isolated_execution

    return validate_and_build(
        args, execute=isolated_execution if execute is None else execute, loader=load
    )


if __name__ == "__main__":
    raise SystemExit(main())
