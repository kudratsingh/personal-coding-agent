"""AGENT-SFT-1 synthetic agent data: generate, specify, validate in the sandbox, filter, mix.

Subcommands, all with explicit paths; every input path passes the sealed-file guard
(``tools.verifier_scan.guard``) and held-out agent-case locations are refused before
anything is opened:

- ``generate --out DIR --tokenizer PATH --families PATH``: build instances from the template
  catalog (``scratchcoder.agent.synth``), one per (template, bug kind) by default
  (``--per-bug``), check each statically (parse, sizes, the ``tools.agent_dev_run plan`` fit
  rule, target shown, case-ID rule, evaluation-family blocklist) and write the kept ones as
  harness v0 case directories with a registry. At most ``--max-per-reply`` instances share
  one reference target (every row of an instance has it as its reply); the record reports
  distinct replies per template beside the distinct replies its seeds can reach. Runs
  nothing.
- ``specs --instances DIR --out PATH``: the natural-language specification of every template,
  the constants of every instance and the independent-oracle file format. Runs nothing.
- ``ledger --ledger PATH [--seed-from DIR ...]``: record earlier sandbox calls (run records
  of evaluation-case validation) as prior entries, once each (keyed by resolved path and
  file hash), and print the totals.
- ``validate-and-build``: everything that executes. It refuses to start without
  ``--receipt``, a passing execution-window receipt written by ``tools.mac_window_check exec``
  for this very command in the last 30 minutes, and without a new ``--out`` directory; it
  refuses a ledger that does not exist or has no prior entries, a ``--max-calls`` above the
  approved cap and a ledger another run holds (the lock is kept for the whole run); it
  refuses when the planned calls exceed what the ledger cap leaves. It also refuses unless
  ``--independent`` verifies against its ``SHA256SUMS`` and that file's SHA-256 equals
  ``--independent-sha256``; every oracle is read (never imported) and checked against the pin
  before the first call, and the pin is written into every validation record, control row and
  row-metadata line. Per instance it drives
  the harness loop with a scripted policy (``scratchcoder.agent.synth.rows``), at
  ``k + 4`` sandbox calls for ``k`` mutants, and writes rows, row metadata, per-instance
  validation records and traces; with ``--controls`` it runs the four negative controls once
  per template (four calls each) before the template's first instance. Only templates whose
  four controls all failed (here, or in ``--controls-from`` against the same oracle) give
  rows; the others' instances are recorded as rejected without calls. Every call is counted
  in the ledger first and refused over the cap.
- ``contamination``: the status-only agent-case contamination check
  (``scratchcoder.agent.synth.contamination``) on generated instances, built rows or a decoded
  replay index; writes the rejected row IDs only, with the input's SHA-256 and kind and the
  registry of every case set checked.
- ``dev-family``: the development task-family rule (``task-clusters-v1``,
  ``scratchcoder.agent.synth.dev_family``) on the accepted instances of every ``--judged``
  build, with pool development and training tasks, the builds' rejected instances and any
  ``--bridge-instances`` set in the family graph, and on the replay index's rows (dropped
  when synthetic nodes add a development task to a row's family). An instance joins families
  only through its family items (task statement without the generator's fixed phrasing,
  specification text, target docstring, reference target); everything else its rows show is
  checked directly against development tasks and the pool tasks in their families (solution
  and canonical AST, prompt Jaccard, shared names). Each judged build must name its instance
  set's registry and each row its instance's registered ``case.json``. Drops every row of an
  excluded or flagged instance and of every instance of its template (or of a template
  holding a bridge with a direct signal, or named by ``--drop-template`` with its
  ``--drop-ruling``), and every replay row on a flag edge to a development task in
  ``--pending-flag-manifest`` unless an adjudication record (``--flag-decisions``) resolves
  each such edge as not equivalent. Writes one status-only result bound to the rows metadata
  and the replay index, and a private ``--evidence`` file with every link the earlier item
  mapping made and this one does not, by category. Runs nothing.
- ``mix``: agent rows plus replay rows at a target ratio of assistant-loss tokens. It needs
  both contamination results, computed on exactly these inputs and against exactly the case
  registries named by ``--case-registry-sha256``, and a dev-family result of this item
  mapping for both sides (with any ``--require-dropped-template`` dropped), caps instances
  per distinct reply, and reports how many pending-flag candidates enter the mix.
- ``manifest``: families, templates, bug and mutation kinds, workflow types, counts, file
  hashes and ledger totals.

Case code, references, mutants and the independent oracle never run on the host: the host
parses them; execution goes only through ``scratchcoder.eval.harness.isolated_execution``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scratchcoder.agent import cases as agent_cases
from scratchcoder.agent import execution
from scratchcoder.agent.synth import (
    GENERATOR_VERSION,
    contamination,
    dev_family,
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
from scratchcoder.eval.tasks import Task
from tools.admission_check import _receipt_image
from tools.verifier_scan import SealedPathError, guard

DEFAULT_MUTANTS = 2
DEFAULT_MAX_PER_REPLY = 3
"""Instances that may share one reference target (the whole assistant turn of each of their
rows): with ``k = 2`` mutants, at most 9 rows train on one reply text."""
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
DEV_CHECKS = (
    f"development tasks: the task-family rule {dev_family.RULE} on both sides (dev-family "
    "result), and 13-gram overlap with development prompts and solutions where a "
    "contamination result names a development file (dev_ngram_sha256)"
)
DEV_SCOPE = (
    f"development tasks: required before mixing; the build counts here come before it. mix "
    f"refuses without a {dev_family.RULE} dev-family result for both sides and records it "
    "with the 13-gram checks (see the mix record; dev_family below when one is given)"
)
"""The manifest's statement: it describes builds, which come before the family rule."""


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
        if any(agent_cases.forbidden_part(part) for part in path.resolve().parts):
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
    per_reply: Counter[str] = Counter()
    try:
        prior_ids, prior = _prior_instances(args.prior_instances or (), per_reply)
    except (OSError, ValueError, KeyError) as error:
        print(f"refused: prior instances: {error}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    per_template: Counter[str] = Counter()
    for template, bug, index in instances.plan_instances(selected, args.per_bug):
        start = per_template[template.id] * args.mutants
        try:
            probe = instances.build(template, bug, index, args.seed_base, encoder=encoder)
            reply = instances.reply_sha256(probe.reference_target)
            if args.max_per_reply and per_reply[reply] >= args.max_per_reply:
                dropped.append(
                    {
                        "template_id": template.id,
                        "bug_kind": bug,
                        "index": index,
                        "problems": [f"reply cap: {args.max_per_reply} instances share it"],
                    }
                )
                continue
            kinds = _ordered_kinds(
                mutation.applicable(
                    probe.reference_target,
                    forbid=[probe.original_target],
                    symmetric=probe.symmetric,
                )
            )
            # Kinds in cycle order from this template's position; a kind left with no mutant
            # once the earlier ones are chosen (its only mutant equals one of theirs, after
            # the canonical rewrite) is passed over for the next.
            chosen: list[str] = []
            for kind in instances.mutation_cycle(kinds, start, len(kinds)):
                try:
                    instances.build(
                        template, bug, index, args.seed_base, mutation_kinds=[*chosen, kind]
                    )
                except mutation.MutationError:
                    continue
                chosen.append(kind)
                if len(chosen) == args.mutants:
                    break
            if len(chosen) < args.mutants:
                raise TemplateError(f"only {len(chosen)} usable mutation kinds")
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
        if instance.id in prior_ids:
            print(f"refused: {instance.id} is already in a prior build", file=sys.stderr)
            return 2
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
        per_reply[reply] += 1
        kept.append(
            {
                "id": instance.id,
                "template_id": template.id,
                "family": template.family.id,
                "workflow_type": template.workflow,
                "bug_kind": bug,
                "mutation_kinds": chosen,
                "applicable_mutation_kinds": kinds,
                "reference_sha256": reply,
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
    # Hash-checked and statically validated again.
    loaded = agent_cases.load_cases(out, synthetic=True)
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
        "max_instances_per_reply": args.max_per_reply,
        "prior_instances": prior,
        "replies": _reply_report(kept, selected, args.capacity_draws, args.seed_base),
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


def _prior_instances(
    directories: Sequence[Path], per_reply: Counter[str]
) -> tuple[set[str], list[dict[str, Any]]]:
    """Count the replies of earlier generated instance sets toward the reply cap.

    Each directory's ``generate.json`` must name the registry the directory holds now, so the
    counts are those of exactly the instances that may be built. Returns the prior instance
    IDs (a new instance may not reuse one) and a record of each directory.
    """
    ids: set[str] = set()
    record: list[dict[str, Any]] = []
    for directory in directories:
        generated = json.loads((directory / "generate.json").read_text(encoding="utf-8"))
        registry = agent_cases.registry_sha256(directory)
        if generated["registry_sha256"] != registry:
            raise ValueError(f"{directory}: generate.json names another registry")
        for row in generated["kept"]:
            if row["id"] in ids:
                raise ValueError(f"{row['id']} appears in two prior builds")
            ids.add(row["id"])
            per_reply[row["reference_sha256"]] += 1
        record.append(
            {
                "directory": str(directory),
                "registry_sha256": registry,
                "instances": len(generated["kept"]),
            }
        )
    return ids, record


def _reply_report(
    kept: Sequence[dict[str, Any]],
    templates: Sequence[Any],
    draws: int,
    seed_base: int,
) -> dict[str, Any]:
    """Distinct replies (reference targets) of the kept instances, per template, beside how
    many distinct replies ``draws`` seeds per bug kind can render (the reach of an
    expansion; size it by distinct replies, not by the call budget)."""
    by_template: dict[str, Counter[str]] = {}
    for row in kept:
        by_template.setdefault(row["template_id"], Counter())[row["reference_sha256"]] += 1
    every = Counter(row["reference_sha256"] for row in kept)
    return {
        "distinct": len(every),
        "max_instances_per_reply": max(every.values(), default=0),
        "capacity_draws_per_bug": draws,
        "capacity_total": sum(
            instances.reply_capacity(template, draws, seed_base) for template in templates
        )
        if draws
        else None,
        "per_template": {
            template.id: {
                "instances": sum(by_template.get(template.id, Counter()).values()),
                "distinct": len(by_template.get(template.id, Counter())),
                "capacity": instances.reply_capacity(template, draws, seed_base) if draws else None,
            }
            for template in templates
        },
    }


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
    agent_cases.load_cases(args.instances, synthetic=True)  # The registered instances.
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
    try:
        with call_ledger.Ledger.locked(args.ledger, args.max_calls) as book:
            skipped: list[dict[str, Any]] = []
            added = call_ledger.prior_from_records(book, args.seed_from or [], skipped=skipped)
            print(json.dumps({"added": added, "skipped": skipped, **book.summary()}))
    except call_ledger.LedgerError as error:
        print(f"refused: ledger: {error}", file=sys.stderr)
        return 2
    return 0


def _oracle_controls(
    args: argparse.Namespace,
    book: call_ledger.Ledger,
    chosen: Sequence[agent_cases.Case],
    records: dict[str, dict[str, Any]],
    oracles: dict[str, Any],
    pinned: independent.Pin,
    image_id: str,
    execute: execution.Execute,
) -> int:
    """Run the reference (a positive control) and the four negative controls against the
    independent oracle's checks alone, once per template (its first selected instance).
    Writes ``oracle-controls.jsonl`` (never read as gating controls) and ``record.json``;
    builds no instance and no row."""
    out: Path = args.out
    usable = [
        c for c in chosen if isinstance(oracles[records[c.id]["template_id"]], independent.Oracle)
    ]
    planned = (len(mutation.CONTROL_KINDS) + 1) * len(usable)
    if planned > book.remaining:
        print(
            f"refused: {planned} planned sandbox calls exceed the {book.remaining} the ledger "
            f"cap leaves ({book.total} of {book.cap} used)",
            file=sys.stderr,
        )
        return 2
    out.mkdir(parents=True)
    target = out / "oracle-controls.jsonl"
    target.touch(exist_ok=False)
    started = utc_now()
    context = {"value": ""}
    counted = book.execute(execute, run=out.name, context=lambda: context["value"])
    stopped = ""
    found: list[dict[str, Any]] = []
    for case in usable:
        record = records[case.id]
        oracle = oracles[record["template_id"]]
        context["value"] = f"{case.id}/oracle-controls"
        try:
            got = rows.run_controls(case, record, oracle, execute=counted, checks="independent")
        except call_ledger.CallCapReached as error:
            stopped = str(error)
            break
        got = [
            {**row, "independent_sums_sha256": pinned.sums_sha256, "oracle_sha256": oracle.sha256}
            for row in got
        ]
        rows.append_jsonl(target, got)
        found.extend(got)
    commit, dirty = git_state()
    # A template's controls count as rejected by the oracle only when the oracle passed its
    # reference (the positive control) and all four controls came back.
    rows_by: dict[str, list[dict[str, Any]]] = {}
    for row in found:
        rows_by.setdefault(row["template_id"], []).append(row)
    reference_ok = {
        t: any(r["control"] == "reference" and r["ok"] for r in got) for t, got in rows_by.items()
    }
    by_template = {
        t: reference_ok[t]
        and {r["control"] for r in got} >= set(mutation.CONTROL_KINDS)
        and all(r["ok"] for r in got)
        for t, got in rows_by.items()
    }
    controls_found = [row for row in found if row["control"] != "reference"]
    summary = {
        "schema": "agent-synth-oracle-controls-v1",
        "generator_version": GENERATOR_VERSION,
        "started_at": started,
        "finished_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "runtime_image_id": image_id,
        "instances_registry_sha256": agent_cases.registry_sha256(args.instances),
        "independent": {"directory": str(args.independent), **pinned.record()},
        "planned_calls": planned,
        "rows": len(found),
        "reference_passed": sorted(t for t, ok in reference_ok.items() if ok),
        "reference_not_passed": sorted(t for t, ok in reference_ok.items() if not ok),
        "controls": len(controls_found),
        # Counted only for templates whose reference the oracle passed.
        "rejected_by_oracle_alone": sum(
            row["ok"] for row in controls_found if reference_ok[row["template_id"]]
        ),
        "templates_all_rejected": sorted(t for t, ok in by_template.items() if ok),
        "templates_not_all_rejected": sorted(t for t, ok in by_template.items() if not ok),
        "templates_without_oracle": sorted(
            records[c.id]["template_id"] for c in chosen if c not in usable
        ),
        "stopped": stopped,
        "ledger": book.summary(),
    }
    atomic_json(out / "record.json", summary)
    keys = ("rows", "controls", "rejected_by_oracle_alone", "reference_not_passed", "stopped")
    print(json.dumps({k: summary[k] for k in keys}))
    return 0 if not stopped else 3


def _prior_controls(paths: Sequence[Path], oracles: dict[str, Any]) -> dict[str, bool]:
    """Templates whose negative controls an earlier build already ran against the oracle
    pinned now: template -> whether every control failed (all four kinds present)."""
    rows_by_template: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by_template.setdefault(str(row["template_id"]), []).append(row)
    found: dict[str, bool] = {}
    for template_id, found_rows in rows_by_template.items():
        oracle = oracles.get(template_id)
        if not isinstance(oracle, independent.Oracle):
            continue
        # Only full-grading controls gate rows; oracle-only measurements never do.
        current = [
            row
            for row in found_rows
            if row.get("oracle_sha256") == oracle.sha256 and row.get("checks", "full") == "full"
        ]
        kinds = {str(row["control"]) for row in current}
        if not current:
            continue
        found[template_id] = kinds >= set(mutation.CONTROL_KINDS) and all(
            row["ok"] for row in current
        )
    return found


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
        loaded = agent_cases.load_cases(args.instances, synthetic=True)
    except (agent_cases.CaseError, OSError, ValueError, KeyError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    try:
        pinned = independent.pin(args.independent, expected=args.independent_sha256)
    except (independent.OracleError, OSError, UnicodeDecodeError) as error:
        print(f"refused: independent oracles: {error}", file=sys.stderr)
        return 2
    try:
        with call_ledger.Ledger.locked(args.ledger, args.max_calls) as book:
            book.require_seeded()
            return _build(args, book, loaded, pinned, image_id, execute=execute, loader=loader)
    except call_ledger.LedgerError as error:
        print(f"refused: ledger: {error}", file=sys.stderr)
        return 2


def _build(
    args: argparse.Namespace,
    book: call_ledger.Ledger,
    loaded: Sequence[agent_cases.Case],
    pinned: independent.Pin,
    image_id: str,
    *,
    execute: execution.Execute,
    loader: Loader,
) -> int:
    """The executing part of validate-and-build, run while the ledger lock is held."""
    out: Path = args.out
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
    if args.oracle_controls_only:
        first: dict[str, agent_cases.Case] = {}
        for case in chosen:
            first.setdefault(records[case.id]["template_id"], case)
        chosen = list(first.values())
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
    # Negative controls gate rows: a template's instances are built only when all four
    # controls failed its grading script, in this run (--controls) or in an earlier build
    # against the same oracle (--controls-from).
    passed = _prior_controls(args.controls_from or [], oracles)
    if args.oracle_controls_only:
        return _oracle_controls(args, book, chosen, records, oracles, pinned, image_id, execute)
    to_control = sorted(t for t in templates if t not in passed) if args.controls else []
    planned = sum(rows.calls_per_instance(len(records[c.id]["mutants"])) for c in chosen)
    planned += len(mutation.CONTROL_KINDS) * len(to_control)
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
    stopped = ""
    totals: Counter[str] = Counter()

    def reject(case_id: str, reason: str, stamp: dict[str, Any], key: str) -> None:
        rows.append_jsonl(
            out / "instances.jsonl",
            [{"instance": case_id, "accepted": False, "reasons": [reason], "calls": 0, **stamp}],
        )
        totals[key] += 1

    for case in chosen:
        record = records[case.id]
        template_id = record["template_id"]
        oracle = oracles[template_id]
        if isinstance(oracle, independent.OracleError):
            reject(case.id, f"oracle: {oracle}", {"independent_sums_sha256": pin_sha}, "no_oracle")
            continue
        stamp = {"independent_sums_sha256": pin_sha, "oracle_sha256": oracle.sha256}
        try:
            if template_id in to_control and template_id not in passed:
                context["value"] = f"{case.id}/controls"
                found = [
                    {**row, **stamp}
                    for row in rows.run_controls(case, record, oracle, execute=counted)
                ]
                rows.append_jsonl(out / "controls.jsonl", found)
                passed[template_id] = all(row["ok"] for row in found)
            if template_id not in passed:
                reject(case.id, "controls_not_run", stamp, "no_controls")
                continue
            if not passed[template_id]:
                reject(case.id, "controls_not_all_failed", stamp, "controls_failed")
                continue
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
        "controls": {
            "rows": len(controls),
            # None, not True, when no control ran: "all failed" needs at least one control.
            "all_failed": all(row["ok"] for row in controls) if controls else None,
            "from": [str(path) for path in args.controls_from or []],
            "templates_passed": sorted(t for t, ok in passed.items() if ok and t in templates),
            "templates_failed": sorted(t for t, ok in passed.items() if not ok),
            "templates_without_controls": sorted(templates - set(passed)),
        },
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
        source, kind = args.instances, "instances"
        digest = agent_cases.registry_sha256(args.instances)
    elif args.rows_meta is not None:
        found = contamination.meta_rows(args.rows_meta)
        source, kind, digest = args.rows_meta, "rows_meta", _sha_file(args.rows_meta)
    else:
        found = contamination.decoded_rows(args.decoded)
        source, kind, digest = args.decoded, "decoded", _sha_file(args.decoded)
    result = contamination.check(index, found)
    result["input"] = str(source)
    result["input_kind"] = kind
    # For instances, the hash of their registry (which pins every file); otherwise the file.
    result["input_sha256"] = digest
    result["index"] = dict(contamination.describe(index))
    atomic_json(args.out, result)
    print(json.dumps({"checked": result["checked"], "rejected": len(result["rejected"])}))
    return 0


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return _hashed_jsonl(path)[0]


def _hashed_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    """The rows of a JSONL file and the SHA-256 of exactly the bytes they were parsed from."""
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return rows, hashlib.sha256(data).hexdigest()


def _pool_tasks(path: Path, split: str) -> list[Task]:
    tasks = [Task(**row) for row in _jsonl(path)]
    if any(task.split != split for task in tasks):
        raise ValueError(f"{path}: every task must be a {split} task")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError(f"{path}: repeated task IDs")
    return tasks


@dataclass
class DevFamilyInputs:
    record: dict[str, Any]
    subjects: list[dev_family.Instance]
    bridges: list[dev_family.Instance]
    spec_texts: dict[str, list[str]]
    user_texts: dict[str, list[str]]
    template_of: dict[str, str]
    bridge_templates: dict[str, set[str]]


ITEM_MAPPING = {
    "version": dev_family.MAPPING,
    "family_prompts": "the template's specification text and the target docstring",
    "family_solutions": "the reference target (every row's reply)",
    "exposure_code": "the original target, the wrong attempts, every module of the original "
    "and the fixed repository and each of its top-level functions and classes, and code "
    "inside each row's user text: solution and canonical AST against every other node in a "
    "family with a development task (pool tasks, accepted and bridge instances), never a "
    "family link",
    "exposure_text": "the generated task statement and every row's user text: prompt "
    "Jaccard 0.5 against those nodes' prompts, and shared defined names with development "
    "tasks",
    "scaffolding": "never a family item: the generated task statement (the generator's "
    "phrasings and traceback framing filled in with test values and helper names), the "
    "harness prompt's fixed blocks, and the catalog's non-target functions",
    "node_ids": dev_family.NAMESPACE + "<instance>",
}
PENDING_FLAG_KEY = "sft_rows_on_pending_dev_crossing_flag_edges_only"
"""The replay stage's count of rows on pending flag edges to development tasks."""


def _dev_family_inputs(args: argparse.Namespace, meta: Sequence[dict[str, Any]]) -> DevFamilyInputs:
    """Subject and bridge instances, their specification texts and row user texts, each
    subject's template, the templates under each bridge node ID, and the record of every
    source.

    Each ``--judged`` pair must belong together: the build's record names the instance set's
    registry, and every row names the registered ``case.json`` of the instance it came from,
    so a superseded set that reuses the instance IDs cannot stand in for the real one."""
    spec_texts = dev_family.load_specs(args.spec_files)
    rows_by_instance: dict[str, list[dict[str, Any]]] = {}
    if len({str(row["id"]) for row in meta}) != len(meta):
        raise ValueError("the rows metadata repeats a row ID")
    for row in meta:
        rows_by_instance.setdefault(str(row["instance"]), []).append(row)
    subjects: list[dev_family.Instance] = []
    bridges: list[dev_family.Instance] = []
    record: dict[str, Any] = {"judged": [], "bridges": []}
    for directory, build in args.judged:
        found, registry = dev_family.read_instances(directory)
        build_record = build / "record.json"
        named = json.loads(build_record.read_text(encoding="utf-8")).get(
            "instances_registry_sha256"
        )
        if named != registry:
            raise ValueError(f"{build} was not built from {directory} as it is (registry)")
        outcome = {}
        for line in _jsonl(build / "instances.jsonl"):
            if str(line["instance"]) in outcome:
                raise ValueError(f"{build}: {line['instance']} is recorded twice")
            outcome[str(line["instance"])] = bool(line["accepted"])
        unknown = set(outcome) - {item.id for item in found}
        if unknown:
            raise ValueError(f"{build}: {len(unknown)} built instances are not in {directory}")
        accepted = [item for item in found if outcome.get(item.id)]
        subjects.extend(accepted)
        bridges.extend(item for item in found if not outcome.get(item.id))
        record["judged"].append(
            {
                "instances": str(directory),
                "registry_sha256": registry,
                "build": str(build),
                "build_record_sha256": _sha_file(build_record),
                "build_instances_sha256": _sha_file(build / "instances.jsonl"),
                "accepted": len(accepted),
                "rejected": sum(1 for item in found if outcome.get(item.id) is False),
                "not_built": sum(1 for item in found if item.id not in outcome),
            }
        )
    for directory in args.bridge_instances or []:
        found, registry = dev_family.read_instances(directory)
        bridges.extend(found)
        record["bridges"].append(
            {
                "instances": str(directory),
                "registry_sha256": registry,
                "count": len(found),
            }
        )
    template_of = {item.id: item.template_id for item in subjects}
    if len(template_of) != len(subjects):
        raise ValueError("an accepted instance ID appears in two builds")
    spec_of = {item.id: item.spec_sha256 for item in subjects}
    for identifier, rows_of in rows_by_instance.items():
        if identifier not in template_of:
            raise ValueError(f"rows of {identifier}, which no build accepted")
        if any(str(row["template_id"]) != template_of[identifier] for row in rows_of):
            raise ValueError(f"rows of {identifier} name another template")
        if any(row.get("spec_sha256") != spec_of[identifier] for row in rows_of):
            raise ValueError(
                f"rows of {identifier} were not built from the judged instance (case.json)"
            )
    bridge_templates: dict[str, set[str]] = {}
    for item in bridges:
        bridge_templates.setdefault(dev_family.NAMESPACE + item.id, set()).add(item.template_id)
    missing = sorted({item.template_id for item in [*subjects, *bridges]} - set(spec_texts))
    if missing:
        raise ValueError(f"no specification text for templates {missing}")
    record["specs"] = [{"path": str(path), "sha256": _sha_file(path)} for path in args.spec_files]
    record["subjects_without_rows"] = sum(1 for item in subjects if item.id not in rows_by_instance)
    record["item_mapping"] = ITEM_MAPPING
    user_texts = {
        identifier: [str(row["user_text"]) for row in rows_of]
        for identifier, rows_of in rows_by_instance.items()
    }
    return DevFamilyInputs(
        record, subjects, bridges, spec_texts, user_texts, template_of, bridge_templates
    )


def _replay_stage(path: Path) -> dict[str, Any]:
    """The replay stage's record, by path and SHA-256, with every count it reports as kept
    on pending (unreviewed) flag edges, so the in-graph figure can be read beside it."""
    data = path.read_bytes()
    document = json.loads(data)
    pending = {
        f"{rule.get('rule', '?')}.{key}": value
        for rule in document.get("rules_applied", ())
        if isinstance(rule, dict)
        for key, value in (rule.get("informational_not_dropped") or {}).items()
        if "pending" in key
    }
    return {
        "record": str(path),
        "record_sha256": hashlib.sha256(data).hexdigest(),
        "format": document.get("format"),
        "pending_flag_rows_reported": pending,
    }


def _pending_flag_block(
    args: argparse.Namespace,
    index: Sequence[dict[str, Any]],
    stage: dict[str, Any],
) -> tuple[dev_family.PendingFlags, dict[str, Any]]:
    """The pending flag edges of the replay rows (every kind), resolved only by existing
    adjudication records, and the result block that cites them. Refuses when the replay
    stage reports no number of rows on such edges, or another one than the manifest gives
    here (a stale or wrong manifest)."""
    manifest_data = args.pending_flag_manifest.read_bytes()
    edges = dev_family.flag_edges(json.loads(manifest_data))
    decisions: list[dict[str, Any]] = []
    record: dict[str, Any] = {"decisions": None, "decisions_sha256": None}
    if args.flag_decisions is not None:
        decisions, digest = _hashed_jsonl(args.flag_decisions)
        record = {"decisions": str(args.flag_decisions), "decisions_sha256": digest}
    kinds = {str(row["id"]): str(row["kind"]) for row in index}
    flags = dev_family.pending_flags(edges, decisions, kinds)
    reported = None
    for key, value in (stage.get("pending_flag_rows_reported") or {}).items():
        if key.endswith("." + PENDING_FLAG_KEY):
            reported = int(value)
    if reported is None:
        raise ValueError(f"the replay stage's record reports no {PENDING_FLAG_KEY} count")
    if reported != len(flags.candidates):
        raise ValueError(
            f"the replay stage reports {reported} rows on pending flag edges; the manifest "
            f"gives {len(flags.candidates)} among the replay rows"
        )
    block = {
        "applied": True,
        "rule": "a replay row on a flag edge to a development task is dropped "
        "(pending_flag_unresolved) unless every such edge is resolved not equivalent by its "
        "own review or by a decision on exactly that pair",
        "manifest": str(args.pending_flag_manifest),
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        **record,
        "edges_to_dev": flags.edges,
        "candidates": len(flags.candidates),
        "candidates_by_kind": dict(sorted(Counter(kinds[row] for row in flags.candidates).items())),
        "candidate_ids": flags.candidates,
        "kept_resolved": len(flags.kept),
        "kept_resolved_decision_lines": flags.kept,
        "excluded": len(flags.excluded),
        "replay_stage_reported": reported,
    }
    return flags, block


def _legacy_comparison(
    dev: Sequence[Task],
    train: Sequence[Task],
    inputs: DevFamilyInputs,
    current: Sequence[dev_family.Node],
    current_bridges: Sequence[dev_family.Node],
    replay: Sequence[str],
    found: dev_family.Assessment,
) -> dict[str, Any]:
    """The links the earlier item mapping made and this one does not (and the reverse),
    each with its category, and the earlier mapping's judgement beside this one's."""
    roles = dev_family.unit_kinds(catalog())
    base = [dev_family.pool_labeled(task) for task in [*dev, *train]]
    specs = inputs.spec_texts
    old = [
        dev_family.legacy_items(
            item, specs[item.template_id], inputs.user_texts.get(item.id, ()), roles
        )
        for item in inputs.subjects
    ] + [
        dev_family.legacy_items(item, specs[item.template_id], (), roles) for item in inputs.bridges
    ]
    new = [
        dev_family.current_items(item, specs[item.template_id])
        for item in [*inputs.subjects, *inputs.bridges]
    ]
    for labeled, node in zip(new, [*current, *current_bridges], strict=True):
        if labeled.node().prompts != node.prompts or labeled.node().solutions != node.solutions:
            raise ValueError(f"{node.id}: the evidence items differ from the judged items")
    changes = dev_family.link_changes([*base, *old], [*base, *new], len(base))
    legacy = dev_family.assess(
        [dev_family.task_node(task) for task in dev],
        [dev_family.task_node(task) for task in train],
        [labeled.node() for labeled in old[: len(inputs.subjects)]],
        [labeled.node() for labeled in old[len(inputs.subjects) :]],
        replay=replay,
    )
    prefix = len(dev_family.NAMESPACE)
    legacy_reason = {j.id[prefix:]: j.reason for j in legacy.subjects}
    transitions = Counter(
        f"{legacy_reason[j.id[prefix:]] or 'clear'} -> {j.reason or 'clear'}"
        for j in found.subjects
    )
    return {
        "changes": changes,
        "legacy": {
            "mapping": dev_family.LEGACY_MAPPING,
            "by_reason": dict(
                sorted(Counter(j.reason or "clear" for j in legacy.subjects).items())
            ),
            "replay_rejected": len(legacy.replay_excluded),
            "families": legacy.families,
        },
        "transitions": dict(sorted(transitions.items())),
        "scaffolding_roles": roles,
    }


def _tracked_location(path: Path) -> str | None:
    """Why ``path`` may not hold a private record: it lies in a Git work tree that does not
    ignore it (a later ``git add`` could stage it). ``None`` when it is outside every work
    tree or ignored there."""
    probe = path.resolve().parent
    while not probe.exists():
        probe = probe.parent
    try:
        top = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if top.returncode != 0:
            return None
        ignored = subprocess.run(
            ["git", "-C", top.stdout.strip(), "check-ignore", "-q", str(path.resolve())],
            capture_output=True,
            check=False,
        )
    except OSError as error:
        return f"cannot check {path} with git: {error}"
    if ignored.returncode == 0:
        return None
    return f"{path} is inside the Git work tree {top.stdout.strip()} and not ignored"


def run_dev_family(args: argparse.Namespace) -> int:
    for path in (args.out, args.evidence):
        if path.exists():
            print(f"refused: {path} exists", file=sys.stderr)
            return 2
        reason = _tracked_location(path)
        if reason is not None:
            print(f"refused: private record: {reason}", file=sys.stderr)
            return 2
    owner = sorted(set(args.drop_template or ()))
    if owner and args.drop_ruling is None:
        print("refused: --drop-template needs --drop-ruling (the ruling's record)", file=sys.stderr)
        return 2
    if not args.bridge_instances:
        print(
            "warning: no --bridge-instances: chains through superseded instance sets are not "
            "in the graph",
            file=sys.stderr,
        )
    try:
        dev = _pool_tasks(args.dev, "dev")
        train = _pool_tasks(args.train, "train")
        meta, meta_sha256 = _hashed_jsonl(args.rows_meta)
        index, index_sha256 = _hashed_jsonl(args.decoded)
        stage = _replay_stage(args.replay_record)
        inputs = _dev_family_inputs(args, meta)
        if len({str(row["id"]) for row in index}) != len(index):
            raise ValueError("the replay index repeats a row ID")
        present = {item.template_id for item in [*inputs.subjects, *inputs.bridges]}
        absent = sorted(set(owner) - present)
        if absent:
            raise ValueError(f"--drop-template names templates no instance has: {absent}")
        specs = inputs.spec_texts
        subjects = [
            dev_family.instance_node(
                item, specs[item.template_id], inputs.user_texts.get(item.id, ())
            )
            for item in inputs.subjects
        ]
        bridges = [
            dev_family.instance_node(item, specs[item.template_id], ()) for item in inputs.bridges
        ]
        replay = [str(row["id"]) for row in index if str(row["kind"]) in set(args.replay_kinds)]
        dev_nodes = [dev_family.task_node(task) for task in dev]
        train_nodes = [dev_family.task_node(task) for task in train]
        found = dev_family.assess(dev_nodes, train_nodes, subjects, bridges, replay=replay)
        # The same judging without any bridge, to count what only bridges reach.
        unbridged = dev_family.assess(dev_nodes, train_nodes, subjects)
        flags, flag_block = _pending_flag_block(args, index, stage)
        comparison = _legacy_comparison(dev, train, inputs, subjects, bridges, replay, found)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"refused: dev-family: {error}", file=sys.stderr)
        return 2
    prefix = len(dev_family.NAMESPACE)
    template_of = inputs.template_of

    def strip(judgements: Sequence[dev_family.Judgement]) -> list[dev_family.Judgement]:
        return [dev_family.Judgement(j.id[prefix:], j.status, j.reason) for j in judgements]

    judged = strip(found.subjects)
    bridge_template_hits = sorted(
        {template for node in found.bridges_hit for template in inputs.bridge_templates[node]}
    )
    dropped = dev_family.expand_templates(judged, template_of, bridge_template_hits, owner)
    dropped_unbridged = dev_family.expand_templates(
        strip(unbridged.subjects), template_of, (), owner
    )
    clear_unbridged = {j.id for j in strip(unbridged.subjects) if j.status == "clear"}
    rejected = sorted(str(row["id"]) for row in meta if str(row["instance"]) in dropped)
    kept = [row for row in meta if str(row["instance"]) not in dropped]
    rows_dropped = {
        reason: sum(1 for row in meta if dropped.get(str(row["instance"])) == reason)
        for reason in dev_family.REASONS
    }
    replay_rejected = sorted(set(found.replay_excluded) | set(flags.excluded))
    changes = comparison["changes"]
    evidence = {
        "schema": "agent-dev-family-evidence-v1",
        "mapping": dev_family.MAPPING,
        "legacy_mapping": dev_family.LEGACY_MAPPING,
        "result": str(args.out),
        "categories": dev_family.CATEGORIES,
        "retained_kinds": sorted(dev_family.RETAINED),
        "scaffolding_source": {
            "task_statement": "generated by scratchcoder.agent.synth.instances (TASK_PHRASES, "
            "TRACEBACK_HEADER, FRAME_LINE) from test values and helper names; exposure text "
            "only, and refused when it matches no phrasing",
            "non_target_functions": "scratchcoder.agent.synth.catalog through "
            "dev_family.unit_kinds (template -> role -> kind)",
            "roles": comparison["scaffolding_roles"],
        },
        "removed_links": changes["removed"],
        "added_links": changes["added"],
        "legacy_judgement": comparison["legacy"],
        "judgement_transitions": comparison["transitions"],
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.evidence, evidence)
    commit, dirty = git_state()
    owner_rows = rows_dropped["owner_ruling"]
    result: dict[str, Any] = {
        "schema": dev_family.SCHEMA,
        "dev_family": dev_family.RULE,
        "rules": dev_family.rules_record(),
        "method": dev_family.METHOD,
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "pool": {
            "dev": str(args.dev),
            "dev_sha256": _sha_file(args.dev),
            "dev_tasks": len(dev),
            "train": str(args.train),
            "train_sha256": _sha_file(args.train),
            "train_tasks": len(train),
        },
        "synthetic": inputs.record,
        "families": found.families,
        "link_changes": {
            "evidence": str(args.evidence),
            "evidence_sha256": _sha_file(args.evidence),
            "removed": changes["removed"]["summary"],
            "added": changes["added"]["summary"],
            "legacy_by_reason": comparison["legacy"]["by_reason"],
            "legacy_replay_rejected": comparison["legacy"]["replay_rejected"],
        },
        "agent": {
            "input": str(args.rows_meta),
            "input_kind": "rows_meta",
            "input_sha256": meta_sha256,
            "checked": len(meta),
            "rejected": rejected,
            "instances": {
                "judged": len(judged),
                "by_status": {
                    status: sorted(item.id for item in judged if item.status == status)
                    for status in ("excluded", "flagged")
                },
                "by_status_count": {
                    status: sum(1 for item in judged if item.status == status)
                    for status in dev_family.STATUSES
                },
                "by_judged_reason": {
                    reason: sum(1 for item in judged if item.reason == reason)
                    for reason in dev_family.JUDGED_REASONS
                },
                "exposure_hits": found.exposure_hits,
                "dropped": dict(sorted(dropped.items())),
                "dropped_by_reason": {
                    reason: sum(1 for value in dropped.values() if value == reason)
                    for reason in dev_family.REASONS
                },
                "templates_dropped": sorted({template_of[i] for i in dropped}),
                "owner_drops": {
                    "templates": owner,
                    "ruling": None if args.drop_ruling is None else str(args.drop_ruling),
                    "ruling_sha256": None
                    if args.drop_ruling is None
                    else _sha_file(args.drop_ruling),
                    "instances": sum(1 for value in dropped.values() if value == "owner_ruling"),
                    "rows": owner_rows,
                },
                "bridges_with_direct_hit": found.bridges_hit,
                "templates_dropped_through_bridge_hits": bridge_template_hits,
                "only_through_bridges": {
                    "judged_not_clear": sorted(
                        j.id for j in judged if j.status != "clear" and j.id in clear_unbridged
                    ),
                    "dropped": sorted(set(dropped) - set(dropped_unbridged)),
                },
            },
            "rows_dropped_by_reason": rows_dropped,
            "rows_dropped_instance_level": sum(
                count
                for reason, count in rows_dropped.items()
                if reason not in ("template", "owner_ruling")
            ),
            "surviving": {
                "rows": len(kept),
                "instances": len({row["instance"] for row in kept}),
                "templates": len({row["template_id"] for row in kept}),
                "families": len({row["family"] for row in kept}),
                "workflow_types": sorted({str(row["workflow_type"]) for row in kept}),
                "rows_by_kind": dict(sorted(Counter(str(row["kind"]) for row in kept).items())),
                "mutation_kinds": sorted(
                    {str(row["mutation_kind"]) for row in kept if row.get("mutation_kind")}
                ),
                "loss_tokens": sum(int(row.get("loss_tokens", 0)) for row in kept),
                **mixing.reply_diversity(kept),
            },
        },
        "replay": {
            "input": str(args.decoded),
            "input_kind": "decoded",
            "input_sha256": index_sha256,
            "kinds": sorted(set(args.replay_kinds)),
            "checked": len(replay),
            "rejected": replay_rejected,
            "rejected_by_reason": {
                "dev_family_new_chain": len(found.replay_new_chain),
                "dev_family_new_link": len(found.replay_new_dev_link),
                "pending_flag_unresolved": len(flags.excluded),
            },
            "rejected_new_chain": len(found.replay_new_chain),
            "rejected_new_dev_link": len(found.replay_new_dev_link),
            "kept_in_dev_family_without_synthetic": len(found.replay_in_dev_family),
            # The in-graph shared-name edges of rows outside every dev family, beside the
            # manifest's pending flag edges that ``pending_flags`` applies.
            "kept_direct_named_edge_without_synthetic": len(found.replay_named),
            "pending_flags": flag_block,
            "replay_stage": stage,
            "not_judged_other_kinds": len(index) - len(replay),
        },
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "agent_rejected": len(rejected),
                "instances_dropped": len(dropped),
                "replay_rejected": len(replay_rejected),
                "pending_flag_candidates": len(flags.candidates),
                "links_removed": changes["removed"]["summary"]["links"],
            }
        )
    )
    return 0


def run_mix(args: argparse.Namespace) -> int:
    if args.out.exists():
        print(f"refused: {args.out} exists", file=sys.stderr)
        return 2
    # Both sides must have been checked, on exactly these inputs, against exactly the named
    # evaluation-case sets.
    checked = {}
    try:
        for side, path, source, kind in (
            ("agent", args.agent_rejected, args.agent_meta, "rows_meta"),
            ("replay", args.replay_rejected, args.replay_index, "decoded"),
        ):
            result = contamination.load_result(path)
            contamination.require_result(
                result, input_path=source, kind=kind, case_registries=args.case_registry_sha256
            )
            checked[side] = {
                "result": str(path),
                "result_sha256": _sha_file(path),
                "input_sha256": result["input_sha256"],
                "dev_ngram_sha256": result.get("index", {}).get("dev_sha256"),
            }
    except (ValueError, KeyError, OSError) as error:
        print(f"refused: contamination: {error}", file=sys.stderr)
        return 2
    # The development task-family rule, required for both sides in addition to the above.
    try:
        family = dev_family.load_result(args.dev_family)
        dev_family.require_result(
            family,
            agent_meta=args.agent_meta,
            replay_index=args.replay_index,
            replay_kinds=args.replay_kinds,
            dev_sha256=[side["dev_ngram_sha256"] for side in checked.values()],
            bridge_registries=args.require_instance_registry or (),
            dropped_templates=args.require_dropped_template or (),
            flag_manifest_sha256=args.require_flag_manifest_sha256,
        )
    except (ValueError, KeyError, OSError) as error:
        print(f"refused: dev-family: {error}", file=sys.stderr)
        return 2
    family_agent = set(family["agent"]["rejected"])
    family_replay = set(family["replay"]["rejected"])
    agent_skip = contamination.rejected_ids(args.agent_rejected) | family_agent
    replay_skip = contamination.rejected_ids(args.replay_rejected) | family_replay
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
    mixed, report = mixing.mix(
        agent, meta, replay, ratio=args.ratio, max_per_reply=args.max_per_reply
    )
    args.out.mkdir(parents=True)
    from scratchcoder.data.sft import write_rows

    write_rows(args.out / "mix.jsonl", mixed)
    pending = family["replay"]["pending_flags"]
    candidates = set(pending["candidate_ids"])
    mixed_ids = {row.id for row in mixed}
    report.update(
        {
            "agent_rows_sha256": _sha_file(args.agent_rows),
            "replay_sha256": _sha_file(args.replay),
            "replay_kinds": list(args.replay_kinds),
            "agent_rejected": len(agent_skip),
            "replay_rejected": len(replay_skip),
            "contamination": checked,
            "dev_family": {
                "result": str(args.dev_family),
                "result_sha256": _sha_file(args.dev_family),
                "rule": family["dev_family"],
                "pool_dev_sha256": family["pool"]["dev_sha256"],
                "pool_train_sha256": family["pool"]["train_sha256"],
                "agent_rejected": len(family_agent),
                "replay_rejected": len(family_replay),
                "instance_registries": sorted(
                    str(part["registry_sha256"])
                    for part in [
                        *family["synthetic"].get("judged", ()),
                        *family["synthetic"].get("bridges", ()),
                    ]
                ),
                "required_instance_registries": sorted(args.require_instance_registry or ()),
                "item_mapping": family["synthetic"]["item_mapping"]["version"],
                "owner_drops": family["agent"]["instances"]["owner_drops"],
                "required_dropped_templates": sorted(args.require_dropped_template or ()),
                "pending_flags": {
                    "candidates": len(candidates),
                    "kept_resolved": pending["kept_resolved"],
                    "excluded": pending["excluded"],
                    "candidates_in_mix": len(candidates & mixed_ids),
                    "manifest_sha256": pending["manifest_sha256"],
                    "decisions_sha256": pending["decisions_sha256"],
                    "replay_stage_reported": pending["replay_stage_reported"],
                    "required_manifest_sha256": args.require_flag_manifest_sha256,
                },
            },
            "dev_method": DEV_CHECKS,
            "case_registries": sorted(args.case_registry_sha256),
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
                **mixing.reply_diversity(meta),
                "controls": summary.get("controls"),
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
            "distinct_replies": len({r["reference_target"] for r in records}),
            "max_instances_per_reply": max(
                Counter(r["reference_target"] for r in records).values(), default=0
            ),
        },
        "contamination_scope": {
            "evaluation_cases": "13-gram overlap, reference AST and defined-name Jaccard "
            "against every registered evaluation case, plus the family blocklist",
            "development_tasks": DEV_SCOPE,
            "benchmarks": "13-gram overlap (at least three word tokens) with benchmark "
            "prompts and reference solutions",
        },
        "builds": builds,
    }
    if args.dev_family is not None:
        family = dev_family.load_result(args.dev_family)
        document["dev_family"] = {
            "result": str(args.dev_family),
            "result_sha256": _sha_file(args.dev_family),
            "rule": family["dev_family"],
            "item_mapping": family["synthetic"]["item_mapping"]["version"],
            "owner_drops": family["agent"]["instances"]["owner_drops"],
            "agent_rows_dropped": len(family["agent"]["rejected"]),
            "instances_dropped": len(family["agent"]["instances"]["dropped"]),
            "dropped_by_reason": family["agent"]["instances"]["dropped_by_reason"],
            "replay_rows_dropped": len(family["replay"]["rejected"]),
            "surviving": family["agent"]["surviving"],
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
    g.add_argument(
        "--max-per-reply",
        type=int,
        default=DEFAULT_MAX_PER_REPLY,
        help="at most this many instances share one reference target (0: no cap)",
    )
    g.add_argument(
        "--prior-instances",
        type=Path,
        nargs="+",
        help="earlier generated sets whose replies count toward --max-per-reply",
    )
    g.add_argument(
        "--capacity-draws",
        type=int,
        default=200,
        help="seeds per bug kind rendered to estimate each template's distinct replies",
    )
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
    v.add_argument(
        "--controls",
        action="store_true",
        help="negative controls once per template (templates not in --controls-from)",
    )
    v.add_argument(
        "--controls-from",
        type=Path,
        nargs="+",
        help="controls.jsonl of earlier builds; templates whose four controls all failed "
        "against the oracle pinned now need not run them again",
    )
    v.add_argument(
        "--oracle-controls-only",
        action="store_true",
        help="run the four negative controls against the independent oracle's checks alone, "
        "once per template, and build nothing (a measurement; these rows never gate)",
    )
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
    d = commands.add_parser("dev-family")
    d.add_argument("--dev", type=Path, required=True, help="the pool's development split")
    d.add_argument("--train", type=Path, required=True, help="the pool's training split")
    d.add_argument(
        "--judged",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("INSTANCES", "BUILD"),
        help="an instance set and its build: accepted instances are judged, the others are "
        "bridges (repeatable)",
    )
    d.add_argument(
        "--bridge-instances",
        type=Path,
        nargs="+",
        help="instance sets whose every instance is an unjudged bridge (superseded sets)",
    )
    d.add_argument(
        "--specs",
        dest="spec_files",
        type=Path,
        nargs="+",
        required=True,
        help="specs.json files giving each template's specification text",
    )
    d.add_argument("--rows-meta", type=Path, required=True)
    d.add_argument("--decoded", type=Path, required=True, help="the decoded replay index")
    d.add_argument("--replay-kinds", nargs="+", default=["code"])
    d.add_argument(
        "--replay-record",
        type=Path,
        required=True,
        help="the replay stage's record.json: its SHA-256 and the pending flag counts it "
        "reports are written beside the in-graph figure; a record without that count, or a "
        "different count of rows on pending flag edges than --pending-flag-manifest gives, "
        "is refused",
    )
    d.add_argument(
        "--pending-flag-manifest",
        type=Path,
        required=True,
        help="the cluster manifest whose flag_edges hold the replay rows' shared-name flag "
        "edges to development tasks: a row on one is dropped unless every such edge is "
        "resolved not equivalent",
    )
    d.add_argument(
        "--flag-decisions",
        type=Path,
        help="adjudication decisions (JSONL with train, dev and final): a decision final "
        "not_equivalent on exactly a flag edge's pair resolves that edge",
    )
    d.add_argument(
        "--drop-template",
        nargs="+",
        metavar="TEMPLATE",
        help="templates the owner ruled out: every instance is dropped (owner_ruling)",
    )
    d.add_argument(
        "--drop-ruling", type=Path, help="the record of the ruling --drop-template cites"
    )
    d.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="where to write the private evidence file: every link the earlier item mapping "
        "made and this one does not, with its category",
    )
    d.add_argument("--out", type=Path, required=True)
    m = commands.add_parser("mix")
    m.add_argument("--agent-rows", type=Path, required=True)
    m.add_argument("--agent-meta", type=Path, required=True)
    m.add_argument("--replay", type=Path, required=True)
    m.add_argument("--replay-index", type=Path, required=True)
    m.add_argument("--replay-kinds", nargs="+", default=["code"])
    m.add_argument(
        "--agent-rejected",
        type=Path,
        required=True,
        help="the contamination result of --agent-meta (contamination --rows-meta)",
    )
    m.add_argument(
        "--replay-rejected",
        type=Path,
        required=True,
        help="the contamination result of --replay-index (contamination --decoded)",
    )
    m.add_argument(
        "--dev-family",
        type=Path,
        required=True,
        help="the dev-family result of --agent-meta and --replay-index (dev-family)",
    )
    m.add_argument(
        "--require-instance-registry",
        nargs="+",
        metavar="SHA256",
        help="registry SHA-256 of instance sets (judged or bridge) the dev-family result "
        "must have held in its graph, such as a superseded set",
    )
    m.add_argument(
        "--require-dropped-template",
        nargs="+",
        metavar="TEMPLATE",
        help="templates the dev-family result must have dropped by owner ruling",
    )
    m.add_argument(
        "--require-flag-manifest-sha256",
        metavar="SHA256",
        help="the SHA-256 of the flag manifest the dev-family result must have applied",
    )
    m.add_argument(
        "--case-registry-sha256",
        nargs="+",
        required=True,
        help="registry SHA-256 of every evaluation-case set both results must cover",
    )
    m.add_argument(
        "--max-per-reply",
        type=int,
        default=DEFAULT_MAX_PER_REPLY,
        help="at most this many instances share one reply (0: no cap)",
    )
    m.add_argument("--ratio", type=float, default=1.0)
    m.add_argument("--out", type=Path, required=True)
    f = commands.add_parser("manifest")
    f.add_argument("--instances", type=Path, required=True)
    f.add_argument("--build", type=Path, nargs="*")
    f.add_argument("--specs", type=Path)
    f.add_argument("--ledger", type=Path)
    f.add_argument("--max-calls", type=int, default=call_ledger.DEFAULT_CAP)
    f.add_argument(
        "--dev-family",
        type=Path,
        help="a dev-family result to record (its SHA-256, dropped counts and survivors)",
    )
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
    "train",
    "dev_family",
    "replay_record",
    "pending_flag_manifest",
    "flag_decisions",
    "drop_ruling",
    "evidence",
)
PATH_LIST_ARGS = (
    "seed_from",
    "cases",
    "benchmarks",
    "build",
    "controls_from",
    "prior_instances",
    "bridge_instances",
    "spec_files",
)


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
    for pair in getattr(args, "judged", None) or []:
        paths.extend(pair)
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
    if args.command == "dev-family":
        return run_dev_family(args)
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
