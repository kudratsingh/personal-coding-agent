"""Build and adjudicate the dev v2 task-family candidate manifest from static signals.

``candidates``
    Reads the pool's train and dev files, the run 1b SFT file (IDs only) and the RL
    candidate file (IDs only), plus the screening record, and writes a *candidate*
    manifest (``scratchcoder.data.clusters``). Nothing is executed and nothing is
    confirmed: every cluster is ``pending``, or ``unclear_quarantined`` when the harness
    could not compare it. The sealed test, holdout and RL-only files are refused by name.
    The output directory must be new; no existing run artifact is changed.
``adjudicate``
    Records one reviewer decision on one cluster, or with ``--member`` on one member of an
    ``unclear_quarantined`` (mixed) cluster, and appends it to the adjudication log. The
    manifest rules (``validate_manifest``) decide which decisions are allowed:
    ``confirmed_equivalent`` and ``not_equivalent`` need matching behavioural results
    from ``tools.task_equivalence``.
``validate``
    Checks a manifest without changing it.

Two views are summarized. The *primary* view joins tasks only on equal docstring-stripped
solution ASTs; it reproduces the audit's overlap counts. The *expanded* view adds the
identifier-canonicalized AST, prompt paraphrase and upstream-ID signals; its extra
matches are candidates for adjudication, not findings.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.clusters import (
    ADJUDICATION_STATUSES,
    FLAG_SIGNALS,
    MANIFEST_FORMAT,
    RULES_VERSION,
    UNION_SIGNALS,
    ClusterRules,
    Edge,
    Item,
    adjudicate,
    cluster_record,
    components,
    find_edges,
    top_level_names,
    validate_manifest,
)
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval.tasks import load_tasks

PRIMARY = ("solution_ast",)
FORBIDDEN = ("test", "holdout", "rl_only")
"""File-name stems this tool refuses to open: the sealed test, the holdout union and the
RL-only pool. The guard is by name (given and resolved), before any read."""
QUARANTINE_RULE = (
    "owner ruling 2026-11-13: the screening candidates whose docstring-stripped reference "
    "solution AST equals a dev task's are provisionally quarantined pending adjudication; "
    "screening.json is not modified"
)


def refuse_sealed(path: Path) -> Path:
    """Raise before reading a sealed test, holdout or RL-only file.

    Both the name given and the name the path resolves to are checked, so a link to a
    refused file is refused too.
    """
    for name_given in (path.name, path.resolve().name):
        stem = name_given.split(".")[0].lower()
        if any(stem == name or stem.startswith(name + "-") for name in FORBIDDEN):
            raise ValueError(f"{path.name}: sealed test, holdout and RL-only files are never read")
    return path


def _ids(path: Path) -> list[str]:
    with refuse_sealed(path).open(encoding="utf-8") as stream:
        return [json.loads(line)["id"] for line in stream if line.strip()]


def load_items(
    train: Path, dev: Path, sft: Path, rl_candidates: Path
) -> tuple[list[Item], dict[str, Any]]:
    """Pool train and dev tasks as cluster items, with ``sft_train``/``rl_candidate`` roles."""
    sft_ids = set(_ids(sft))
    candidate_ids = set(_ids(rl_candidates))
    items: list[Item] = []
    for path, split in ((train, "train"), (dev, "dev")):
        for task in load_tasks(refuse_sealed(path)):
            if task.split != split:
                raise ValueError(f"{path.name}: expected only {split} tasks")
            roles = tuple(
                role
                for role, members in (("sft_train", sft_ids), ("rl_candidate", candidate_ids))
                if task.id in members
            )
            if roles and split != "train":
                raise ValueError("SFT rows and RL candidates must be pool train tasks")
            items.append(
                Item(
                    task.id,
                    split,
                    task.source,
                    task.prompt,
                    task.solution,
                    task.mode,
                    task.entry_point,
                    roles,
                )
            )
    train_ids = {item.id for item in items if item.split == "train"}
    if not candidate_ids <= train_ids:
        raise ValueError("every RL candidate must be a pool train task")
    info = {
        "sft_rows": len(sft_ids),
        "sft_rows_in_pool_train": len(sft_ids & train_ids),
        "rl_candidates": len(candidate_ids),
    }
    return items, info


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _dev_partners(
    items: Sequence[Item], edges: Iterable[Edge], signals: Collection[str]
) -> dict[int, set[int]]:
    """Dev index to the train indices it shares a direct edge with, under ``signals``."""
    found: dict[int, set[int]] = {}
    for edge in edges:
        if not any(name in edge.signals for name in signals):
            continue
        for x, y in ((edge.a, edge.b), (edge.b, edge.a)):
            if items[x].split == "dev" and items[y].split == "train":
                found.setdefault(x, set()).add(y)
    return found


def view_counts(
    items: Sequence[Item],
    edges: Sequence[Edge],
    signals: Collection[str],
    eligible: Collection[str],
) -> dict[str, Any]:
    """Overlap counts for one signal view; IDs are left out so the summary stays short."""
    direct = _dev_partners(items, edges, signals)
    sft = {
        d for d, partners in direct.items() if any("sft_train" in items[p].roles for p in partners)
    }
    rlc = {
        d
        for d, partners in direct.items()
        if any("rl_candidate" in items[p].roles for p in partners)
    }
    candidates = {
        items[p].id
        for partners in direct.values()
        for p in partners
        if "rl_candidate" in items[p].roles
    }
    cross_edges = [
        edge
        for edge in edges
        if any(name in edge.signals for name in signals)
        and {items[edge.a].split, items[edge.b].split} == {"train", "dev"}
    ]
    sft_edges = [
        edge
        for edge in cross_edges
        if "sft_train" in items[edge.a].roles or "sft_train" in items[edge.b].roles
    ]
    groups = components(len(items), edges, signals)
    family_sft = family_rlc = family_train = 0
    dev_groups = dev_group_members = 0
    for members in groups:
        dev = [i for i in members if items[i].split == "dev"]
        train = [i for i in members if items[i].split == "train"]
        if len(dev) > 1:
            dev_groups += 1
            dev_group_members += len(dev)
        if dev and train:
            family_train += len(dev)
            family_sft += len(dev) * any("sft_train" in items[i].roles for i in train)
            family_rlc += len(dev) * any("rl_candidate" in items[i].roles for i in train)
    return {
        "signals": sorted(signals),
        "direct": {
            "dev_tasks_matching_sft": len(sft),
            "dev_tasks_matching_pool_train": len(direct),
            "dev_tasks_matching_rl_candidates": len(rlc),
            "dev_tasks_matching_rl_candidates_and_sft": len(rlc & sft),
            "dev_train_edges": len(cross_edges),
            "distinct_train_partners": len({p for v in direct.values() for p in v}),
            "dev_sft_edges": len(sft_edges),
            "distinct_sft_partners": len(
                {p for d in sft for p in direct[d] if "sft_train" in items[p].roles}
            ),
            "distinct_rl_candidates_matched": len(candidates),
            "eligible_rl_candidates_matched": len(candidates & set(eligible)),
        },
        "family": {
            "dev_tasks_in_training_exposed_families": family_sft,
            "dev_tasks_in_rl_candidate_families": family_rlc,
            "dev_tasks_in_pool_train_families": family_train,
            "dev_internal_groups": dev_groups,
            "dev_internal_group_members": dev_group_members,
        },
    }


def screening_quarantine(
    items: Sequence[Item],
    edges: Sequence[Edge],
    screening: Mapping[str, Any],
    cluster_of: Mapping[str, str],
) -> dict[str, Any]:
    """The provisional quarantine list and the eligible recount; screening is not modified."""
    eligible = set(screening["eligible"])
    reasons = {task["id"]: task["reason"] for task in screening["tasks"]}
    by_id = {item.id: index for index, item in enumerate(items)}

    def matched(signals: Collection[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for dev, partners in _dev_partners(items, edges, signals).items():
            for partner in partners:
                if "rl_candidate" in items[partner].roles:
                    found.setdefault(items[partner].id, []).append(items[dev].id)
        return {key: sorted(value) for key, value in sorted(found.items())}

    def describe(candidate: str, devs: Sequence[str]) -> dict[str, Any]:
        item = items[by_id[candidate]]
        return {
            "id": candidate,
            "split": item.split,
            "source": item.source,
            "top_level_names": list(top_level_names(item.solution)),
            "screening_reason": reasons.get(candidate)
            or ("kept" if candidate in eligible else None),
            "eligible": candidate in eligible,
            "matched_dev_tasks": list(devs),
            "cluster_id": cluster_of.get(candidate),
            "status": "provisional_quarantine_pending_adjudication",
        }

    primary = matched(PRIMARY)
    expanded = matched(UNION_SIGNALS)
    quarantined = [describe(key, value) for key, value in primary.items()]
    extra = [
        {**describe(key, value), "status": "listed_for_adjudication_not_quarantined"}
        for key, value in expanded.items()
        if key not in primary
    ]
    held = {entry["id"] for entry in quarantined if entry["eligible"]}
    extra_eligible = {entry["id"] for entry in extra if entry["eligible"]}
    floor = screening["rule"]["floor"]
    return {
        "rule": QUARANTINE_RULE,
        "signals": list(PRIMARY),
        "candidates": quarantined,
        "recount": {
            "eligible_before": len(eligible),
            "quarantined": len(quarantined),
            "quarantined_eligible": len(held),
            "eligible_after": len(eligible - held),
            "floor": floor,
            "floor_met": len(eligible - held) >= floor,
            "note": "before any further filtering; run 2 remains blocked",
        },
        "additional_expanded_candidates": {
            "signals": list(UNION_SIGNALS),
            "status": "listed for adjudication only; the owner's ruling covers the primary list",
            "candidates": extra,
            "sensitivity": {
                "eligible_if_also_quarantined": len(eligible - held - extra_eligible),
                "additional_eligible": len(extra_eligible),
            },
        },
    }


def build_manifest(
    items: Sequence[Item],
    edges: Sequence[Edge],
    screening: Mapping[str, Any],
    provenance: Mapping[str, Any],
    rules: ClusterRules,
) -> dict[str, Any]:
    """Clusters with a dev member (expanded view), flag edges, views and the quarantine."""
    groups = components(len(items), edges)
    cluster_of: dict[str, str] = {}
    records = []
    train_only = 0
    for members in groups:
        if len(members) < 2:
            continue
        splits = {items[i].split for i in members}
        if "dev" not in splits:
            train_only += 1
            continue
        roles = {role for i in members for role in items[i].roles}
        categories = []
        if splits == {"dev"}:
            categories.append("dev_internal")
        else:
            categories.append("dev_vs_pool_train")
            if "sft_train" in roles:
                categories.append("training_exposed_dev")
            if "rl_candidate" in roles:
                categories.append("dev_vs_rl_candidate")
        if len({items[i].mode for i in members}) > 1:
            categories.append("function_io_mixed")
        record = cluster_record(items, members, edges, categories=categories)
        for index in members:
            cluster_of[items[index].id] = record["cluster_id"]
        records.append(record)
    flags = [
        {
            "a": items[edge.a].id,
            "b": items[edge.b].id,
            "splits": [items[edge.a].split, items[edge.b].split],
            "signals": {k: v for k, v in edge.signals.items() if k in FLAG_SIGNALS},
            "evidence": edge.signals["evidence"],
            "review": "pending",
        }
        for edge in edges
        if not edge.union and {items[edge.a].split, items[edge.b].split} == {"train", "dev"}
    ]
    eligible = screening["eligible"]
    statuses: dict[str, int] = dict.fromkeys(ADJUDICATION_STATUSES, 0)
    for record in records:
        statuses[record["adjudication"]["status"]] += 1
    categories_count: dict[str, int] = {}
    for record in records:
        for name in record["categories"]:
            categories_count[name] = categories_count.get(name, 0) + 1
    manifest = {
        "format": MANIFEST_FORMAT,
        "kind": "candidate",
        "created_at": utc_now(),
        "provenance": dict(provenance),
        "rules": {**asdict(rules), "union_signals": list(UNION_SIGNALS)},
        "summary": {
            "items": {
                "train": sum(item.split == "train" for item in items),
                "dev": sum(item.split == "dev" for item in items),
            },
            "edges": len(edges),
            "union_edges": sum(edge.union for edge in edges),
            "clusters_with_dev": len(records),
            "train_only_clusters_not_listed": train_only,
            "categories": categories_count,
            "adjudication_statuses": statuses,
            "cross_split_flag_edges": len(flags),
            "behavior_pairs_pending": sum(len(record["behavior"]) for record in records),
            "views": {
                "primary": view_counts(items, edges, PRIMARY, eligible),
                "expanded": view_counts(items, edges, UNION_SIGNALS, eligible),
            },
        },
        "clusters": records,
        "flag_edges": flags,
        "screening_quarantine": screening_quarantine(items, edges, screening, cluster_of),
        "existing_weights": {
            "rule": (
                "owner ruling 2026-11-13: dev families confirmed as training-exposed (a "
                "confirmed_equivalent cluster containing a run 1b SFT row) are excluded from "
                "the repaired evaluation of existing weights; until adjudicated they are "
                "listed here as pending"
            ),
            "pending_clusters": [
                record["cluster_id"]
                for record in records
                if "training_exposed_dev" in record["categories"]
            ],
        },
    }
    validate_manifest(manifest)
    return manifest


def candidates(args: argparse.Namespace) -> int:
    output: Path = args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; a candidate manifest is never overwritten")
    rules = ClusterRules()
    paths = {
        "pool_train": args.train,
        "pool_dev": args.dev,
        "sft_train": args.sft,
        "rl_candidates": args.rl_candidates,
        "screening": args.screening,
    }
    for path in paths.values():
        refuse_sealed(path)
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    items, info = load_items(args.train, args.dev, args.sft, args.rl_candidates)
    screening = json.loads(args.screening.read_text(encoding="utf-8"))
    if screening.get("format") != "rl-screening-v1":
        raise SystemExit("unknown screening record format")
    edges = find_edges(items, rules)
    commit, dirty = git_state()
    module = Path(__file__).resolve().parents[1] / "src" / "scratchcoder" / "data" / "clusters.py"
    provenance = {
        "inputs": {
            name: {"path": _relative(path, args.root), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "input_counts": info,
        "tool": {
            "command": "python -m tools.task_clusters candidates",
            "git_commit": commit,
            "dirty": dirty,
            "rules_version": RULES_VERSION,
            "clusters_py_sha256": file_sha256(module),
            "tool_sha256": file_sha256(Path(__file__).resolve()),
        },
        "scope": (
            "static signals only; no code was executed; the sealed test, holdout and RL-only "
            "files were not opened; screening.json and every run artifact are read only"
        ),
    }
    manifest = build_manifest(items, edges, screening, provenance, rules)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "record.json",
        {
            "schema_version": 1,
            "kind": "dev-v2-cluster-candidates",
            "created_at": manifest["created_at"],
            "provenance": provenance,
            "manifest_sha256": file_sha256(output / "manifest.json"),
            "summary": manifest["summary"],
            "screening_recount": manifest["screening_quarantine"]["recount"],
            "paid_runs": 0,
        },
    )
    print(
        json.dumps(
            {
                "summary": manifest["summary"],
                "recount": manifest["screening_quarantine"]["recount"],
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


def record_decision(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    decided_at = utc_now()
    adjudicate(
        manifest,
        args.cluster,
        status=args.status,
        reviewer=args.reviewer,
        decided_at=decided_at,
        rationale=args.rationale,
        uncertainty=args.uncertainty,
        member=args.member,
    )
    manifest["kind"] = (
        "adjudicated"
        if all(item["adjudication"]["status"] != "pending" for item in manifest["clusters"])
        else "partially_adjudicated"
    )
    atomic_json(args.manifest, manifest)
    entry = {
        "cluster_id": args.cluster,
        "member": args.member,
        "status": args.status,
        "reviewer": args.reviewer,
        "decided_at": decided_at,
        "rationale": args.rationale,
        "uncertainty": args.uncertainty,
        "manifest_sha256": file_sha256(args.manifest),
    }
    with (args.manifest.parent / "adjudication-log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, sort_keys=True))
    return 0


def check_manifest(args: argparse.Namespace) -> int:
    validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    print("manifest valid")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("candidates", help="write the static candidate manifest")
    make.add_argument("--train", type=Path, required=True, help="pool train.jsonl")
    make.add_argument("--dev", type=Path, required=True, help="pool dev.jsonl")
    make.add_argument("--sft", type=Path, required=True, help="SFT train.jsonl (IDs only)")
    make.add_argument("--rl-candidates", type=Path, required=True, help="IDs only")
    make.add_argument("--screening", type=Path, required=True, help="screening.json")
    make.add_argument("--output", type=Path, required=True, help="a new directory")
    make.add_argument(
        "--root", type=Path, default=Path(os.getcwd()), help="paths recorded relative to this"
    )
    decide = commands.add_parser("adjudicate", help="record one reviewer decision")
    decide.add_argument("--manifest", type=Path, required=True)
    decide.add_argument("--cluster", required=True)
    decide.add_argument(
        "--status", required=True, choices=[s for s in ADJUDICATION_STATUSES if s != "pending"]
    )
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--rationale", required=True)
    decide.add_argument("--uncertainty")
    decide.add_argument(
        "--member", help="decide one member of an unclear_quarantined cluster (a task ID)"
    )
    check = commands.add_parser("validate", help="check a manifest")
    check.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"candidates": candidates, "adjudicate": record_decision, "validate": check_manifest}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
