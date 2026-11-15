"""Pilot intake: one row range of a pinned pool source file, adapted and deduplicated.

Static only: nothing here runs task, reference or generated code, and nothing opens a sealed
split (the pool's test, holdout union or RL-only file). It prepares training candidates for
reference verification. Protection against the sealed test split is a separate, approved
step (``tools.holdout_integrity_scan``) that runs on this command's output.

Rows ``[--start-row, --start-row + --rows)`` of the pool spec's OpenCodeInstruct file are
read in file order. The file's size and SHA-256 are checked against the pool's completed
download receipt first. Each row then goes through these steps, in order, and the first
that fails is its drop reason:

1. ``unsupported_schema``: the reviewed adapter (``adapt_row``) cannot adapt it.
2. ``over_context`` / ``oversized_for_sandbox``: the build's intake screen (the public
   prompt plus reference must fit the spec's context, and code plus tests the sandbox's
   input limit).
3. ``over_rl_prompt_tokens``: the public prompt is longer than the spec's RL prompt limit.
4. ``choice_contamination``: the build's ARC-Easy rule. Any field shares an n-gram with an
   ARC-Easy test question or its options.
5. ``benchmark_contamination``: the build's benchmark rule (``build_pool``). This covers a
   HumanEval, HumanEval+ or MBPP+ ID, a benchmark prompt near-duplicate (5-word shingle
   Jaccard at least 0.85), or a benchmark n-gram in any field.
6. ``id_in_pool`` / ``duplicate_id``: the ID is already a pool train or dev task, or
   repeats an earlier row of the range.
7. ``dev_contamination``: any field shares an n-gram with a dev task's text (the build's
   RL-only rule, applied to dev only).
8. ``near_duplicate``: the prompt near-duplicates a pool train or dev task or an earlier
   kept candidate (5-word shingle Jaccard at least 0.85).
9. ``dev_paraphrase``: the build's holdout-paraphrase rule against every dev prompt
   (3-word shingle Jaccard at least 0.5, or at least 0.25 when the two define a function
   of the same name).
10. ``dev_family``: the frozen task-family rules (``task-clusters-v1``) put it in one family
    with a dev task. Families are computed over the dev split, the pool's train split and
    every adapted row of the range, so a row dropped for another reason still links its
    family-mates (the pool build likewise assigned families before dropping paraphrases).

Rows the pool build already read (before the source's ``rows`` in its first file) are
refused: they include the pool's holdouts. Where a rule is stricter than the build's train
intake it says so: the RL prompt limit and dev n-grams are the build's RL-only rules (here
over all four fields, the solution included), and near-duplicates keep the first row in file
order.

Outputs, written under ``--out`` only when none of them exists:

- ``candidates.jsonl``: the kept tasks, train split, in file order;
- ``rows.jsonl``: every row's index, ID (when adapted) and outcome (``kept`` or the reason);
- ``remainder.jsonl``: with ``--remainder``, those pool train tasks, for the integrity scan;
- ``record.json``: the row range, every input's SHA-256, the rules, counts by reason, the
  tool commit and the software versions.

Exit codes: 0 done; 2 an input was refused (sealed, wrong hash, too few rows); 3 an output
already exists.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import scratchcoder
from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.adapters import adapt_row, canonical_id
from scratchcoder.data.arc import arc_splits
from scratchcoder.data.clusters import RULES_VERSION, Item, UnionFind, find_edges
from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.filters import ContaminationIndex
from scratchcoder.data.manifest import ManifestEntry
from scratchcoder.data.pool import SimilarityIndex
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.tasks import Task, load_tasks, save_tasks
from scratchcoder.sandbox.runner import Limits
from scratchcoder.tokenizer.bulk import FastEncoder
from tools.pilot_data import _load_tokenizer
from tools.real_samples import HUMANEVAL_SHA256
from tools.task_pool import (
    COLUMNS,
    DATASETS,
    HoldoutParaphrases,
    _parquet_rows,
    _provenance,
    _screen,
    _source,
    _verified,
    load_spec,
)
from tools.verifier_scan import SealedPathError, guard

KIND = "opencode"
REASONS = (
    "unsupported_schema",
    "over_context",
    "oversized_for_sandbox",
    "over_rl_prompt_tokens",
    "choice_contamination",
    "benchmark_contamination",
    "id_in_pool",
    "duplicate_id",
    "dev_contamination",
    "near_duplicate",
    "dev_paraphrase",
    "dev_family",
)
OUTPUTS = ("candidates.jsonl", "rows.jsonl", "remainder.jsonl", "record.json")
ARC_SOURCE = "allenai/ai2_arc"


class Refused(Exception):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def humaneval_tasks(path: Path) -> list[Task]:
    """HumanEval as the pool build adapts it, from the pinned gzip (hash-checked)."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HUMANEVAL_SHA256:
        raise Refused("original HumanEval bytes differ")
    entry = ManifestEntry(
        source="openai/human-eval",
        revision="6d43fb980f9fee3c892a914eda09951f772ad10d",
        url="https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz",
        license="MIT",
        attribution="OpenAI HumanEval",
        sha256=HUMANEVAL_SHA256,
        size_bytes=len(raw),
        purpose="eval",
        license_verified=True,
    )
    return [
        adapt_row("openai/human-eval", json.loads(line), entry, split="test")
        for line in gzip.decompress(raw).splitlines()
    ]


def arc_questions(path: Path, record: Path) -> list[ChoiceTask]:
    """The ARC-Easy test questions from the pinned Pod export, checked against its record."""
    meta = json.loads(record.read_text(encoding="utf-8"))
    if meta.get("kind") != "pod-task-export" or meta.get("export") != "arc":
        raise Refused("not an ARC export record")
    # Before reading: the export must be the very file its record names, at that size.
    named = Path(meta["output"]["path"])
    if not named.exists() or not path.samefile(named) or path.is_symlink():
        raise Refused("the ARC export is not the file its record names")
    if path.stat().st_size != meta["output"]["bytes"]:
        raise Refused("the ARC export's size differs from its record")
    if _sha(path) != meta["output"]["sha256"]:
        raise Refused("the ARC export's SHA-256 differs from its record")
    pinned = asdict(arc_splits()["test"].manifest)
    if meta["source"]["manifests"]["test"] != pinned or meta["questions"] != (
        arc_splits()["test"].task_count
    ):
        raise Refused("the ARC export record does not name the reviewed ARC-Easy test split")
    revision = pinned["revision"]
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["labels"], row["choices"] = tuple(row["labels"]), tuple(row["choices"])
        question = ChoiceTask(**row)
        if question.source != ARC_SOURCE or question.source_revision != revision:
            raise Refused("an ARC export question has other provenance")
        questions.append(question)
    if len(questions) != meta["questions"]:
        raise Refused("the ARC export's question count differs from its record")
    return questions


def fields(task: Task) -> tuple[str, ...]:
    return (task.prompt, task.solution or "", task.public_tests, task.hidden_tests)


def _item(task: Task, split: str) -> Item:
    return Item(
        task.id, split, task.source, task.prompt, task.solution, task.mode, task.entry_point
    )


def dev_family_ids(
    candidates: Sequence[Task],
    dev: Sequence[Task],
    train: Sequence[Task],
    range_bridges: Sequence[Task] = (),
) -> set[str]:
    """Candidates in one task family with any dev task.

    Pool train and every other adapted row of the range (``range_bridges``: rows dropped for
    another reason) join families as bridges, as in the pool build, which assigned families
    before it dropped paraphrases. A bridge row whose ID repeats a pool or range ID is
    scanned under an internal ID and joined to the item that holds that ID."""
    items = [
        *(_item(task, "dev") for task in dev),
        *(_item(task, "train") for task in train),
        *(_item(task, "pilot") for task in candidates),
    ]
    position = {item.id: index for index, item in enumerate(items)}
    if len(position) != len(items):
        raise ValueError("dev, pool train and candidate IDs must be unique")
    joins: list[tuple[int, int]] = []
    for number, task in enumerate(range_bridges):
        if task.id in position:
            joins.append((position[task.id], len(items)))
            items.append(_item(replace(task, id=f"\x00range-{number}"), "bridge"))
        else:
            position[task.id] = len(items)
            items.append(_item(task, "bridge"))
    joined = UnionFind(len(items))
    for edge in find_edges(items):
        if edge.union:
            joined.union(edge.a, edge.b)
    for a, b in joins:
        joined.union(a, b)
    dev_roots = {joined.find(index) for index, item in enumerate(items) if item.split == "dev"}
    return {
        item.id
        for index, item in enumerate(items)
        if item.split == "pilot" and joined.find(index) in dev_roots
    }


def intake(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    source = _source(spec, KIND)
    receipts = json.loads(args.receipts.read_text(encoding="utf-8"))
    if receipts.get("status") != "completed" or receipts.get("spec_sha256") != spec["sha256"]:
        raise Refused("completed download receipts for this exact spec are required")
    receipt = next(
        (
            item
            for item in receipts["files"]
            if item["kind"] == KIND and item["filename"] == args.file
        ),
        None,
    )
    if receipt is None or args.file not in source["files"]:
        raise Refused(f"{args.file} is not a receipted {KIND} file of the spec")
    if (
        source["rows"] is not None
        and args.file == source["files"][0]
        and args.start_row < source["rows"]
    ):
        raise Refused(f"rows before {source['rows']} of {args.file} were read by the pool build")
    path = Path(receipt["path"])
    guard([path])
    if not _verified(path, receipt["size_bytes"], receipt["sha256"]):
        raise Refused(f"{args.file}: bytes differ from the receipt")
    tokenizer = _load_tokenizer(
        Path(spec["tokenizer"]["path"]),
        expected_identity=spec["tokenizer"]["identity"],
        expected_file_sha256=spec["tokenizer"]["file_sha256"],
        expected_vocab_size=spec["tokenizer"]["vocab_size"],
        expected_bos_id=256,
    )
    encoder = FastEncoder(tokenizer)
    limits = Limits(memory_mb=spec["verification"]["memory_mb"])
    min_alpha = spec["protection"]["min_alpha_tokens"]
    prompt_limit = spec["splits"]["rl_prompt_tokens"]
    benchmarks = humaneval_tasks(args.humaneval)
    for name, given in (("humaneval-plus", args.humaneval_plus), ("mbpp-plus", args.mbpp_plus)):
        benchmarks.extend(read_release(given, release(name)))
    choices = arc_questions(args.arc_export, args.arc_record)
    train = list(load_tasks(args.pool_train))
    dev = list(load_tasks(args.pool_dev))
    if any(task.split != "train" for task in train) or any(task.split != "dev" for task in dev):
        raise Refused("--pool-train and --pool-dev must hold only their own split")

    choice_index = ContaminationIndex(
        (text for question in choices for text in (question.question, question.public_prompt())),
        min_alpha_tokens=min_alpha,
    )
    benchmark_index = ContaminationIndex(
        (part for task in benchmarks for part in fields(task)), min_alpha_tokens=min_alpha
    )
    benchmark_prompts = SimilarityIndex()
    for task in benchmarks:
        benchmark_prompts.add(task.prompt)
    benchmark_ids = {canonical_id(task.id) for task in benchmarks}
    dev_index = ContaminationIndex(
        (part for task in dev for part in fields(task)), min_alpha_tokens=min_alpha
    )
    duplicates = SimilarityIndex()
    for task in [*train, *dev]:
        duplicates.add(task.prompt)
    paraphrases = HoldoutParaphrases(dev)
    pool_ids = {canonical_id(task.id) for task in [*train, *dev]}

    provenance = _provenance(source, receipt)
    counts: Counter[str] = Counter()
    outcomes: list[dict[str, Any]] = []
    survivors: list[tuple[int, Task]] = []
    bridges: list[Task] = []
    seen: set[str] = set()
    taken = 0

    def drop(row: int, identifier: str | None, reason: str) -> None:
        counts[reason] += 1
        outcomes.append({"row": row, "id": identifier, "reason": reason})

    for offset, row in enumerate(_parquet_rows(path, COLUMNS[KIND], start=args.start_row)):
        if offset >= args.rows:
            break
        taken += 1
        index = args.start_row + offset
        try:
            task = adapt_row(DATASETS[KIND], row, provenance)
        except (ValueError, SyntaxError, KeyError, TypeError, json.JSONDecodeError):
            drop(index, None, "unsupported_schema")
            continue
        screen = (
            _screen(task, task.solution, encoder, spec["context_tokens"], limits)
            if (task.solution is not None)
            else "unsupported_schema"
        )
        if screen is not None:
            drop(index, task.id, screen)
        elif len(encoder.encode(task.public_prompt())) > prompt_limit:
            drop(index, task.id, "over_rl_prompt_tokens")
        elif any(choice_index.contains(text) for text in fields(task)):
            drop(index, task.id, "choice_contamination")
        elif (
            canonical_id(task.id) in benchmark_ids
            or benchmark_prompts.contains(task.prompt)
            or any(benchmark_index.contains(text) for text in fields(task))
        ):
            drop(index, task.id, "benchmark_contamination")
        elif canonical_id(task.id) in pool_ids:
            drop(index, task.id, "id_in_pool")
        elif task.id in seen:
            drop(index, task.id, "duplicate_id")
        elif any(dev_index.contains(text) for text in fields(task)):
            drop(index, task.id, "dev_contamination")
        elif duplicates.contains(task.prompt):
            drop(index, task.id, "near_duplicate")
        elif paraphrases.matches(task):
            drop(index, task.id, "dev_paraphrase")
        else:
            duplicates.add(task.prompt)
            survivors.append((index, task))
        if not survivors or survivors[-1][1] is not task:
            bridges.append(task)
        seen.add(task.id)
    if taken != args.rows:
        raise Refused(f"{args.file} holds {taken} rows from row {args.start_row}, not {args.rows}")
    family = dev_family_ids([task for _, task in survivors], dev, train, bridges)
    kept = []
    for index, task in survivors:
        if task.id in family:
            drop(index, task.id, "dev_family")
        else:
            kept.append(replace(task, split="train"))
            outcomes.append({"row": index, "id": task.id, "reason": "kept"})
    counts["kept"] = len(kept)
    counts["rows"] = taken
    remainder: list[Task] = []
    if args.remainder is not None:
        wanted = json.loads(args.remainder.read_text(encoding="utf-8"))["admitted"]
        by_id = {task.id: task for task in train}
        missing = [identifier for identifier in wanted if identifier not in by_id]
        if missing or len(set(wanted)) != len(wanted):
            raise Refused("every remainder ID must be a distinct pool train task")
        remainder = [by_id[identifier] for identifier in sorted(wanted)]
    save_tasks(args.out / "candidates.jsonl", kept)
    with (args.out / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for entry in sorted(outcomes, key=lambda item: item["row"]):
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
    if args.remainder is not None:
        save_tasks(args.out / "remainder.jsonl", remainder)
    commit, dirty = git_state()
    inputs = {
        name: {"path": str(given), "sha256": _sha(given)}
        for name, given in (
            ("spec", args.spec),
            ("receipts", args.receipts),
            ("source_file", path),
            ("pool_train", args.pool_train),
            ("pool_dev", args.pool_dev),
            ("humaneval", args.humaneval),
            ("humaneval_plus", args.humaneval_plus),
            ("mbpp_plus", args.mbpp_plus),
            ("arc_export", args.arc_export),
            ("arc_record", args.arc_record),
            *((("remainder_ids", args.remainder),) if args.remainder is not None else ()),
        )
    }
    record = {
        "kind": "pilot-intake",
        "schema_version": 1,
        "created_at": utc_now(),
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "file": args.file,
            "license": source["license"],
            "terms_reviewed_on": source["terms_reviewed_on"],
            "attribution": source["attribution"],
            "obligations": source["obligations"],
            "start_row": args.start_row,
            "rows": args.rows,
        },
        "rules": {
            "order": list(REASONS),
            "family_rules": RULES_VERSION,
            "context_tokens": spec["context_tokens"],
            "rl_prompt_tokens": prompt_limit,
            "min_alpha_tokens": min_alpha,
            "tokenizer_identity": spec["tokenizer"]["identity"],
        },
        "counts": {reason: counts.get(reason, 0) for reason in (*REASONS, "kept", "rows")},
        "remainder_tasks": len(remainder) if args.remainder is not None else None,
        "inputs": inputs,
        "outputs": {
            name: _sha(args.out / name) for name in OUTPUTS[:-1] if (args.out / name).exists()
        },
        "tool": {
            "command": "python -m tools.pilot_intake",
            "git_commit": commit,
            "dirty": dirty,
            "python": platform.python_version(),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "uv_lock_sha256": _sha(Path("uv.lock")) if Path("uv.lock").exists() else None,
            "package": str(Path(scratchcoder.__file__).parent),
        },
        "executed_code": False,
        "sealed_files_opened": False,
    }
    atomic_json(args.out / "record.json", record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("spec", "receipts", "pool-train", "pool-dev", "humaneval"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("humaneval-plus", "mbpp-plus", "arc-export", "arc-record", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--file", required=True, help="the spec's file name, as receipted")
    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--remainder", type=Path, help="admitted RL remainder record (IDs)")
    args = parser.parse_args(argv)
    if args.start_row < 0 or args.rows < 1:
        print("refused: --start-row must be >= 0 and --rows >= 1", file=sys.stderr)
        return 2
    try:
        # Every task or record input is refused if sealed. The benchmark files are pinned
        # by hash instead: the ARC export's name says "test" (ARC-Easy's test split, a
        # public benchmark), which the name guard would refuse.
        guard([args.spec, args.receipts, args.pool_train, args.pool_dev, args.arc_record])
        guard([args.humaneval, args.humaneval_plus, args.mbpp_plus])
        if args.remainder is not None:
            guard([args.remainder])
        if any((args.out / name).exists() for name in OUTPUTS):
            print("refused: an output already exists", file=sys.stderr)
            return 3
        args.out.mkdir(parents=True, exist_ok=True)
        record = intake(args)
    except (Refused, SealedPathError, ValueError, KeyError, TypeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(record["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
