"""Offline sample review: pinned Parquet text, explicit code licenses, existing filters."""

import argparse
import gzip
import importlib
import json
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.arc import read_arc_easy
from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.filters import (
    ContaminationIndex,
    Document,
    FilterConfig,
    content_hash,
    document_split,
    filter_documents,
)
from scratchcoder.data.manifest import TRAINING_LICENSES
from scratchcoder.data.shards import file_sha256

HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"


def declared_license(text: str) -> str | None:
    """Recognize an original-file declaration, never infer rights from dataset tags."""
    header = text[:16384]
    identifiers = re.findall(r"SPDX-License-Identifier:\s*([^\r\n]+)", header)
    if identifiers:
        # Compound, conflicting, or unfamiliar expressions need separate review.
        names = {identifier.strip().removesuffix("*/").strip() for identifier in identifiers}
        return next(iter(names)) if len(names) == 1 and names <= TRAINING_LICENSES else None
    if re.search(r"GNU (?:GENERAL|LESSER|AFFERO)|CC-BY-(?:SA|NC)|non-commercial", header, re.I):
        return None
    if "Permission is hereby granted, free of charge" in header and all(
        marker in header for marker in ("The above copyright notice", "THE SOFTWARE IS PROVIDED")
    ):
        return "MIT"
    if re.search(r"Licensed under the Apache License, Version 2\.0", header, re.I):
        return "Apache-2.0"
    return None


def sample_rows(
    rows: Iterable[dict[str, Any]], domain: str, target: int
) -> Iterator[dict[str, Any]]:
    """Ordered whole-document prefix reaching the UTF-8 target, never clipping a file."""
    total = 0
    text_key = "text" if domain == "web" else "content"
    for row in rows:
        text = row[text_key]
        if not isinstance(text, str):
            raise ValueError("sample text must be a string")
        yield row
        total += len(text.encode("utf-8"))
        if total >= target:
            return
    if total < target:
        raise ValueError(f"container ended at {total} bytes before target {target}")


def parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet = importlib.import_module("pyarrow.parquet")
    with parquet.ParquetFile(path, arrow_extensions_enabled=False) as source:
        for batch in source.iter_batches(batch_size=256, use_threads=False):
            yield from batch.to_pylist()


def contamination_inputs(
    evalplus_directory: Path, arc_directory: Path, humaneval_path: Path
) -> tuple[list[str], dict[str, Any]]:
    texts = []
    identities: dict[str, Any] = {}
    expected = HUMANEVAL_SHA256
    if file_sha256(humaneval_path) != expected:
        raise ValueError("original HumanEval exclusion bytes differ")
    original = [
        json.loads(line) for line in gzip.decompress(humaneval_path.read_bytes()).splitlines()
    ]
    for row in original:
        texts.extend([row["prompt"], row["canonical_solution"], row["test"]])
    identities["humaneval"] = {"tasks": len(original), "release_sha256": expected}
    for name in ("humaneval-plus", "mbpp-plus"):
        spec = release(name)
        tasks = read_release(evalplus_directory / (name + ".jsonl.gz"), spec)
        for task in tasks:
            texts.extend([task.prompt, task.hidden_tests, task.solution or ""])
        identities[name] = {"tasks": len(tasks), "release_sha256": spec.manifest.sha256}
    choices, metadata = read_arc_easy(arc_directory)
    texts.extend(task.question for task in choices)
    identities["arc-easy"] = metadata
    identities["final_pool_holdout"] = {
        "tasks": 0,
        "available": False,
        "limitation": "final verified pool is not built; repeat exclusion before corpus training",
    }
    identities["policy"] = "normalized 13-word spans; shorter complete fields also indexed"
    return texts, identities


def process_sample(
    rows: Iterable[dict[str, Any]],
    *,
    domain: str,
    source: str,
    target: int,
    contamination: ContaminationIndex,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    documents = []
    provenance: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for number, row in enumerate(sample_rows(rows, domain, target)):
        text = row["text" if domain == "web" else "content"]
        counts["sample_documents"] += 1
        counts["sample_text_bytes"] += len(text.encode())
        identifier = str(row.get("id", number))
        license_name = "ODC-By-1.0" if domain == "web" else declared_license(text)
        if license_name is None:
            counts["missing_or_unapproved_file_license"] += 1
            continue
        documents.append(Document(identifier, source, text, domain))
        provenance[identifier] = dict(
            id=identifier,
            license=license_name,
            sha256=content_hash(text),
            repository=row.get("max_stars_repo_name"),
            path=row.get("max_stars_repo_path"),
            url=row.get("url"),
        )
    kept, filters = filter_documents(documents, contamination)
    counts["license_admitted_documents"] = len(documents)
    splits: dict[str, Counter[str]] = {}
    licenses: Counter[str] = Counter()
    with (output / "documents.jsonl").open("x", encoding="utf-8") as stream:
        for document in kept:
            split = document_split(content_hash(document.text), web=domain == "web")
            split_counts = splits.setdefault(split, Counter())
            size = len(document.text.encode())
            split_counts["documents"] += 1
            split_counts["text_bytes"] += size
            licenses[provenance[document.id]["license"]] += 1
            stream.write(
                json.dumps(
                    dict(asdict(document), split=split, provenance=provenance[document.id]),
                    ensure_ascii=False,
                )
                + "\n"
            )
    filter_counts = filters.get(source, {})
    filter_counts = {
        name: filter_counts.get(name, 0)
        for name in (
            "seen",
            "kept",
            "unknown_domain",
            "too_short",
            "too_large",
            "long_line",
            "low_alphanumeric",
            "generated",
            "duplicate",
            "contamination",
        )
    }
    report = dict(
        domain=domain,
        source=source,
        target_text_bytes=target,
        **counts,
        filters=filter_counts,
        filter_configuration=asdict(FilterConfig()),
        kept_documents=len(kept),
        kept_text_bytes=sum(len(doc.text.encode()) for doc in kept),
        file_license_counts=dict(licenses),
        splits={name: dict(value) for name, value in splits.items()},
        document_file_sha256=file_sha256(output / "documents.jsonl"),
        # ByteBPE's pre-merge alphabet represents each UTF-8 byte as one token, no BOS.
        tokens=sum(len(doc.text.encode()) for doc in kept),
        token_count_protocol="pre-merge byte alphabet, no BOS; frozen-BPE recount in step 4",
    )
    atomic_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--evalplus-directory", type=Path, required=True)
    parser.add_argument("--arc-directory", type=Path, required=True)
    parser.add_argument("--humaneval-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = json.loads(args.receipts.read_text())
    if receipts["status"] != "completed" or args.output.exists():
        parser.error("completed verified receipts and a new output directory required")
    for entry in receipts["files"]:
        path = Path(entry["path"])
        if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("raw source size/hash differs before parsing")
    started = time.perf_counter()
    texts, coverage = contamination_inputs(
        args.evalplus_directory, args.arc_directory, args.humaneval_path
    )
    index = ContaminationIndex(texts)
    reports = []
    for entry in receipts["files"]:
        domain = entry["domain"]
        report = process_sample(
            parquet_rows(Path(entry["path"])),
            domain=domain,
            source=entry["source"],
            target=500_000_000 if domain == "web" else 1_000_000_000,
            contamination=index,
            output=args.output / domain,
        )
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    commit, dirty = git_state()
    atomic_json(
        args.output / "report.json",
        dict(
            created_at=utc_now(),
            git_commit=commit,
            dirty=dirty,
            sources=reports,
            download_receipts=receipts,
            contamination_coverage=coverage,
            elapsed_seconds=time.perf_counter() - started,
            paid_runs=0,
            final_training_admission=False,
            limitations=[
                "Ordered prefixes, not representative random source samples",
                "Missing final verified pool holdout; no corpus training authorized",
                "File license declarations checked, not a legal rights warranty",
            ],
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
