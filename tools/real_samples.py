"""Offline sample review: pinned Parquet text, explicit code licenses, existing filters."""

import argparse
import gzip
import importlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
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
from scratchcoder.data.shards import file_sha256
from tools.header_licenses import (
    REPOSITORY_ALLOWLIST,
    HeaderLicense,
    classify_file,
    classify_header,
    repository_license_ids,
)

HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"


def declared_license(text: str) -> str | None:
    """Compatibility helper; classification never infers rights from dataset tags."""
    return classify_header(text).license_id


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


RepositoryLicenses = Mapping[tuple[str, str], tuple[str, ...]]
"""Verified (max_stars_repo_name, max_stars_repo_path) -> upstream license IDs."""

REPOSITORY_EVIDENCE = "upstream repository license metadata"


def process_sample(
    rows: Iterable[dict[str, Any]],
    *,
    domain: str,
    source: str,
    target: int,
    contamination: ContaminationIndex,
    output: Path,
    repository_licenses: RepositoryLicenses | None = None,
) -> dict[str, Any]:
    """Review one ordered prefix.

    With ``repository_licenses``, a Python file's repository metadata comes only
    from that verified mapping, keyed by its exact repository and path; a file
    absent from it has no repository metadata. Without it, an explicit
    ``repository_license`` row field is used, as before.
    """
    output.mkdir(parents=True, exist_ok=False)
    documents = []
    provenance: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    rule_bytes: Counter[str] = Counter()
    rule_files: Counter[str] = Counter()
    license_classes: dict[str, Counter[str]] = {}
    rejected_repository_licenses: Counter[str] = Counter()
    rejected_repository_license_ids: Counter[str] = Counter()
    repository_id_outcomes: dict[str, Counter[str]] = {}
    repository_lookups: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    def audit(document: Document, reason: str) -> None:
        rule_bytes[reason] += len(document.text.encode())
        rule_files[reason] += 1

    for number, row in enumerate(sample_rows(rows, domain, target)):
        text = row["text" if domain == "web" else "content"]
        counts["sample_documents"] += 1
        counts["sample_text_bytes"] += len(text.encode())
        identifier = str(row.get("id", number))
        listed: Any = None
        if domain != "web":
            if repository_licenses is None:
                listed = row.get("repository_license")
            else:
                key = (str(row.get("max_stars_repo_name")), str(row.get("max_stars_repo_path")))
                listed = repository_licenses.get(key)
                repository_lookups["found" if listed is not None else "absent"] += 1
        classification = (
            HeaderLicense("permissive", "ODC-By-1.0", "owner-reviewed web source terms")
            if domain == "web"
            else classify_file(text, listed)
        )
        size = len(text.encode())
        via_repository = classification.category == "repository_disallowed" or (
            classification.evidence == REPOSITORY_EVIDENCE
        )
        if via_repository:
            outcome = "admitted" if classification.category == "permissive" else "rejected"
            for item in repository_license_ids(listed):
                tally = repository_id_outcomes.setdefault(item, Counter())
                tally[outcome + "_files"] += 1
                tally[outcome + "_text_bytes"] += size
                if outcome == "rejected" and item not in REPOSITORY_ALLOWLIST:
                    rejected_repository_license_ids[item] += 1
        license_name = classification.license_id
        license_counts = license_classes.setdefault(classification.category, Counter())
        license_counts["files"] += 1
        license_counts["text_bytes"] += len(text.encode())
        if classification.category != "permissive":
            if classification.category == "repository_disallowed":
                rejected_repository_licenses[classification.license_id or "unknown"] += 1
            counts["missing_or_unapproved_file_license"] += 1
            counts["missing_or_unapproved_file_license_bytes"] += len(text.encode())
            reason = "license_" + classification.category
            rule_bytes[reason] += len(text.encode())
            rule_files[reason] += 1
            if classification.category == "unrecognized" and len(examples) < 10:
                examples.append(
                    dict(
                        id=identifier,
                        repository=row.get("max_stars_repo_name"),
                        path=row.get("max_stars_repo_path"),
                        sha256=content_hash(text),
                        evidence=classification.evidence,
                        first_five_lines=text.splitlines()[:5],
                    )
                )
            continue
        documents.append(Document(identifier, source, text, domain))
        provenance[identifier] = dict(
            id=identifier,
            license=license_name,
            license_class=classification.category,
            license_evidence=classification.evidence,
            license_basis="repository_metadata" if via_repository else "inline_or_source_terms",
            repository_licenses=list(repository_license_ids(listed)) if via_repository else None,
            sha256=content_hash(text),
            repository=row.get("max_stars_repo_name"),
            path=row.get("max_stars_repo_path"),
            url=row.get("url"),
        )
    kept, filters = filter_documents(documents, contamination, on_decision=audit)
    counts["license_admitted_documents"] = (
        counts["sample_documents"] - counts["missing_or_unapproved_file_license"]
    )
    splits: dict[str, Counter[str]] = {}
    licenses: Counter[str] = Counter()
    kept_license_ids: Counter[str] = Counter()
    kept_basis: dict[str, Counter[str]] = {}
    with (output / "documents.jsonl").open("x", encoding="utf-8") as stream:
        for document in kept:
            split = document_split(content_hash(document.text), web=domain == "web")
            split_counts = splits.setdefault(split, Counter())
            size = len(document.text.encode())
            split_counts["documents"] += 1
            split_counts["text_bytes"] += size
            record = provenance[document.id]
            licenses[record["license"] or "unresolved"] += 1
            for item in (record["license"] or "unresolved").split(" AND "):
                kept_license_ids[item] += 1
            basis = kept_basis.setdefault(record["license_basis"], Counter())
            basis["files"] += 1
            basis["text_bytes"] += size
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
        primary_decisions={
            name: dict(files=rule_files[name], text_bytes=rule_bytes[name])
            for name in (
                "license_copyleft",
                "license_restricted",
                "license_unrecognized",
                "license_missing_repository_license",
                "license_repository_disallowed",
                *filter_counts,
            )
            if name != "seen"
        },
        license_classes={name: dict(value) for name, value in license_classes.items()},
        rejected_repository_license_counts=dict(rejected_repository_licenses),
        rejected_repository_license_id_counts=dict(
            sorted(rejected_repository_license_ids.items(), key=lambda x: (-x[1], x[0]))
        ),
        repository_license_id_outcomes={
            name: dict(value)
            for name, value in sorted(
                repository_id_outcomes.items(),
                key=lambda x: (-(x[1]["admitted_files"] + x[1]["rejected_files"]), x[0]),
            )
        },
        repository_metadata_lookups=dict(repository_lookups),
        final_training_admission=False,
        license_policy=(
            "header-classifier-v3; explicit header first; absent header uses verified "
            "repository license metadata, admitted only when every listed ID is on the D-05 "
            "allowlist; missing metadata is separate"
        ),
        contamination_ngram_words=contamination.n,
        contamination_short_fields="complete normalized fields shorter than n are also indexed",
        filter_configuration=asdict(FilterConfig()),
        kept_documents=len(kept),
        kept_text_bytes=sum(len(doc.text.encode()) for doc in kept),
        file_license_counts=dict(licenses),
        kept_license_id_counts=dict(sorted(kept_license_ids.items(), key=lambda x: (-x[1], x[0]))),
        kept_license_basis={name: dict(value) for name, value in kept_basis.items()},
        splits={name: dict(value) for name, value in splits.items()},
        document_file_sha256=file_sha256(output / "documents.jsonl"),
        # ByteBPE's pre-merge alphabet represents each UTF-8 byte as one token, no BOS.
        tokens=sum(len(doc.text.encode()) for doc in kept),
        token_count_protocol="pre-merge byte alphabet, no BOS; frozen-BPE recount in step 4",
    )
    atomic_json(
        output / "unrecognized-examples.json", dict(examples=examples, raw_private_material=True)
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
    parser.add_argument(
        "--repository-licenses",
        type=Path,
        help="report.json of a completed tools.stack_licenses recovery for this Python sample",
    )
    args = parser.parse_args()
    receipts = json.loads(args.receipts.read_text())
    if receipts["status"] != "completed" or args.output.exists():
        parser.error("completed verified receipts and a new output directory required")
    for entry in receipts["files"]:
        path = Path(entry["path"])
        if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("raw source size/hash differs before parsing")
    mapping: RepositoryLicenses | None = None
    mapping_identity: dict[str, Any] | None = None
    if args.repository_licenses is not None:
        from tools.stack_licenses import load_mapping

        mapping, mapping_identity = load_mapping(args.repository_licenses)
        python_entry = next(item for item in receipts["files"] if item["domain"] == "python")
        if mapping_identity["sample_parquet_sha256"] != python_entry["sha256"]:
            raise ValueError("license mapping was built for a different Python sample")
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
            repository_licenses=mapping if domain == "python" else None,
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
            repository_license_mapping=mapping_identity,
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
