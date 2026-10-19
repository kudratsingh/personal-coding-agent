"""One real-data BPE training call, immutable identity, then held-out measurements."""

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.shards import file_sha256
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder, encode_many


def documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def prefix(texts: Iterable[str], target: int) -> list[str]:
    selected = []
    remaining = target
    for text in texts:
        if remaining <= 0:
            break
        part = text.encode()[:remaining].decode("utf-8", errors="ignore")
        selected.append(part)
        remaining -= len(part.encode())
        if part != text:
            break
    return selected


def balanced_selection(
    web: Iterable[str], python: Iterable[str], target: int
) -> tuple[list[str], list[str], int]:
    left, right = prefix(web, target), prefix(python, target)
    while True:
        a, b = sum(len(text.encode()) for text in left), sum(len(text.encode()) for text in right)
        if a == b:
            if not a:
                raise ValueError("nonempty balanced training data required")
            return left, right, a
        target = min(a, b)
        left, right = prefix(left, target), prefix(right, target)


def corpus_hash(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def measure(path: Path, tokenizer: ByteBPE, workers: int) -> dict[str, Any]:
    splits: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    encoded = encode_many(tokenizer, (row["text"] for row in documents(path)), workers=workers)
    for row, ids in zip(documents(path), encoded, strict=True):
        counts = splits.setdefault(row["split"], dict(documents=0, text_bytes=0, tokens=0))
        counts["documents"] += 1
        counts["text_bytes"] += len(row["text"].encode())
        counts["tokens"] += len(ids)
    held_out = splits["validation"]
    total = {
        name: sum(counts[name] for counts in splits.values())
        for name in ("documents", "text_bytes", "tokens")
    }
    real_probe = next(row["text"][:256] for row in documents(path) if row["split"] == "validation")
    assert tokenizer.encode(real_probe) == FastEncoder(tokenizer).encode(real_probe)
    assert tokenizer.decode(tokenizer.encode(real_probe)) == real_probe
    return dict(
        **total,
        tokens_with_document_bos=total["tokens"] + total["documents"],
        splits=splits,
        held_out_bytes_per_token=held_out["text_bytes"] / held_out["tokens"],
        elapsed_seconds=time.perf_counter() - started,
        workers=workers,
        real_probe_fast_reference_equal=True,
        real_probe_characters=len(real_probe),
        protocol="all retained documents; no BOS in tokens; validation never used for BPE training",
    )


def freeze(
    sample_root: Path,
    output: Path,
    *,
    vocab_size: int = 32768,
    domain_byte_cap: int = 100_000_000,
    workers: int = 4,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("freeze output already exists; retraining requires owner approval")
    inputs = json.loads((sample_root / "report.json").read_text())
    by_domain = {source["domain"]: source for source in inputs["sources"]}
    for domain in ("web", "python"):
        if (
            file_sha256(sample_root / domain / "documents.jsonl")
            != by_domain[domain]["document_file_sha256"]
        ):
            raise ValueError("reviewed sample identity differs")
    target = min(
        domain_byte_cap,
        by_domain["web"]["splits"]["stage1"]["text_bytes"],
        by_domain["python"]["splits"]["train"]["text_bytes"],
    )
    web, code, size = balanced_selection(
        (
            row["text"]
            for row in documents(sample_root / "web/documents.jsonl")
            if row["split"] == "stage1"
        ),
        (
            row["text"]
            for row in documents(sample_root / "python/documents.jsonl")
            if row["split"] == "train"
        ),
        target,
    )
    commit, dirty = git_state()
    record: dict[str, Any] = dict(
        created_at=utc_now(),
        git_commit=commit,
        dirty=dirty,
        status="training",
        requested_vocab_size=vocab_size,
        domain_byte_cap=domain_byte_cap,
        training_bytes={"web": size, "python": size},
        training_total_bytes=size * 2,
        training_documents={"web": len(web), "python": len(code)},
        training_corpus_hashes={"web": corpus_hash(web), "python": corpus_hash(code)},
        source_document_hashes={
            domain: source["document_file_sha256"] for domain, source in by_domain.items()
        },
        selection="stage1 web / train Python; ordered equal valid-UTF8 prefixes; no validation",
        hardware=dict(
            system=platform.system(), machine=platform.machine(), processor=platform.processor()
        ),
        paid_runs=0,
        retraining_requires_owner_approval=True,
    )
    output.mkdir(parents=True)
    atomic_json(output / "record.json", record)
    started = time.perf_counter()
    tokenizer = ByteBPE.train(chain(web, code), vocab_size)
    tokenizer.save(output / "tokenizer.json")
    record.update(
        status="frozen-measurements-pending",
        training_seconds=time.perf_counter() - started,
        actual_vocab_size=tokenizer.vocab_size,
        merge_count=len(tokenizer.merges),
        tokenizer_hash=tokenizer.sha256,
        tokenizer_file_sha256=file_sha256(output / "tokenizer.json"),
        merge_table_sha256=hashlib.sha256(
            json.dumps(tokenizer.merges, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    atomic_json(output / "record.json", record)
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("status", "training_seconds", "tokenizer_hash", "merge_table_sha256")
            }
        ),
        flush=True,
    )
    del web, code
    measurements = {}
    for domain in ("web", "python"):
        measurements[domain] = measure(sample_root / domain / "documents.jsonl", tokenizer, workers)
        atomic_json(output / "measurements.json", measurements)
        print(json.dumps({"domain": domain, **measurements[domain]}), flush=True)
    assert ByteBPE.load(output / "tokenizer.json").sha256 == tokenizer.sha256
    record.update(status="completed", completed_at=utc_now(), measurements=measurements)
    atomic_json(output / "record.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.samples, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
