"""Synthetic development material; never represents approved external data."""

from pathlib import Path
from typing import Any

from scratchcoder.common.records import atomic_json
from scratchcoder.data.filters import (
    ContaminationIndex,
    Document,
    content_hash,
    document_split,
    filter_documents,
)
from scratchcoder.data.shards import write_shard
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder


def prepare_fixture(
    directory: Path, *, vocab_size: int
) -> tuple[ByteBPE, dict[str, Path], dict[str, Any]]:
    documents: list[Document] = []
    for i in range(1000):
        web = f"The river near village {i} flows past green hills and a quiet forest.\n"
        code = f"def add_{i}(left, right):\n    return left + right + {i}\n"
        documents.extend(
            (
                Document(content_hash(web), "synthetic-web", web, "web"),
                Document(content_hash(code), "synthetic-code", code, "python"),
            )
        )
    kept, report = filter_documents(documents, ContaminationIndex([]))
    splits: dict[str, list[Document]] = {}
    for document in kept:
        key = document.domain + "-" + document_split(document.id, web=document.domain == "web")
        splits.setdefault(key, []).append(document)
    corpus = [
        document.text
        for key, items in splits.items()
        if not key.endswith("validation")
        for document in items
    ]
    tokenizer = ByteBPE.train(corpus, vocab_size)
    tokenizer.save(directory / "tokenizer.json")
    encoder = FastEncoder(tokenizer)
    paths: dict[str, Path] = {}
    for name, items in splits.items():
        paths[name] = directory / f"{name}.bin"
        write_shard(
            paths[name],
            (encoder.encode(document.text) for document in items),
            vocab_size=tokenizer.vocab_size,
            tokenizer_hash=tokenizer.sha256,
        )
    metadata = {
        "synthetic": True,
        "external_data": False,
        "filter_counts": report,
        "tokenizer_hash": tokenizer.sha256,
        "split_document_ids": {
            name: [document.id for document in items] for name, items in splits.items()
        },
    }
    atomic_json(directory / "manifest.json", metadata)
    atomic_json(directory / "stage1-sources.json", {"web": paths["web-stage1"].name})
    atomic_json(
        directory / "stage2-sources.json",
        {"web": paths["web-stage2"].name, "python": paths["python-train"].name},
    )
    return tokenizer, paths, metadata
