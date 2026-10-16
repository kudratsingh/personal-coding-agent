"""Explicit local held-out domain loss/bits-per-byte reports, never training."""

import hashlib
import json
from pathlib import Path

from scratchcoder.common.records import git_state, utc_now
from scratchcoder.data.shards import TokenShard, file_sha256
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.harness import save_report
from scratchcoder.train.validation import shard_windows, validate


def evaluate_validation_checkpoint(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    sources_path: Path,
    output: Path,
    scorecard: Path,
    documents: int,
    sequence_length: int | None = None,
) -> None:
    if type(documents) is not int or documents < 1:
        raise ValueError("positive document sample limit required")
    model, tokenizer = load_checkpoint(checkpoint, tokenizer_path)
    length = model.config.context if sequence_length is None else sequence_length
    if type(length) is not int or not 1 <= length <= model.config.context:
        raise ValueError("validation sequence length must fit model context")
    sources = json.loads(sources_path.read_text())
    if (
        not isinstance(sources, dict)
        or not sources
        or any(
            not isinstance(name, str) or not name.strip() or not isinstance(path, str)
            for name, path in sources.items()
        )
    ):
        raise ValueError("nonempty domain-to-shard-path map required")
    shards = {
        name: TokenShard(
            sources_path.parent / path,
            tokenizer_hash=tokenizer.sha256,
            vocab_size=tokenizer.vocab_size,
            bos_id=tokenizer.bos_id,
        )
        for name, path in sources.items()
    }
    metrics = {}
    counts = {}
    for name, shard in shards.items():
        windows = shard_windows(shard, tokenizer, documents=documents, sequence_length=length)
        metrics[name] = validate(model, windows)
        counts[name] = {"seen_documents": len(windows), "available_documents": len(shard)}
    identity = {name: shard.identity for name, shard in shards.items()}
    commit, dirty = git_state()
    report = {
        "schema_version": 1,
        "status": "completed",
        "suite": "validation-bits-per-byte",
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
        "checkpoint_hash": file_sha256(checkpoint),
        "tokenizer_hash": tokenizer.sha256,
        "task_hash": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "configuration": {
            "document_limit_per_domain": documents,
            "sequence_length": length,
            "sampling": "first-document-prefixes",
            "special_token_bytes": 0,
        },
        "source_hashes": identity,
        "document_counts": counts,
        "metrics": metrics,
        "per_task": [],
    }
    save_report(report, output, scorecard=scorecard)
