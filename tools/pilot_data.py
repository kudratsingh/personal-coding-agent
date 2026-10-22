"""Build and verify a hash-bound, offline English stage-1 pilot shard pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scratchcoder.data.shards import TokenShard, file_sha256, write_shard
from scratchcoder.tokenizer.bpe import SPECIAL_TOKENS, ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder, encode_many

_REQUIRED_ROW_FIELDS = {"id", "domain", "split", "text"}
_ARTIFACTS = {
    "train_shard": ("train.bin", "train.bin.json"),
    "validation_shard": ("validation.bin", "validation.bin.json"),
    "tokenizer": ("tokenizer.json",),
}


def _sha256_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _reject_symlink(path: Path) -> None:
    """Reject symlinks in a path, including any existing parent component."""
    candidate = Path(os.path.abspath(path))
    parts = candidate.parents[::-1]
    current = Path(candidate.anchor)
    for part in (*parts, candidate):
        if part == current:
            continue
        current = part
        if current.is_symlink():
            raise ValueError(f"symlink path is not allowed: {path}")


def _load_tokenizer(
    path: Path,
    *,
    expected_identity: str,
    expected_file_sha256: str,
    expected_vocab_size: int,
    expected_bos_id: int,
) -> ByteBPE:
    _reject_symlink(path)
    if not path.is_file():
        raise ValueError("tokenizer path must be a regular file")
    _sha256_text(expected_identity, "expected tokenizer identity")
    _sha256_text(expected_file_sha256, "expected tokenizer file SHA-256")
    if file_sha256(path) != expected_file_sha256:
        raise ValueError("tokenizer file SHA-256 differs from expected identity")
    tokenizer = ByteBPE.load(path)
    if tokenizer.sha256 != expected_identity:
        raise ValueError("tokenizer identity differs from expected identity")
    if (
        type(expected_vocab_size) is not int
        or tokenizer.vocab_size != expected_vocab_size
        or type(expected_bos_id) is not int
        or tokenizer.bos_id != expected_bos_id
        or tokenizer.bos_id != 256
    ):
        raise ValueError("tokenizer vocabulary or special-token expectations differ")
    return tokenizer


def _source_rows(path: Path, expected_sha256: str) -> Iterator[dict[str, Any]]:
    """Verify every source byte, then stream parsed rows in source order."""
    _reject_symlink(path)
    _sha256_text(expected_sha256, "expected source SHA-256")
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("reviewed source file SHA-256 differs")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed source JSONL row {number}") from exc
                if (
                    not isinstance(row, dict)
                    or not _REQUIRED_ROW_FIELDS <= row.keys()
                    or not isinstance(row["id"], str)
                    or not row["id"]
                    or not isinstance(row["domain"], str)
                    or not isinstance(row["split"], str)
                    or not isinstance(row["text"], str)
                    or not row["text"]
                ):
                    raise ValueError(f"malformed source row schema at row {number}")
                yield row
    except UnicodeDecodeError as exc:
        raise ValueError("malformed source JSONL encoding") from exc


def _selection_hash(document_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for identifier in document_ids:
        encoded = identifier.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def build_pilot_data(
    source_path: Path,
    expected_source_sha256: str,
    tokenizer_path: Path,
    expected_tokenizer_identity: str,
    expected_tokenizer_file_sha256: str,
    output_dir: Path,
    *,
    train_token_cap: int = 66_000_000,
    validation_documents: int = 32,
    expected_vocab_size: int = 65_536,
    expected_bos_id: int = 256,
) -> dict[str, Any]:
    """Build shards from reviewed local rows; never fits or downloads a tokenizer/data."""
    if type(train_token_cap) is not int or train_token_cap < 1:
        raise ValueError("train_token_cap must be a positive integer")
    if type(validation_documents) is not int or validation_documents < 1:
        raise ValueError("validation_documents must be a positive integer")
    _reject_symlink(output_dir)
    if output_dir.exists():
        raise ValueError("pilot output directory must not already exist")

    tokenizer = _load_tokenizer(
        tokenizer_path,
        expected_identity=expected_tokenizer_identity,
        expected_file_sha256=expected_tokenizer_file_sha256,
        expected_vocab_size=expected_vocab_size,
        expected_bos_id=expected_bos_id,
    )

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    train_tokens = 0
    train_complete = False
    validation_complete = False
    seen: set[str] = set()
    selection_encoder = FastEncoder(tokenizer)
    for row in _source_rows(source_path, expected_source_sha256):
        if row["domain"] != "web":
            continue
        if row["split"] == "stage1" and not train_complete:
            if row["id"] in seen:
                raise ValueError("duplicate selected document ID")
            seen.add(row["id"])
            train_rows.append(row)
            train_tokens += len(selection_encoder.encode(row["text"])) + 1
            if train_tokens >= train_token_cap:
                train_complete = True
        elif row["split"] == "validation" and not validation_complete:
            if row["id"] in seen:
                raise ValueError("training and validation document IDs must be distinct")
            if any(item["id"] == row["id"] for item in validation_rows):
                raise ValueError("duplicate selected validation document ID")
            seen.add(row["id"])
            validation_rows.append(row)
            validation_complete = len(validation_rows) >= validation_documents
        if train_complete and validation_complete:
            break

    if not train_complete or not validation_complete:
        raise ValueError(
            "reviewed source ended before training cap and validation selection were met"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.bin"
    validation_path = output_dir / "validation.bin"
    tokenizer_copy = output_dir / "tokenizer.json"
    shutil.copyfile(tokenizer_path, tokenizer_copy)
    train_ids = [row["id"] for row in train_rows]
    validation_ids = [row["id"] for row in validation_rows]
    train_encodings = encode_many(tokenizer, (row["text"] for row in train_rows))
    validation_encodings = encode_many(tokenizer, (row["text"] for row in validation_rows))
    write_shard(
        train_path,
        train_encodings,
        vocab_size=tokenizer.vocab_size,
        tokenizer_hash=tokenizer.sha256,
        bos_id=tokenizer.bos_id,
    )
    write_shard(
        validation_path,
        validation_encodings,
        vocab_size=tokenizer.vocab_size,
        tokenizer_hash=tokenizer.sha256,
        bos_id=tokenizer.bos_id,
    )
    train_shard = TokenShard(
        train_path,
        tokenizer_hash=tokenizer.sha256,
        vocab_size=tokenizer.vocab_size,
        bos_id=tokenizer.bos_id,
    )
    validation_shard = TokenShard(
        validation_path,
        tokenizer_hash=tokenizer.sha256,
        vocab_size=tokenizer.vocab_size,
        bos_id=tokenizer.bos_id,
    )
    actual_train_tokens = sum(length for _, length in train_shard.offsets)
    validation_token_count = sum(length for _, length in validation_shard.offsets)
    if actual_train_tokens != train_tokens:
        raise ValueError("internal training token accounting mismatch")
    artifacts: dict[str, dict[str, Any]] = {}
    for key, names in _ARTIFACTS.items():
        artifacts[key] = {
            "files": {
                name: {
                    "path": name,
                    "sha256": file_sha256(output_dir / name),
                    "size_bytes": (output_dir / name).stat().st_size,
                }
                for name in names
            }
        }
    all_ids = train_ids + validation_ids
    binding: dict[str, Any] = {
        "schema_version": 1,
        "status": "built-not-launched",
        "authorization": "No launch or training authorization is asserted by this binding.",
        "selection": (
            "English web stage1 training and first validation documents in source order; "
            "whole documents; no text changes or token clipping."
        ),
        "source": {
            "path": Path(os.path.relpath(source_path.absolute(), output_dir.absolute())).as_posix(),
            "sha256": expected_source_sha256,
        },
        "tokenizer": {
            "path": "tokenizer.json",
            "identity": tokenizer.sha256,
            "file_sha256": file_sha256(tokenizer_copy),
            "vocab_size": tokenizer.vocab_size,
            "bos_id": tokenizer.bos_id,
            "special_tokens": list(SPECIAL_TOKENS),
        },
        "artifacts": artifacts,
        "train": {
            "documents": len(train_ids),
            "tokens_including_bos": actual_train_tokens,
            "token_cap": train_token_cap,
            "whole_document_overshoot": actual_train_tokens - train_token_cap,
            "document_ids": train_ids,
            "selection_sha256": _selection_hash(train_ids),
        },
        "validation": {
            "documents": len(validation_ids),
            "tokens_including_bos": validation_token_count,
            "document_ids": validation_ids,
            "selection_sha256": _selection_hash(validation_ids),
        },
        "selection_sha256": _selection_hash(all_ids),
    }
    binding_path = output_dir / "binding.json"
    binding_path.write_text(json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return binding


_AUDIT_METHOD = (
    "exact token-level containment; document boundaries ignored; "
    "no near-duplicate/n-gram statistics"
)
_DUPLICATE_EXAMPLE_LIMIT = 5


class LeakageError(ValueError):
    """Validation content was found verbatim in the training shard; carries the audit."""

    def __init__(self, message: str, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


def _document_digest(tokens: npt.NDArray[np.uint16]) -> str:
    return hashlib.sha256(np.ascontiguousarray(tokens, dtype="<u2").tobytes()).hexdigest()


def _find_window(
    stream: npt.NDArray[np.uint16], window: npt.NDArray[np.uint16]
) -> tuple[int | None, int]:
    """Return (first offset of ``window`` in ``stream`` or None, longest window prefix found)."""
    size = int(stream.size)
    count = int(window.size)
    if count == 0:
        return None, 0
    candidates = np.flatnonzero(stream == window[0])
    found = 1 if candidates.size else 0
    for i in range(1, count):
        if candidates.size == 0:
            break
        candidates = candidates[candidates + i < size]
        candidates = candidates[stream[candidates + i] == window[i]]
        if candidates.size:
            found = i + 1
    return (int(candidates[0]) if found == count else None), found


def audit_overlap(
    train: TokenShard,
    validation: TokenShard,
    *,
    sequence_length: int,
    validation_documents: int,
) -> dict[str, Any]:
    """Audit exact token-level leakage of validation content into the training stream.

    For each of the first ``validation_documents`` validation documents this reports whether
    its full token content repeats a training document (under any index) and whether its
    validation window, the first ``sequence_length + 1`` tokens exactly as
    ``scratchcoder.train.validation.shard_windows`` takes them, appears without its leading
    BOS as a contiguous run anywhere in the training token stream, document boundaries
    ignored. The longest found window prefix exposes near misses. Exact training-document
    repeats are counted as well. This is exact containment only: it does not compute
    near-duplicate, n-gram, or paraphrase statistics.
    """
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if type(validation_documents) is not int or validation_documents < 1:
        raise ValueError("validation_documents must be a positive integer")
    if train.vocab_size != validation.vocab_size or train.bos_id != validation.bos_id:
        raise ValueError("training and validation shards must share vocabulary and BOS")
    _reject_symlink(train.path)
    stream: npt.NDArray[np.uint16] = np.fromfile(train.path, dtype="<u2")
    if int(stream.size) != sum(length for _, length in train.offsets):
        raise ValueError("training shard size differs from its document offsets")
    first_by_digest: dict[str, int] = {}
    duplicate_examples: list[list[int]] = []
    duplicates = 0
    for index, (offset, length) in enumerate(train.offsets):
        first = first_by_digest.setdefault(
            _document_digest(stream[offset : offset + length]), index
        )
        if first != index:
            duplicates += 1
            if len(duplicate_examples) < _DUPLICATE_EXAMPLE_LIMIT:
                duplicate_examples.append([first, index])
    audited = min(validation_documents, len(validation))
    reports: list[dict[str, Any]] = []
    leaks = 0
    for index in range(audited):
        document: npt.NDArray[np.uint16] = np.asarray(validation[index], dtype="<u2")
        duplicate = first_by_digest.get(_document_digest(document))
        window = document[1 : sequence_length + 1]
        found_at, prefix = _find_window(stream, window)
        if duplicate is not None or found_at is not None:
            leaks += 1
        reports.append(
            {
                "index": index,
                "tokens": int(document.size),
                "duplicate_of_training_index": duplicate,
                "window_tokens": int(window.size),
                "window_found_in_training_at": found_at,
                "longest_window_prefix_found": prefix,
            }
        )
    return {
        "schema_version": 1,
        "sequence_length": sequence_length,
        "train_tokens": int(stream.size),
        "train_documents": len(train),
        "validation_documents_audited": audited,
        "duplicate_training_documents": duplicates,
        "duplicate_training_examples": duplicate_examples,
        "validation": reports,
        "leaks": leaks,
        "method": _AUDIT_METHOD,
    }


def verify_binding(
    binding_path: Path,
    *,
    leakage_audit: bool = True,
    sequence_length: int = 1024,
    validation_documents: int | None = None,
) -> dict[str, Any]:
    """Verify all bound bytes, strict relative paths, metadata, tokenizer and shard headers.

    With ``leakage_audit`` the verified shards are also audited with :func:`audit_overlap`;
    a leak raises :class:`LeakageError`, otherwise the audit is attached to the returned
    dictionary as ``leakage_audit``. ``binding.json`` on disk is never rewritten.
    """
    if validation_documents is not None and (
        type(validation_documents) is not int or validation_documents < 1
    ):
        raise ValueError("validation_documents must be a positive integer")
    _reject_symlink(binding_path)
    if not binding_path.is_file():
        raise ValueError("binding path must be a regular file")
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid binding JSON") from exc
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != 1
        or binding.get("status") != "built-not-launched"
        or binding.get("authorization")
        != "No launch or training authorization is asserted by this binding."
    ):
        raise ValueError("unsupported pilot binding")
    source = binding.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {"path", "sha256"}
        or not isinstance(source["path"], str)
        or not source["path"]
    ):
        raise ValueError("invalid source binding")
    _sha256_text(source["sha256"], "source SHA-256")
    root = binding_path.parent
    _reject_symlink(root)
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_ARTIFACTS):
        raise ValueError("binding is missing required artifacts")
    verified: dict[str, Path] = {}
    for key, names in _ARTIFACTS.items():
        entry = artifacts[key]
        files = entry.get("files") if isinstance(entry, dict) else None
        if not isinstance(files, dict) or set(files) != set(names):
            raise ValueError("binding artifact list differs from required files")
        for name in names:
            record = files[name]
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256", "size_bytes"}
                or record["path"] != name
            ):
                raise ValueError("unsafe or malformed artifact path")
            path = root / name
            _reject_symlink(path)
            if (
                not path.is_file()
                or type(record["size_bytes"]) is not int
                or path.stat().st_size != record["size_bytes"]
            ):
                raise ValueError("artifact is missing or has changed size")
            digest = _sha256_text(record["sha256"], "artifact SHA-256")
            if file_sha256(path) != digest:
                raise ValueError("artifact SHA-256 differs from binding")
            verified[f"{key}:{name}"] = path
    tokenizer_meta = binding.get("tokenizer")
    if not isinstance(tokenizer_meta, dict) or tokenizer_meta.get("path") != "tokenizer.json":
        raise ValueError("invalid tokenizer binding")
    tokenizer = ByteBPE.load(verified["tokenizer:tokenizer.json"])
    if (
        tokenizer.sha256 != tokenizer_meta.get("identity")
        or tokenizer.vocab_size != tokenizer_meta.get("vocab_size")
        or tokenizer.bos_id != tokenizer_meta.get("bos_id")
        or list(SPECIAL_TOKENS) != tokenizer_meta.get("special_tokens")
        or file_sha256(verified["tokenizer:tokenizer.json"]) != tokenizer_meta.get("file_sha256")
    ):
        raise ValueError("tokenizer identity or special-token metadata differs")
    shards: dict[str, TokenShard] = {}
    for name, section in (("train_shard", "train"), ("validation_shard", "validation")):
        shard_path = verified[f"{name}:{'train.bin' if section == 'train' else 'validation.bin'}"]
        shard = shards[section] = TokenShard(
            shard_path,
            tokenizer_hash=tokenizer.sha256,
            vocab_size=tokenizer.vocab_size,
            bos_id=tokenizer.bos_id,
        )
        declared = binding.get(section)
        if (
            not isinstance(declared, dict)
            or type(declared.get("documents")) is not int
            or declared["documents"] < 1
            or len(shard) != declared["documents"]
            or type(declared.get("tokens_including_bos")) is not int
            or declared["tokens_including_bos"] < declared["documents"]
        ):
            raise ValueError("shard document count differs from binding")
        actual_tokens = sum(length for _, length in shard.offsets)
        if actual_tokens != declared["tokens_including_bos"]:
            raise ValueError(f"{section} token count differs from binding")
        if section == "train" and (
            type(declared.get("token_cap")) is not int
            or declared["token_cap"] < 1
            or actual_tokens < declared["token_cap"]
            or type(declared.get("whole_document_overshoot")) is not int
            or declared["whole_document_overshoot"] != actual_tokens - declared["token_cap"]
        ):
            raise ValueError("training token count differs from binding")
    train_ids = binding["train"].get("document_ids")
    validation_ids = binding["validation"].get("document_ids")
    if (
        not isinstance(train_ids, list)
        or not isinstance(validation_ids, list)
        or any(not isinstance(item, str) for item in train_ids + validation_ids)
        or len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids) & set(validation_ids)
        or binding["train"].get("selection_sha256") != _selection_hash(train_ids)
        or binding["validation"].get("selection_sha256") != _selection_hash(validation_ids)
        or binding.get("selection_sha256") != _selection_hash(train_ids + validation_ids)
    ):
        raise ValueError("document selection metadata is invalid")
    if leakage_audit:
        audit = audit_overlap(
            shards["train"],
            shards["validation"],
            sequence_length=sequence_length,
            validation_documents=(
                validation_documents
                if validation_documents is not None
                else binding["validation"]["documents"]
            ),
        )
        if audit["leaks"] > 0:
            leaking = [
                item["index"]
                for item in audit["validation"]
                if item["duplicate_of_training_index"] is not None
                or item["window_found_in_training_at"] is not None
            ]
            raise LeakageError(
                f"validation content leaks into training: {audit['leaks']} of "
                f"{audit['validation_documents_audited']} audited validation documents "
                f"(validation indices {leaking[:10]})",
                audit,
            )
        binding["leakage_audit"] = audit
    return binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-identity", required=True)
    parser.add_argument("--tokenizer-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-token-cap", type=int, default=66_000_000)
    parser.add_argument("--validation-documents", type=int, default=32)
    args = parser.parse_args()
    build_pilot_data(
        args.source,
        args.source_sha256,
        args.tokenizer,
        args.tokenizer_identity,
        args.tokenizer_file_sha256,
        args.output,
        train_token_cap=args.train_token_cap,
        validation_documents=args.validation_documents,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
