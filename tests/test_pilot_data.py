"""Authored-fixture checks for offline pilot shard binding."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.shards import file_sha256
from scratchcoder.tokenizer.bpe import ByteBPE
from tools.pilot_data import build_pilot_data, verify_binding


def _fixture(tmp_path: Path) -> tuple[Path, Path, ByteBPE, str, str]:
    source = tmp_path / "reviewed.jsonl"
    rows = [
        {"id": "train-a", "domain": "web", "split": "stage1", "text": "alpha beta"},
        {"id": "held-a", "domain": "web", "split": "validation", "text": "held out one"},
        {"id": "train-b", "domain": "web", "split": "stage1", "text": "gamma delta"},
        {"id": "held-b", "domain": "web", "split": "validation", "text": "held out two"},
        {"id": "python-a", "domain": "python", "split": "train", "text": "print(1)"},
        {"id": "web-other", "domain": "web", "split": "stage2", "text": "exclude me"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tokenizer_path = tmp_path / "frozen.json"
    tokenizer = ByteBPE()
    tokenizer.save(tokenizer_path)
    return source, tokenizer_path, tokenizer, file_sha256(source), file_sha256(tokenizer_path)


def _build(tmp_path: Path, *, cap: int = 8) -> tuple[Path, dict[str, Any]]:
    source, tokenizer_path, tokenizer, source_hash, tokenizer_file_hash = _fixture(tmp_path)
    output = tmp_path / "pilot"
    binding = build_pilot_data(
        source,
        source_hash,
        tokenizer_path,
        tokenizer.sha256,
        tokenizer_file_hash,
        output,
        train_token_cap=cap,
        validation_documents=2,
        expected_vocab_size=tokenizer.vocab_size,
    )
    return output, binding


def test_build_binds_whole_documents_and_keeps_validation_separate(tmp_path: Path) -> None:
    output, binding = _build(tmp_path)
    assert binding["status"] == "built-not-launched"
    assert binding["train"]["document_ids"] == ["train-a"]
    assert binding["train"]["tokens_including_bos"] == 11
    assert binding["train"]["whole_document_overshoot"] == 3
    assert binding["validation"]["document_ids"] == ["held-a", "held-b"]
    assert set(binding["train"]["document_ids"]).isdisjoint(binding["validation"]["document_ids"])
    assert verify_binding(output / "binding.json") == binding


def test_old_source_hash_rejects_changed_source_before_output(tmp_path: Path) -> None:
    source, tokenizer_path, tokenizer, _, tokenizer_file_hash = _fixture(tmp_path)
    output = tmp_path / "pilot"
    old_frozen_source_hash = "8096430180bb736f9a977c973bb30756290674de3096efeb95351d980312509b"
    with pytest.raises(ValueError, match="source file SHA-256"):
        build_pilot_data(
            source,
            old_frozen_source_hash,
            tokenizer_path,
            tokenizer.sha256,
            tokenizer_file_hash,
            output,
            train_token_cap=1,
            validation_documents=1,
            expected_vocab_size=tokenizer.vocab_size,
        )
    assert not output.exists()


def test_malformed_selected_row_fails_before_output_creation(tmp_path: Path) -> None:
    source, tokenizer_path, tokenizer, _, tokenizer_file_hash = _fixture(tmp_path)
    source.write_text('{"id":"broken","domain":"web"}\n', encoding="utf-8")
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match="schema"):
        build_pilot_data(
            source,
            file_sha256(source),
            tokenizer_path,
            tokenizer.sha256,
            tokenizer_file_hash,
            output,
            train_token_cap=1,
            validation_documents=1,
            expected_vocab_size=tokenizer.vocab_size,
        )
    assert not output.exists()


def test_stops_parsing_after_cap_and_validation_selection(tmp_path: Path) -> None:
    _, tokenizer_path, tokenizer, _, tokenizer_file_hash = _fixture(tmp_path)
    source = tmp_path / "bounded.jsonl"
    rows = [
        {"id": "train", "domain": "web", "split": "stage1", "text": "training"},
        {"id": "held-a", "domain": "web", "split": "validation", "text": "validation a"},
        {"id": "held-b", "domain": "web", "split": "validation", "text": "validation b"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + "not-json\n",
        encoding="utf-8",
    )
    binding = build_pilot_data(
        source,
        file_sha256(source),
        tokenizer_path,
        tokenizer.sha256,
        tokenizer_file_hash,
        tmp_path / "bounded-pilot",
        train_token_cap=1,
        validation_documents=2,
        expected_vocab_size=tokenizer.vocab_size,
    )
    assert binding["train"]["document_ids"] == ["train"]
    assert binding["validation"]["document_ids"] == ["held-a", "held-b"]


@pytest.mark.parametrize("mutation", ["shard", "sidecar", "tokenizer", "binding"])
def test_verifier_rejects_mutated_required_artifacts(tmp_path: Path, mutation: str) -> None:
    output, _ = _build(tmp_path)
    target = {
        "shard": output / "train.bin",
        "sidecar": output / "train.bin.json",
        "tokenizer": output / "tokenizer.json",
        "binding": output / "binding.json",
    }[mutation]
    if mutation == "binding":
        binding = json.loads(target.read_text())
        binding["train"]["documents"] += 1
        target.write_text(json.dumps(binding), encoding="utf-8")
    else:
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_binding(output / "binding.json")


def test_verifier_rejects_path_traversal_and_symlink(tmp_path: Path) -> None:
    output, _ = _build(tmp_path)
    binding_path = output / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding["artifacts"]["train_shard"]["files"]["train.bin"]["path"] = "../train.bin"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        verify_binding(binding_path)

    binding["artifacts"]["train_shard"]["files"]["train.bin"]["path"] = "train.bin"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    (output / "train.bin").unlink()
    (output / "train.bin").symlink_to(tmp_path / "outside.bin")
    with pytest.raises(ValueError, match="symlink"):
        verify_binding(binding_path)


def test_build_rejects_tokenizer_identity_mismatch_before_output(tmp_path: Path) -> None:
    source, tokenizer_path, tokenizer, source_hash, tokenizer_file_hash = _fixture(tmp_path)
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match="tokenizer identity"):
        build_pilot_data(
            source,
            source_hash,
            tokenizer_path,
            hashlib.sha256(b"wrong").hexdigest(),
            tokenizer_file_hash,
            output,
            train_token_cap=1,
            validation_documents=1,
            expected_vocab_size=tokenizer.vocab_size,
        )
    assert not output.exists()
