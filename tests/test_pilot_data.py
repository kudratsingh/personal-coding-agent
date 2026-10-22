"""Authored-fixture checks for offline pilot shard binding."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scratchcoder.data.shards import TokenShard, file_sha256, write_shard
from scratchcoder.tokenizer.bpe import ByteBPE
from tools.pilot_data import LeakageError, audit_overlap, build_pilot_data, verify_binding
from tools.pilot_data_audit import main as audit_main


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
    verified = verify_binding(output / "binding.json")
    assert verified.pop("leakage_audit")["leaks"] == 0
    assert verified == binding


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


def _build_rows(
    tmp_path: Path, rows: list[dict[str, Any]], *, cap: int, validation_documents: int
) -> Path:
    _, tokenizer_path, tokenizer, _, tokenizer_file_hash = _fixture(tmp_path)
    source = tmp_path / "custom.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "custom-pilot"
    build_pilot_data(
        source,
        file_sha256(source),
        tokenizer_path,
        tokenizer.sha256,
        tokenizer_file_hash,
        output,
        train_token_cap=cap,
        validation_documents=validation_documents,
        expected_vocab_size=tokenizer.vocab_size,
    )
    return output / "binding.json"


def _leaking_binding(tmp_path: Path) -> Path:
    """Validation text repeats a training row's text under a different document ID."""
    rows = [
        {"id": "train-a", "domain": "web", "split": "stage1", "text": "alpha beta"},
        {"id": "held-copy", "domain": "web", "split": "validation", "text": "alpha beta"},
    ]
    return _build_rows(tmp_path, rows, cap=1, validation_documents=1)


def _shards(binding_path: Path) -> tuple[TokenShard, TokenShard]:
    tokenizer = ByteBPE.load(binding_path.parent / "tokenizer.json")

    def load(name: str) -> TokenShard:
        return TokenShard(
            binding_path.parent / name,
            tokenizer_hash=tokenizer.sha256,
            vocab_size=tokenizer.vocab_size,
            bos_id=tokenizer.bos_id,
        )

    return load("train.bin"), load("validation.bin")


def test_verify_attaches_clean_leakage_audit_without_rewriting_binding(tmp_path: Path) -> None:
    output, binding = _build(tmp_path)
    binding_path = output / "binding.json"
    before = file_sha256(binding_path)
    verified = verify_binding(binding_path)
    audit = verified["leakage_audit"]
    assert audit["schema_version"] == 1
    assert audit["leaks"] == 0
    assert audit["train_tokens"] == binding["train"]["tokens_including_bos"] == 11
    assert audit["train_documents"] == 1
    assert audit["validation_documents_audited"] == 2
    assert audit["duplicate_training_documents"] == 0
    assert audit["duplicate_training_examples"] == []
    assert "exact token-level containment" in audit["method"]
    for index, item in enumerate(audit["validation"]):
        assert item["index"] == index
        assert item["duplicate_of_training_index"] is None
        assert item["window_found_in_training_at"] is None
        assert item["window_tokens"] == item["tokens"] - 1 == len("held out one")
        assert item["longest_window_prefix_found"] < item["window_tokens"]
    assert file_sha256(binding_path) == before
    assert "leakage_audit" not in json.loads(binding_path.read_text(encoding="utf-8"))
    assert (
        verify_binding(binding_path, validation_documents=1)["leakage_audit"][
            "validation_documents_audited"
        ]
        == 1
    )
    assert "leakage_audit" not in verify_binding(binding_path, leakage_audit=False)


def test_verify_rejects_validation_text_repeated_under_another_id(tmp_path: Path) -> None:
    binding_path = _leaking_binding(tmp_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["train"]["document_ids"] == ["train-a"]
    assert binding["validation"]["document_ids"] == ["held-copy"]
    with pytest.raises(ValueError, match="leak") as info:
        verify_binding(binding_path)
    assert isinstance(info.value, LeakageError)
    audit = info.value.audit
    assert audit["leaks"] == 1
    assert audit["validation"][0]["duplicate_of_training_index"] == 0
    assert audit["validation"][0]["window_found_in_training_at"] == 1
    assert verify_binding(binding_path, leakage_audit=False)["validation"]["documents"] == 1


def test_verify_finds_validation_window_inside_a_training_document(tmp_path: Path) -> None:
    rows = [
        {"id": "train-a", "domain": "web", "split": "stage1", "text": "xx held out one yy"},
        {"id": "held-a", "domain": "web", "split": "validation", "text": "held out one"},
    ]
    binding_path = _build_rows(tmp_path, rows, cap=1, validation_documents=1)
    train, validation = _shards(binding_path)
    audit = audit_overlap(train, validation, sequence_length=1024, validation_documents=1)
    item = audit["validation"][0]
    assert item["duplicate_of_training_index"] is None
    assert item["window_tokens"] == len("held out one")
    assert item["window_found_in_training_at"] == 1 + len("xx ")
    assert item["longest_window_prefix_found"] == item["window_tokens"]
    assert audit["leaks"] == 1
    with pytest.raises(ValueError, match="leak"):
        verify_binding(binding_path)
    assert verify_binding(binding_path, leakage_audit=False)["train"]["document_ids"] == ["train-a"]


def test_audit_overlap_on_hand_built_shards(tmp_path: Path) -> None:
    train_documents = [[1, 2, 3, 4, 5], [7, 8, 9], [1, 2, 3, 4, 5], [10, 11, 12, 13]]
    validation_documents = [
        [7, 8, 9],  # exact repeat of training document 1
        [1, 2, 99],  # prefix 1,2 exists; the full window does not
        [50, 51],  # nothing in common
        [10, 11, 12, 13, 14, 15],  # only the window fits inside training document 3
        [7, 8, 9],  # beyond the audited count
    ]
    for name, documents in (
        ("train.bin", train_documents),
        ("validation.bin", validation_documents),
    ):
        write_shard(tmp_path / name, documents, vocab_size=300, tokenizer_hash="t", bos_id=256)
    train = TokenShard(tmp_path / "train.bin", tokenizer_hash="t", vocab_size=300, bos_id=256)
    validation = TokenShard(
        tmp_path / "validation.bin", tokenizer_hash="t", vocab_size=300, bos_id=256
    )
    audit = audit_overlap(train, validation, sequence_length=3, validation_documents=4)
    assert audit["train_tokens"] == 21
    assert audit["train_documents"] == 4
    assert audit["validation_documents_audited"] == 4
    assert audit["duplicate_training_documents"] == 1
    assert audit["duplicate_training_examples"] == [[0, 2]]
    assert audit["leaks"] == 2
    by_index = {item["index"]: item for item in audit["validation"]}
    assert by_index[0]["duplicate_of_training_index"] == 1
    assert by_index[0]["window_found_in_training_at"] == 7
    assert by_index[0]["window_tokens"] == by_index[0]["longest_window_prefix_found"] == 3
    assert by_index[1]["duplicate_of_training_index"] is None
    assert by_index[1]["window_found_in_training_at"] is None
    assert by_index[1]["longest_window_prefix_found"] == 2
    assert by_index[1]["tokens"] == 4
    assert by_index[1]["window_tokens"] == 3  # clipped to sequence_length
    assert by_index[2]["window_found_in_training_at"] is None
    assert by_index[2]["longest_window_prefix_found"] == 0
    assert by_index[3]["duplicate_of_training_index"] is None
    assert by_index[3]["window_tokens"] == 3
    assert by_index[3]["window_found_in_training_at"] == 17
    assert by_index[3]["tokens"] == 7
    with pytest.raises(ValueError, match="sequence_length"):
        audit_overlap(train, validation, sequence_length=0, validation_documents=1)
    with pytest.raises(ValueError, match="validation_documents"):
        audit_overlap(train, validation, sequence_length=3, validation_documents=0)


def test_audit_cli_writes_report_and_signals_leaks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean").mkdir()
    (tmp_path / "leaking").mkdir()
    output, _ = _build(tmp_path / "clean")
    report = tmp_path / "clean-report.json"
    assert audit_main(["--binding", str(output / "binding.json"), "--output", str(report)]) == 0
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["leaks"] == 0
    assert written["validation_documents_audited"] == 2
    assert written["binding_sha256"] == file_sha256(output / "binding.json")
    assert "no leak" in capsys.readouterr().out

    leaking = _leaking_binding(tmp_path / "leaking")
    report = tmp_path / "leaking-report.json"
    assert audit_main(["--binding", str(leaking), "--output", str(report)]) == 1
    assert json.loads(report.read_text(encoding="utf-8"))["leaks"] == 1
    assert "leaks into training" in capsys.readouterr().err

    missing = tmp_path / "missing-report.json"
    assert audit_main(["--binding", str(tmp_path / "absent.json"), "--output", str(missing)]) == 1
    assert not missing.exists()
    assert "verification failed" in capsys.readouterr().err
