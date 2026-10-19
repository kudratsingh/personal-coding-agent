import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from scratchcoder.data.shards import file_sha256
from scratchcoder.tokenizer.bpe import ByteBPE
from tools.freeze_tokenizer import balanced_selection, corpus_hash, freeze


def test_balancing_preserves_utf8_and_document_boundaries() -> None:
    web, code, size = balanced_selection(["éé"], ["abc"], 3)
    assert web == ["é"] and code == ["ab"] and size == 2
    assert sum(len(text.encode()) for text in web) == sum(len(text.encode()) for text in code)
    assert corpus_hash(["ab", "c"]) != corpus_hash(["a", "bc"])


def test_freeze_single_training_call_excludes_validation_and_records_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "samples"
    sources = []
    for domain, split in (("web", "stage1"), ("python", "train")):
        path = root / domain / "documents.jsonl"
        path.parent.mkdir(parents=True)
        training = "authored training characters and punctuation a+b+c éé\n" * 3
        rows = [
            dict(text=training, split=split),
            dict(text="VALIDATION_ONLY characters and Unicode λ\n", split="validation"),
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        sources.append(
            dict(
                domain=domain,
                document_file_sha256=file_sha256(path),
                splits={split: dict(text_bytes=len(training.encode()))},
            )
        )
    (root / "report.json").write_text(json.dumps(dict(sources=sources)))
    original_train = ByteBPE.train
    calls = []

    def train(cls: type[ByteBPE], corpus: Iterable[str], vocab_size: int) -> ByteBPE:
        texts = list(corpus)
        assert all("VALIDATION_ONLY" not in text for text in texts)
        calls.append(texts)
        return original_train(texts, vocab_size)

    monkeypatch.setattr(ByteBPE, "train", classmethod(train))
    output = tmp_path / "frozen"
    record = freeze(root, output, vocab_size=300, domain_byte_cap=80, workers=1)
    assert len(calls) == 1
    assert record["training_bytes"]["web"] == record["training_bytes"]["python"] == 80
    assert record["status"] == "completed"
    tokenizer = ByteBPE.load(output / "tokenizer.json")
    assert record["tokenizer_hash"] == tokenizer.sha256
    assert len(record["merge_table_sha256"]) == 64
    for domain in ("web", "python"):
        measurements = record["measurements"][domain]
        assert measurements["documents"] == 2
        assert measurements["tokens_with_document_bos"] == measurements["tokens"] + 2
        heldout = measurements["splits"]["validation"]
        assert measurements["held_out_bytes_per_token"] == heldout["text_bytes"] / heldout["tokens"]
        assert measurements["real_probe_fast_reference_equal"]
    assert json.loads((output / "record.json").read_text()) == record
    with pytest.raises(ValueError, match="retraining requires owner approval"):
        freeze(root, output, vocab_size=300, domain_byte_cap=80, workers=1)
    assert len(calls) == 1
