import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig
from scratchcoder.data.shards import write_shard
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE


def checkpoint_fixture(root: Path) -> tuple[Path, Path, ModelConfig]:
    tokenizer = ByteBPE()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    config = ModelConfig(vocab_size=272, width=16, heads=2, layers=1, context=16)
    model = GPT(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(config)},
            "tokenizer_hash": tokenizer.sha256,
        },
        checkpoint,
    )
    return checkpoint, tokenizer_path, config


def test_validation_checkpoint_cli_and_reference_metrics(tmp_path: Path) -> None:
    checkpoint, tokenizer_path, _ = checkpoint_fixture(tmp_path)
    tokenizer = ByteBPE.load(tokenizer_path)
    for name in ("web", "python"):
        write_shard(
            tmp_path / f"{name}.bin",
            [[97, 98, 99], [100, 101]],
            vocab_size=272,
            tokenizer_hash=tokenizer.sha256,
        )
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"web": "web.bin", "python": "python.bin"}))
    output = tmp_path / "evaluation"
    scorecard = tmp_path / "scorecard.jsonl"
    args = [
        "eval-validation",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(tokenizer_path),
        "--sources",
        str(sources),
        "--output",
        str(output),
        "--scorecard",
        str(scorecard),
        "--documents",
        "1",
    ]
    assert main(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["suite"] == "validation-bits-per-byte"
    assert report["document_counts"]["web"] == {"seen_documents": 1, "available_documents": 2}
    for domain in ("web", "python"):
        assert report["metrics"][domain]["loss"] == pytest.approx(math.log(272), abs=1e-6)
        assert report["metrics"][domain]["bits_per_byte"] == pytest.approx(math.log2(272), abs=1e-6)
    assert json.loads(scorecard.read_text())["source_hashes"] == report["source_hashes"]
    with pytest.raises(ValueError, match="document"):
        main(args[:-1] + ["0"])


def test_shared_checkpoint_loader_rejects_identity_and_vocabulary_mismatch(tmp_path: Path) -> None:
    checkpoint, tokenizer_path, _ = checkpoint_fixture(tmp_path)
    saved = torch.load(checkpoint, weights_only=True)
    saved["tokenizer_hash"] = "incorrect"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="identities"):
        load_checkpoint(checkpoint, tokenizer_path)
    saved["tokenizer_hash"] = ByteBPE.load(tokenizer_path).sha256
    saved["config"]["model"]["vocab_size"] = 273
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="vocabulary"):
        load_checkpoint(checkpoint, tokenizer_path)


@pytest.mark.parametrize("vocab_size,bos_id", [(273, 256), (272, 257)])
def test_validation_rejects_semantic_shard_mismatch_before_report(
    tmp_path: Path, vocab_size: int, bos_id: int
) -> None:
    checkpoint, tokenizer_path, _ = checkpoint_fixture(tmp_path)
    tokenizer = ByteBPE.load(tokenizer_path)
    write_shard(
        tmp_path / "fixture.bin",
        [[97, 98]],
        vocab_size=vocab_size,
        bos_id=bos_id,
        tokenizer_hash=tokenizer.sha256,
    )
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"web": "fixture.bin"}))
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="tokenizer"):
        main(
            [
                "eval-validation",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer_path),
                "--sources",
                str(sources),
                "--output",
                str(output),
                "--scorecard",
                str(tmp_path / "scores.jsonl"),
                "--documents",
                "1",
            ]
        )
    assert not output.exists()
