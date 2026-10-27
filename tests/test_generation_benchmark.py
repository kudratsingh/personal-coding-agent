import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.benchmark import model_config, run_generation_benchmark
from scratchcoder.engine.generation import SAMPLER_VERSION
from scratchcoder.eval.checkpoint import load_checkpoint
from scratchcoder.eval.command import sampler_record
from scratchcoder.eval.harness import EvaluationConfig
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import ByteBPE

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_fixture(root: Path) -> tuple[Path, Path]:
    tokenizer = ByteBPE()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    config = ModelConfig(vocab_size=272, width=16, heads=2, layers=1, context=24)
    torch.manual_seed(0)
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "model": GPT(config).state_dict(),
            "config": {"model": asdict(config)},
            "tokenizer_hash": tokenizer.sha256,
        },
        checkpoint,
    )
    return checkpoint, tokenizer_path


def run_cli(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    assert main(["generation-benchmark", *arguments]) == 0
    result: dict[str, Any] = json.loads(capsys.readouterr().out)
    return result


def test_benchmark_cli_reports_random_weight_throughput_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_cli(
        [
            "--config",
            str(ROOT / "configs/smoke.yaml"),
            "--batch-sizes",
            "1",
            "3",
            "--prompt-lengths",
            "4",
            "9",
            "--max-new-tokens",
            "5",
            "--top-p",
            "0.9",
            "--repeats",
            "2",
        ],
        capsys,
    )
    assert result["kind"] == "generation-benchmark"
    assert result["sampler_version"] == SAMPLER_VERSION
    assert (result["device"], result["precision"]) == ("cpu", "float32")
    assert result["model"]["kind"] == "random-initialization"
    assert result["model"]["config"] == asdict(model_config(ROOT / "configs/smoke.yaml"))
    assert result["settings"]["top_p"] == 0.9 and result["settings"]["temperature"] == 1.0
    rows = result["results"]
    assert [row["batch_size"] for row in rows] == [1, 3]
    assert [row["prompt_tokens"] for row in rows] == [4, 4 + 9 + 4]
    # One unsorted batch of lengths 4, 9, 4 is left-padded to 9 slots per row.
    assert [row["prompt_slots"] for row in rows] == [4, 3 * 9]
    assert [row["prompt_padding_fraction"] for row in rows] == pytest.approx([0, 10 / 27])
    assert [row["generated_tokens"] for row in rows] == [5, 15]
    for row in rows:
        assert len(row["seconds"]) == 2 and row["tokens_per_second"] > 0
        assert row["prefill_seconds"] > 0
        assert row["peak_memory_bytes"] > 0 and row["peak_memory_kind"] == "process-max-rss"


def test_benchmark_loads_a_checkpoint_on_the_requested_device(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    result = run_cli(
        [
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--batch-sizes",
            "2",
            "--prompt-lengths",
            "20",
            "--max-new-tokens",
            "8",
            "--precision",
            "bfloat16",
            "--temperature",
            "0",
        ],
        capsys,
    )
    assert result["model"]["kind"] == "checkpoint"
    assert result["model"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert result["precision"] == "bfloat16"
    # Each row's budget is min(8, 24 - 20).
    assert result["results"][0]["generated_tokens"] == 2 * 4


def test_json_run_specifications_name_their_model() -> None:
    assert model_config(ROOT / "configs/rehearsal-a40-s-b.json") == ModelConfig(
        vocab_size=65536, width=768, layers=12, heads=12, context=1024
    )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"config": None}, "exactly one"),
        ({"checkpoint": Path("x.pt")}, "exactly one"),
        ({"precision": "float16"}, "precision"),
        ({"prompt_lengths": [64]}, "room"),
        ({"prompt_lengths": []}, "positive"),
        ({"batch_sizes": [0]}, "positive"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"repeats": 0}, "repeats"),
        ({"device": "gpu"}, "device"),
    ],
)
def test_benchmark_rejects_invalid_requests(changes: dict[str, Any], message: str) -> None:
    request: dict[str, Any] = {
        "checkpoint": None,
        "tokenizer": None,
        "config": ROOT / "configs/smoke.yaml",
        "device": "cpu",
        "precision": "float32",
        "batch_sizes": [1],
        "prompt_lengths": [4],
        "max_new_tokens": 2,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "seed": 0,
        "repeats": 1,
    }
    with pytest.raises(ValueError, match=message):
        run_generation_benchmark(**{**request, **changes})


def test_checkpoint_without_tokenizer_is_rejected(tmp_path: Path) -> None:
    checkpoint, _ = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="tokenizer"):
        run_generation_benchmark(
            checkpoint=checkpoint,
            tokenizer=None,
            config=None,
            device="cpu",
            precision="float32",
            batch_sizes=[1],
            prompt_lengths=[4],
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            top_p=1.0,
            seed=0,
            repeats=1,
        )


def test_checkpoint_loader_resolves_devices_without_fallback(tmp_path: Path) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    model, _ = load_checkpoint(checkpoint, tokenizer, device="cpu")
    assert next(model.parameters()).device == torch.device("cpu") and not model.training
    with pytest.raises(ValueError, match="device"):
        load_checkpoint(checkpoint, tokenizer, device="gpu")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            load_checkpoint(checkpoint, tokenizer, device="cuda")


def test_sampled_reports_name_the_sampler() -> None:
    assert sampler_record(EvaluationConfig()) == {}
    assert sampler_record(EvaluationConfig(temperature=0.8, samples=4)) == {
        "sampler_version": SAMPLER_VERSION
    }
