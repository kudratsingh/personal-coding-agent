"""Generation into completions files: prompts, seeds, budgets, extraction and the CLI."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from eval_fixtures import TOKENIZER, chat_tokens, io_task, pool_task, save_task_file

from scratchcoder.cli import main
from scratchcoder.common.config import ModelConfig
from scratchcoder.data import chat
from scratchcoder.engine.generation import Completion, generate_many
from scratchcoder.eval import protocols
from scratchcoder.eval.completions import Sampling, read_completions, validate_against_tasks
from scratchcoder.eval.generate import generate_completions, generation_device
from scratchcoder.eval.harness import attempt_seed
from scratchcoder.model.gpt import GPT


def checkpoint_fixture(root: Path, *, context: int = 128) -> tuple[Path, Path]:
    tokenizer_path = root / "tokenizer.json"
    TOKENIZER.save(tokenizer_path)
    config = ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=context)
    torch.manual_seed(0)
    model = GPT(config)
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": asdict(config)},
            "tokenizer_hash": TOKENIZER.sha256,
        },
        checkpoint,
    )
    return checkpoint, tokenizer_path


def tasks_fixture(root: Path) -> tuple[Path, list[Any]]:
    tasks = [pool_task("t/2"), pool_task("t/1"), io_task("t/3")]
    return save_task_file(root / "dev.jsonl", tasks), tasks


def test_greedy_chat_generation_matches_direct_batched_generation(tmp_path: Path) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    tasks_path, tasks = tasks_fixture(tmp_path)
    output = tmp_path / "completions.jsonl"
    header, digest = generate_completions(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        suite="pool",
        split="dev",
        tasks_path=tasks_path,
        protocol="chat-v1",
        output=output,
        batch_size=2,
    )
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    completions = read_completions(output, sha256=digest)
    validate_against_tasks(completions, tasks_path)
    assert header.protocol_sha256 == protocols.protocol_for("chat-v1", "pool").sha256()
    assert header.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert header.tokenizer_sha256 == TOKENIZER.sha256 and header.model_context == 128
    assert header.sampler_version is None and header.device == "cpu"
    model = GPT(ModelConfig(vocab_size=272, width=16, layers=1, heads=2, context=128))
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    model.eval()
    ordered = sorted(tasks, key=lambda task: task.id)
    prompts = [chat.task_prompt(TOKENIZER, task) for task in ordered]
    direct = generate_many(
        model,
        prompts,
        batch_size=3,
        max_new_tokens=512,
        stops=[chat.CODE_STOP_IDS] * 3,
    )
    for task, prompt, expected in zip(ordered, prompts, direct, strict=True):
        row = completions.row(task.id, 0)
        assert list(row.tokens) == expected.tokens
        assert row.stop_reason == expected.finish_reason
        assert row.budget == 128 - len(prompt) == expected.budget
        assert row.prompt_tokens == len(prompt)
        assert row.seed == attempt_seed(0, task.id, 0)
        assert row.completion == TOKENIZER.decode(expected.tokens, skip_special=True)
        assert row.program == chat.extract_program(expected.tokens, TOKENIZER)


def test_sampled_attempts_use_harness_seeds_whatever_the_batch(tmp_path: Path) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    tasks_path, _ = tasks_fixture(tmp_path)
    sampling = Sampling(samples=3, temperature=0.9, top_p=0.95, top_k=50, seed=7)
    files = []
    for batch_size in (1, 9):
        output = tmp_path / f"sampled-{batch_size}.jsonl"
        generate_completions(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            suite="pool",
            split="dev",
            tasks_path=tasks_path,
            protocol="chat-v1",
            output=output,
            sampling=sampling,
            batch_size=batch_size,
        )
        files.append(read_completions(output))
    small, large = files
    assert small.header.sampler_version == "philox4x32-10-inverse-cdf-v1"
    assert [row.seed for row in small.rows] == [
        attempt_seed(7, task_id, sample) for task_id in ("t/1", "t/2", "t/3") for sample in range(3)
    ]
    # Same seeds, different batches: identical tokens on CPU float32.
    assert [row.tokens for row in small.rows] == [row.tokens for row in large.rows]
    assert len({row.tokens for row in small.rows if row.task_id == "t/1"}) > 1


def fake_generate(answers: dict[int, list[int]]) -> Any:
    calls: list[dict[str, Any]] = []

    def run(model: GPT, prompts: list[list[int]], **options: Any) -> list[Completion]:
        calls.append({"prompts": prompts, **options})
        out = []
        for prompt in prompts:
            budget = min(options["max_new_tokens"], model.config.context - len(prompt))
            tokens = answers.get(len(prompt), [])[:budget]
            stopped = bool(tokens) and tokens[-1] in (options["stops"] or [[]])[0]
            if not stopped:
                tokens = (tokens + [ord(" ")] * budget)[:budget]
            out.append(Completion(tokens, "stop" if stopped else "length", budget))
        return out

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_context_limits_extraction_and_stop_reasons(tmp_path: Path) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path, context=64)
    long_task = pool_task("t/long", prompt="x" * 200)
    tasks = [pool_task("t/1"), long_task]
    tasks_path = save_task_file(tmp_path / "dev.jsonl", tasks)
    prompt = chat.task_prompt(TOKENIZER, tasks[0])
    program = "x=1"
    fake = fake_generate({len(prompt): list(chat_tokens(program))})
    output = tmp_path / "completions.jsonl"
    with pytest.raises(ValueError, match="greedy"):
        Sampling(samples=2)
    generate_completions(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        suite="pool",
        split="dev",
        tasks_path=tasks_path,
        protocol="chat-v1",
        output=output,
        generate=fake,
    )
    completions = read_completions(output)
    limited = completions.row("t/long", 0)
    assert limited.stop_reason == "context_limit" and limited.tokens == () and limited.budget == 0
    assert limited.program is None
    # The context-limited prompt never reaches generation.
    assert fake.calls[0]["prompts"] == [prompt]
    assert fake.calls[0]["stops"] == [chat.CODE_STOP_IDS]
    assert fake.calls[0]["seeds"] is None
    answered = completions.row("t/1", 0)
    assert answered.stop_reason == "stop" and answered.program == program


def test_base_pool_generation_has_no_stop_tokens_and_truncates_text(tmp_path: Path) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    task = io_task("t/io")
    tasks_path = save_task_file(tmp_path / "dev.jsonl", [task])
    prompt = protocols.protocol_for("base-pool-completion-v1", "pool").prompt_ids(TOKENIZER, task)
    text = "n = int(input())\nprint(n + 1)\nassert True\n"
    fake = fake_generate({len(prompt): list(text.encode())})
    output = tmp_path / "base.jsonl"
    generate_completions(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        suite="pool",
        split="dev",
        tasks_path=tasks_path,
        protocol="base-pool-completion-v1",
        output=output,
        generate=fake,
    )
    row = read_completions(output).row("t/io", 0)
    assert fake.calls[0]["stops"] is None and fake.calls[0]["max_new_tokens"] == 512
    assert row.stop_reason == "length" and row.program == "n = int(input())\nprint(n + 1)"


def test_benchmark_generation_records_the_task_32_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    release = tmp_path / "release.jsonl.gz"
    release.write_bytes(b"fixture release bytes")
    tasks = [
        pool_task(f"humaneval/{index}", prompt=f"def f{index}(x):\n", split="eval")
        for index in (31, 32)
    ]
    monkeypatch.setattr(protocols, "read_release", lambda path, specification: tasks)
    output = tmp_path / "he.jsonl"
    header, _ = generate_completions(
        checkpoint=checkpoint,
        tokenizer_path=tokenizer,
        suite="humaneval-plus",
        split="eval",
        tasks_path=release,
        protocol="base-completion-rstrip-v1",
        output=output,
        generate=fake_generate({}),
    )
    assert header.excluded_tasks == (
        {"task_id": "humaneval/32", "reason": "known numerical-tolerance failure"},
    )
    completions = read_completions(output)
    assert completions.task_ids == ["humaneval/31"]
    row = completions.row("humaneval/31", 0)
    assert row.program == "def f31(x):" + " " * row.budget
    assert row.budget == 128 - len(b"def f31(x):") and header.max_new_tokens == 256


def test_devices_are_explicit_and_outputs_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert generation_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="cpu, cuda"):
        generation_device("gpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS requested but unavailable"):
        generation_device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert generation_device("mps") == torch.device("mps")
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    tasks_path, _ = tasks_fixture(tmp_path)
    output = tmp_path / "exists.jsonl"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        generate_completions(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            suite="pool",
            split="dev",
            tasks_path=tasks_path,
            protocol="chat-v1",
            output=output,
        )
    assert output.read_text() == "keep"
    with pytest.raises(ValueError, match="protocol"):
        generate_completions(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            suite="pool",
            split="dev",
            tasks_path=tasks_path,
            protocol="base-completion-rstrip-v1",
            output=tmp_path / "new.jsonl",
        )


def test_cli_writes_the_file_and_prints_its_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint, tokenizer = checkpoint_fixture(tmp_path)
    tasks_path, _ = tasks_fixture(tmp_path)
    output = tmp_path / "cli.jsonl"
    arguments = [
        "eval-generate",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(tokenizer),
        "--suite",
        "pool",
        "--tasks",
        str(tasks_path),
        "--protocol",
        "chat-v1",
        "--output",
        str(output),
        "--samples",
        "2",
        "--temperature",
        "0.8",
        "--top-p",
        "0.95",
        "--batch-size",
        "4",
    ]
    with pytest.raises(SystemExit):
        main(arguments)  # The pool needs an explicit split.
    capsys.readouterr()
    assert main([*arguments, "--split", "dev"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert printed["rows"] == 6 and printed["split"] == "dev"
    header = read_completions(output).header
    assert header.sampling == Sampling(samples=2, temperature=0.8, top_p=0.95)
    assert header.batch_size == 4
