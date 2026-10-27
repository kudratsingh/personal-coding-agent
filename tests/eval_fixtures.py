"""Shared authored fixtures for the evaluation-pipeline tests (never real pool data)."""

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from scratchcoder.data import chat
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import SAMPLER_VERSION
from scratchcoder.eval import protocols
from scratchcoder.eval.completions import (
    CompletionRow,
    CompletionsHeader,
    Sampling,
    prompt_sha256,
    task_ids_sha256,
    write_completions,
)
from scratchcoder.eval.protocols import protocol_for
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.tokenizer.bpe import ByteBPE

TOKENIZER = ByteBPE()  # No merges: every ordinary token is one UTF-8 byte.
CHECKPOINT = "c" * 64


def pool_task(identifier: str, **fields: object) -> Task:
    values: dict[str, object] = {
        "id": identifier,
        "prompt": "Write add_one(x).",
        "public_tests": "assert add_one(1) == 2",
        "hidden_tests": "assert add_one(2) == 3\nassert add_one(-1) == 0",
        "source": "synthetic-a",
        "license": "fixture",
        "written_by": "model",
        "split": "dev",
        "solution": "def add_one(x):\n    return x + 1",
    }
    values.update(fields)
    return Task(**values)  # type: ignore[arg-type]


def io_task(identifier: str, cases: int = 3, **fields: object) -> Task:
    hidden = json.dumps([{"input": f"{i}\n", "output": f"{i + 1}\n"} for i in range(cases)])
    values: dict[str, object] = {
        "prompt": "Read n and print n + 1.",
        "public_tests": json.dumps([{"input": "5\n", "output": "6\n"}]),
        "hidden_tests": hidden,
        "mode": "io",
        "source": "synthetic-io",
        "written_by": "human",
        "solution": "print(int(input()) + 1)",
    }
    values.update(fields)
    return pool_task(identifier, **values)


def save_task_file(path: Path, tasks: Sequence[Task]) -> Path:
    save_tasks(path, tasks)
    return path


def chat_tokens(program: str | None) -> tuple[int, ...]:
    """Generated IDs of a chat-v1 answer holding ``program`` (no block when None)."""
    if program is None:
        return (*TOKENIZER.encode("I cannot."), chat.ASSISTANT_END)
    return (
        chat.PYTHON_START,
        *TOKENIZER.encode(program),
        chat.PYTHON_END,
        chat.ASSISTANT_END,
    )


def write_fixture_completions(
    path: Path,
    tasks_path: Path,
    tasks: Sequence[Task],
    program_for: Callable[[Task, int], str | None],
    *,
    split: str = "dev",
    sampling: Sampling | None = None,
    protocol: str = "chat-v1",
    checkpoint: str = CHECKPOINT,
    context: int = 4096,
) -> str:
    """A chat-v1 pool completions file whose attempts answer with ``program_for``."""
    sampling = sampling or Sampling()
    definition = protocol_for(protocol, "pool")
    rows = []
    for task in tasks:
        prompt = definition.prompt_ids(TOKENIZER, task)
        for sample, seed in enumerate(sampling.seeds(task.id)):
            program = program_for(task, sample)
            tokens = chat_tokens(program)
            budget = definition.budget(len(prompt), context)
            rows.append(
                CompletionRow(
                    task_id=task.id,
                    sample=sample,
                    seed=seed,
                    prompt_tokens=len(prompt),
                    prompt_sha256=prompt_sha256(prompt),
                    tokens=tokens,
                    completion=TOKENIZER.decode(tokens, skip_special=True),
                    program=definition.candidate(
                        task, tokens=tokens, completion="", decoder=TOKENIZER
                    ),
                    stop_reason="stop",
                    budget=budget,
                )
            )
    header = CompletionsHeader(
        suite="pool",
        split=split,
        protocol=definition.name,
        protocol_sha256=definition.sha256(),
        protocol_record=definition.record(),
        checkpoint_sha256=checkpoint,
        tokenizer_sha256=TOKENIZER.sha256,
        tokenizer_vocab_size=TOKENIZER.vocab_size,
        model_context=context,
        task_file_sha256=file_sha256(tasks_path),
        task_count=len(tasks),
        task_ids_sha256=task_ids_sha256(task.id for task in tasks),
        sampling=sampling,
        max_new_tokens=definition.cap,
        precision="float32",
        device="cpu",
        rows=len(rows),
        created_at="2026-10-26T00:00:00+00:00",
        sampler_version=None if sampling.greedy else SAMPLER_VERSION,
        generation_seconds=1.5,
    )
    return write_completions(path, header, rows)


def benchmark_tasks() -> list[Task]:
    tasks = []
    for index in range(3):
        prompt = f"def f{index}(x):\n    '''Return x.\n    >>> f{index}(1)\n    1\n    '''\n"
        tasks.append(
            pool_task(
                f"HumanEval/{index}",
                prompt=prompt,
                split="eval",
                entry_point=f"f{index}",
                public_tests=f"assert callable(f{index})",
                hidden_tests=(
                    f"def check(candidate):\n    assert candidate(2) == 2\ncheck(f{index})"
                ),
                solution=prompt + "    return x\n",
                source="openai/human-eval",
                written_by="human",
            )
        )
    return tasks


def write_benchmark_completions(
    root: Path, tasks: list[Task], completions: dict[str, str], *, suite: str = "humaneval"
) -> tuple[Path, Path]:
    release = root / "release.jsonl.gz"
    release.write_bytes(b"pinned release bytes (fixture)")
    definition = protocols.protocol_for("base-completion-rstrip-v1", suite)
    rows = []
    for task in tasks:
        prompt = definition.prompt_ids(TOKENIZER, task)
        text = completions[task.id]
        tokens = tuple(text.encode())
        rows.append(
            CompletionRow(
                task_id=task.id,
                sample=0,
                seed=Sampling().seeds(task.id)[0],
                prompt_tokens=len(prompt),
                prompt_sha256=prompt_sha256(prompt),
                tokens=tokens,
                completion=text,
                program=definition.candidate(task, tokens=tokens, completion=text),
                stop_reason="length",
                budget=len(tokens),
            )
        )
    header = CompletionsHeader(
        suite=suite,
        split="eval",
        protocol=definition.name,
        protocol_sha256=definition.sha256(),
        protocol_record=definition.record(),
        checkpoint_sha256=CHECKPOINT,
        tokenizer_sha256=TOKENIZER.sha256,
        tokenizer_vocab_size=TOKENIZER.vocab_size,
        model_context=1024,
        task_file_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),
        task_count=len(tasks),
        task_ids_sha256=task_ids_sha256(task.id for task in tasks),
        sampling=Sampling(),
        max_new_tokens=256,
        precision="bfloat16",
        device="cuda:0",
        rows=len(rows),
        created_at="2026-10-26T00:00:00+00:00",
    )
    path = root / "he.jsonl"
    write_completions(path, header, rows)
    return path, release
