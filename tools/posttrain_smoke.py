"""CPU smoke of post-training run 1: a tiny model, fixture tasks and an in-memory bucket.

``smoke`` runs every stage of run 1 (probe, ARC on three parents, base evaluation, SFT,
ARC and evaluation of the selected SFT weights, RL screening) and the pretraining rehearsal's
finalization on CPU, with nothing touching a network. The fixtures stand in for the real inputs:

- three parent checkpoints of a 1-layer model, in the parent-checkpoint format;
- a byte-level BPE tokenizer with real merges, trained on the fixtures' own kind of text
  (``smoke_tokenizer``), so every prompt path runs the encoder the Pod runs;
- authored pool dev and test tasks and RL candidates, each exported for the Pod with
  ``scratchcoder pod-task-export``'s code, so the worker sees placeholders, never tests;
- authored benchmark tasks served in place of the pinned HumanEval and EvalPlus releases
  (``patched_benchmarks``), and four ARC-style questions;
- SFT rows rendered from authored train tasks.

The tests build workers from the same fixtures (``fixtures``, ``make_worker``) to exercise
each stop. No fixture is real pool data.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from scratchcoder.common.config import ModelConfig
from scratchcoder.data import chat
from scratchcoder.data.sft import SFTRow, write_rows
from scratchcoder.data.shards import file_sha256
from scratchcoder.eval import protocols
from scratchcoder.eval.choices import ChoiceTask
from scratchcoder.eval.export import write_export
from scratchcoder.eval.tasks import Task, load_tasks, save_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.rl.rollout import SCREENING_SEED
from scratchcoder.tokenizer.bpe import FIRST_MERGE, ByteBPE
from tools.posttrain_approved import run1_stages
from tools.posttrain_run import PosttrainWorker as Worker
from tools.posttrain_run import RunLimits, Specification
from tools.rehearsal_run import SMOKE_CREDENTIALS, MemoryBucket

SMOKE_POD_ID = "smoke-posttrain-1"
PARENTS = {"a-stage1-final": 11, "a-stage2-final": 12, "b-mixed-final": 13}
"""Each parent checkpoint and the seed of its random weights."""


def _task(identifier: str, split: str, index: int, **fields: Any) -> Task:
    values: dict[str, Any] = {
        "id": identifier,
        "prompt": f"Write add_{index}(x) returning x + {index}.",
        "public_tests": f"assert add_{index}(1) == {1 + index}",
        "hidden_tests": f"assert add_{index}(2) == {2 + index}\nassert add_{index}(0) == {index}",
        "source": "synthetic-smoke",
        "license": "fixture",
        "written_by": "model",
        "split": split,
        "solution": f"def add_{index}(x):\n    return x + {index}",
    }
    values.update(fields)
    return Task(**values)


def _io_task(identifier: str, split: str, index: int) -> Task:
    cases = [{"input": f"{n}\n", "output": f"{n + index}\n"} for n in range(3)]
    return _task(
        identifier,
        split,
        index,
        prompt=f"Read n and print n + {index}.",
        public_tests=json.dumps([{"input": "5\n", "output": f"{5 + index}\n"}]),
        hidden_tests=json.dumps(cases),
        mode="io",
        written_by="human",
        solution=f"print(int(input()) + {index})",
    )


@dataclass(frozen=True)
class Fixtures:
    root: Path
    tokenizer: ByteBPE
    inputs: dict[str, Path]
    full_tasks: dict[str, Path]
    """The full task files the exports came from (with hidden tests): the Mac's side."""


def _parent(path: Path, config: ModelConfig, tokenizer: ByteBPE, seed: int, name: str) -> None:
    torch.manual_seed(seed)
    model = GPT(config)
    state = {
        "config": {"model": asdict(config)},
        "tokenizer_hash": tokenizer.sha256,
        "model": model.state_dict(),
        "run_id": f"smoke-{name}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


SMOKE_MERGES = 96
"""Merges of the smoke tokenizer: enough that prompts, rows and labels all merge."""


def smoke_tokenizer() -> ByteBPE:
    """A small BPE tokenizer with real merges, trained on text like the fixtures'."""
    corpus = [
        f"Write add_{index}(x) returning x + {index}. assert add_{index}(1) == {1 + index}\n"
        f"def add_{index}(x):\n    return x + {index}\nRead n and print n + {index}.\n"
        f"print(int(input()) + {index})\nWhich number is {index}? Say hello. Hello there."
        for index in range(48)
    ]
    return ByteBPE.train(corpus, FIRST_MERGE + SMOKE_MERGES)


def smoke_model(tokenizer: ByteBPE) -> ModelConfig:
    return ModelConfig(
        vocab_size=tokenizer.vocab_size, width=32, layers=1, heads=2, context=192, qk_norm=False
    )


def fixtures(root: Path) -> Fixtures:
    """Write every run-1 input of the smoke under ``root``; return their paths."""
    tokenizer = smoke_tokenizer()
    inputs: dict[str, Path] = {}
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    inputs["tokenizer"] = tokenizer_path
    config = smoke_model(tokenizer)
    for name, seed in PARENTS.items():
        path = root / name / "checkpoint.pt"
        _parent(path, config, tokenizer, seed, name)
        inputs[name] = path
    full: dict[str, Path] = {}
    for split, offset in (("dev", 10), ("test", 20)):
        tasks = [
            _task(f"smoke/{split}/1", split, offset + 1),
            _task(f"smoke/{split}/2", split, offset + 2),
            _io_task(f"smoke/{split}/3", split, offset + 3),
        ]
        source = root / "mac" / f"{split}.jsonl"
        save_tasks(source, tasks)
        full[split] = source
        export = root / f"pool-{split}" / f"{split}-export.jsonl"
        write_export(
            "pool", tasks_path=source, output=export, record=root / "mac" / f"{split}.json"
        )
        inputs[f"pool-{split}"] = export
    train = [_task(f"smoke/train/{index}", "train", index) for index in range(16)]
    train_path = root / "mac" / "train.jsonl"
    save_tasks(train_path, train)
    full["train"] = train_path
    candidates = root / "rl-candidates-list" / "rl_candidates.jsonl"
    candidates.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_text(
        "".join(json.dumps({"id": task.id, "source": task.source}) + "\n" for task in train[12:])
    )
    inputs["rl-candidates-list"] = candidates
    export = root / "rl-candidates" / "candidates-export.jsonl"
    write_export(
        "candidates",
        tasks_path=train_path,
        candidates_path=candidates,
        output=export,
        record=root / "mac" / "candidates.json",
    )
    inputs["rl-candidates"] = export
    rows = [
        SFTRow.of(task.id, task.source, chat.render_code_task(tokenizer, task))
        for task in train[:12]
    ]
    write_rows(root / "sft-train" / "train.jsonl", rows)
    inputs["sft-train"] = root / "sft-train" / "train.jsonl"
    dev_tasks = list(load_tasks(full["dev"]))
    write_rows(
        root / "sft-code-dev" / "code_dev.jsonl",
        [
            SFTRow.of(task.id, task.source, chat.render_code_task(tokenizer, task))
            for task in dev_tasks
        ],
    )
    inputs["sft-code-dev"] = root / "sft-code-dev" / "code_dev.jsonl"
    conversation = chat.render_conversation(
        tokenizer,
        [chat.Message("user", "Say hello."), chat.Message("assistant", "Hello there.")],
    )
    write_rows(
        root / "sft-language-dev" / "language_dev.jsonl",
        [SFTRow.of("smoke/language/1", "language-smoke", conversation)],
    )
    inputs["sft-language-dev"] = root / "sft-language-dev" / "language_dev.jsonl"
    for suite, identifiers in (
        ("humaneval", ("humaneval/31", "humaneval/33")),
        ("humaneval-plus", ("humaneval/31", "humaneval/32")),
        ("mbpp-plus", ("Mbpp/2", "Mbpp/3")),
    ):
        tasks = [
            _task(
                identifier,
                "eval",
                40 + index,
                prompt=f"def f{index}(x):\n",
                entry_point=f"f{index}",
            )
            for index, identifier in enumerate(identifiers)
        ]
        path = root / suite / "release.jsonl"
        save_tasks(path, tasks)
        inputs[suite] = path
    questions = [
        ChoiceTask(
            id=f"smoke-arc-{index}",
            question=f"Which number is {index}?",
            labels=("A", "B", "C"),
            choices=(str(index), str(index + 1), str(index + 2)),
            answer="A",
            source="allenai/ai2_arc",
            source_revision="0" * 40,
            license="CC-BY-SA-4.0",
            split="test",
        )
        for index in range(4)
    ]
    arc = root / "arc-easy-test" / "arc-easy-test.jsonl"
    arc.parent.mkdir(parents=True, exist_ok=True)
    arc.write_text(
        "".join(json.dumps(asdict(question), sort_keys=True) + "\n" for question in questions)
    )
    inputs["arc-easy-test"] = arc
    return Fixtures(root, tokenizer, inputs, full)


SMOKE_TASKS = {"pool": 3, "humaneval": 2, "humaneval-plus": 1, "mbpp-plus": 2}
"""Each suite's scored fixture tasks: HumanEval+ always excludes ``humaneval/32``."""
KINDS = {
    "tokenizer": "tokenizer",
    "a-stage1-final": "checkpoint",
    "a-stage2-final": "checkpoint",
    "b-mixed-final": "checkpoint",
    "pool-dev": "pool-export",
    "pool-test": "pool-export",
    "rl-candidates": "candidates-export",
    "rl-candidates-list": "candidate-list",
    "sft-train": "sft-rows",
    "sft-code-dev": "sft-rows",
    "sft-language-dev": "sft-rows",
    "humaneval": "benchmark",
    "humaneval-plus": "benchmark",
    "mbpp-plus": "benchmark",
    "arc-easy-test": "arc-export",
}
EXISTING = {
    "tokenizer": "smoke-build/tokenizer/tokenizer.json",
    "a-stage1-final": "rehearsal-runs/A/smoke-a/stage1-final/checkpoint.pt",
    "a-stage2-final": "rehearsal-runs/A/smoke-a/stage2-final/checkpoint.pt",
    "b-mixed-final": "rehearsal-runs/B/smoke-b/mixed-final/checkpoint.pt",
}


def smoke_specification(found: Fixtures) -> dict[str, Any]:
    """Run 1's specification at the smoke's scale: the stages of record on tiny inputs."""
    inputs = {}
    for name, kind in KINDS.items():
        path = found.inputs[name]
        inputs[name] = {
            "kind": kind,
            "key": EXISTING.get(name),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    stages = run1_stages()
    for stage in stages:
        for entry in stage.get("files", []):
            entry["planned_tasks"] = SMOKE_TASKS[entry["suite"]]
    return {
        "schema_version": 1,
        "kind": "posttrain-run-specification",
        "run": "1",
        "model": asdict(smoke_model(found.tokenizer)),
        "limits": {
            "ceiling_usd": 3.0,
            "quoted_gpu_hourly": 0.59,
            "quoted_total_hourly": 0.6,
            "self_destruct_margin_seconds": 180,
            "watchdog_margin_usd": 0.01,
            "mac_idle_seconds": 600,
            "pod_idle_seconds": 900,
            "max_transfer_seconds": 900,
            "finalization_reserve_seconds": 900,
            "startup_seconds": 1080,
        },
        "planning": {
            "generation_tokens_per_second": {"4": 50.0},
            "training_slots_per_second": 100.0,
            "forward_tokens_per_second": 100.0,
            "probe_seconds": 30,
            "arc_seconds_per_checkpoint": 5,
            "checkpoint_load_seconds": 1,
            "compile_seconds": 1,
            "weights_save_seconds": 1,
        },
        "probe": {
            "batch_sizes": [1, 4],
            "prompt_lengths": 4,
            "max_new_tokens": 4,
            "temperature": 0.8,
            "top_p": 0.95,
            "precision": "float32",
            "seed": 0,
            "repeats": 1,
            "minimum_tokens_per_second": {"4": 1.0},
        },
        "generation": {"batch_size": 4, "precision": "float32"},
        "sft": {
            "train_rows": 12,
            "planned_dev_tokens": 1_000,
            "learning_rate": 6e-05,
            "warmup_fraction": 0.03,
            "min_lr_ratio": 0.1,
            "schedule": "cosine",
            "beta1": 0.9,
            "beta2": 0.95,
            "adam_epsilon": 1e-08,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "batch_size": 2,
            "accumulation_steps": 2,
            "sequence_length": 192,
            "seed": 0,
            "passes": 1,
            "precision": "float32",
            "attention": "reference",
            "optimizer_foreach": False,
            "compile": False,
            "dev_batch_size": 4,
            "speed_check_updates": 2,
            "speed_check_warmup_updates": 0,
            "reference_slots_per_second": 1.0,
            "minimum_speed_fraction": 0.9,
        },
        "screening": {
            "samples": 8,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 16,
            "seed_base": SCREENING_SEED,
            "planned_tasks": 4,
            "chunk_tasks": 2,
        },
        "stages": stages,
        "inputs": inputs,
        "r2": {"storage_cap_usd_per_month": 5.0},
    }


@contextmanager
def patched_benchmarks() -> Iterator[None]:
    """Serve the fixture benchmark files in place of the pinned releases (the smoke only)."""

    def humaneval(path: Path) -> list[Task]:
        return list(load_tasks(path))

    def release(path: Path, specification: object) -> list[Task]:
        return list(load_tasks(path))

    with (
        mock.patch.object(protocols, "read_humaneval", humaneval),
        mock.patch.object(protocols, "read_release", release),
    ):
        yield


def make_worker(
    found: Fixtures,
    root: Path,
    *,
    bucket: MemoryBucket,
    changes: Callable[[dict[str, Any]], None] | None = None,
    limits: RunLimits | None = None,
    pod_created: float | None = None,
    **options: Any,
) -> Worker:
    """A CPU worker for run 1 on the smoke fixtures; ``changes`` edits the specification."""
    raw = smoke_specification(found)
    if changes is not None:
        changes(raw)
    spec = Specification.from_dict(raw)
    return Worker(
        spec,
        root,
        inputs=found.inputs,
        store=bucket.store(SMOKE_CREDENTIALS),
        pod_id=SMOKE_POD_ID,
        pod_created=time.time() if pod_created is None else pod_created,
        limits=limits,
        **options,
    )


def smoke(output: Path, *, bucket: MemoryBucket | None = None) -> dict[str, Any]:
    """Run 1 on CPU, offline, into an in-memory bucket; return a summary."""
    torch.set_num_threads(1)
    if output.exists() or output.is_symlink():
        raise FileExistsError("smoke output must be a new directory")
    found = fixtures(output / "inputs")
    bucket = MemoryBucket() if bucket is None else bucket
    worker = make_worker(found, output / "run", bucket=bucket)
    with patched_benchmarks():
        report = worker.execute()
    return summary(report, bucket)


def summary(report: Mapping[str, Any], bucket: MemoryBucket) -> dict[str, Any]:
    prefix = report["r2"]["prefix"]
    return {
        "status": report["status"],
        "stop_reason": report.get("stop_reason"),
        "stages": {stage["name"]: stage["status"] for stage in report["stages"]},
        "artifacts": len(report["artifacts"]),
        "bucket_objects": len(bucket.objects),
        "objects_outside_prefix": sorted(
            key for key in bucket.objects if not key.startswith(prefix)
        ),
        "probe_rates": (report.get("probe") or {}).get("measured_tokens_per_second"),
        "sft_selected": (report.get("sft") or {}).get("selected"),
    }
