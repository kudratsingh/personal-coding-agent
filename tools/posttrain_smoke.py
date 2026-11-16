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


def fixtures(root: Path, *, config: ModelConfig | None = None) -> Fixtures:
    """Write every run-1 input of the smoke under ``root``; return their paths. The parents
    are of ``config`` (default ``smoke_model``)."""
    tokenizer = smoke_tokenizer()
    inputs: dict[str, Path] = {}
    tokenizer_path = root / "tokenizer" / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    inputs["tokenizer"] = tokenizer_path
    config = smoke_model(tokenizer) if config is None else config
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


def smoke_specification(found: Fixtures, kinds: Mapping[str, str] = KINDS) -> dict[str, Any]:
    """Run 1's specification at the smoke's scale: the stages of record on tiny inputs."""
    inputs = {}
    for name, kind in kinds.items():
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


# Run 1b ------------------------------------------------------------------------------------

SMOKE_POD_ID_1B = "smoke-posttrain-1b"
KINDS_1B = {name: kind for name, kind in KINDS.items() if not name.startswith("a-stage")}
SMOKE_SFT_1B = {
    "learning_rate": 3e-3,
    "passes": 30,
    "weight_decay": 0.0,
}
"""The smoke's SFT trains its 1-layer model hard enough on the fixture rows to learn the
chat-v1 format, so the real format gate passes on it; ``failing_gate`` undoes this."""


def smoke_model_1b(tokenizer: ByteBPE) -> ModelConfig:
    """Run 1b's smoke model: two layers of width 64, enough to learn where the fixture's
    code ends, which the real format gate's close check needs."""
    return ModelConfig(
        vocab_size=tokenizer.vocab_size, width=64, layers=2, heads=2, context=192, qk_norm=False
    )


def omit_rows_like_run_1b(found: Fixtures) -> None:
    """Rewrite the fixture's SFT train and code dev rows as run 1b's data build does: the
    rows ``omits_public_tests`` names leave their public tests out."""
    from scratchcoder.data.sft import omits_public_tests

    for name, split in (("sft-train", "train"), ("sft-code-dev", "dev")):
        path = found.inputs[name]
        tasks = list(load_tasks(found.full_tasks[split]))
        if split == "train":
            tasks = tasks[:12]
        path.unlink()
        write_rows(
            path,
            [
                SFTRow.of(
                    task.id,
                    task.source,
                    chat.render_code_task(
                        found.tokenizer, task, public_tests=not omits_public_tests(task.id)
                    ),
                )
                for task in tasks
            ],
        )


def fixtures_1b(root: Path) -> Fixtures:
    """Run 1b's smoke inputs: run 1's fixtures without run A's parents, the SFT rows built
    with the public-test omission rule and the MBPP+ fixtures under MBPP+ IDs."""
    found = fixtures(root, config=smoke_model_1b(smoke_tokenizer()))
    omit_rows_like_run_1b(found)
    mbpp = found.inputs["mbpp-plus"]
    tasks = [
        Task(**{**task.__dict__, "id": f"mbpp/{index + 2}"})
        for index, task in enumerate(load_tasks(mbpp))
    ]
    mbpp.unlink()
    save_tasks(mbpp, tasks)
    inputs = {name: path for name, path in found.inputs.items() if name in KINDS_1B}
    return Fixtures(found.root, found.tokenizer, inputs, found.full_tasks)


def smoke_specification_1b(found: Fixtures) -> dict[str, Any]:
    """Run 1b's specification at the smoke's scale: run 1b's stages, the special rows and
    the format gate of record on tiny inputs, the gate's counts the fixture's."""
    from scratchcoder.data.sft import omits_public_tests, read_rows
    from tools.posttrain_approved import RUN1B_SFT, run1b_stages

    raw = smoke_specification(Fixtures(found.root, found.tokenizer, found.inputs, {}), KINDS_1B)
    stages = run1b_stages()
    with_tests = sum(
        not omits_public_tests(row.id) for row in read_rows(found.inputs["sft-code-dev"])
    )
    for stage in stages:
        for entry in stage.get("files", []):
            entry["planned_tasks"] = SMOKE_TASKS[entry["suite"]]
        if stage["kind"] == "format-gate":
            stage.update(row_count=max(1, with_tests), smoke_prompts=3, benchmark_prompts=2)
    raw.update(
        run="1b",
        model=asdict(smoke_model_1b(found.tokenizer)),
        stages=stages,
        limits={**raw["limits"], "ceiling_usd": 1.5},
        planning={**raw["planning"], "format_gate_seconds": 5},
        sft={
            **raw["sft"],
            **{key: RUN1B_SFT[key] for key in ("special_rows_init", "special_rows_seed")},
            "special_rows_lr_scale": RUN1B_SFT["special_rows_lr_scale"],
            **SMOKE_SFT_1B,
        },
    )
    return raw


def failing_gate(raw: dict[str, Any]) -> None:
    """The smoke's SFT at run 1's peak rate for one pass: too little to learn the format,
    so the format gate fails and the run stops before anything reads the weights."""
    raw["sft"].update(learning_rate=6e-05, passes=1, weight_decay=0.1)


def make_worker_1b(
    found: Fixtures,
    root: Path,
    *,
    bucket: MemoryBucket,
    changes: Callable[[dict[str, Any]], None] | None = None,
    limits: RunLimits | None = None,
    pod_created: float | None = None,
    **options: Any,
) -> Worker:
    """A CPU worker for run 1b on its smoke fixtures; ``changes`` edits the specification."""
    raw = smoke_specification_1b(found)
    if changes is not None:
        changes(raw)
    spec = Specification.from_dict(raw)
    return Worker(
        spec,
        root,
        inputs=found.inputs,
        store=bucket.store(SMOKE_CREDENTIALS),
        pod_id=SMOKE_POD_ID_1B,
        pod_created=time.time() if pod_created is None else pod_created,
        limits=limits,
        **options,
    )


def smoke_1b(
    output: Path,
    *,
    bucket: MemoryBucket | None = None,
    changes: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run 1b on CPU, offline, into an in-memory bucket; return a summary."""
    torch.set_num_threads(1)
    if output.exists() or output.is_symlink():
        raise FileExistsError("smoke output must be a new directory")
    found = fixtures_1b(output / "inputs")
    bucket = MemoryBucket() if bucket is None else bucket
    worker = make_worker_1b(found, output / "run", bucket=bucket, changes=changes)
    with patched_benchmarks():
        report = worker.execute()
    return {**summary(report, bucket), "format_gate": report.get("format_gate")}


# Run 1c ------------------------------------------------------------------------------------

SMOKE_POD_ID_1C = "smoke-posttrain-1c"
KINDS_1C = {
    "tokenizer": "tokenizer",
    "sft-weights": "checkpoint",
    "sft-train": "sft-rows",
    "sft-code-dev": "sft-rows",
    "sft-language-dev": "sft-rows",
    "pool-dev": "pool-export",
    "humaneval": "benchmark",
    "humaneval-plus": "benchmark",
    "mbpp-plus": "benchmark",
}
"""Run 1c's inputs: no pool test export, no RL or ARC input, no base checkpoint."""
SMOKE_COPIES_1C = 30
"""Each fixture train row appears this many times under its own ID, so the smoke's one pass
(run 1c's) trains its model as much as run 1b's smoke's 30 passes over the 12 rows do."""
SMOKE_SFT_1C = {"learning_rate": 5e-3, "weight_decay": 0.0}
"""Hard enough on the fixture rows that the real format gate passes on the final weights of
the one pass (run 1b's smoke also had the special rows' 10x rate, which run 1c drops);
``failing_gate_1c`` undoes it."""
SMOKE_IMAGE = "runpod/pytorch:smoke-tag"
"""The image name the smoke's environment record carries: a tag, as the provider reports."""


def fixtures_1c(root: Path) -> Fixtures:
    """Run 1c's smoke inputs: run 1b's fixtures (the public-test omission, the MBPP+ IDs), the
    parent written by ``rl.checkpoint.save_weights`` (the format of run 1b's SFT weights, with
    its ``format``, ``step`` and ``metadata`` keys) under ``sft-weights``, and the train rows
    repeated under their own IDs for the one pass."""
    from scratchcoder.data.sft import read_rows
    from scratchcoder.rl.checkpoint import save_weights

    found = fixtures_1b(root)
    config = smoke_model_1b(found.tokenizer)
    torch.manual_seed(PARENTS["b-mixed-final"])
    parent = root / "sft-weights" / "checkpoint.pt"
    save_weights(
        parent,
        GPT(config),
        tokenizer_hash=found.tokenizer.sha256,
        run_id="smoke-sft-weights",
        step=0,
        metadata={"smoke": "the imported parent"},
    )
    original = list(read_rows(found.inputs["sft-train"]))
    train = root / "sft-train-1c" / "train.jsonl"
    write_rows(
        train,
        [
            SFTRow(f"{row.id}/copy-{copy}", row.source, row.ids, row.mask)
            for copy in range(SMOKE_COPIES_1C)
            for row in original
        ],
    )
    inputs = {name: path for name, path in found.inputs.items() if name in KINDS_1C}
    inputs.update({"sft-weights": parent, "sft-train": train})
    return Fixtures(found.root, found.tokenizer, inputs, found.full_tasks)


def smoke_specification_1c(found: Fixtures) -> dict[str, Any]:
    """Run 1c's specification at the smoke's scale: its stages, one pass, no special-row
    redraw, the final weights kept, the parent's dev loss at update 0, the format gate and the
    estimate stop (the estimate projected from this specification, as the bind does)."""
    from scratchcoder.data.sft import omits_public_tests, read_rows
    from tools.posttrain_approved import RUN1C_SFT, run1c_estimate_usd, run1c_stages
    from tools.posttrain_ops import estimate_limit_usd

    raw = smoke_specification(Fixtures(found.root, found.tokenizer, found.inputs, {}), KINDS_1C)
    stages = run1c_stages()
    with_tests = sum(
        not omits_public_tests(row.id) for row in read_rows(found.inputs["sft-code-dev"])
    )
    for stage in stages:
        for entry in stage.get("files", []):
            entry["planned_tasks"] = SMOKE_TASKS[entry["suite"]]
        if stage["kind"] == "format-gate":
            stage.update(row_count=max(1, with_tests), smoke_prompts=3, benchmark_prompts=2)
    rows = sum(1 for _ in read_rows(found.inputs["sft-train"]))
    sft = {
        **raw["sft"],
        **SMOKE_SFT_1C,
        "train_rows": rows,
        "passes": RUN1C_SFT["passes"],
        "kept_weights": RUN1C_SFT["kept_weights"],
        "parent_dev_loss": RUN1C_SFT["parent_dev_loss"],
    }
    sft["planned_updates"] = -(-rows // (sft["batch_size"] * sft["accumulation_steps"]))
    raw.update(
        run="1c",
        model=asdict(smoke_model_1b(found.tokenizer)),
        stages=stages,
        limits={
            **raw["limits"],
            "ceiling_usd": 1.5,
            "gate_estimate_usd": None,
            "cost_limit_usd": None,
        },
        planning={**raw["planning"], "format_gate_seconds": 5},
        sft=sft,
    )
    estimate = run1c_estimate_usd(raw)
    raw["limits"].update(
        gate_estimate_usd=estimate, cost_limit_usd=estimate_limit_usd(1.5, estimate)
    )
    return raw


def failing_gate_1c(raw: dict[str, Any]) -> None:
    """Run 1c's smoke SFT at the real peak rate and weight decay: too little for the random
    parent to learn the format in one pass, so the gate fails and the evaluation never runs."""
    raw["sft"].update(learning_rate=6e-05, weight_decay=0.1)


def make_worker_1c(
    found: Fixtures,
    root: Path,
    *,
    bucket: MemoryBucket,
    changes: Callable[[dict[str, Any]], None] | None = None,
    limits: RunLimits | None = None,
    pod_created: float | None = None,
    **options: Any,
) -> Worker:
    """A CPU worker for run 1c on its smoke fixtures, its report carrying the environment
    record the Pod run uploads, built by the Pod run's own ``run_extra`` from a receipt that
    names the smoke image; ``changes`` edits the specification."""
    from tools.posttrain_run import run_extra

    raw = smoke_specification_1c(found)
    if changes is not None:
        changes(raw)
    spec = Specification.from_dict(raw)
    extra = run_extra(spec, {"pod_image": SMOKE_IMAGE}, options.pop("extra", {}))
    return Worker(
        spec,
        root,
        inputs=found.inputs,
        store=bucket.store(SMOKE_CREDENTIALS),
        pod_id=SMOKE_POD_ID_1C,
        pod_created=time.time() if pod_created is None else pod_created,
        limits=limits,
        extra=extra,
        **options,
    )


def smoke_1c(
    output: Path,
    *,
    bucket: MemoryBucket | None = None,
    changes: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run 1c on CPU, offline, into an in-memory bucket; return a summary."""
    torch.set_num_threads(1)
    if output.exists() or output.is_symlink():
        raise FileExistsError("smoke output must be a new directory")
    found = fixtures_1c(output / "inputs")
    bucket = MemoryBucket() if bucket is None else bucket
    worker = make_worker_1c(found, output / "run", bucket=bucket, changes=changes)
    with patched_benchmarks():
        report = worker.execute()
    sft = report.get("sft") or {}
    return {
        **summary(report, bucket),
        "format_gate": report.get("format_gate"),
        "sft_kept_weights": sft.get("kept_weights"),
        "sft_quarters": [entry["update"] for entry in sft.get("quarters", [])],
        "sft_parent_dev_loss": sft.get("parent_dev_loss"),
        "sft_dev_loss_change": sft.get("dev_loss_change"),
        "limits": {
            key: report["limits"][key]
            for key in ("gate_estimate_usd", "limit_usd", "worker_cost_stop_usd", "wall_seconds")
        },
        "environment_uploaded": any(
            key.endswith("/environment/environment.json") for key in bucket.objects
        ),
    }
