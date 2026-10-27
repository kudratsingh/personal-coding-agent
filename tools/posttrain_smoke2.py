"""CPU smoke of post-training run 2: RL with a Mac bridge server thread, the control, ARC, eval.

``smoke`` runs every stage of run 2 (RL, the more-SFT control, ARC-Easy on both selected
checkpoints, the seven evaluation files of each) and run 1's finalization on CPU, offline, into
an in-memory bucket. The Mac's side runs in a thread: a real ``BridgeServer`` over the
directory transport (``DirectoryTransport``, the fake SSH transport with the remote program's
semantics), dated by the Pod's clock, serving the Pod worker's real ``BridgeClient``. The
fixtures stand in for the real inputs:

- run 1's selected SFT weights: a 1-layer model saved weights-only, as run 1 saves them;
- the smoke tokenizer, pool exports, benchmark files and ARC questions of run 1's smoke;
- authored train and dev tasks whose hidden tests carry markers (``HIDDEN_MARKERS``), a kept
  screening record, and the three Pod files ``scratchcoder rl-pod-export`` writes from them,
  so the worker holds withheld tasks and the Mac's facts, never a hidden test;
- scripted programs (``scripted_programs``): the model generates as usual, and each completion
  is then replaced by a program for its task, correct for even seeds and for greedy, so rewards
  vary and dev checks pass and fail. Without it a 1-layer random model writes no program and
  no dev check would cross the bridge.

By default the Mac scores with fakes (``FakeRewardSource``, a dev executor that compares each
program with the task's reference); with ``docker`` the real ``LocalRewardSource`` and
``SandboxDevExecutor`` run every program in the pinned sandbox image. No fixture is real data.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from scratchcoder.data.chat import PYTHON_END, PYTHON_START
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import Completion, generate_many
from scratchcoder.eval.tasks import Task, save_tasks
from scratchcoder.model.gpt import GPT
from scratchcoder.rl import dev as dev_module
from scratchcoder.rl import rollout as rollout_module
from scratchcoder.rl.bridge_mac import BridgeServer, DirectoryTransport, SandboxDevExecutor
from scratchcoder.rl.checkpoint import save_weights
from scratchcoder.rl.fakes import FakeRewardSource
from scratchcoder.rl.pod_export import write_rl_pod_export
from scratchcoder.rl.rewards import LocalRewardSource, RewardSource
from scratchcoder.rl.rollout import ROLLOUT_SEED
from scratchcoder.tokenizer.bpe import ByteBPE
from scratchcoder.tokenizer.bulk import FastEncoder
from tools.posttrain_approved import bridge_section, run2_stages
from tools.posttrain_run import RunLimits, Specification
from tools.posttrain_run2 import Run2Worker
from tools.posttrain_smoke import (
    SMOKE_TASKS,
    Fixtures,
    fixtures,
    patched_benchmarks,
    smoke_model,
    summary,
)
from tools.rehearsal_run import SMOKE_CREDENTIALS, MemoryBucket

SMOKE2_POD_ID = "smoke-posttrain-2"
RUN1_SFT_KEY = "posttrain-runs/1/smoke-posttrain-1/sft-weights/checkpoint.pt"
HIDDEN_MARKERS = tuple(str(4_242_400 + index) for index in range(12))
"""Each fixture task's hidden tests carry one of these numbers, found nowhere else: a scan of
every file the Pod side holds for them finds a leaked hidden test."""
KEPT = 6
DEV_TASKS = 3


def _function_task(identifier: str, split: str, index: int, marker: str) -> Task:
    return Task(
        id=identifier,
        prompt=f"Write add_{index}(x) returning x + {index}.",
        public_tests=f"assert add_{index}(1) == {1 + index}",
        hidden_tests=f"assert add_{index}({marker}) == {int(marker) + index}\n"
        f"assert add_{index}(0) == {index}",
        source="synthetic-smoke",
        license="fixture",
        written_by="model",
        split=split,
        solution=f"def add_{index}(x):\n    return x + {index}",
    )


def _io_task(identifier: str, split: str, index: int, marker: str) -> Task:
    cases = [
        {"input": f"{marker}\n", "output": f"{int(marker) + index}\n"},
        {"input": "2\n", "output": f"{2 + index}\n"},
    ]
    return Task(
        id=identifier,
        prompt=f"Read n and print n + {index}.",
        public_tests=json.dumps([{"input": "5\n", "output": f"{5 + index}\n"}]),
        hidden_tests=json.dumps(cases),
        source="synthetic-smoke",
        license="fixture",
        written_by="human",
        split=split,
        mode="io",
        solution=f"print(int(input()) + {index})",
    )


@dataclass(frozen=True)
class Fixtures2:
    root: Path
    run1: Fixtures
    tokenizer: ByteBPE
    inputs: dict[str, Path]
    """The Pod's inputs: withheld task files, the facts, exports, weights, tokenizer."""
    mac: dict[str, Path]
    """The Mac's side: the full train and dev files and the kept screening record."""
    kept: tuple[Task, ...]
    dev: tuple[Task, ...]
    """The full records the Mac scores against (the kept tasks and the dev subset)."""


def fixtures2(root: Path) -> Fixtures2:
    """Write every run-2 input of the smoke under ``root``; return their paths."""
    run1 = fixtures(root / "run1")
    tokenizer = run1.tokenizer
    config = smoke_model(tokenizer)
    torch.manual_seed(21)
    sft = root / "sft-weights" / "checkpoint.pt"
    save_weights(
        sft, GPT(config), tokenizer_hash=tokenizer.sha256, run_id="posttrain-1-sft", step=3
    )
    markers = iter(HIDDEN_MARKERS)
    train = [_function_task(f"smoke2/train/{i}", "train", 50 + i, next(markers)) for i in range(4)]
    train += [_io_task(f"smoke2/train/{i}", "train", 50 + i, next(markers)) for i in range(4, 7)]
    dev = [_function_task(f"smoke2/dev/{i}", "dev", 70 + i, next(markers)) for i in range(3)]
    dev += [_io_task(f"smoke2/dev/{i}", "dev", 70 + i, next(markers)) for i in range(3, 5)]
    mac = {
        "train": root / "mac" / "train.jsonl",
        "dev": root / "mac" / "dev.jsonl",
        "kept": root / "mac" / "screening.json",
    }
    save_tasks(mac["train"], train)
    save_tasks(mac["dev"], dev)
    kept_ids = [task.id for task in train[:KEPT]]
    mac["kept"].write_text(
        json.dumps({"format": "rl-screening-v1", "outcome": "kept", "kept": kept_ids})
    )
    export = root / "rl-pod-export"
    write_rl_pod_export(
        train_tasks=mac["train"],
        kept=mac["kept"],
        dev_tasks=mac["dev"],
        output=export,
        dev_size=DEV_TASKS,
    )
    from scratchcoder.rl.dev import dev_subset

    inputs = {
        "tokenizer": run1.inputs["tokenizer"],
        "sft-weights": sft,
        "rl-tasks": export / "rl-tasks.jsonl",
        "rl-dev": export / "rl-dev.jsonl",
        "rl-facts": export / "rl-facts.json",
        **{
            name: run1.inputs[name]
            for name in (
                "pool-dev",
                "pool-test",
                "humaneval",
                "humaneval-plus",
                "mbpp-plus",
                "arc-easy-test",
            )
        },
    }
    return Fixtures2(
        root,
        run1,
        tokenizer,
        inputs,
        mac,
        tuple(train[:KEPT]),
        tuple(dev_subset(dev, DEV_TASKS)),
    )


KINDS2 = {
    "tokenizer": "tokenizer",
    "sft-weights": "checkpoint",
    "rl-tasks": "rl-tasks",
    "rl-dev": "rl-dev-tasks",
    "rl-facts": "rl-facts",
    "pool-dev": "pool-export",
    "pool-test": "pool-export",
    "humaneval": "benchmark",
    "humaneval-plus": "benchmark",
    "mbpp-plus": "benchmark",
    "arc-easy-test": "arc-export",
}
EXISTING2 = {"tokenizer": "smoke-build/tokenizer/tokenizer.json", "sft-weights": RUN1_SFT_KEY}
SMOKE_RL: dict[str, Any] = {
    "updates": 3,
    "tasks_per_update": 2,
    "group_size": 4,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 16,
    "learning_rate": 1e-3,
    "warmup_updates": 1,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "micro_batch": 4,
    "generation_batch": 4,
    "optimizer_foreach": False,
    "shuffle_seed": 0,
    "rollout_seed": ROLLOUT_SEED,
    "dev_checks": [0, 2, 3],
    "patience": 3,
    "collapse_distinct": 2.0,
    "collapse_updates": 5,
    "leakage_rate": 0.01,
    "leakage_min_rewarded": 100,
    "check_at_stop": True,
    "check_at_external_stop": False,
    "attention": "reference",
    "dev_tasks": DEV_TASKS,
    "dev_samples": 2,
    "dev_batch_size": 4,
    "min_tasks": 2,
}
SMOKE_CONTROL: dict[str, Any] = {
    "batch_size": 2,
    "accumulation_steps": 2,
    "learning_rate": 6e-05,
    "warmup_updates": 1,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "seed": 0,
    "pass_checks": [1, 2],
    "max_passes": 50,
    "tolerance": 0.05,
    "optimizer_foreach": False,
    "check_at_stop": True,
    "check_at_external_stop": False,
}
SMOKE_BRIDGE: dict[str, Any] = {
    "root": "/workspace/posttrain-bridge",
    "workers": 8,
    "mac_poll_seconds": 2.0,
    "pod_poll_seconds": 0.5,
    "ssh_timeout_seconds": 120.0,
    "ssh_list_timeout_seconds": 30.0,
    "ssh_retries": 4,
    "awaiting_slack_seconds": 60.0,
}


def smoke_specification2(
    found: Fixtures2, *, rl: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Run 2's specification at the smoke's scale, every gate value filled from the files."""
    rl_section = dict(SMOKE_RL if rl is None else rl)
    inputs = {}
    for name, kind in KINDS2.items():
        path = found.inputs[name]
        inputs[name] = {
            "kind": kind,
            "key": EXISTING2.get(name),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    stages = run2_stages()
    for stage in stages:
        for entry in stage.get("files", []):
            entry["planned_tasks"] = SMOKE_TASKS[entry["suite"]]
    return {
        "schema_version": 1,
        "kind": "posttrain-run-specification",
        "run": "2",
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
            "startup_seconds": 900,
        },
        "planning": {
            "generation_tokens_per_second": {"4": 50.0},
            "training_slots_per_second": 100.0,
            "arc_seconds_per_checkpoint": 5,
            "checkpoint_load_seconds": 1,
            "weights_save_seconds": 1,
            "reward_wait_seconds_per_update": 1,
            "dev_wait_seconds_per_check": 1,
        },
        "generation": {"batch_size": 4, "precision": "float32"},
        "rl": rl_section,
        "control": dict(SMOKE_CONTROL),
        "bridge": bridge_section(SMOKE_BRIDGE, rl_section),
        "stages": stages,
        "inputs": inputs,
        "r2": {"storage_cap_usd_per_month": 5.0},
    }


# Scripted programs -----------------------------------------------------------------------------

_FUNCTION = re.compile(r"Write add_(\d+)\(x\)")
_IO = re.compile(r"print n \+ (\d+)\.")


def program_for(prompt: str, seed: int | None) -> str:
    """A program for the task in ``prompt``: correct for greedy (``seed`` None) and for an even
    seed, off by one otherwise (two characters longer, so the fake reward, which reads the
    length, differs too); a no-op for a prompt it does not know."""
    wrong = "+1" if seed is not None and seed % 2 == 1 else ""
    found = _FUNCTION.search(prompt)
    if found is not None:
        index = int(found.group(1))
        return f"def add_{index}(x):\n    return x + {index}{wrong}"
    found = _IO.search(prompt)
    if found is not None:
        index = int(found.group(1))
        return f"print(int(input()) + {index}{wrong})"
    return "pass"


@contextmanager
def scripted_programs(tokenizer: ByteBPE) -> Iterator[None]:
    """Let the model generate as usual, then replace each RL and dev-check completion by a
    closed python block holding ``program_for`` its prompt (see the module docstring)."""
    encoder = FastEncoder(tokenizer)
    original = generate_many  # The real one: both modules call it by this name.

    def generate(model: GPT, prompts: Sequence[Sequence[int]], **options: Any) -> list[Completion]:
        completions = original(model, prompts, **options)
        seeds = options.get("seeds")
        rows = []
        for index, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            text = tokenizer.decode(prompt, skip_special=True)
            code = program_for(text, None if seeds is None else int(seeds[index]))
            tokens = [PYTHON_START, *encoder.encode(code), PYTHON_END]
            rows.append(Completion(tokens, "stop", completion.budget))
        return rows

    with (
        mock.patch.object(rollout_module, "generate_many", generate),
        mock.patch.object(dev_module, "generate_many", generate),
    ):
        yield


# The Mac's side --------------------------------------------------------------------------------


@dataclass
class ReferenceDev:
    """The Mac's fake dev executor: a program passes when it is its task's reference."""

    calls: int = 0

    def __call__(self, pairs: Sequence[tuple[Task, str]]) -> list[bool]:
        self.calls += 1
        return [code.strip() == (task.solution or "").strip() for task, code in pairs]


@dataclass
class MacSide:
    server: BridgeServer
    stop: threading.Event
    thread: threading.Thread
    errors: list[BaseException]


def mac_server(
    found: Fixtures2,
    spec: Specification,
    *,
    pod_root: Path,
    state: Path,
    wall: Callable[[], float] = time.time,
    docker: bool = False,
    reward_source: RewardSource | None = None,
) -> BridgeServer:
    """The Mac's ``BridgeServer`` for the smoke's Pod, with the specification's Mac bounds."""
    bridge = spec.bridge
    if docker:
        rewards: RewardSource = LocalRewardSource(
            found.kept, timeout=float(bridge["mac_reward_timeout_seconds"])
        )
        dev: Any = SandboxDevExecutor(timeout=float(bridge["mac_dev_timeout_seconds"]))
    else:
        rewards = reward_source or FakeRewardSource()
        dev = ReferenceDev()
    return BridgeServer(
        DirectoryTransport(pod_root, wall=wall),
        reward_tasks=found.kept,
        dev_tasks=found.dev,
        pod_task_sha256={
            "reward": file_sha256(found.inputs["rl-tasks"]),
            "dev": file_sha256(found.inputs["rl-dev"]),
        },
        state=state,
        reward_source=rewards,
        dev_execute=dev,
        reward_timeout=float(bridge["mac_reward_timeout_seconds"]),
        dev_timeout=float(bridge["mac_dev_timeout_seconds"]),
        wall=wall,
    )


@contextmanager
def mac_thread(server: BridgeServer, *, poll: float = 0.01) -> Iterator[MacSide]:
    """Serve in a thread until the block ends; re-raise anything the server raised."""
    stop = threading.Event()
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            server.serve(poll_seconds=poll, should_stop=stop.is_set, progress_seconds=0.05)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=serve, name="smoke-mac-bridge", daemon=True)
    thread.start()
    side = MacSide(server, stop, thread, errors)
    try:
        yield side
    finally:
        stop.set()
        thread.join(timeout=60)
        server.close()
    if thread.is_alive():
        raise RuntimeError("the Mac bridge thread did not stop")
    if errors:
        raise errors[0]


# The worker ------------------------------------------------------------------------------------


def make_run2_worker(
    found: Fixtures2,
    root: Path,
    *,
    bucket: MemoryBucket,
    changes: Callable[[dict[str, Any]], None] | None = None,
    limits: RunLimits | None = None,
    pod_created: float | None = None,
    **options: Any,
) -> Run2Worker:
    """A CPU worker for run 2 on the smoke fixtures; ``changes`` edits the specification."""
    raw = smoke_specification2(found)
    if changes is not None:
        changes(raw)
    spec = Specification.from_dict(raw)
    clock = options.get("clock", time.time)
    options.setdefault("bridge_root", root.parent / f"{root.name}-bridge")
    return Run2Worker(
        spec,
        root,
        inputs=found.inputs,
        store=bucket.store(SMOKE_CREDENTIALS),
        pod_id=SMOKE2_POD_ID,
        pod_created=clock() if pod_created is None else pod_created,
        limits=limits,
        **options,
    )


def smoke(
    output: Path,
    *,
    bucket: MemoryBucket | None = None,
    docker: bool = False,
    changes: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run 2 on CPU, offline, into an in-memory bucket, the Mac in a thread; return a summary."""
    torch.set_num_threads(1)
    if output.exists() or output.is_symlink():
        raise FileExistsError("smoke output must be a new directory")
    found = fixtures2(output / "inputs")
    bucket = MemoryBucket() if bucket is None else bucket
    worker = make_run2_worker(found, output / "run", bucket=bucket, changes=changes)
    server = mac_server(
        found, worker.plan, pod_root=worker.bridge_root, state=output / "mac-state", docker=docker
    )
    with patched_benchmarks(), scripted_programs(found.tokenizer), mac_thread(server):
        report = worker.execute()
    return summary2(report, bucket)


def summary2(report: Mapping[str, Any], bucket: MemoryBucket) -> dict[str, Any]:
    value = summary(report, bucket)
    value.pop("probe_rates", None)
    value.pop("sft_selected", None)
    rl = report.get("rl") or {}
    control = report.get("more_sft") or {}
    selection = report.get("selection") or {}
    value.update(
        rl={
            key: rl.get(key)
            for key in ("stop_reason", "updates", "optimizer_steps", "busy_seconds")
        },
        more_sft={
            key: control.get(key)
            for key in (
                "stop_reason",
                "updates",
                "budget_seconds",
                "busy_seconds",
                "mismatch_fraction",
            )
        },
        selected={
            arm: ((selection.get(arm) or {}).get("selected") or {}).get("label")
            for arm in ("rl", "more-sft")
        },
        identical_endpoints=(selection.get("identical_endpoints") or {}).get("identical"),
        bridge_waits=len((report.get("bridge") or {}).get("waits", [])),
    )
    return value
