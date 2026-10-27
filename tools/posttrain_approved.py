"""The owner-approved definition of post-training run 1, in one place.

``run1_specification()`` is ``configs/rehearsal-a40-s-post-1.json`` (a test keeps the two
equal), and ``check_run1`` is what the readiness receipt applies to the specification it is
given: every approved value must be there unchanged. The approved values come from the
post-training proposal as approved on 2026-10-26 and the owner's later rulings:

- **Ceiling:** $3.00 at the quoted $0.59/h GPU and $0.60/h total, so the worker stops at
  $3.00, the watchdog at $2.99 and the in-Pod timer at 17,820 s (``posttrain_ops``).
- **Probe:** batch 1, 64, 128 and 256 on 256 real chat-v1 prompt lengths around the pool dev
  median, top-p 0.95 at temperature 0.8, bf16; it stops the run below 5,000 generated tok/s
  at batch 64 or 10,000 at batch 256.
- **ARC-Easy:** label log-likelihood, float32, on run A's ``stage1-final`` and
  ``stage2-final`` and run B's ``mixed-final``, then on the selected SFT weights.
- **Evaluation:** pool dev and test greedy plus n = 8 at temperature 0.8, top-p 0.95, seed 0;
  HumanEval, HumanEval+ (task 32 excluded) and MBPP+ greedy. Base B under
  ``base-pool-completion-v1`` and ``base-completion-rstrip-v1``; SFT under ``chat-v1``.
- **SFT:** one pass over the 3,390 rows of record, 16 x 2 rows per update, peak 6e-5, 3%
  warmup, cosine to 10%, a fresh AdamW (0.9 / 0.95, weight decay 0.1, clip 1.0), bf16 with
  fused attention and compile; dev loss at each quarter, the lowest kept; a speed check
  after 50 updates against 90% of 56,765 slots/s.
- **RL screening:** the 1,000 candidates (AceCode excluded), 8 samples at temperature 1.0 and
  top-p 1.0, at most min(256, 1,024 - prompt) new tokens, ``SCREENING_SEED``.
- **Inputs:** run B's ``mixed-final`` (SHA-256 56339e3f...), run A's two finals (labels and
  SHA-256s from the runs' controller records), the tokenizer of the S build, the SFT data of
  record, the RL candidate list, the pinned benchmark releases and the Pod exports.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scratchcoder.data import evalplus, humaneval
from scratchcoder.eval import protocols
from scratchcoder.rl.rollout import SCREENING_SEED

SPECIFICATION_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "rehearsal-a40-s-post-1.json"
)
PARENTS = ("a-stage1-final", "a-stage2-final", "b-mixed-final")
POOL_TASKS = 2_000
BENCHMARK_TASKS = {"humaneval": 164, "humaneval-plus": 163, "mbpp-plus": 378}

LIMITS = {
    "ceiling_usd": 3.0,
    "quoted_gpu_hourly": 0.59,
    "quoted_total_hourly": 0.6,
    "self_destruct_margin_seconds": 180,
    "watchdog_margin_usd": 0.01,
    "mac_idle_seconds": 600,
    "pod_idle_seconds": 900,
    "max_transfer_seconds": 900,
    "finalization_reserve_seconds": 900,
    "startup_seconds": 1_080,
}
PLANNING = {
    "generation_tokens_per_second": {"64": 5_000.0, "256": 10_000.0},
    "training_slots_per_second": 56_765.0,
    "forward_tokens_per_second": 94_862.0,
    "probe_seconds": 360,
    "arc_seconds_per_checkpoint": 54,
    "checkpoint_load_seconds": 20,
    "compile_seconds": 120,
    "weights_save_seconds": 5,
}
"""Planning rates: the proposal's planning speeds and run B's measured training and forward
rates; the stage seconds are estimates for the launch rule and the projection only."""
PROBE = {
    "batch_sizes": [1, 64, 128, 256],
    "prompt_lengths": 256,
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "precision": "bfloat16",
    "seed": 0,
    "repeats": 1,
    "minimum_tokens_per_second": {"64": 5_000.0, "256": 10_000.0},
}
GENERATION = {"batch_size": 64, "precision": "bfloat16"}
SFT = {
    "train_rows": 3_390,
    "planned_dev_tokens": 827_850,
    "learning_rate": 6e-05,
    "warmup_fraction": 0.03,
    "min_lr_ratio": 0.1,
    "schedule": "cosine",
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "batch_size": 16,
    "accumulation_steps": 2,
    "sequence_length": 1_024,
    "seed": 0,
    "passes": 1,
    "precision": "bfloat16",
    "attention": "fused",
    "optimizer_foreach": True,
    "compile": True,
    "dev_batch_size": 16,
    "speed_check_updates": 50,
    "speed_check_warmup_updates": 2,
    "reference_slots_per_second": 56_765.0,
    "minimum_speed_fraction": 0.9,
}
SCREENING = {
    "samples": 8,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
    "seed_base": SCREENING_SEED,
    "planned_tasks": 1_000,
    "chunk_tasks": 64,
}
MODEL = {
    "vocab_size": 65_536,
    "width": 768,
    "layers": 12,
    "heads": 12,
    "context": 1_024,
    "qk_norm": False,
}
INPUTS: dict[str, dict[str, Any]] = {
    "b-mixed-final": {
        "kind": "checkpoint",
        "key": "rehearsal-runs/B/sn8esbi4mfiqkb/mixed-final/checkpoint.pt",
        "size_bytes": 1_623_502_847,
        "sha256": "56339e3f4fa98dda348dc5c4d2a146443504da7959ef619d412d034a1d47c9e3",
    },
    "a-stage1-final": {
        "kind": "checkpoint",
        "key": "rehearsal-runs/A/g9id1yi7fzq5mq/stage1-final/checkpoint.pt",
        "size_bytes": 1_623_502_783,
        "sha256": "024759f3414f70bbc7a351d5c828e3e941bc2a0e6d20fde2f472d1c52dbd9ed2",
    },
    "a-stage2-final": {
        "kind": "checkpoint",
        "key": "rehearsal-runs/A/g9id1yi7fzq5mq/stage2-final/checkpoint.pt",
        "size_bytes": 1_623_502_847,
        "sha256": "59a164f86ff08770eb739d6167909cc96d5143a0ac61b1c66d83410deea684b2",
    },
    "tokenizer": {
        "kind": "tokenizer",
        "key": "rehearsal-s-20261025/tokenizer/tokenizer.json",
        "size_bytes": 717_466,
        "sha256": "13e2eb08dc4930f645d5582b23f2b39a58c3735dbc264720b8c78fef1c21c23e",
    },
    "sft-train": {
        "kind": "sft-rows",
        "key": None,
        "size_bytes": 6_774_446,
        "sha256": "0d5467b21914318dec4183d7017e1b94cd30b0b3e155ea22aa960b907b62ab91",
    },
    "sft-code-dev": {
        "kind": "sft-rows",
        "key": None,
        "size_bytes": 3_484_529,
        "sha256": "757cffcbca3e1042ee5fea4d4326d71ac3d1000a96465bd5fa7e70e0faff3b9b",
    },
    "sft-language-dev": {
        "kind": "sft-rows",
        "key": None,
        "size_bytes": 573_795,
        "sha256": "22c84bf3a3a28e079818c9ce4ba81fb16495935b98c4b58952f4592d2d6dbe99",
    },
    "rl-candidates-list": {
        "kind": "candidate-list",
        "key": None,
        "size_bytes": 102_662,
        "sha256": "8301ff29d7d5e854f8d75050f3d0e0a43a4ac8cd0375d92a5c63e5da893005fe",
    },
    "pool-dev": {
        "kind": "pool-export",
        "key": None,
        "size_bytes": 3_912_528,
        "sha256": "8e59c314a9918235ecff65ad3f2a82f684557d2f06bd2f6f9191a2e282a76045",
    },
    "pool-test": {"kind": "pool-export", "key": None, "size_bytes": None, "sha256": None},
    "rl-candidates": {
        "kind": "candidates-export",
        "key": None,
        "size_bytes": 2_161_291,
        "sha256": "936412a05db2d7a3174e42e1c058230ec0cae125716c75df9c049134d7f48181",
    },
    "humaneval": {
        "kind": "benchmark",
        "key": None,
        "size_bytes": humaneval.DATA_BYTES,
        "sha256": humaneval.DATA_SHA256,
    },
    "humaneval-plus": {
        "kind": "benchmark",
        "key": None,
        "size_bytes": evalplus.release("humaneval-plus").manifest.size_bytes,
        "sha256": evalplus.release("humaneval-plus").manifest.sha256,
    },
    "mbpp-plus": {
        "kind": "benchmark",
        "key": None,
        "size_bytes": evalplus.release("mbpp-plus").manifest.size_bytes,
        "sha256": evalplus.release("mbpp-plus").manifest.sha256,
    },
    "arc-easy-test": {
        "kind": "arc-export",
        "key": None,
        "size_bytes": 1_090_258,
        "sha256": "56fd1ec0da7497b3399cb6e3f2f0900887f236eba4c9414bc4271f4702b82870",
    },
}
"""Every input. The pool test export is pinned only at the go: the test split is sealed, and
the runbook exports it then. Every other pin was computed from files that may be read."""


def eval_files(base: bool, *, pool_tasks: int = POOL_TASKS) -> list[dict[str, Any]]:
    """The seven completions files of an evaluation stage under the base or chat protocols."""
    pool = protocols.BASE_POOL if base else protocols.CHAT
    benchmark = protocols.BASE_BENCHMARK if base else protocols.CHAT
    files = []
    for split in ("dev", "test"):
        for name, samples, temperature, top_p in (("greedy", 1, 0.0, 1.0), ("n8", 8, 0.8, 0.95)):
            files.append(
                {
                    "name": f"pool-{split}-{name}",
                    "suite": "pool",
                    "split": split,
                    "tasks": f"pool-{split}",
                    "protocol": pool,
                    "samples": samples,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": 0,
                    "planned_tasks": pool_tasks,
                }
            )
    for suite, tasks in BENCHMARK_TASKS.items():
        files.append(
            {
                "name": f"{suite}-greedy",
                "suite": suite,
                "split": "eval",
                "tasks": suite,
                "protocol": benchmark,
                "samples": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
                "planned_tasks": tasks,
            }
        )
    return files


def run1_stages() -> list[dict[str, Any]]:
    """Run 1's stages in order."""
    return [
        {"name": "probe", "kind": "probe", "checkpoint": "b-mixed-final", "prompts": "pool-dev"},
        {
            "name": "arc-base",
            "kind": "arc",
            "checkpoints": list(PARENTS),
            "questions": "arc-easy-test",
        },
        {
            "name": "eval-base",
            "kind": "eval",
            "checkpoint": "b-mixed-final",
            "files": eval_files(True),
        },
        {
            "name": "sft",
            "kind": "sft",
            "parent": "b-mixed-final",
            "train": "sft-train",
            "dev": ["sft-code-dev", "sft-language-dev"],
        },
        {"name": "arc-sft", "kind": "arc", "checkpoints": ["sft"], "questions": "arc-easy-test"},
        {"name": "eval-sft", "kind": "eval", "checkpoint": "sft", "files": eval_files(False)},
        {
            "name": "screen",
            "kind": "screen",
            "checkpoint": "sft",
            "tasks": "rl-candidates",
            "candidates": "rl-candidates-list",
        },
    ]


def run1_specification() -> dict[str, Any]:
    """The run-1 specification of record."""
    return {
        "schema_version": 1,
        "kind": "posttrain-run-specification",
        "run": "1",
        "model": dict(MODEL),
        "limits": dict(LIMITS),
        "planning": json.loads(json.dumps(PLANNING)),
        "probe": json.loads(json.dumps(PROBE)),
        "generation": dict(GENERATION),
        "sft": dict(SFT),
        "screening": dict(SCREENING),
        "stages": run1_stages(),
        "inputs": json.loads(json.dumps(INPUTS)),
        "r2": {"storage_cap_usd_per_month": 5.0},
    }


APPROVED_SECTIONS = ("model", "limits", "probe", "generation", "sft", "screening", "stages", "r2")
"""Sections that must equal the approved run exactly. ``planning`` holds estimates; its two
planning speeds and the measured training and forward rates are checked one by one."""


def check_run1(raw: Mapping[str, Any]) -> None:
    """Refuse a run-1 specification that differs from the approved one in any approved value.

    Every approved section must be equal; the planning speeds and the measured rates must be
    the approved ones; every input pin the approved specification holds must be equal, and
    the pool test export's pin may be filled in at the go only.
    """
    approved = run1_specification()
    if raw.get("run") != "1" or raw.get("kind") != approved["kind"]:
        raise ValueError("not a run-1 post-training specification")
    differ = [name for name in APPROVED_SECTIONS if raw.get(name) != approved[name]]
    planning = raw.get("planning", {})
    for name in (
        "generation_tokens_per_second",
        "training_slots_per_second",
        "forward_tokens_per_second",
    ):
        if planning.get(name) != approved["planning"][name]:
            differ.append(f"planning.{name}")
    inputs = raw.get("inputs", {})
    if set(inputs) != set(approved["inputs"]):
        differ.append("inputs")
    else:
        for name, pin in approved["inputs"].items():
            given = inputs[name]
            for field, value in pin.items():
                if value is not None and given.get(field) != value:
                    differ.append(f"inputs.{name}.{field}")
                elif value is None and name != "pool-test" and given.get(field) is not None:
                    differ.append(f"inputs.{name}.{field}")
    if differ:
        raise ValueError(f"the specification differs from the approved run 1 in {differ}")


def write_specification(path: Path = SPECIFICATION_PATH) -> None:
    """Write ``run1_specification()`` in the repository's JSON style."""
    path.write_text(json.dumps(run1_specification(), indent=2) + "\n", encoding="utf-8")
