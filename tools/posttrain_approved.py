"""The owner-approved definitions of post-training runs 1 and 2, in one place.

Run 2 (below run 1's values): ``run2_specification()`` is the tracked template
``configs/rehearsal-a40-s-post-2.json``, every gate value ``null`` (the ceiling, the gate's
estimate and the cost limit it derives, run 1's selected SFT weights and its SFT evaluation
artifacts, the three ``rl-pod-export`` files and the pool test export);
``python -m tools.posttrain_approved run2-gate`` fills them from the gate's own files and writes
the launch specification, and ``check_run2`` is what the readiness receipt applies to it. Its
protocol is the proposal's, frozen at approval (RL, the more-SFT control, ARC-Easy and the
seven evaluation files of each selected checkpoint), with the owner's later rulings: the
1,000 RL candidates without AceCode and the reward-wait bounds derived from the channel's
budgets (``posttrain_ops.bridge_timeouts``) in place of a flat 420 s.

``run1_specification()`` is ``configs/rehearsal-a40-s-post-1.json`` (a test keeps the two
equal), and ``check_run1`` is what the readiness receipt applies to the specification it is
given: every approved value must be there unchanged. The approved values come from the
post-training proposal as approved on 2026-10-26 and the owner's later rulings:

- **Ceiling:** $3.00 at the quoted $0.59/h GPU and $0.60/h total, so the worker stops at
  $3.00, the watchdog at $2.99 and the in-Pod timer at 17,820 s (``posttrain_ops``).
- **Probe:** batch 1, 64, 128 and 256 on 256 real chat-v1 prompt lengths around the pool dev
  median, top-p 0.95 at temperature 0.8, bf16; it stops the run below 8,279 generated tok/s
  at batch 128 or 9,268 at batch 256 (``GENERATION_FLOORS``).
- **Generation:** evaluation and screening at batch 128, in bf16; run 2's evaluation shares it.
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

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scratchcoder.data import evalplus, humaneval
from scratchcoder.eval import protocols
from scratchcoder.rl.rollout import ROLLOUT_SEED, SCREENING_SEED

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
GENERATION_FLOORS = {"128": 8_279.0, "256": 9_268.0}
"""The probe's floors and the planning generation speeds of both runs: 5% under the rates run
1's first launch measured, rounded down (8,714.9 tok/s at batch 128 and 9,756.7 at batch 256;
it measured 4,607 at batch 64, below that launch's 5,000 floor, and nothing generates at 64)."""
PLANNING = {
    "generation_tokens_per_second": dict(GENERATION_FLOORS),
    "training_slots_per_second": 56_765.0,
    "forward_tokens_per_second": 94_862.0,
    "probe_seconds": 360,
    "arc_seconds_per_checkpoint": 54,
    "checkpoint_load_seconds": 20,
    "compile_seconds": 120,
    "weights_save_seconds": 5,
}
"""Planning rates: the probe's floors as the generation speeds and run B's measured training and
forward rates; the stage seconds are estimates for the launch rule and the projection only."""
PROBE = {
    "batch_sizes": [1, 64, 128, 256],
    "prompt_lengths": 256,
    "max_new_tokens": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "precision": "bfloat16",
    "seed": 0,
    "repeats": 1,
    "minimum_tokens_per_second": dict(GENERATION_FLOORS),
}
GENERATION = {"batch_size": 128, "precision": "bfloat16"}
"""Evaluation and screening generation. Each row samples with its own seed, so the batch
changes only how rows are grouped, not which seed a row uses."""
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


# Run 2 --------------------------------------------------------------------------------------

SPECIFICATION_PATH_2 = SPECIFICATION_PATH.with_name("rehearsal-a40-s-post-2.json")
RUN2_LIMITS: dict[str, Any] = {
    **LIMITS,
    "ceiling_usd": None,
    "gate_estimate_usd": None,
    "cost_limit_usd": None,
    "startup_seconds": 900,
}
"""Run 2's limits: the ceiling (about $5.80) and the estimate are fixed at the owner's gate
from run 1's measured rates, and the cost limit every other limit derives from is the lower of
the ceiling and 120% of the estimate (the owner's run 2 gate rulings); setup fetches only the
541 MB SFT weights, so its allowance is 900 s."""
RUN2_PLANNING = {
    "generation_tokens_per_second": dict(GENERATION_FLOORS),
    "training_slots_per_second": 56_765.0,
    "arc_seconds_per_checkpoint": 54,
    "checkpoint_load_seconds": 20,
    "weights_save_seconds": 5,
    "reward_wait_seconds_per_update": 48,
    "dev_wait_seconds_per_check": 300,
}
"""Run 1's probe floors at the batches run 2 generates at (128 for the evaluation, 256 for the
rollouts and the dev checks), run B's measured training rate, and the Mac's time: about 48 s
of rewards per update (256 attempts on 8 workers) and 300 s per 1,800-program dev check. The
gate's projection re-prices them from run 1's measured rates."""
RUN2_RL: dict[str, Any] = {
    "updates": 100,
    "tasks_per_update": 32,
    "group_size": 8,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 256,
    "learning_rate": 3e-06,
    "warmup_updates": 5,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "micro_batch": 16,
    "generation_batch": 256,
    "optimizer_foreach": True,
    "shuffle_seed": 0,
    "rollout_seed": ROLLOUT_SEED,
    "dev_checks": [0, 12, 25, 50, 75, 100],
    "patience": 3,
    "collapse_distinct": 2.0,
    "collapse_updates": 5,
    "leakage_rate": 0.01,
    "leakage_min_rewarded": 100,
    "check_at_stop": True,
    "check_at_external_stop": False,
    "attention": "fused",
    "dev_tasks": 200,
    "dev_samples": 8,
    "dev_batch_size": 256,
    "min_tasks": 320,
}
"""The frozen RL protocol: 100 updates of 32 tasks x G = 8 at temperature 1.0 and top-p 1.0,
at most min(256, 1,024 - prompt) new tokens; tasks shuffled with seed 0 and reshuffled each
pass; a fresh AdamW (0.9 / 0.95, weight decay 0.1) at 3e-6 after 5 warmup updates, clip 1.0,
no KL; dev checks at 0, 12, 25, 50, 75 and 100 on the 200-task subset (greedy and n = 8),
generated at batch 256 (the owner's run 2 gate rulings: every dev check of both arms, check 0
included, reads this one value); the collapse, patience-3 and 1% leakage stops. The kernel and
the optimizer path match SFT (fused attention, the foreach AdamW)."""
RUN2_CONTROL: dict[str, Any] = {
    "batch_size": 16,
    "accumulation_steps": 2,
    "learning_rate": 6e-05,
    "warmup_updates": 10,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-08,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "seed": 0,
    "pass_checks": [1, 2, 4, 8],
    "max_passes": 400,
    "tolerance": 0.05,
    "optimizer_foreach": True,
    "check_at_stop": True,
    "check_at_external_stop": False,
}
"""The more-SFT control: SFT's optimizer at a constant 6e-5 after 10 warmup updates, 16 x 2
rows of the kept tasks' references per update, checks after 1, 2, 4 and 8 passes and at the
budget's end (the start is RL's update-0 check), the budget RL's busy seconds within 5%."""
RUN2_BRIDGE_SETTINGS: dict[str, Any] = {
    "root": "/workspace/posttrain-bridge",
    "workers": 8,
    "mac_poll_seconds": 2.0,
    "pod_poll_seconds": 0.5,
    "ssh_timeout_seconds": 120.0,
    "ssh_list_timeout_seconds": 30.0,
    "ssh_retries": 4,
    "awaiting_slack_seconds": 60.0,
}
"""The bridge's settings, the inputs of ``posttrain_ops.bridge_timeouts``: the Pod's bridge
root, the Mac's 8 sandbox workers, both sides' poll intervals, the SSH timeouts and retries
(``scratchcoder rl-bridge`` defaults) and the watchdog's slack on top of a wait's timeout."""


def bridge_section(settings: Mapping[str, Any], rl: Mapping[str, Any]) -> dict[str, Any]:
    """A specification's ``bridge`` object: the settings and every bound they derive."""
    from tools.posttrain_ops import DERIVED_BRIDGE_KEYS, bridge_timeouts

    derived = bridge_timeouts(
        attempts_per_request=int(rl["tasks_per_update"]) * int(rl["group_size"]),
        dev_programs=int(rl["dev_tasks"]) * (int(rl["dev_samples"]) + 1),
        workers=int(settings["workers"]),
        mac_poll_seconds=float(settings["mac_poll_seconds"]),
        pod_poll_seconds=float(settings["pod_poll_seconds"]),
        ssh_timeout_seconds=float(settings["ssh_timeout_seconds"]),
        ssh_list_timeout_seconds=float(settings["ssh_list_timeout_seconds"]),
        ssh_retries=int(settings["ssh_retries"]),
        awaiting_slack_seconds=float(settings["awaiting_slack_seconds"]),
        pod_idle_seconds=float(LIMITS["pod_idle_seconds"]),
    )
    return {**settings, **{key: getattr(derived, key) for key in DERIVED_BRIDGE_KEYS}}


RUN2_INPUTS: dict[str, dict[str, Any]] = {
    "tokenizer": dict(INPUTS["tokenizer"]),
    "sft-weights": {"kind": "checkpoint", "key": None, "size_bytes": None, "sha256": None},
    "rl-tasks": {"kind": "rl-tasks", "key": None, "size_bytes": None, "sha256": None},
    "rl-dev": {"kind": "rl-dev-tasks", "key": None, "size_bytes": None, "sha256": None},
    "rl-facts": {"kind": "rl-facts", "key": None, "size_bytes": None, "sha256": None},
    "pool-dev": dict(INPUTS["pool-dev"]),
    "pool-test": dict(INPUTS["pool-test"]),
    "humaneval": dict(INPUTS["humaneval"]),
    "humaneval-plus": dict(INPUTS["humaneval-plus"]),
    "mbpp-plus": dict(INPUTS["mbpp-plus"]),
    "arc-easy-test": dict(INPUTS["arc-easy-test"]),
}
"""Run 2's inputs. Fixed at the gate (``null`` here): run 1's selected SFT weights (the
``sft-weights`` label of run 1's verified report: key, size, SHA-256), the three
``rl-pod-export`` files and the pool test export. The rest are run 1's pins."""
GATED_INPUTS = ("sft-weights", "rl-tasks", "rl-dev", "rl-facts", "pool-test")


def sft_results_template() -> dict[str, dict[str, Any]]:
    """Run 2's ``sft_results``, every pin ``null`` until the gate: run 1's ARC report and seven
    completions files of the SFT weights (``posttrain_run.SFT_RESULTS``). An arm whose selected
    weights equal the SFT weights reuses them in place of a duplicate evaluation."""
    from tools.posttrain_run import SFT_RESULTS

    return {name: {"key": None, "size_bytes": None, "sha256": None} for name in SFT_RESULTS}


def run2_stages() -> list[dict[str, Any]]:
    """Run 2's stages in order."""
    common = {"parent": "sft-weights", "tasks": "rl-tasks", "dev": "rl-dev", "facts": "rl-facts"}
    return [
        {"name": "rl", "kind": "rl", **common},
        {"name": "more-sft", "kind": "more-sft", **common},
        {
            "name": "arc-post",
            "kind": "arc",
            "checkpoints": ["rl", "more-sft"],
            "questions": "arc-easy-test",
        },
        {"name": "eval-rl", "kind": "eval", "checkpoint": "rl", "files": eval_files(False)},
        {
            "name": "eval-more-sft",
            "kind": "eval",
            "checkpoint": "more-sft",
            "files": eval_files(False),
        },
    ]


def run2_specification(gate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run 2's specification: the tracked template (``gate`` None, every gate value ``null``),
    or the launch specification with the gate's values (``fill_gate``)."""
    raw: dict[str, Any] = {
        "schema_version": 1,
        "kind": "posttrain-run-specification",
        "run": "2",
        "model": dict(MODEL),
        "limits": dict(RUN2_LIMITS),
        "planning": json.loads(json.dumps(RUN2_PLANNING)),
        "generation": dict(GENERATION),
        "rl": json.loads(json.dumps(RUN2_RL)),
        "control": json.loads(json.dumps(RUN2_CONTROL)),
        "bridge": bridge_section(RUN2_BRIDGE_SETTINGS, RUN2_RL),
        "stages": run2_stages(),
        "inputs": json.loads(json.dumps(RUN2_INPUTS)),
        "sft_results": sft_results_template(),
        "r2": {"storage_cap_usd_per_month": 5.0},
    }
    return raw if gate is None else fill_gate(raw, gate)


def _sha256_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.strip("0123456789abcdef"):
        raise ValueError(f"{name} must be 64 lowercase hex digits")
    return value


GATE_KEYS = ("ceiling_usd", "gate_estimate_usd", *GATED_INPUTS, "sft-results")
"""What the gate gives: the owner's ceiling, the projection's estimate, each gated pin and run
1's SFT evaluation artifacts (``sft-results``)."""


def fill_gate(raw: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """The template with the gate's values: ``ceiling_usd``, ``gate_estimate_usd`` and the
    cost limit they derive (``posttrain_ops.estimate_limit_usd``), for each gated input its pin
    (``sft-weights``: key, size_bytes, sha256; the others: size_bytes, sha256), and
    ``sft-results``, the pins of run 1's SFT evaluation artifacts (key, size_bytes, sha256)."""
    from tools.posttrain_ops import estimate_limit_usd

    if set(gate) != set(GATE_KEYS):
        raise ValueError(f"the gate gives exactly {list(GATE_KEYS)}")
    filled: dict[str, Any] = json.loads(json.dumps(raw))
    ceiling = gate["ceiling_usd"]
    if isinstance(ceiling, bool) or not isinstance(ceiling, int | float) or not 0 < ceiling:
        raise ValueError("the gate's ceiling must be a positive number of dollars")
    filled["limits"]["ceiling_usd"] = float(ceiling)
    estimate = gate["gate_estimate_usd"]
    if (
        isinstance(estimate, bool)
        or not isinstance(estimate, int | float)
        or not math.isfinite(estimate)
        or not 0 < estimate
    ):
        raise ValueError("the gate's estimate must be a positive, finite number of dollars")
    filled["limits"]["gate_estimate_usd"] = float(estimate)
    filled["limits"]["cost_limit_usd"] = estimate_limit_usd(float(ceiling), float(estimate))
    for name in GATED_INPUTS:
        pin = gate[name]
        fields = (
            {"key", "size_bytes", "sha256"} if name == "sft-weights" else {"size_bytes", "sha256"}
        )
        if not isinstance(pin, Mapping) or set(pin) != fields:
            raise ValueError(f"the gate's {name} pin gives exactly {sorted(fields)}")
        if type(pin["size_bytes"]) is not int or pin["size_bytes"] < 1:
            raise ValueError(f"the gate's {name} size must be a positive integer")
        _sha256_hex(pin["sha256"], f"the gate's {name} SHA-256")
        filled["inputs"][name].update(pin)
    results = gate["sft-results"]
    if not isinstance(results, Mapping) or set(results) != set(filled["sft_results"]):
        raise ValueError(f"the gate's sft-results pin exactly {sorted(filled['sft_results'])}")
    for name, pin in results.items():
        if not isinstance(pin, Mapping) or set(pin) != {"key", "size_bytes", "sha256"}:
            raise ValueError(f"the gate's sft-results {name} gives key, size_bytes and sha256")
        _sha256_hex(pin["sha256"], f"the gate's sft-results {name} SHA-256")
        filled["sft_results"][name] = dict(pin)
    from tools.posttrain_ops import specification_limits
    from tools.posttrain_run import check_sft_results

    check_sft_results(filled)  # Each pin is run 1's artifact beside its SFT weights.
    specification_limits(filled)  # The limits the cost limit derives must exist.
    fit = gate_launch_fit(filled)
    if not fit["fits"]:
        raise ValueError(
            f"the gate's cost limit of ${fit['limit_usd']:.2f} gives a "
            f"{fit['lifetime_seconds']:,.0f} s lifetime, which leaves "
            f"{fit['setup_allowance_seconds']:,.0f} s for setup after the "
            f"{fit['planning_seconds_needed']:,.0f} s the launch rule needs at the planning "
            f"rates, less than the planned {fit['startup_seconds']:,.0f} s: the readiness "
            "receipt would refuse the launch after the Pod has billed its setup; run 2 needs "
            "the owner's ruling (a higher ceiling or another stop) before this gate"
        )
    return filled


LAUNCH_AT_GATE_RULE = (
    "the gate's derived limits must hold the planned setup (limits.startup_seconds) plus the "
    "whole run at the planning rates (the launch rule's need), so a gate the readiness "
    "receipt would refuse after the Pod has billed its setup is refused before any billing"
)


def gate_launch_fit(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The launch rule at the gate: the lifetime the filled specification's limits give
    (``posttrain_ops.lifetime_seconds``) less the planned setup, against the run at the
    planning rates (``posttrain_run.launch_fit``). ``setup_allowance_seconds`` is what the
    limits leave for setup once the planning need is held."""
    from tools.posttrain_ops import lifetime_seconds
    from tools.posttrain_run import Specification, launch_fit

    spec = Specification.from_dict(raw)
    limits = spec.ceiling
    lifetime = lifetime_seconds(limits)
    startup = float(spec.limits["startup_seconds"])
    fit = launch_fit(spec, remaining_seconds=lifetime - startup)
    needed = float(fit["planning_seconds_needed"])
    return {
        "rule": LAUNCH_AT_GATE_RULE,
        "limit_usd": float(getattr(limits, "limit_usd", limits.worker_cost_stop_usd)),
        "lifetime_seconds": lifetime,
        "startup_seconds": startup,
        "planning_seconds_needed": needed,
        "setup_allowance_seconds": lifetime - needed,
        "fits": bool(fit["fits_at_planning"]),
    }


RUN2_APPROVED_SECTIONS = (
    "model",
    "planning",
    "generation",
    "rl",
    "control",
    "bridge",
    "stages",
    "r2",
)
"""Sections of run 2 that must equal the approved template exactly."""


def check_run2(raw: Mapping[str, Any]) -> None:
    """Refuse a run-2 launch specification that differs from the approved one, or in which any
    gate value is still ``null``: every approved section equal, the limits equal but for the
    gate's ceiling, estimate and cost limit, every input pin equal but for the gate's. The
    estimate must be a positive, finite number of dollars and the written cost limit exactly
    the lower of the ceiling and 120% of it (``posttrain_ops.specification_limits``)."""
    from tools.posttrain_ops import RUN2_GATE_LIMITS, specification_limits
    from tools.posttrain_run import check_sft_results, gate_nulls

    approved = run2_specification()
    if raw.get("run") != "2" or raw.get("kind") != approved["kind"]:
        raise ValueError("not a run-2 post-training specification")
    missing = gate_nulls(raw)
    if missing:
        raise ValueError(f"run 2's gate values are still null: {missing}")
    differ = [name for name in RUN2_APPROVED_SECTIONS if raw.get(name) != approved[name]]
    limits = raw.get("limits", {})
    if not isinstance(limits, Mapping) or {
        key: value for key, value in limits.items() if key not in RUN2_GATE_LIMITS
    } != {key: value for key, value in approved["limits"].items() if key not in RUN2_GATE_LIMITS}:
        differ.append("limits")
    inputs = raw.get("inputs", {})
    if not isinstance(inputs, Mapping) or set(inputs) != set(approved["inputs"]):
        differ.append("inputs")
    else:
        for name, pin in approved["inputs"].items():
            for field, value in pin.items():
                if value is not None and inputs[name].get(field) != value:
                    differ.append(f"inputs.{name}.{field}")
    if differ:
        raise ValueError(f"the specification differs from the approved run 2 in {differ}")
    specification_limits(raw)
    check_sft_results(raw)


def write_specification_2(path: Path = SPECIFICATION_PATH_2) -> None:
    """Write the tracked run-2 template (``run2_specification()``)."""
    path.write_text(json.dumps(run2_specification(), indent=2) + "\n", encoding="utf-8")


def run1_sft_weights(report_path: Path) -> dict[str, Any]:
    """The pin of run 1's selected SFT weights from its verified ``report.json``: the R2 key,
    size and SHA-256 of the ``sft-weights`` label's ``checkpoint.pt``."""
    from tools.posttrain_ops import REPORT_KIND, SUCCESS_STATUSES, report_artifacts, run_prefix

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("kind") != REPORT_KIND
        or report.get("run") != "1"
    ):
        raise ValueError(f"{report_path} is not post-training run 1's report")
    if report.get("status") not in SUCCESS_STATUSES:
        raise ValueError(f"run 1's report is {report.get('status')!r}, not completed")
    prefix = run_prefix("1", str(report.get("pod_id")))
    rows = [
        row
        for row in report_artifacts(report, prefix=prefix)
        if (row["label"], row["name"]) == ("sft-weights", "checkpoint.pt")
    ]
    if len(rows) != 1:
        raise ValueError("run 1's report lists no single sft-weights/checkpoint.pt")
    row = rows[0]
    return {"key": row["key"], "size_bytes": row["bytes"], "sha256": row["sha256"]}


def run1_sft_results(report_path: Path) -> dict[str, dict[str, Any]]:
    """The pins of run 1's SFT evaluation artifacts from its verified ``report.json``: the R2
    key, size and SHA-256 of each ``posttrain_run.SFT_RESULTS`` artifact (``arc-sft-sft`` and
    the seven ``eval-sft-<file>`` labels)."""
    from tools.posttrain_ops import report_artifacts, run_prefix
    from tools.posttrain_run import SFT_RESULTS

    report = json.loads(report_path.read_text(encoding="utf-8"))
    prefix = run_prefix("1", str(report.get("pod_id")))
    rows = {f"{row['label']}/{row['name']}": row for row in report_artifacts(report, prefix=prefix)}
    missing = [name for name in SFT_RESULTS if name not in rows]
    if missing:
        raise ValueError(f"run 1's report lists none of {missing}")
    return {
        name: {
            "key": rows[name]["key"],
            "size_bytes": rows[name]["bytes"],
            "sha256": rows[name]["sha256"],
        }
        for name in SFT_RESULTS
    }


def pod_export_pins(directory: Path) -> dict[str, dict[str, Any]]:
    """The pins of the three ``rl-pod-export`` files, hashed from the bytes in ``directory``
    (the record is only cross-checked), each file checked to hold no hidden test."""
    from scratchcoder.data.shards import file_sha256
    from scratchcoder.rl.pod_export import FILES, RECORD, read_record
    from tools.posttrain_binding import require_export

    record = read_record(directory / RECORD)
    kinds = {"rl-tasks": "rl-tasks", "rl-dev": "rl-dev-tasks", "rl-facts": "rl-facts"}
    pins = {}
    for name, file in FILES.items():
        path = directory / file
        pin = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        listed = record["files"][name]
        if (listed.get("bytes"), listed.get("sha256")) != (pin["size_bytes"], pin["sha256"]):
            raise ValueError(f"{path} differs from its rl-pod-export record")
        require_export(name, kinds[name], path)
        pins[name] = pin
    return pins


MEASURED_SCENARIO = "run 1's measured rates"
"""The run 2 projection's scenario whose total is the gate's estimate."""


def gate_estimate(feasibility: Path, *, run1_report: Path) -> float:
    """The gate's estimate, read from the run 2 projection's own output
    (``python -m tools.pilot_feasibility posttrain ... --output``), never typed: the total in
    dollars of its scenario at run 1's measured rates. The projection must be of the approved
    template, from this run 1 report, at the approved quoted price or below it."""
    from scratchcoder.data.shards import file_sha256

    report = json.loads(feasibility.read_text(encoding="utf-8"))
    template = (json.dumps(run2_specification(), indent=2) + "\n").encode()
    if (
        not isinstance(report, dict)
        or report.get("kind") != "posttrain-feasibility"
        or report.get("run") != "2"
    ):
        raise ValueError(f"{feasibility} is not run 2's cost projection")
    if report.get("specification_sha256") != hashlib.sha256(template).hexdigest():
        raise ValueError("the projection priced another specification than the run-2 template")
    sources = (report.get("measured") or {}).get("sources") or {}
    if sources.get("report_sha256") != file_sha256(run1_report):
        raise ValueError("the projection's measured rates are not from this run 1 report")
    limits = RUN2_LIMITS
    if report.get("quoted_total_hourly") != limits["quoted_total_hourly"]:
        raise ValueError("the projection is not priced at the specification's quoted rate")
    gpu = report.get("quoted_gpu_hourly")
    if not isinstance(gpu, int | float) or gpu > limits["quoted_gpu_hourly"]:
        raise ValueError("the projection's GPU price is above the approved quote: ask the owner")
    estimate = (report.get("scenarios") or {}).get(MEASURED_SCENARIO, {}).get("total_usd")
    if (
        isinstance(estimate, bool)
        or not isinstance(estimate, int | float)
        or not math.isfinite(estimate)
        or estimate <= 0
        or report.get("gate_estimate_usd") != estimate
    ):
        raise ValueError("the projection gives no estimate at run 1's measured rates")
    return float(estimate)


def gate_values(
    *,
    ceiling_usd: float,
    feasibility: Path,
    run1_report: Path,
    rl_pod_export: Path,
    pool_test_export: Path,
) -> dict[str, Any]:
    """Every run-2 gate value from the gate's own files: the owner's ceiling, the run 2
    projection's estimate, run 1's verified report (its SFT weights and its SFT evaluation
    artifacts), the ``rl-pod-export`` directory and the pool test export."""
    from scratchcoder.data.shards import file_sha256
    from tools.posttrain_binding import require_export

    count = require_export("pool-test", "pool-export", pool_test_export)
    if count != POOL_TASKS:
        raise ValueError(f"the pool test export holds {count} tasks, not {POOL_TASKS}")
    return {
        "ceiling_usd": ceiling_usd,
        "gate_estimate_usd": gate_estimate(feasibility, run1_report=run1_report),
        "sft-weights": run1_sft_weights(run1_report),
        **pod_export_pins(rl_pod_export),
        "pool-test": {
            "size_bytes": pool_test_export.stat().st_size,
            "sha256": file_sha256(pool_test_export),
        },
        "sft-results": run1_sft_results(run1_report),
    }


def main(argv: list[str] | None = None) -> int:
    """``run2-gate``: write run 2's launch specification from the gate's values (Mac);
    ``write-template``: rewrite the tracked run-2 template."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("run2-gate", help="fill run 2's gate values (Mac, at the gate)")
    gate.add_argument("--ceiling-usd", type=float, required=True, help="the owner's ceiling")
    gate.add_argument(
        "--feasibility",
        type=Path,
        required=True,
        help="the run 2 projection's --output at run 1's measured rates (its estimate)",
    )
    gate.add_argument("--run1-report", type=Path, required=True, help="run 1's verified report")
    gate.add_argument("--rl-pod-export", type=Path, required=True, help="the rl-pod-export dir")
    gate.add_argument("--pool-test-export", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True, help="the new launch specification")
    template = commands.add_parser("write-template", help="rewrite the tracked run-2 template")
    template.add_argument("--output", type=Path, default=SPECIFICATION_PATH_2)
    args = parser.parse_args(argv)
    if args.command == "write-template":
        write_specification_2(args.output)
        return 0
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"{args.output} already exists; a launch spec is never rewritten")
        values = gate_values(
            ceiling_usd=args.ceiling_usd,
            feasibility=args.feasibility,
            run1_report=args.run1_report,
            rl_pod_export=args.rl_pod_export,
            pool_test_export=args.pool_test_export,
        )
        filled = run2_specification(values)
        check_run2(filled)
        from tools.posttrain_run import Specification

        Specification.from_dict(filled)
    except (OSError, ValueError, KeyError) as error:
        print(f"posttrain-approved: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    data = (json.dumps(filled, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    limits = filled["limits"]
    fit = gate_launch_fit(filled)
    printed = {
        "specification": str(args.output),
        "sha256": hashlib.sha256(data).hexdigest(),
        **{key: limits[key] for key in ("ceiling_usd", "gate_estimate_usd", "cost_limit_usd")},
        **{
            key: round(fit[key])
            for key in ("lifetime_seconds", "planning_seconds_needed", "setup_allowance_seconds")
        },
    }
    print(json.dumps(printed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
