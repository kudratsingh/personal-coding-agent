"""The SFT format gate: a CPU check that SFT weights speak chat-v1 before paid evaluation.

Token-weighted dev loss can look healthy while the few format tokens are wrong: a model
that puts most of its mass on ``<|assistant_end|>`` after ``<|assistant_start|>`` loses
one token's worth of loss per row and produces an empty completion for every prompt. The
gate measures those tokens directly and runs no generated code. Four checks:

- ``first_token``: on ``ROWS`` code-dev rows chosen by ``selection_key``, the generation
  prompt (the row up to and including ``<|assistant_start|>``, which is exactly what
  chat-v1 renders for the task) must have ``<|python_start|>`` as its argmax in every
  row, with a mean probability of at least ``MIN_P_PYTHON_START``.
- ``close``: teacher-forced on the same rows, ``<|python_end|>`` at the true end of the
  code must have a mean probability of at least ``MIN_P_PYTHON_END``, and
  ``<|assistant_end|>`` right after it at least ``MIN_P_ASSISTANT_END``.
- ``separation``: the largest pairwise cosine among the tied embedding rows of the seven
  special tokens chat-v1 uses must be below ``MAX_SPECIAL_COSINE``.
- ``greedy_smoke``: greedy chat-v1 generation through ``generate_many`` (the evaluation's
  length-sorted, left-padded batches) on ``SMOKE_PROMPTS`` pool dev tasks chosen by the
  same key must extract a nonempty program from at least ``MIN_SMOKE_SHARE`` of them.
  The programs are hashed, never run.

The gate passes only if all four pass. ``run_gate`` writes one JSON record with every
measurement, the thresholds and the SHA-256 of each input. See
``docs/walkthroughs/sft-format-gate.md``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

from scratchcoder.common.records import atomic_json, git_state, utc_now
from scratchcoder.data.chat import (
    ASSISTANT_END,
    ASSISTANT_START,
    BOS,
    PYTHON_END,
    PYTHON_START,
    USER_END,
    USER_START,
    TextDecoder,
    TextEncoder,
    is_special,
)
from scratchcoder.data.sft import SFTRow, read_rows
from scratchcoder.data.shards import file_sha256
from scratchcoder.engine.generation import generate_many
from scratchcoder.eval.protocols import CHAT, load_suite_tasks, protocol_for
from scratchcoder.eval.tasks import Task
from scratchcoder.model.gpt import GPT
from scratchcoder.tokenizer.bpe import SPECIAL_TOKENS

FORMAT = "scratchcoder-sft-format-gate"
VERSION = 1
SELECTION_SALT = "sft-format-gate-1"
ROWS = 32
SMOKE_PROMPTS = 16
SMOKE_BATCH_SIZE = 16
MIN_P_PYTHON_START = 0.9
"""Every code row's first assistant token is ``<|python_start|>``: a learned format puts
nearly all its mass there, so 0.9 is reachable with margin and a coin flip between two
specials (the failure this gate exists for) is far below it."""
MIN_P_PYTHON_END = 0.5
"""Where code ends is genuinely uncertain (another function could follow), so the bar is
that closing is the more likely choice on average, not a near-certainty."""
MIN_P_ASSISTANT_END = 0.5
"""After ``<|python_end|>`` the turn always ends in training; 0.5 is a loose floor."""
MAX_SPECIAL_COSINE = 0.9
"""Trained rows of distinct tokens point in clearly different directions; rows that never
received a useful gradient drift together toward one shared direction."""
MIN_SMOKE_SHARE = 0.75
"""12 of 16 prompts: a format failure gives about none, an imperfect model most."""
CHAT_SPECIALS = (
    BOS,
    USER_START,
    USER_END,
    ASSISTANT_START,
    ASSISTANT_END,
    PYTHON_START,
    PYTHON_END,
)
"""The special tokens a chat-v1 code conversation contains."""


def thresholds() -> dict[str, Any]:
    return {
        "rows": ROWS,
        "smoke_prompts": SMOKE_PROMPTS,
        "min_p_python_start": MIN_P_PYTHON_START,
        "argmax_python_start": "every row",
        "min_p_python_end": MIN_P_PYTHON_END,
        "min_p_assistant_end": MIN_P_ASSISTANT_END,
        "max_special_cosine": MAX_SPECIAL_COSINE,
        "special_cosine_comparison": "strictly below",
        "min_smoke_share": MIN_SMOKE_SHARE,
    }


def name(token: int) -> str:
    return SPECIAL_TOKENS[token - 256] if is_special(token) else f"ordinary:{token}"


def selection_key(identifier: str) -> str:
    """Rows and tasks are taken in this order; it does not depend on file order."""
    return hashlib.sha256(f"{SELECTION_SALT}|{identifier}".encode()).hexdigest()


def assistant_index(row: SFTRow) -> int:
    """The index of a code row's ``<|assistant_start|>``; refuses any other row shape.

    A chat-v1 code row ends ``<|assistant_start|><|python_start|>{code}<|python_end|>
    <|assistant_end|>`` with nonempty code free of special tokens, after one user turn.
    """
    ids = row.ids
    if ids.count(ASSISTANT_START) != 1:
        raise ValueError(f"row {row.id} is not a one-exchange chat-v1 code row")
    start = ids.index(ASSISTANT_START)
    if (
        ids[:2] != (BOS, USER_START)
        or ids[start - 1] != USER_END
        or start + 4 > len(ids)
        or ids[start + 1] != PYTHON_START
        or ids[-2:] != (PYTHON_END, ASSISTANT_END)
        or any(is_special(token) for token in ids[start + 2 : -2])
    ):
        raise ValueError(f"row {row.id} is not a one-exchange chat-v1 code row")
    return start


def select_rows(rows: Iterable[SFTRow], count: int, context: int) -> list[SFTRow]:
    """The first ``count`` code rows by ``selection_key`` whose inputs fit ``context``."""
    items = list(rows)
    for row in items:
        assistant_index(row)
    fitting = [row for row in items if len(row.ids) - 1 <= context]
    chosen = sorted(fitting, key=lambda row: selection_key(row.id))[:count]
    if len(chosen) != count:
        raise ValueError(f"the format gate needs {count} code rows that fit the context")
    return chosen


def select_tasks(
    tasks: Iterable[Task], count: int, encoder: TextEncoder, context: int
) -> list[Task]:
    """The first ``count`` tasks by ``selection_key`` whose chat-v1 prompt can generate."""
    protocol = protocol_for(CHAT, "pool")
    runnable = [
        task
        for task in tasks
        if protocol.budget(len(protocol.prompt_ids(encoder, task)), context) >= 1
    ]
    chosen = sorted(runnable, key=lambda task: selection_key(task.id))[:count]
    if len(chosen) != count:
        raise ValueError(f"the format gate needs {count} pool tasks with room to generate")
    return chosen


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def judge_first_token(argmaxes: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    if not argmaxes or len(argmaxes) != len(probabilities):
        raise ValueError("first-token values need one argmax and probability per row")
    rows_argmax = sum(token == PYTHON_START for token in argmaxes)
    mean = _mean(probabilities)
    return {
        "mean_p_python_start": mean,
        "min_p_python_start": min(probabilities),
        "rows_argmax_python_start": rows_argmax,
        "rows": len(argmaxes),
        "passed": rows_argmax == len(argmaxes) and mean >= MIN_P_PYTHON_START,
    }


def judge_close(python_end: Sequence[float], assistant_end: Sequence[float]) -> dict[str, Any]:
    if not python_end or len(python_end) != len(assistant_end):
        raise ValueError("close values need two probabilities per row")
    mean_end, mean_after = _mean(python_end), _mean(assistant_end)
    return {
        "mean_p_python_end": mean_end,
        "mean_p_assistant_end_after_python_end": mean_after,
        "min_p_python_end": min(python_end),
        "min_p_assistant_end_after_python_end": min(assistant_end),
        "passed": mean_end >= MIN_P_PYTHON_END and mean_after >= MIN_P_ASSISTANT_END,
    }


def judge_separation(cosines: Sequence[Sequence[float]]) -> dict[str, Any]:
    size = len(CHAT_SPECIALS)
    if len(cosines) != size or any(len(row) != size for row in cosines):
        raise ValueError(f"separation needs a {size} x {size} cosine matrix")
    pairs = [(cosines[i][j], i, j) for i in range(size) for j in range(i + 1, size)]
    largest, left, right = max(pairs)
    return {
        "max_cosine": largest,
        "max_pair": [name(CHAT_SPECIALS[left]), name(CHAT_SPECIALS[right])],
        "mean_cosine": _mean([value for value, _, _ in pairs]),
        "passed": largest < MAX_SPECIAL_COSINE,
    }


def judge_smoke(extracted: Sequence[bool]) -> dict[str, Any]:
    if not extracted:
        raise ValueError("the greedy smoke needs at least one prompt")
    required = math.ceil(MIN_SMOKE_SHARE * len(extracted))
    count = sum(extracted)
    return {
        "programs": count,
        "prompts": len(extracted),
        "required": required,
        "passed": count >= required,
    }


@torch.no_grad()
def measure_rows(model: GPT, rows: Sequence[SFTRow]) -> list[dict[str, Any]]:
    """First-token and close probabilities of each row, in float32."""
    measured = []
    for row in rows:
        start = assistant_index(row)
        prompt = torch.tensor([list(row.ids[: start + 1])])
        first = torch.softmax(model(prompt)[0, -1].float(), dim=-1)
        inputs = torch.tensor([list(row.ids[:-1])])
        forced = torch.softmax(model(inputs)[0].float(), dim=-1)
        end = len(row.ids) - 2  # The index of <|python_end|>; logits at t predict ids[t + 1].
        measured.append(
            {
                "id": row.id,
                "prompt_tokens": start + 1,
                "code_tokens": end - start - 2,
                "argmax": int(first.argmax()),
                "argmax_name": name(int(first.argmax())),
                "p_python_start": float(first[PYTHON_START]),
                "p_assistant_end": float(first[ASSISTANT_END]),
                "p_python_end_at_close": float(forced[end - 1, PYTHON_END]),
                "p_assistant_end_after_python_end": float(forced[end, ASSISTANT_END]),
            }
        )
    return measured


@torch.no_grad()
def special_cosines(model: GPT) -> list[list[float]]:
    """Pairwise cosines of the tied embedding rows of ``CHAT_SPECIALS``."""
    rows = model.embedding.weight.detach().float()[list(CHAT_SPECIALS)]
    unit = torch.nn.functional.normalize(rows, dim=-1)
    return [[float(value) for value in row] for row in unit @ unit.T]


@torch.no_grad()
def special_geometry(model: GPT) -> dict[str, Any]:
    """Recorded, not judged: the rows' norms and their smallest pairwise distance.

    With the embedding tied to the output, ``h . (e_i - e_j)`` is the logit gap between
    tokens ``i`` and ``j``, so a short difference vector keeps the gap small.
    """
    rows = model.embedding.weight.detach().float()[list(CHAT_SPECIALS)]
    distances = torch.cdist(rows, rows)
    size = len(CHAT_SPECIALS)
    pairs = [(float(distances[i, j]), i, j) for i in range(size) for j in range(i + 1, size)]
    nearest, left, right = min(pairs)
    return {
        "norms": [float(value) for value in rows.norm(dim=-1)],
        "min_pair_distance": nearest,
        "min_distance_pair": [name(CHAT_SPECIALS[left]), name(CHAT_SPECIALS[right])],
    }


def greedy_smoke(
    model: GPT,
    encoder: TextEncoder,
    decoder: TextDecoder,
    tasks: Sequence[Task],
    *,
    batch_size: int = SMOKE_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Greedy chat-v1 completions of ``tasks`` and what the protocol extracts from each."""
    protocol = protocol_for(CHAT, "pool")
    prompts = [protocol.prompt_ids(encoder, task) for task in tasks]
    completions = generate_many(
        model,
        prompts,
        batch_size=batch_size,
        max_new_tokens=protocol.cap,
        stops=[protocol.stop_ids] * len(prompts),
        precision="float32",
    )
    results = []
    for task, prompt, completion in zip(tasks, prompts, completions, strict=True):
        tokens = list(completion.tokens)
        program = protocol.candidate(task, tokens=tokens, completion="", decoder=decoder)
        results.append(
            {
                "task_id": task.id,
                "prompt_tokens": len(prompt),
                "generated_tokens": len(tokens),
                "first_tokens": tokens[:8],
                "stop_reason": completion.finish_reason,
                "opened": PYTHON_START in tokens,
                "closed": program is not None,
                "program_extracted": program is not None and bool(program.strip()),
                "program_sha256": None
                if program is None
                else hashlib.sha256(program.encode()).hexdigest(),
            }
        )
    return results


def evaluate_gate(
    model: GPT,
    encoder: TextEncoder,
    decoder: TextDecoder,
    rows: Iterable[SFTRow],
    tasks: Iterable[Task],
    *,
    row_count: int = ROWS,
    smoke_prompts: int = SMOKE_PROMPTS,
    batch_size: int = SMOKE_BATCH_SIZE,
) -> dict[str, Any]:
    """Every check's measurements and verdict; ``passed`` only if all four pass.

    Each smoke task's prompt that also names a chosen row must equal that row's prompt,
    so the rows and the tasks are known to come from the same tokenizer and renderer.
    """
    model.eval()
    context = model.config.context
    chosen = select_rows(rows, row_count, context)
    measured = measure_rows(model, chosen)
    tasks_chosen = select_tasks(tasks, smoke_prompts, encoder, context)
    protocol = protocol_for(CHAT, "pool")
    by_id = {row.id: row for row in chosen}
    matched = 0
    for task in tasks_chosen:
        if task.id in by_id:
            row = by_id[task.id]
            if list(row.ids[: assistant_index(row) + 1]) != protocol.prompt_ids(encoder, task):
                raise ValueError(f"row {task.id} and its task render different prompts")
            matched += 1
    smoke = greedy_smoke(model, encoder, decoder, tasks_chosen, batch_size=batch_size)
    cosines = special_cosines(model)
    checks = {
        "first_token": judge_first_token(
            [row["argmax"] for row in measured], [row["p_python_start"] for row in measured]
        ),
        "close": judge_close(
            [row["p_python_end_at_close"] for row in measured],
            [row["p_assistant_end_after_python_end"] for row in measured],
        ),
        "separation": {
            **judge_separation(cosines),
            **special_geometry(model),
            "tokens": [name(token) for token in CHAT_SPECIALS],
            "cosines": cosines,
        },
        "greedy_smoke": {
            **judge_smoke([row["program_extracted"] for row in smoke]),
            "opened": sum(row["opened"] for row in smoke),
            "closed": sum(row["closed"] for row in smoke),
            "batch_size": batch_size,
            "max_new_tokens": protocol.cap,
            "precision": "float32",
        },
    }
    checks["first_token"]["mean_p_assistant_end"] = _mean(
        [row["p_assistant_end"] for row in measured]
    )
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "failed_checks": [label for label, check in checks.items() if not check["passed"]],
        "checks": checks,
        "thresholds": thresholds(),
        "row_ids": [row.id for row in chosen],
        "smoke_task_ids": [task.id for task in tasks_chosen],
        "smoke_prompts_matching_rows": matched,
        "rows": measured,
        "smoke": smoke,
        "protocol_sha256": protocol.sha256(),
        "selection": f"ascending sha256('{SELECTION_SALT}|' + id) among rows and tasks that "
        "fit the model context",
    }


def run_gate(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    rows_path: Path,
    tasks_path: Path,
    output: Path,
    threads: int = 4,
) -> dict[str, Any]:
    """Load the checkpoint on CPU, evaluate the gate, write ``output``; return the record."""
    from scratchcoder.eval.checkpoint import load_checkpoint
    from scratchcoder.tokenizer.bulk import FastEncoder

    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if output.exists():
        raise FileExistsError(f"{output} already exists; gate records are not replaced")
    started = time.perf_counter()
    model, tokenizer = load_checkpoint(checkpoint, tokenizer_path)
    torch.set_num_threads(threads)  # load_checkpoint leaves one thread.
    tasks, _ = load_suite_tasks("pool", "dev", tasks_path)
    gate = evaluate_gate(model, FastEncoder(tokenizer), tokenizer, read_rows(rows_path), tasks)
    commit, dirty = git_state()
    record = {
        "format": FORMAT,
        "version": VERSION,
        **gate,
        "inputs": {
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "tokenizer_sha256": file_sha256(tokenizer_path),
            "tokenizer_identity": tokenizer.sha256,
            "rows_sha256": file_sha256(rows_path),
            "tasks_sha256": file_sha256(tasks_path),
            "model": {
                "vocab_size": model.config.vocab_size,
                "width": model.config.width,
                "layers": model.config.layers,
                "context": model.config.context,
            },
        },
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "seconds": time.perf_counter() - started,
        "created_at": utc_now(),
        "git_commit": commit,
        "dirty": dirty,
    }
    atomic_json(output, record)
    return record
