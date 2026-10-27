"""Evaluation suites and generation protocols: prompt, budget, stops and extraction.

A protocol fixes how a task becomes a prompt, how many tokens may follow it, which tokens
stop generation and how a program is taken from what was generated. Its record (a JSON
object) is hashed, and every completions file names the record's SHA-256, so a changed
protocol can never be mistaken for the one a file was generated under.

Suites:

- ``pool``: the rehearsal task pool's ``dev`` or ``test`` split (a normalized task JSONL).
- ``humaneval``: OpenAI's original 164 tasks, the pinned release the reference gate scored.
- ``humaneval-plus``: the pinned EvalPlus release with ``humaneval/32`` excluded (163 tasks).
- ``mbpp-plus``: the pinned EvalPlus release (378 tasks).

Protocols:

- ``chat-v1`` for every suite: the pool user turn is the task prompt and public test, the
  benchmark user turn is the benchmark prompt. At most ``min(512, context - prompt)`` new
  tokens; generation stops at ``<|python_end|>`` or ``<|assistant_end|>``; the program is
  the first closed python block, and with none the attempt scores failed without running.
- ``base-pool-completion-v1`` for the pool only: the task text and public test in a module
  docstring; at most ``min(512, context - prompt)`` new tokens; the program is the
  completion before the first stop string of the task's mode.
- ``base-completion-rstrip-v1`` for the benchmarks only: the stripped benchmark prompt; at
  most ``min(256, context - prompt)`` new tokens; the program is the stripped prompt plus
  the completion before the first original stop string.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scratchcoder.data import chat
from scratchcoder.data.evalplus import read_release, release
from scratchcoder.data.humaneval import read_humaneval
from scratchcoder.eval import prompt as prompts
from scratchcoder.eval.tasks import Task, load_tasks
from scratchcoder.tokenizer.bpe import SPECIAL_TOKENS

POOL = "pool"
HUMANEVAL = "humaneval"
HUMANEVAL_PLUS = "humaneval-plus"
MBPP_PLUS = "mbpp-plus"
BENCHMARK_SUITES = (HUMANEVAL, HUMANEVAL_PLUS, MBPP_PLUS)
SUITES = (POOL, *BENCHMARK_SUITES)
POOL_SPLITS = ("dev", "test")
BENCHMARK_SPLIT = "eval"

CHAT = chat.PROTOCOL
BASE_POOL = prompts.POOL_PROTOCOL
BASE_BENCHMARK = "base-completion-rstrip-v1"
PROTOCOLS = (CHAT, BASE_POOL, BASE_BENCHMARK)
CHAT_MAX_NEW_TOKENS = 512
BASE_BENCHMARK_MAX_NEW_TOKENS = 256
RECORD_VERSION = 1
FIND_ZERO = "humaneval/32"
FIND_ZERO_REASON = "known numerical-tolerance failure"


def canonical_json(value: Any) -> bytes:
    """The one byte encoding that records are hashed in: sorted keys, no spaces."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record)).hexdigest()


def family(suite: str) -> str:
    """``pool`` for the task pool, ``benchmark`` for HumanEval, HumanEval+ and MBPP+."""
    if suite == POOL:
        return "pool"
    if suite in BENCHMARK_SUITES:
        return "benchmark"
    raise ValueError(f"suite must be one of {', '.join(SUITES)}")


@dataclass(frozen=True)
class Protocol:
    """One generation protocol for one suite family (``pool`` or ``benchmark``)."""

    name: str
    family: str

    def __post_init__(self) -> None:
        allowed = {
            (CHAT, "pool"),
            (CHAT, "benchmark"),
            (BASE_POOL, "pool"),
            (BASE_BENCHMARK, "benchmark"),
        }
        if (self.name, self.family) not in allowed:
            raise ValueError(
                f"protocol {self.name!r} is not defined for {self.family} suites; use chat-v1, "
                "base-pool-completion-v1 (pool) or base-completion-rstrip-v1 (benchmarks)"
            )

    @property
    def cap(self) -> int:
        """The protocol's new-token cap; the budget is ``min(cap, context - prompt)``."""
        return BASE_BENCHMARK_MAX_NEW_TOKENS if self.name == BASE_BENCHMARK else CHAT_MAX_NEW_TOKENS

    @property
    def stop_ids(self) -> tuple[int, ...]:
        """Token IDs that end generation early; base protocols stop on strings afterwards."""
        return chat.CODE_STOP_IDS if self.name == CHAT else ()

    def record(self) -> dict[str, Any]:
        """The protocol's full description; its SHA-256 identifies the protocol."""
        mode = {
            CHAT: "chat" if self.family == "pool" else "chat-benchmark",
            BASE_POOL: "base-pool",
            BASE_BENCHMARK: "base",
        }[self.name]
        described = prompts.prompt_protocol(mode)
        extraction = {
            CHAT: "first closed python block of the generated token IDs; with none, no "
            "program, scored failed without execution",
            BASE_POOL: "completion text before the first stop string of the task's mode",
            BASE_BENCHMARK: "stripped prompt + completion text before the first stop string",
        }[self.name]
        return {
            "record_version": RECORD_VERSION,
            "name": self.name,
            "family": self.family,
            "prompt_stripped": described["prompt_stripped"],
            "description": described["protocol"],
            "prompt_tokens": "chat-v1 renderer"
            if self.name == CHAT
            else "tokenizer.encode(prompt text), no special tokens",
            "max_new_tokens": f"min({self.cap}, context - prompt tokens)",
            "generation_stop_tokens": [SPECIAL_TOKENS[token - 256] for token in self.stop_ids],
            "completion_text": "tokenizer.decode(generated tokens, skip_special=True)",
            "extraction": extraction,
            "context_limit": "no room to generate: recorded as context_limit, scored failed",
        }

    def sha256(self) -> str:
        return record_sha256(self.record())

    def prompt_ids(self, encoder: chat.TextEncoder, task: Task) -> list[int]:
        if self.name == CHAT:
            if self.family == "pool":
                return chat.task_prompt(encoder, task)
            return chat.benchmark_prompt(encoder, task)
        if self.name == BASE_POOL:
            return encoder.encode(prompts.pool_prompt(task))
        return encoder.encode(prompts.base_prompt(task.prompt))

    def budget(self, prompt_tokens: int, context: int) -> int:
        """New tokens allowed after a prompt; zero means a context limit."""
        if type(prompt_tokens) is not int or prompt_tokens < 0 or context < 1:
            raise ValueError("prompt length and context must be nonnegative integers")
        return max(0, min(self.cap, context - prompt_tokens))

    def candidate(
        self,
        task: Task,
        *,
        tokens: Sequence[int],
        completion: str,
        decoder: chat.TextDecoder | None = None,
    ) -> str | None:
        """The program this protocol takes from one generation, or None (no program).

        chat-v1 reads the generated token IDs, so it needs ``decoder`` (the tokenizer);
        the base protocols read ``completion``, the decoded text with special tokens
        skipped.
        """
        if self.name == CHAT:
            if decoder is None:
                raise ValueError("chat-v1 extraction reads token IDs and needs the tokenizer")
            return chat.extract_program(tokens, decoder)
        if self.name == BASE_POOL:
            return prompts.truncate_pool_completion(completion, mode=task.mode)
        return prompts.base_prompt(task.prompt) + prompts.truncate_completion(completion)


def protocol_for(name: str, suite: str) -> Protocol:
    if name not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {', '.join(PROTOCOLS)}")
    return Protocol(name, family(suite))


def load_suite_tasks(suite: str, split: str, path: Path) -> tuple[list[Task], list[dict[str, str]]]:
    """A suite's scored tasks in ID order and the tasks it excludes, with reasons.

    A pool file must hold only tasks of ``split`` (``dev`` or ``test``). A benchmark file
    must be the pinned release (its size and SHA-256 are checked) and ``split`` is
    ``eval``. HumanEval+ always excludes ``humaneval/32``.
    """
    exclusions: list[dict[str, str]] = []
    if suite == POOL:
        if split not in POOL_SPLITS:
            raise ValueError("a pool suite's split is dev or test")
        tasks = list(load_tasks(path))
        if not tasks or any(task.split != split for task in tasks):
            raise ValueError(f"the pool task file must hold only {split} tasks")
    else:
        family(suite)
        if split != BENCHMARK_SPLIT:
            raise ValueError("a benchmark suite's split is eval")
        if suite == HUMANEVAL:
            tasks = read_humaneval(path)
        else:
            tasks = read_release(path, release(suite))
        if suite == HUMANEVAL_PLUS:
            if sum(task.id == FIND_ZERO for task in tasks) != 1:
                raise ValueError("HumanEval+ must hold exactly one task 32")
            tasks = [task for task in tasks if task.id != FIND_ZERO]
            exclusions.append({"task_id": FIND_ZERO, "reason": FIND_ZERO_REASON})
    return sorted(tasks, key=lambda task: task.id), exclusions
