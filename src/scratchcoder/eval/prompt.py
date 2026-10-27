"""Recorded prompt/stop protocols for base-model scoring, shared with reference tooling.

``base-completion-rstrip-v1`` scores HumanEval-style prefix completion.
``base-pool-completion-v1`` scores pool tasks with a base model: the task text and
public test sit in a module docstring, and the model writes the program after it. Its
stop strings depend on the task's mode, so truncation always names the mode.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from scratchcoder.data import chat
from scratchcoder.eval.tasks import Task

BASE_STOPS = ("\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n#")
POOL_STOPS_BY_MODE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "function": ("\nif __name__", "\nprint(", "\nassert", '\n"""', "\n# Test"),
        "io": ("\nassert", '\n"""', "\n# Test"),
    }
)
"""base-pool-completion-v1's stop strings by task mode.

An I/O task's program runs at the top level: it prints its answer there, and a program
written as ``main()`` is often called from ``if __name__ == "__main__":``. So
``\\nprint(`` and ``\\nif __name__`` would cut a correct I/O program short; both stop
function-mode completions only, where top-level code after the function is example usage.
Both amendments were made on 2026-10-26, before the protocol's first scored use, and the
table is part of the protocol record, so the record's hash changed with each."""
POOL_MAX_NEW_TOKENS = 512
POOL_PROTOCOL = "base-pool-completion-v1"


def base_prompt(text: str) -> str:
    return text.rstrip()


def truncate_completion(text: str, *, stops: Sequence[str] = BASE_STOPS) -> str:
    offsets = [text.find(stop) for stop in stops if stop in text]
    return text[: min(offsets)] if offsets else text


def pool_prompt(task: Task) -> str:
    """The task text and public test in a module docstring, then a newline.

    The docstring holds chat-v1's user turn (prompt, blank line, public test) on its
    own lines between triple double quotes. A triple double quote inside that text is
    written with backslashes before each quote, so the docstring cannot close early.
    """
    body = chat.task_user_text(task).replace('"""', '\\"\\"\\"')
    return '"""\n' + body + '\n"""\n'


def pool_budget(prompt_tokens: int, *, context: int = 1024, cap: int = POOL_MAX_NEW_TOKENS) -> int:
    """New tokens allowed after a pool prompt: ``min(cap, context - prompt_tokens)``.

    A result below one means the prompt does not leave room to generate.
    """
    if type(prompt_tokens) is not int or prompt_tokens < 1 or context < 1 or cap < 1:
        raise ValueError("prompt length, context and cap must be positive integers")
    return min(cap, context - prompt_tokens)


def pool_stops(mode: str) -> tuple[str, ...]:
    """The base-pool stop strings for a task's mode, ``function`` or ``io``."""
    if mode not in POOL_STOPS_BY_MODE:
        raise ValueError("a pool task's mode is function or io")
    return POOL_STOPS_BY_MODE[mode]


def truncate_pool_completion(text: str, *, mode: str) -> str:
    """The completion before its first stop string for ``mode``; the retained program.

    ``mode`` is the task's ``mode`` and has no default, so no caller can silently
    apply the other mode's stops. The prompt ends with a newline, so the completion
    starts a line: a stop's leading newline may be that final prompt newline, and a
    completion that opens with ``assert`` keeps nothing.
    """
    stops = pool_stops(mode)
    line = "\n" + text
    offsets = [line.find(stop) for stop in stops if stop in line]
    return text[: max(min(offsets) - 1, 0)] if offsets else text


def prompt_protocol(mode: str = "base", *, stops: Sequence[str] = BASE_STOPS) -> dict[str, Any]:
    if mode == "base":
        return {
            "prompt_stripped": True,
            "protocol": {
                "name": "base-completion-rstrip-v1",
                "prompt_format": "raw-task-prompt.rstrip()",
                "stop_strings": list(stops),
                "candidate_assembly": "stripped-prompt+retained-completion",
            },
        }
    if mode == "base-pool":
        return {
            "prompt_stripped": True,
            "protocol": {
                "name": POOL_PROTOCOL,
                "prompt_format": 'module docstring: """\\n{prompt}\\n\\n{public test}\\n"""\\n',
                "prompt_strip": "task prompt.strip() and public test.strip()",
                "stop_strings_by_mode": {
                    task_mode: list(stops) for task_mode, stops in POOL_STOPS_BY_MODE.items()
                },
                "max_new_tokens": f"min({POOL_MAX_NEW_TOKENS}, context - prompt tokens)",
                "candidate_assembly": "retained-completion",
            },
        }
    if mode in {"chat", "chat-benchmark"}:
        return chat.protocol("pool" if mode == "chat" else "benchmark")
    return {
        "prompt_stripped": False,
        "protocol": {
            "name": "legacy-instruction-v1",
            "prompt_format": "public-prompt-with-public-tests",
            "stop_strings": [],
            "candidate_assembly": "legacy-fence-extraction-and-entry-point-heuristic",
        },
    }
