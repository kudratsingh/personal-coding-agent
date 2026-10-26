"""Recorded prompt/stop protocols for base-model scoring, shared with reference tooling.

``base-completion-rstrip-v1`` scores HumanEval-style prefix completion.
``base-pool-completion-v1`` scores pool tasks with a base model: the task text and
public test sit in a module docstring, and the model writes the program after it.
"""

from collections.abc import Sequence
from typing import Any

from scratchcoder.data import chat
from scratchcoder.eval.tasks import Task

BASE_STOPS = ("\nclass ", "\ndef ", "\nif __name__", "\nprint(", "\n#")
POOL_STOPS = ("\nif __name__", "\nprint(", "\nassert", '\n"""', "\n# Test")
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


def truncate_pool_completion(text: str, *, stops: Sequence[str] = POOL_STOPS) -> str:
    """The completion before its first stop string; the retained text is the program.

    The prompt ends with a newline, so the completion starts a line: a stop's leading
    newline may be that final prompt newline, and a completion that opens with
    ``assert`` keeps nothing.
    """
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
            "prompt_stripped": False,
            "protocol": {
                "name": POOL_PROTOCOL,
                "prompt_format": 'module docstring: """\\n{prompt}\\n\\n{public test}\\n"""\\n',
                "stop_strings": list(POOL_STOPS),
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
