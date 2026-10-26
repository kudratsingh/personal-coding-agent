"""chat-v1: the post-training conversation format, its loss mask and code extraction.

A conversation is rendered to token IDs as::

    <|bos|>
    <|user_start|>{user text}<|user_end|>
    <|assistant_start|>{assistant content}<|assistant_end|>
    ... further user/assistant turns ...

For a code task the user text is the task prompt, a blank line and the public test,
and the assistant content is one python block,
``<|python_start|>{code}<|python_end|>``. A language turn's content is its text.

Message text is encoded without special-token injection, so a literal
``"<|user_end|>"`` inside a message stays ordinary bytes.

``mask[i]`` is true when ``ids[i]`` carries loss: as a next-token target it is
predicted from ``ids[:i]``. Exactly the tokens after each ``<|assistant_start|>``, up
to and including that turn's ``<|assistant_end|>``, carry loss. ``<|bos|>``, every
user-turn token and every ``<|assistant_start|>`` do not. See
``docs/walkthroughs/chat-format.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from scratchcoder.data.onepass import IGNORE
from scratchcoder.eval.tasks import Task
from scratchcoder.tokenizer.bpe import FIRST_MERGE, SPECIAL_TOKENS

PROTOCOL = "chat-v1"
SPECIAL_IDS = {token: 256 + index for index, token in enumerate(SPECIAL_TOKENS)}
BOS = SPECIAL_IDS["<|bos|>"]
USER_START = SPECIAL_IDS["<|user_start|>"]
USER_END = SPECIAL_IDS["<|user_end|>"]
ASSISTANT_START = SPECIAL_IDS["<|assistant_start|>"]
ASSISTANT_END = SPECIAL_IDS["<|assistant_end|>"]
PYTHON_START = SPECIAL_IDS["<|python_start|>"]
PYTHON_END = SPECIAL_IDS["<|python_end|>"]
CODE_STOP_IDS = (PYTHON_END, ASSISTANT_END)
"""Generation for a code task may stop at either; nothing after the first block scores."""
ROLES = ("user", "assistant")


class TextEncoder(Protocol):
    """``ByteBPE`` and ``FastEncoder`` both encode text without special tokens."""

    def encode(self, text: str) -> list[int]: ...


class TextDecoder(Protocol):
    def decode(self, ids: Sequence[int]) -> str: ...


class MaskedTokens(Protocol):
    """A token sequence with its loss mask, such as ``Rendered`` or an SFT row."""

    @property
    def ids(self) -> tuple[int, ...]: ...

    @property
    def mask(self) -> tuple[bool, ...]: ...


def is_special(token: int) -> bool:
    return 256 <= token < FIRST_MERGE


@dataclass(frozen=True)
class Message:
    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError("a chat-v1 message role is user or assistant")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("a chat-v1 message needs nonempty text")


@dataclass(frozen=True)
class Rendered:
    """Token IDs and the aligned loss mask of one chat-v1 conversation."""

    ids: tuple[int, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.mask) or len(self.ids) < 2:
            raise ValueError("a rendered conversation has aligned ids and mask of length 2+")
        if self.mask[0]:
            raise ValueError("the first token is never a target")

    @property
    def assistant_tokens(self) -> int:
        """The number of next-token targets that carry loss."""
        return sum(self.mask)

    def inputs_targets(self) -> tuple[list[int], list[int]]:
        """``ids[:-1]`` and ``ids[1:]``, with targets outside the mask set to ``IGNORE``."""
        return list(self.ids[:-1]), [
            token if keep else IGNORE
            for token, keep in zip(self.ids[1:], self.mask[1:], strict=True)
        ]


def _content(encoder: TextEncoder, text: str) -> list[int]:
    ids = encoder.encode(text)
    if not ids or any(type(token) is not int or token < 0 or is_special(token) for token in ids):
        raise ValueError("message text must encode to nonempty ordinary (non-special) tokens")
    return ids


def program_text(code: str) -> str:
    """Code as it appears in a python block: CRLF to LF, no leading blank lines, rstrip."""
    lines = code.replace("\r\n", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    text = "\n".join(lines).rstrip()
    if not text:
        raise ValueError("a python block needs nonempty code")
    return text


def task_user_text(task: Task) -> str:
    """The chat-v1 user turn of a pool task: prompt, a blank line, the public test.

    Function-mode public tests are the assertion text; I/O-mode public tests are the
    stored JSON list of input/output cases. Surrounding whitespace is stripped from both.
    """
    return task.prompt.strip() + "\n\n" + task.public_tests.strip()


def benchmark_user_text(task: Task) -> str:
    """The chat-v1 user turn of a HumanEval, HumanEval+ or MBPP+ task: its prompt alone."""
    return task.prompt.strip()


def _turns(encoder: TextEncoder, messages: Sequence[Message]) -> tuple[list[int], list[bool]]:
    ids, mask = [BOS], [False]
    for position, message in enumerate(messages):
        if message.role != ROLES[position % 2]:
            raise ValueError("chat-v1 turns alternate user, assistant, starting with user")
        if message.role == "user":
            body = [USER_START, *_content(encoder, message.text), USER_END]
            ids.extend(body)
            mask.extend([False] * len(body))
        else:
            reply = [*_content(encoder, message.text), ASSISTANT_END]
            ids.extend([ASSISTANT_START, *reply])
            mask.extend([False] + [True] * len(reply))
    return ids, mask


def render_conversation(encoder: TextEncoder, messages: Sequence[Message]) -> Rendered:
    """A language conversation: one or more user/assistant exchanges, ending on assistant."""
    if not messages or messages[-1].role != "assistant":
        raise ValueError("a training conversation ends with an assistant turn")
    ids, mask = _turns(encoder, messages)
    return Rendered(tuple(ids), tuple(mask))


def render_code_task(encoder: TextEncoder, task: Task, code: str | None = None) -> Rendered:
    """A pool task and its code (the reference solution by default) as one exchange."""
    source = task.solution if code is None else code
    if source is None:
        raise ValueError("a code-task row needs a reference solution or explicit code")
    ids, mask = _turns(encoder, [Message("user", task_user_text(task))])
    reply = [PYTHON_START, *_content(encoder, program_text(source)), PYTHON_END, ASSISTANT_END]
    ids.extend([ASSISTANT_START, *reply])
    mask.extend([False] + [True] * len(reply))
    return Rendered(tuple(ids), tuple(mask))


def generation_prompt(encoder: TextEncoder, messages: Sequence[Message]) -> list[int]:
    """The conversation so far (ending on a user turn), then ``<|assistant_start|>``."""
    if not messages or messages[-1].role != "user":
        raise ValueError("a generation prompt ends with a user turn")
    ids, _ = _turns(encoder, messages)
    return [*ids, ASSISTANT_START]


def task_prompt(encoder: TextEncoder, task: Task) -> list[int]:
    """The generation prompt of a pool task; the prefix of its training row."""
    return generation_prompt(encoder, [Message("user", task_user_text(task))])


def benchmark_prompt(encoder: TextEncoder, task: Task) -> list[int]:
    """The generation prompt of a HumanEval, HumanEval+ or MBPP+ task."""
    return generation_prompt(encoder, [Message("user", benchmark_user_text(task))])


def extract_program(generated: Iterable[int], decoder: TextDecoder) -> str | None:
    """The code of the first closed python block in generated IDs, or None.

    ``generated`` holds only the tokens after the prompt. The block is the tokens after
    the first ``<|python_start|>`` up to the next special token. It is a program only if
    that token is ``<|python_end|>``: any other special token there (another start, an
    ``<|assistant_end|>``) or the end of the generation means no program, which scores 0.
    The block must also open inside the first assistant turn: any special token before
    the first ``<|python_start|>`` (such as ``<|assistant_end|>`` followed by an invented
    later turn) means no program.
    """
    tokens = list(generated)
    if PYTHON_START not in tokens:
        return None
    start = tokens.index(PYTHON_START) + 1
    if any(is_special(token) for token in tokens[: start - 1]):
        return None
    for end in range(start, len(tokens)):
        if is_special(tokens[end]):
            return decoder.decode(tokens[start:end]) if tokens[end] == PYTHON_END else None
    return None


def protocol(user_turn: str = "pool") -> dict[str, Any]:
    """The recorded chat-v1 description for reports; ``user_turn`` is pool or benchmark."""
    if user_turn not in {"pool", "benchmark"}:
        raise ValueError("user_turn must be pool or benchmark")
    return {
        "prompt_stripped": False,
        "protocol": {
            "name": PROTOCOL,
            "prompt_format": "<|bos|><|user_start|>{user}<|user_end|><|assistant_start|>",
            "user_turn": (
                "task prompt.strip() + blank line + public test.strip()"
                if user_turn == "pool"
                else "benchmark prompt.strip()"
            ),
            "stop_tokens": [SPECIAL_TOKENS[token - 256] for token in CODE_STOP_IDS],
            "candidate_assembly": "first-closed-python-block-else-no-program",
        },
    }
