"""Conservative recovery of an intact function from a python block whose closing marker is
missing (``--recovery intact-function``). Host-side text and ``ast`` work only; nothing runs.

It applies only when the reply opens a block (``<|python_start|>``, with no special token
before it, as :func:`scratchcoder.data.chat.extract_program` requires) and has **no**
``<|python_end|>`` anywhere after it. The block text is the decoded tokens after the start up
to the next special token (or the end of the generation). Two rules can fire; each yields a
**verbatim prefix** of that text, never edited, re-indented or completed:

- ``eos_whole_parse``: the block ended at ``<|assistant_end|>`` and the generation stopped
  there (finish ``stop``, not ``length``), and the whole block text parses. The model closed its
  turn; only the block marker is missing.
- ``followed_by_top_level``: the longest prefix that (a) ends just before a later column-0 line
  that is not blank and not a comment and (b) parses as a module. Because the prefix parses
  completely and the next line starts at column 0, every top-level statement in the prefix,
  including any function, ended before the generation ran out.

The prefix then goes through the same reply rules and editor as a normal reply
(:func:`scratchcoder.agent.workspace.parse_reply`): exactly one shown function, nothing it
depends on dropped, the function's own lines taken verbatim. Recovery never looks at test
outcomes, and a text with a carriage return is not recovered (normalizing it would change
characters).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from scratchcoder.data.chat import ASSISTANT_END, PYTHON_END, PYTHON_START, TextDecoder, is_special

MODES = ("off", "intact-function")
RULES = ("eos_whole_parse", "followed_by_top_level")


@dataclass(frozen=True)
class Recovery:
    code: str
    """A verbatim prefix of the block text (lines ``1..end_line``)."""
    rule: str
    end_line: int
    """1-based last line of the block text that was kept."""
    block_lines: int
    """Lines in the whole (unclosed) block text."""

    def record(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("code")
        return row


def _parses(text: str) -> bool:
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def block_text(tokens: Sequence[int], decoder: TextDecoder) -> tuple[str, int | None] | None:
    """The unclosed block's text and the special token that ended it (None at the end of the
    generation), or None when recovery does not apply."""
    tokens = list(tokens)
    if PYTHON_START not in tokens:
        return None
    start = tokens.index(PYTHON_START) + 1
    if any(is_special(token) for token in tokens[: start - 1]):
        return None
    if PYTHON_END in tokens[start:]:
        return None
    end = next((i for i in range(start, len(tokens)) if is_special(tokens[i])), len(tokens))
    ender = tokens[end] if end < len(tokens) else None
    return decoder.decode(tokens[start:end]), ender


def recover_text(text: str, *, ended_by_eos: bool) -> Recovery | None:
    """Apply the two rules to an unclosed block's text."""
    if "\r" in text:
        return None
    lines = text.split("\n")
    if (
        ended_by_eos
        and _parses(text)
        and any(isinstance(node, ast.FunctionDef) for node in ast.parse(text).body)
    ):
        return Recovery(text, "eos_whole_parse", len(lines), len(lines))
    starts = [
        index
        for index, line in enumerate(lines)
        if index and line and not line[0].isspace() and not line.startswith("#")
    ]
    for following in reversed(starts):
        prefix = "\n".join(lines[:following])
        if not _parses(prefix):
            continue
        if not any(isinstance(node, ast.FunctionDef) for node in ast.parse(prefix).body):
            continue
        return Recovery(prefix, "followed_by_top_level", following, len(lines))
    return None


def recover(tokens: Sequence[int], decoder: TextDecoder, finish_reason: str) -> Recovery | None:
    """The recovered prefix of a reply without a closed block, or None."""
    found = block_text(tokens, decoder)
    if found is None:
        return None
    text, ender = found
    ended_by_eos = ender == ASSISTANT_END and finish_reason == "stop" and tokens[-1] == ender
    return recover_text(text, ended_by_eos=ended_by_eos)
