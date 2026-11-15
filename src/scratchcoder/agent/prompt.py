"""The mode-H prompt: one chat-v1 user turn built from budgeted blocks, counted exactly.

The checkpoint was trained on one user turn (a task and a test) answered by one python block,
so every turn of the loop is rendered as a fresh single-turn conversation::

    {task}

    MAP
    {path: functions, one line per source module}

    SNIPPET {path}::{name}
    {function source}
    ... (one per located function, nearest first)

    PREVIOUS ATTEMPT
    {removed and added lines of the last applied edit}

    RESULT run_tests ...
    {condensed failure}

    {instruction}

Each block has a token budget (:data:`BUDGETS`), counted with the frozen tokenizer. Over
budget, a block is truncated (and the truncation logged); if the whole prompt is still over
:data:`PROMPT_BUDGET`, the previous attempt is dropped, then the map, then the farthest
snippets (the nearest one always stays). If even that does not fit, :class:`ContextLimit` is
raised and the instance ends as a context failure. The generation cap is
``min(GENERATION_CAP, CONTEXT - prompt tokens)``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from scratchcoder.data.chat import Message, TextEncoder, generation_prompt
from scratchcoder.eval.harness import ContextLimitError

CONTEXT = 1024
PROMPT_BUDGET = 700
GENERATION_CAP = 324
"""Planned generation cap: the room a full 700-token prompt leaves in the 1,024 context."""
BUDGETS = {"task": 120, "map": 80, "snippet": 250, "previous": 100, "observation": 150}
FUNCTION_BUDGET = 180
INSTRUCTION = "Reply with the complete corrected version of one function shown above."
ELLIPSIS = "    ..."


class ContextLimit(ContextLimitError):
    """The prompt cannot fit its budget even after every allowed reduction."""


@dataclass(frozen=True)
class Snippet:
    key: str
    """``path::name`` of the function shown."""
    source: str


@dataclass(frozen=True)
class Blocks:
    task: str
    map: Sequence[str]
    snippets: Sequence[Snippet]
    observation: str
    previous: Sequence[str] = ()


@dataclass(frozen=True)
class Prompt:
    ids: tuple[int, ...]
    text: str
    block_tokens: dict[str, int]
    """Tokens of each block's text encoded alone (the total is counted on the whole prompt,
    so block counts need not add up to it exactly)."""
    shown: tuple[str, ...]
    """The function keys the prompt shows, nearest first."""
    truncations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tokens(self) -> int:
        return len(self.ids)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(",".join(map(str, self.ids)).encode()).hexdigest()

    def generation_cap(self) -> int:
        return min(GENERATION_CAP, CONTEXT - self.tokens)


def _count(encoder: TextEncoder, text: str) -> int:
    return len(encoder.encode(text)) if text else 0


def _fit_lines(encoder: TextEncoder, lines: Sequence[str], budget: int, tail: str) -> list[str]:
    """The longest prefix of ``lines`` that, with ``tail`` appended when shortened, fits."""
    if _count(encoder, "\n".join(lines)) <= budget:
        return list(lines)
    kept = list(lines)
    while kept:
        kept.pop()
        if _count(encoder, "\n".join([*kept, tail])) <= budget:
            return [*kept, tail]
    return []


def _fit_text(encoder: TextEncoder, text: str, budget: int) -> str:
    """``text`` cut at a line, then at a character, so it fits ``budget`` tokens."""
    if _count(encoder, text) <= budget:
        return text
    lines = _fit_lines(encoder, text.split("\n"), budget, "...")
    if lines:
        return "\n".join(lines)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _count(encoder, text[:middle] + "...") <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low] + "..."


def _text(blocks: Blocks, include_map: bool, snippets: Sequence[Snippet], previous: bool) -> str:
    parts = [blocks.task.strip()]
    if include_map and blocks.map:
        parts.append("MAP\n" + "\n".join(blocks.map))
    for snippet in snippets:
        parts.append(f"SNIPPET {snippet.key}\n{snippet.source}")
    if previous and blocks.previous:
        parts.append("PREVIOUS ATTEMPT\n" + "\n".join(blocks.previous))
    if blocks.observation:
        parts.append(blocks.observation)
    parts.append(INSTRUCTION)
    return "\n\n".join(parts)


def build(encoder: TextEncoder, blocks: Blocks) -> Prompt:
    """Budget the blocks and render the prompt; raises :class:`ContextLimit`."""
    truncations: list[str] = []
    task = blocks.task.strip()
    if _count(encoder, task) > BUDGETS["task"]:
        raise ContextLimit("the task statement is over its budget")
    map_lines = _fit_lines(encoder, blocks.map, BUDGETS["map"], "...")
    if list(map_lines) != list(blocks.map):
        truncations.append("map")
    snippets: list[Snippet] = []
    used = 0
    for index, snippet in enumerate(blocks.snippets):
        lines = snippet.source.split("\n")
        fitted = _fit_lines(encoder, lines, FUNCTION_BUDGET, ELLIPSIS)
        if fitted != lines:
            truncations.append(f"function:{snippet.key}")
        source = "\n".join(fitted)
        cost = _count(encoder, f"SNIPPET {snippet.key}\n{source}")
        if index and used + cost > BUDGETS["snippet"]:
            truncations.append(f"snippet_dropped:{snippet.key}")
            continue
        if not index and cost > BUDGETS["snippet"]:
            raise ContextLimit("the nearest function does not fit the snippet budget")
        snippets.append(Snippet(snippet.key, source))
        used += cost
    previous = list(blocks.previous)
    if (
        previous
        and _count(encoder, "PREVIOUS ATTEMPT\n" + "\n".join(previous)) > BUDGETS["previous"]
    ):
        truncations.append("previous_dropped")
        previous = []
    observation = _fit_text(encoder, blocks.observation, BUDGETS["observation"])
    if observation != blocks.observation:
        truncations.append("observation")
    fitted_blocks = Blocks(task, map_lines, snippets, observation, previous)
    include_map, keep_previous = True, bool(previous)

    def render() -> tuple[str, list[int]]:
        text = _text(fitted_blocks, include_map, snippets, keep_previous)
        return text, generation_prompt(encoder, [Message("user", text)])

    text, ids = render()
    if len(ids) > PROMPT_BUDGET and keep_previous:
        keep_previous = False
        truncations.append("previous_dropped_for_total")
        text, ids = render()
    if len(ids) > PROMPT_BUDGET and include_map:
        include_map = False
        truncations.append("map_dropped_for_total")
        text, ids = render()
    while len(ids) > PROMPT_BUDGET and len(snippets) > 1:
        truncations.append(f"snippet_dropped_for_total:{snippets.pop().key}")
        text, ids = render()
    if len(ids) > PROMPT_BUDGET:
        raise ContextLimit("the prompt is over its budget after every reduction")
    block_tokens = {
        "task": _count(encoder, task),
        "map": _count(encoder, "MAP\n" + "\n".join(map_lines)) if include_map and map_lines else 0,
        "snippet": sum(_count(encoder, f"SNIPPET {s.key}\n{s.source}") for s in snippets),
        "previous": (
            _count(encoder, "PREVIOUS ATTEMPT\n" + "\n".join(previous)) if keep_previous else 0
        ),
        "observation": _count(encoder, observation),
        "instruction": _count(encoder, INSTRUCTION),
    }
    return Prompt(
        tuple(ids),
        text,
        block_tokens,
        tuple(snippet.key for snippet in snippets),
        tuple(truncations),
    )
