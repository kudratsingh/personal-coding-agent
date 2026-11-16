"""The mode-H prompt: one chat-v1 user turn built from budgeted blocks, counted exactly.

The checkpoint was trained on one user turn (a task and a test) answered by one python block,
so every turn of the loop is rendered as a fresh single-turn conversation::

    {task}

    MAP
    {path: functions, one line per source module}

    SNIPPET {path}::{name}
    {function source}
    ... (one per located function, nearest first)

    CHANGES SO FAR
    {removed and added lines of the workspace against the original repository}

    RESULT run_tests ...
    {condensed failure}

    {instruction}

Each block has a token budget (:data:`BUDGETS`), counted with the frozen tokenizer. Over
budget, a block is truncated (and the truncation logged); if the whole prompt is still over
:data:`PROMPT_BUDGET`, the changes-so-far block is dropped, then the map, then the farthest
snippets (the nearest one always stays). If even that does not fit, :class:`ContextLimit` is
raised and the instance ends as a context failure. The generation cap is
``min(GENERATION_CAP, CONTEXT - prompt tokens - prefill tokens)``; the prefill (``--prefill-name``,
:func:`name_prefill`) is empty unless that option is on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from scratchcoder.data.chat import PYTHON_START, Message, TextEncoder, generation_prompt
from scratchcoder.eval.harness import ContextLimitError

CONTEXT = 1024
PROMPT_BUDGET = 700
GENERATION_CAP = 324
"""Planned generation cap: the room a full 700-token prompt leaves in the 1,024 context."""
BUDGETS = {"task": 120, "map": 80, "snippet": 250, "previous": 100, "observation": 150}
FUNCTION_BUDGET = 180
PREVIOUS_HEADER = "CHANGES SO FAR"
"""Header of the ``previous`` block: every applied edit so far, as removed and added lines of
the current workspace against the original repository (cumulative, not only the last edit)."""
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
    tests: Sequence[str] = ()
    """Public test function sources, used only by the training layout."""


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
    prefill: tuple[int, ...] = ()
    """Reply tokens the harness writes before generation (``--prefill-name``; empty when off).
    Not part of ``ids``, ``tokens`` or ``sha256``."""
    prefill_text: str = ""
    sources: tuple[str, ...] = ()
    """The source of each shown function as the prompt renders it, in ``shown`` order."""
    prefill_skipped: str = ""
    """With ``--prefill-name``: why this prompt got no prefill (:data:`PREFILL_SKIPS`), or
    empty."""

    @property
    def tokens(self) -> int:
        return len(self.ids)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(",".join(map(str, self.ids)).encode()).hexdigest()

    def generation_cap(self) -> int:
        """New tokens allowed: the planned cap, or the room the prompt and any prefill leave."""
        return min(GENERATION_CAP, CONTEXT - self.tokens - len(self.prefill))


def snippet_header(key: str) -> str:
    """The header line of a shown function (its rendered snippet is header + source)."""
    return f"SNIPPET {key}\n"


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
        parts.append(snippet_header(snippet.key) + snippet.source)
    if previous and blocks.previous:
        parts.append(PREVIOUS_HEADER + "\n" + "\n".join(blocks.previous))
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
        cost = _count(encoder, snippet_header(snippet.key) + source)
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
        and _count(encoder, PREVIOUS_HEADER + "\n" + "\n".join(previous)) > BUDGETS["previous"]
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
        "snippet": sum(_count(encoder, snippet_header(s.key) + s.source) for s in snippets),
        "previous": (
            _count(encoder, PREVIOUS_HEADER + "\n" + "\n".join(previous)) if keep_previous else 0
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
        sources=tuple(snippet.source for snippet in snippets),
    )


LAYOUTS = ("harness", "training")
TESTS_BUDGET = 150
"""Token budget of the training layout's public-test block (the observation's budget)."""


def build_training(encoder: TextEncoder, blocks: Blocks) -> Prompt:
    """The training-shaped layout: one user turn shaped like the SFT pool rows::

        {task}

        {function source}          (each shown function, nearest first, no header)

        {public test source}       (the failing public tests' functions)

        {condensed failure}        (repair turns only: the caller passes "" on turn 1)

        {instruction}

    The same budgets apply where a block exists (task 120, functions 250 with 180 each,
    observation 150, prompt 700); the tests get :data:`TESTS_BUDGET`. Over the total, the
    farthest functions are dropped, then the tests are cut to the first; still over is a
    :class:`ContextLimit`. There is no map and no changes block: the shown code is the current
    workspace."""
    truncations: list[str] = []
    task = blocks.task.strip()
    if _count(encoder, task) > BUDGETS["task"]:
        raise ContextLimit("the task statement is over its budget")
    snippets: list[Snippet] = []
    used = 0
    for index, snippet in enumerate(blocks.snippets):
        lines = snippet.source.split("\n")
        fitted = _fit_lines(encoder, lines, FUNCTION_BUDGET, ELLIPSIS)
        if fitted != lines:
            truncations.append(f"function:{snippet.key}")
        source = "\n".join(fitted)
        cost = _count(encoder, source)
        if index and used + cost > BUDGETS["snippet"]:
            truncations.append(f"snippet_dropped:{snippet.key}")
            continue
        snippets.append(Snippet(snippet.key, source))
        used += cost
    tests: list[str] = []
    for index, test in enumerate(blocks.tests):
        candidate = [*tests, test]
        if _count(encoder, "\n\n".join(candidate)) <= TESTS_BUDGET:
            tests = candidate
        elif not index:
            tests = [_fit_text(encoder, test, TESTS_BUDGET)]
            truncations.append("tests")
        else:
            truncations.append("tests_dropped")
    observation = _fit_text(encoder, blocks.observation, BUDGETS["observation"])
    if observation != blocks.observation:
        truncations.append("observation")

    def render() -> tuple[str, list[int]]:
        parts = [task, *(snippet.source for snippet in snippets), *tests]
        if observation:
            parts.append(observation)
        parts.append(INSTRUCTION)
        text = "\n\n".join(parts)
        return text, generation_prompt(encoder, [Message("user", text)])

    text, ids = render()
    while len(ids) > PROMPT_BUDGET and len(snippets) > 1:
        truncations.append(f"snippet_dropped_for_total:{snippets.pop().key}")
        text, ids = render()
    if len(ids) > PROMPT_BUDGET and len(tests) > 1:
        truncations.append("tests_dropped_for_total")
        tests = tests[:1]
        text, ids = render()
    if len(ids) > PROMPT_BUDGET:
        raise ContextLimit("the prompt is over its budget after every reduction")
    block_tokens = {
        "task": _count(encoder, task),
        "snippet": sum(_count(encoder, snippet.source) for snippet in snippets),
        "tests": _count(encoder, "\n\n".join(tests)),
        "observation": _count(encoder, observation),
        "instruction": _count(encoder, INSTRUCTION),
    }
    return Prompt(
        tuple(ids),
        text,
        block_tokens,
        tuple(snippet.key for snippet in snippets),
        tuple(truncations),
        sources=tuple(snippet.source for snippet in snippets),
    )


def build_layout(encoder: TextEncoder, blocks: Blocks, layout: str) -> Prompt:
    if layout == "harness":
        return build(encoder, blocks)
    if layout == "training":
        return build_training(encoder, blocks)
    raise ValueError(f"unknown prompt layout {layout!r}")


SEPARATOR_TOKENS = 2
"""Tokens allowed for one blank-line separator (``"\\n\\n"``) that a later turn adds between
blocks or snippets: the separator itself plus one for byte-pair drift at the joins."""
SEPARATOR_ALLOWANCE = 2 * SEPARATOR_TOKENS
"""Tokens allowed for the two separators that the observation and changes-so-far blocks add to
a prompt (the plan's worst case adds both blocks)."""


def most_snippets(header_tokens: Sequence[int]) -> int:
    """The most snippets one prompt can show: every snippet costs at least its header, and the
    snippets shown share :data:`BUDGETS` ``["snippet"]``, so this is the largest number of the
    cheapest headers whose sum fits that budget (at least one: the nearest always stays)."""
    count, used = 0, 0
    for tokens in sorted(header_tokens):
        if used + tokens > BUDGETS["snippet"]:
            break
        count, used = count + 1, used + tokens
    return max(count, 1)


def fit_bound(
    prompt_tokens: int,
    reference_tokens: Sequence[int],
    *,
    snippet_tokens: int,
    snippets_shown: int,
    header_tokens: Sequence[int],
) -> dict[str, Any]:
    """The static fit check of a case (plan section 3.5). ``prompt_tokens`` is its turn-1
    prompt without the observation and changes-so-far blocks, ``snippet_tokens`` that
    prompt's snippet block (each snippet's cost, separators not included), ``snippets_shown``
    how many snippets it shows, ``header_tokens`` the header cost (:func:`snippet_header`) of
    every function any turn could show (the top-level functions of the case's allowed source
    files: an edit can change a function but never add one), and ``reference_tokens`` the
    tokens of each target's reference function.

    The worst-case prompt of any turn grows the snippet block to its full budget (a later turn
    can show a longer function, such as an edited one grown up to :data:`FUNCTION_BUDGET`, or
    more located functions), adds one :data:`SEPARATOR_TOKENS` for each snippet beyond the
    turn-1 count up to :func:`most_snippets` (the turn-1 separators are already in
    ``prompt_tokens``), adds both feedback blocks at their full budgets and allows
    :data:`SEPARATOR_ALLOWANCE` tokens for their separators. The case fits when:

    1. that worst-case prompt is within :data:`PROMPT_BUDGET`;
    2. the whole fix, with a 20% margin, fits the context after it:
       ``worst + 1.2 x sum(reference) <= CONTEXT``;
    3. the largest single function (one per turn), with the margin and its two block tokens
       (python start and end), fits the generation cap: ``1.2 x largest + 2 <= GENERATION_CAP``.

    ``PROMPT_BUDGET + GENERATION_CAP == CONTEXT``, so a prompt within condition 1 always leaves
    the full :data:`GENERATION_CAP`; the context never shrinks the cap of a case that can fit.
    Block costs are counted on each block alone, so the bound is an estimate within the
    per-join drift allowance, not an exact count. This is planning only: at run time the
    budgets are enforced by truncation, drops and :class:`ContextLimit`, and the cap is
    computed from the real prompt."""
    extra_snippets = max(0, most_snippets(header_tokens) - snippets_shown)
    snippet_separators = extra_snippets * SEPARATOR_TOKENS
    worst = (
        prompt_tokens
        - snippet_tokens
        + max(snippet_tokens, BUDGETS["snippet"])
        + snippet_separators
        + BUDGETS["observation"]
        + BUDGETS["previous"]
        + SEPARATOR_ALLOWANCE
    )
    largest = max(reference_tokens, default=0)
    total_bound = worst + 1.2 * sum(reference_tokens)
    turn_bound = 1.2 * largest + 2
    return {
        "worst_prompt_tokens": worst,
        "snippet_separator_tokens": snippet_separators,
        "generation_cap": GENERATION_CAP,
        "total_bound": total_bound,
        "largest_function_bound": turn_bound,
        "fits": worst <= PROMPT_BUDGET and total_bound <= CONTEXT and turn_bound <= GENERATION_CAP,
    }


PREFILL_OPEN = "<|python_start|>"
"""How a prefill's block opener is written in ``prefill_text``: in chat-v1 the opening of a
python block (``` ```python ``` plus a newline in plain markdown) is this one special token."""


PREFILL_SKIPS = ("decorated",)
"""Why a prompt that shows a function gets no name prefill. ``decorated``: the nearest shown
function has decorators. The editor replaces a function's whole span, decorators included,
with the reply's function, and a reply opened with ``def <name>(`` cannot write decorators
before it, so a prefilled edit would silently delete them; the prefill writes only the name
prefix, never other lines of the shown source, so it does not apply to that turn."""


def name_prefill(prompt: Prompt, encoder: TextEncoder) -> Prompt:
    """``prompt`` with the name prefill of ``--prefill-name``: the reply is started with a
    python block and ``def <name>(``, where ``<name>`` is the function of the nearest shown
    snippet (``prompt.shown[0]``, the harness's own selection, already displayed in the
    prompt). It reads nothing but the prompt; a prompt that shows no function gets no prefill,
    and one whose nearest shown function is decorated gets none either, with
    ``prefill_skipped`` set (:data:`PREFILL_SKIPS`).

    The prefill tokens are not part of ``prompt.ids`` (whose hash is unchanged); they count
    against the context (:meth:`Prompt.generation_cap`) but are not generated tokens."""
    if not prompt.shown:
        return prompt
    if prompt.sources and prompt.sources[0].startswith("@"):
        return replace(prompt, prefill_skipped="decorated")
    name = prompt.shown[0].split("::", 1)[1]
    text = f"def {name}("
    return replace(
        prompt, prefill=(PYTHON_START, *encoder.encode(text)), prefill_text=PREFILL_OPEN + text
    )
