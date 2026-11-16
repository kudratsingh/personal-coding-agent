"""The static fit plan of one case: its turn-1 prompt size and the section 3.5 fit bound.

``tools.agent_dev_run plan`` and the synthetic-data generator both call :func:`plan_case`,
so a generated instance is admitted by exactly the rule a development case is. Nothing here
executes code.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from scratchcoder.agent import loop, prompt, workspace
from scratchcoder.agent.cases import Case
from scratchcoder.data.chat import TextEncoder


def plan_case(
    case: Case, encoder: TextEncoder, *, failing: Iterable[str] | None = None
) -> dict[str, Any]:
    """The turn-1 prompt (without its observation) located from ``failing`` (every public
    test when None, as ``plan`` does), whether every target is shown, and
    :func:`scratchcoder.agent.prompt.fit_bound` for it."""
    candidates = workspace.locate(case.files, case.public_tests, failing, case.allowed_files)
    built = prompt.build(encoder, loop.blocks_for(case, case.files, candidates, "", case.files))
    reference = [len(encoder.encode(case.reference_function(k))) for k in case.targets]
    snippet = built.block_tokens["snippet"]
    showable = [
        function.key
        for path in sorted(set(case.allowed_files) & set(case.files))
        if not workspace.is_test_file(path)
        for function in workspace.functions(case.files[path], path).values()
    ]
    headers = [len(encoder.encode(prompt.snippet_header(key))) for key in showable]
    return {
        "case": case.id,
        "prompt_tokens_without_observation": built.tokens,
        "block_tokens": built.block_tokens,
        "shown": list(built.shown),
        "targets_shown": all(target in built.shown for target in case.targets),
        "reference_fix_tokens": reference,
        **prompt.fit_bound(
            built.tokens,
            reference,
            snippet_tokens=snippet,
            snippets_shown=len(built.shown),
            header_tokens=headers,
        ),
    }
