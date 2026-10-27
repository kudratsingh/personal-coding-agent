"""Tasks as the Pod may hold them: prompts, public tests and references, no hidden tests.

The Pod runs the RL loop and the dev checks, but a hidden test never leaves the Mac. A task
file shipped to the Pod therefore carries a fixed placeholder in ``hidden_tests`` (a ``Task``
cannot be empty there), and the Mac sends what the Pod needs from the real tests as a facts
file (``scratchcoder.rl.facts``). ``is_withheld`` recognises the placeholder, so code that
would otherwise read the hidden tests refuses a withheld task instead of silently computing
from the placeholder.
"""

from __future__ import annotations

import json
from dataclasses import replace

from scratchcoder.eval.tasks import Task

WITHHELD_FUNCTION_TESTS = "# the hidden tests stay on the Mac"
WITHHELD_IO_TESTS = json.dumps([{"input": "", "output": ""}])
"""The placeholders: a comment for a function task, one empty case for an I/O task (whose
tests must be a nonempty list of cases)."""


def is_withheld(task: Task) -> bool:
    """Whether ``task``'s hidden tests are the placeholder, not the real tests."""
    return task.hidden_tests in (WITHHELD_FUNCTION_TESTS, WITHHELD_IO_TESTS)


def withheld_task(task: Task) -> Task:
    """``task`` as the Pod may hold it: the hidden tests replaced by the placeholder, and a
    reference solution kept only for a train task (the control's targets), never for a dev or
    test task."""
    placeholder = WITHHELD_IO_TESTS if task.mode == "io" else WITHHELD_FUNCTION_TESTS
    return replace(
        task, hidden_tests=placeholder, solution=task.solution if task.split == "train" else None
    )


def refuse_withheld(tasks: list[Task] | tuple[Task, ...], what: str) -> None:
    """Raise if any task is withheld: ``what`` says what needed the real hidden tests."""
    withheld = [task.id for task in tasks if is_withheld(task)]
    if withheld:
        raise ValueError(
            f"{what} reads each task's hidden tests, but {len(withheld)} tasks withhold "
            f"them (e.g. {withheld[0]}): pass the facts the Mac computed from the real tests"
        )
