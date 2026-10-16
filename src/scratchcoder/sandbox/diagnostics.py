"""Safe allowlisted failure categories, never raw candidate/test output."""

import re

from scratchcoder.sandbox.runner import Result


def failure_category(result: Result) -> str:
    if result.status != "failed":
        return result.status
    if result.exit_code is not None and result.exit_code < 0:
        return f"signal_{-result.exit_code}"
    for name in (
        "SyntaxError",
        "IndentationError",
        "AssertionError",
        "NameError",
        "TypeError",
        "MemoryError",
        "OverflowError",
        "ZeroDivisionError",
        "ModuleNotFoundError",
        "ImportError",
        "AttributeError",
        "IndexError",
        "KeyError",
        "ValueError",
        "RecursionError",
        "RuntimeError",
    ):
        if re.search(r"(?m)^" + name + r"(?::|$)", result.stdout):
            return name
    return "other_failure"
