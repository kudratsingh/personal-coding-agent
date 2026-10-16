import pytest

from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import Result


def test_failure_diagnostics_never_guess_oom_from_a_killed_process() -> None:
    assert failure_category(Result("failed", exit_code=-9)) == "signal_9"
    assert failure_category(Result("failed", "MemoryError: private output")) == "MemoryError"
    assert failure_category(Result("failed", "private output")) == "other_failure"
    assert failure_category(Result("unavailable", "private output")) == "unavailable"


@pytest.mark.parametrize(
    "name",
    [
        "ModuleNotFoundError",
        "ImportError",
        "AttributeError",
        "IndexError",
        "KeyError",
        "ValueError",
        "RecursionError",
        "RuntimeError",
    ],
)
def test_dependency_and_common_categories_never_return_messages(name: str) -> None:
    output = f"Traceback fixture\n{name}: private-module SECRET_PRIVATE_TEXT\n"
    assert failure_category(Result("failed", output)) == name
    assert "PRIVATE" not in failure_category(Result("failed", output))
    assert failure_category(Result("passed", output)) == "passed"


def test_categories_are_anchored_allowlisted_and_observational() -> None:
    for output in (
        "private message mentions ModuleNotFoundError: but is not a category",
        "UnreviewedException: private ModuleNotFoundError",
        "ValueErrorSimilar: private message",
    ):
        assert failure_category(Result("failed", output)) == "other_failure"
    # Existing precedence and status/signal behavior do not change.
    mixed = "ModuleNotFoundError: private\nAssertionError: private\n"
    assert failure_category(Result("failed", mixed)) == "AssertionError"
    assert failure_category(Result("failed", mixed, exit_code=-9)) == "signal_9"
    assert failure_category(Result("output_limit", mixed)) == "output_limit"
