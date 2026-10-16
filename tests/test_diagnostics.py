from scratchcoder.sandbox.diagnostics import failure_category
from scratchcoder.sandbox.runner import Result


def test_failure_diagnostics_never_guess_oom_from_a_killed_process() -> None:
    assert failure_category(Result("failed", exit_code=-9)) == "signal_9"
    assert failure_category(Result("failed", "MemoryError: private output")) == "MemoryError"
    assert failure_category(Result("failed", "private output")) == "other_failure"
    assert failure_category(Result("unavailable", "private output")) == "unavailable"
