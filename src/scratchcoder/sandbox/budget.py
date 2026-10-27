"""The wall-time deadline rule for one sandboxed I/O call.

The task pool's verification (``tools.task_pool.limits_for``) and the reinforcement-learning
reward channel (``scratchcoder.sandbox.reward``) both take an I/O task's deadline from
:func:`io_seconds`, so a task gets the same deadline from both: a reference solution that the
pool verification passed in time is not cut short when the reward channel scores it.
"""

MAX_SECONDS = 60.0
"""The largest deadline the runner accepts (``Limits.timeout_seconds``)."""
IO_BASE_SECONDS = 2.0
"""The pool verification's ``timeout_seconds`` in ``configs/task-pool-rehearsal.json``."""
IO_SECONDS_PER_CASE = 0.25
"""The pool verification's ``io_seconds_per_case`` in ``configs/task-pool-rehearsal.json``."""


def io_seconds(
    cases: int,
    *,
    base_seconds: float = IO_BASE_SECONDS,
    seconds_per_case: float = IO_SECONDS_PER_CASE,
) -> float:
    """The deadline for one call that runs ``cases`` I/O cases in sequence: ``base_seconds``
    plus ``seconds_per_case`` for every case, at most 60 s. All cases share this one deadline,
    so a case may use whatever time the cases before it left."""
    if type(cases) is not int or cases < 0:
        raise ValueError("the case count must be a non-negative integer")
    seconds = base_seconds
    seconds += seconds_per_case * cases
    return min(MAX_SECONDS, seconds)
