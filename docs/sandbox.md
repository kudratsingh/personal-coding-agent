# Generated-code execution

The development runner is implemented; its dedicated Linux containment workflow verifies functional and abuse cases. Default local tests skip container execution unless explicitly enabled. Production use remains subject to the [sandbox ADR](adr/0004-development-sandbox.md).

All model-generated programs must execute through a dedicated isolation boundary. Required controls include restricted filesystem access, no inherited secrets or network, and hard limits on runtime, memory, process count, file size, and captured output.

Before enabling evaluation or reinforcement learning, verify containment of infinite loops, memory exhaustion, process spawning, filesystem escape attempts, environment reads, network calls, and excessive output. A Python subprocess by itself does not establish this isolation boundary.

Keep hidden tests and reference solutions separate from model-visible prompts and repair feedback. Document the selected isolation mechanism and remaining limitations when it is implemented.

The backend uses a pinned Linux/amd64 Python image and the fixed local `/var/run/docker.sock` Unix socket. Pull the exact image specified in `scratchcoder.sandbox.runner.IMAGE` before running it; automatic pulls and remote Docker contexts are not used. If Docker is unavailable, execution fails closed. See the [walkthrough](walkthroughs/sandbox.md).

An explicit dependency profile can instead use an owner-controlled locally built immutable image ID. `sandbox/evalplus/` contains a dedicated Python 3.12.13/NumPy 2.5.4 build with a hash-required binary wheel and no dependency resolution. It is never built from the whole workspace, published, or selected by model output. Enabling system site packages is explicit under isolated Python mode; the default stdlib profile retains `-S`. Candidate threads are bounded through fixed BLAS/OMP settings. All filesystem/network/user/resource controls remain in place; see [ADR 0006](adr/0006-evaluation-runtime.md).

The default input cap remains 128,000 bytes. Larger benchmark scripts require an explicit bounded `Limits.input_bytes` (maximum 8 MiB); the file bound follows that declaration. Reports using another runtime or limits must record them and cannot claim identical upstream timing/memory policy merely because the same tests run. Dedicated NumPy CI builds the pinned image and exercises both functionality and isolation before benchmark use.

## Reward channel for reinforcement learning

Reinforcement-learning rewards and RL candidate screening use a separate grading path in `scratchcoder.sandbox.reward` that never sends the expected test values into the container. Evaluation scoring of record is unchanged and still runs hidden tests in-process through the runner above.

- **Function tasks.** The host parses each supported hidden assertion — `assert CALL == LITERAL`, `assert LITERAL == CALL`, `assert CALL is None/True/False`, and `assert not CALL`, where `LITERAL` is any `ast.literal_eval` value and `CALL` contains no comparison — and keeps the expected literal. Only `CALL`'s source enters the container with the candidate. The container's trusted supervisor runs the candidate in a child process as the main script, then the tests' imports, as the evaluation program does, evaluates each `CALL`, and reports a type-tagged encoding of the value (subclasses as their base type); the host validates it, rebuilds the value from built-in types only, and decides the check as the assertion itself would: Python `==` against the kept literal (so `2.0` passes `== 2` and `True` passes `== 1`), identity for `is None/True/False`, and truthiness for `assert not CALL`. No candidate-defined method runs on the host; a result of a candidate-defined type cannot be encoded and fails, even where the in-process assertion could pass it. Any assertion outside the supported forms makes the whole task "not parseable" for this channel (`is_parseable` is `False`), and such tasks are screened out of RL rather than graded in-process.
- **I/O tasks.** Only the inputs enter the container. Cases run in turn from a fork server (one clean interpreter per attempt, a fresh fork per case) under one deadline shared by all cases, with no stop at the first failure; the container returns each case's bounded stdout, and the host compares with expected outputs it never shared. Cases that do not run count as failed.
- **Budgets.** A function task's attempt has 8 s, 4 s per call. An I/O task's attempt has the task pool verification's deadline, 2 s plus 0.25 s per case, at most 60 s (`scratchcoder.sandbox.budget.io_seconds`, also used by `tools/task_pool.py`), with the same container limits; as in the pool verification, a case may use whatever the earlier cases left. Each result records the budget it ran under.
- **Reward.** `passed / total`, with the denominator taken from the task file. A value that cannot be encoded (an always-equal object, a live handle, a function) is a failure; forging the reported value earns nothing, because the candidate never learns which value would pass.
- **Host statuses.** The supervisor tags a `started` frame and a `final` frame with a per-invocation nonce. Before `started`, an absent binary, exit 125 or an unreachable daemon is `unavailable` and any other failure is `host_error`; a docker CLI still running 15 s past the budget is `host_timeout`. These are retried up to a cap and are never scored as a reward, not even zero. After `started`, everything is scored: an out-of-memory kill is `resource_kill`, and a lost or malformed report is `malformed_report` (0.0), so a candidate cannot turn its own failure into a retried host failure. PID 1 ignores candidate signals and is non-dumpable, and it kills every straggler after the run and after each I/O case.
- **Static audit.** `audit_attempt` scans an attempt for risky identifiers and patterns (frame and code-object introspection, `inspect`, `excepthook`, `atexit`, `os._exit`, `os.kill`, comparison overrides, builtins tampering, `/proc`, `/tmp`, `__file__`, `sys.modules`, `gc`, `ctypes`, `signal`, dynamic imports) and returns the flags for monitoring; it is a signal, not a verdict.

This path reuses every isolation flag of the development runner unchanged; it adds grading integrity, not a stronger isolation boundary. The evaluation runner keeps its statuses, but a call ended by its host watchdog keeps the status `timeout` and is marked `host_stalled`: such a result is not a verdict on the candidate, and callers must retry it, never score it. The task pool's verification retries such stalls; the evaluation harness does not yet, so there a host stall still counts as a failed sample. See the [reward-channel walkthrough](walkthroughs/reward-channel.md) and the [sandbox ADR](adr/0004-development-sandbox.md) amendment for the guarantees, limits and residual risks.

## Local readiness observation (2026-10-20)

The local Docker daemon was already reachable; the missing pinned stdlib image
caused the earlier `unavailable` / exit **125** result. After the owner authorized
that exact image download, the configured socket/image worked without changing
the runner or unrelated containers. Docker **29.8.0**, arm64 host, pinned
**linux/amd64** runtime (emulated on this laptop).

`SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_sandbox.py tests/test_evaluation_integration.py`
passed **13/13** cases in **14.30 seconds**, including functional execution,
resource/escape attempts, environment isolation and output bounds. The full local
check with stdlib containment enabled passed **692 tests / 7 explicit optional
NumPy-runtime skips**. No additional NumPy image was downloaded and no task-32
diagnostics were reopened. This establishes the stated development-runtime
checks, **not production-host approval** or a new security guarantee.
