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

- **Function tasks.** The host parses each supported hidden assertion — `assert CALL == LITERAL`, `assert LITERAL == CALL`, `assert CALL is None/True/False`, and `assert not CALL`, where `LITERAL` is any `ast.literal_eval` value — and keeps the expected literal. Only `CALL`'s source enters the container with the candidate. The container's trusted supervisor runs the candidate in a child process, evaluates each `CALL`, and reports a type-preserving encoding of the value; the host decodes it and compares type and value against the kept literal. Any assertion outside the supported forms makes the whole task "not parseable" for this channel (`is_parseable` is `False`), and such tasks are screened out of RL rather than graded in-process.
- **I/O tasks.** Only the inputs enter the container. Every case runs under a per-case deadline and an overall per-attempt wall budget (default 8 s), with no stop at the first failure; the container returns each case's bounded stdout, and the host compares with expected outputs it never shared. Cases that do not run count as failed.
- **Reward.** `passed / total`, with the denominator taken from the task file. A value that cannot be encoded (an always-equal object, a live handle, a function) is a failure; forging the reported value earns nothing, because the candidate never learns which value would pass.
- **Host statuses.** A host stall or a docker CLI that never returns is `host_timeout`, a missing daemon or absent image is `unavailable`, and unparseable supervisor framing is `host_error`. These are retried up to a cap and are never scored as a reward, not even zero. A resource/OOM kill of the container is a distinct status scored as a failed `0.0`, separate from a candidate timeout.
- **Static audit.** `audit_attempt` scans an attempt for risky identifiers and patterns (frame and code-object introspection, `inspect`, `excepthook`, `atexit`, `os._exit`, `__eq__` overrides, builtins tampering, `/proc`, `__file__`, `sys.modules`, `gc`, `ctypes`, `signal`, and `open` of a `/tmp` path) and returns the flags for monitoring; it is a signal, not a verdict.

This path reuses every isolation flag of the development runner unchanged; it adds grading integrity, not a stronger isolation boundary. See the [reward-channel walkthrough](walkthroughs/reward-channel.md) and the [sandbox ADR](adr/0004-development-sandbox.md) amendment for the guarantees, limits and residual risks.

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
