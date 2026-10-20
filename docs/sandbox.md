# Generated-code execution

The development runner is implemented; its dedicated Linux containment workflow verifies functional and abuse cases. Default local tests skip container execution unless explicitly enabled. Production use remains subject to the [sandbox ADR](adr/0004-development-sandbox.md).

All model-generated programs must execute through a dedicated isolation boundary. Required controls include restricted filesystem access, no inherited secrets or network, and hard limits on runtime, memory, process count, file size, and captured output.

Before enabling evaluation or reinforcement learning, verify containment of infinite loops, memory exhaustion, process spawning, filesystem escape attempts, environment reads, network calls, and excessive output. A Python subprocess by itself does not establish this isolation boundary.

Keep hidden tests and reference solutions separate from model-visible prompts and repair feedback. Document the selected isolation mechanism and remaining limitations when it is implemented.

The backend uses a pinned Linux/amd64 Python image and the fixed local `/var/run/docker.sock` Unix socket. Pull the exact image specified in `scratchcoder.sandbox.runner.IMAGE` before running it; automatic pulls and remote Docker contexts are not used. If Docker is unavailable, execution fails closed. See the [walkthrough](walkthroughs/sandbox.md).

An explicit dependency profile can instead use an owner-controlled locally built immutable image ID. `sandbox/evalplus/` contains a dedicated Python 3.12.13/NumPy 2.5.4 build with a hash-required binary wheel and no dependency resolution. It is never built from the whole workspace, published, or selected by model output. Enabling system site packages is explicit under isolated Python mode; the default stdlib profile retains `-S`. Candidate threads are bounded through fixed BLAS/OMP settings. All filesystem/network/user/resource controls remain in place; see [ADR 0006](adr/0006-evaluation-runtime.md).

The default input cap remains 128,000 bytes. Larger benchmark scripts require an explicit bounded `Limits.input_bytes` (maximum 8 MiB); the file bound follows that declaration. Reports using another runtime or limits must record them and cannot claim identical upstream timing/memory policy merely because the same tests run. Dedicated NumPy CI builds the pinned image and exercises both functionality and isolation before benchmark use.

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
