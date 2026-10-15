# Generated-code execution

The development runner is implemented; its dedicated Linux containment workflow verifies functional and abuse cases. Default local tests skip container execution unless explicitly enabled. Production use remains subject to the [sandbox ADR](adr/0004-development-sandbox.md).

All model-generated programs must execute through a dedicated isolation boundary. Required controls include restricted filesystem access, no inherited secrets or network, and hard limits on runtime, memory, process count, file size, and captured output.

Before enabling evaluation or reinforcement learning, verify containment of infinite loops, memory exhaustion, process spawning, filesystem escape attempts, environment reads, network calls, and excessive output. A Python subprocess by itself does not establish this isolation boundary.

Keep hidden tests and reference solutions separate from model-visible prompts and repair feedback. Document the selected isolation mechanism and remaining limitations when it is implemented.

The backend uses a pinned Linux/amd64 Python image and the fixed local `/var/run/docker.sock` Unix socket. Pull the exact image specified in `scratchcoder.sandbox.runner.IMAGE` before running it; automatic pulls and remote Docker contexts are not used. If Docker is unavailable, execution fails closed. See the [walkthrough](walkthroughs/sandbox.md).
