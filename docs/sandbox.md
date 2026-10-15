# Generated-code execution

The runner is planned and has not been implemented or security-tested.

All model-generated programs must execute through a dedicated isolation boundary. Required controls include restricted filesystem access, no inherited secrets or network, and hard limits on runtime, memory, process count, file size, and captured output.

Before enabling evaluation or reinforcement learning, verify containment of infinite loops, memory exhaustion, process spawning, filesystem escape attempts, environment reads, network calls, and excessive output. A Python subprocess by itself does not establish this isolation boundary.

Keep hidden tests and reference solutions separate from model-visible prompts and repair feedback. Document the selected isolation mechanism and remaining limitations when it is implemented.
