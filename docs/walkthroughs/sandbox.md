# Isolated execution boundary

The host serializes candidate code, tests, and limits as data. It starts a fixed hardened Docker command and passes that JSON through standard input. No repository directory, secret, or daemon socket is mounted in the container. All candidate execution occurs under `scratchcoder.sandbox` inside the container.

A trusted container supervisor writes the candidate into its temporary filesystem and starts a child interpreter with a small explicit environment. Function tests require actual completion after assertions; I/O tests compare captured output for supplied input cases. A bounded reader enforces output limits, and process/container limits contain resource abuse. Host timeout cleanup removes the exact run container.

For an infinite loop, the supervisor reaches the wall deadline and kills the child. If the whole container is killed by its memory limit, the host reports a resource failure. Neither path grants a passing result or falls back to host Python.

See [ADR 0004](../adr/0004-development-sandbox.md) for the threat boundary and grading limitations. Integration tests run only when explicitly enabled on a disposable Docker worker; absence of the daemon is not a passed security test.

Reinforcement-learning rewards use a different supervisor through the same hardened command, described in the [reward-channel walkthrough](reward-channel.md): the host keeps the expected values and sends only each hidden assertion's call (or each I/O case's input) into the container, so a candidate cannot forge a pass, and host failures are reported distinctly from candidate outcomes so they are retried rather than scored. That path leaves this one, the evaluation path of record, unchanged.

Questions: why no bind mounts (deny host files); why fixed local daemon (avoid remote paid resources); why bound capture twice (protect supervisor and host); why kill the container (contain surviving descendants); why refuse a host fallback (preserve the isolation contract).
