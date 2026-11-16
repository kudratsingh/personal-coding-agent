"""Harness v0 of the coding agent: fix or implement one function in a small repository.

The host locates a target function by static analysis, budgets a chat-v1 prompt, asks a policy
(the model) for one function, applies it as an AST span replacement, runs the public tests in
the sandbox for feedback, retries within budget, then grades the final workspace with hidden
correctness and preservation checks. Workspace and model code never run on the host: every
execution goes through :func:`scratchcoder.eval.harness.isolated_execution`. See
``docs/walkthroughs/agent-loop.md``.
"""
