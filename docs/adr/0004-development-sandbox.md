# ADR 0004: fail-closed development code runner

**Status:** Accepted for disposable development workers; production backend pending review
**Date:** 2026-10-15

## Decision

Execute generated programs only inside a pinned Linux/amd64 Docker runtime through the sandbox module. Use a fixed local Unix daemon endpoint, no host mounts or inherited credentials, no network, a read-only root filesystem, an unprivileged user, dropped capabilities, and memory/process/CPU/file/output limits. Missing infrastructure returns an explicit non-passing result, never host execution.

The container runs a trusted supervisor and a separate candidate process. The supervisor checks completion, bounds output, and reports structured outcomes. Host-side capture is also bounded; every run cleans up its exact randomly named container.

## Limits

This is a development containment boundary, not a kernel security guarantee or tamper-proof grader. Generated Python can introspect its own process; hidden-test secrecy means tests are excluded from model prompts, not guaranteed inaccessible at runtime. Do not expose this runner as a public service. Production isolation and host approval remain a separate gate before large-scale untrusted workloads.

## Verification

Unit tests inspect the command and fail-closed behavior. An explicitly enabled disposable Linux CI worker runs functional/I/O tests and abuse cases for loops, allocations, forks, root writes, environment/repository access, network, and output floods. Local hosts without a running daemon skip integration tests explicitly.
