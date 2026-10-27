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

## Amendment (2026-10-26): reinforcement-learning reward channel and host statuses

This amendment records a reward-grading path added beside the runner and accepts, for one
rehearsal only, where that path runs.

### The reward channel and its guarantees

Reinforcement-learning rewards and RL candidate screening use a separate channel that never
puts the expected test values in the container. For a supported hidden assertion
`assert CALL == LITERAL` (and the close variants `assert CALL is None/True/False` and
`assert not CALL`), the host parses `LITERAL` with `ast.literal_eval` and keeps it; only
`CALL`'s source enters the container with the candidate. The container's trusted supervisor
runs the candidate in a child process, evaluates each `CALL` there, and reports a
type-preserving encoding of its value; the host decodes it and compares type and value against
the literal it kept. The reward is `passed / total`, with the denominator taken from the task
file. For I/O tasks every case runs under a per-case deadline and an overall per-attempt wall
budget; only the inputs enter the container, the container returns each case's bounded stdout,
and the host compares with expected outputs it never shared. Cases or calls that do not run
count as failed.

This closes the in-program grading hole: because the candidate never learns which value would
pass, forging the reported value — including flooding the result channel, printing fake output,
or faking a completion — earns no more than genuinely producing the value, and a value that
cannot be encoded (an object whose `__eq__` always returns `True`, a live handle, a function)
is a failure. Assertions outside the supported forms make a task not parseable for this
channel; those tasks are screened out of RL rather than graded in-process. A static audit flags
risky identifiers and patterns on each attempt as a monitoring signal, not a verdict.

**Limits.** This adds grading integrity, not a stronger isolation boundary: it reuses every
container flag unchanged. It is not a tamper-proof grader against a candidate that simply
guesses the right output, and it does not judge whether a task's tests are strong. Evaluation
scoring of record is unchanged and still runs the hidden tests in-process in the container.

### Distinct host-side statuses

The channel distinguishes host failures from candidate outcomes. A host stall or a docker CLI
that never returns is `host_timeout`; a missing daemon, missing binary or absent pinned image
is `unavailable`; a supervisor that returns no parseable framing is `host_error`. These are
retried up to a cap and are never scored as a reward — not even zero — so a grader problem can
never teach the policy to cause one. A resource kill (an out-of-memory or resource-limit kill of
the container) is a distinct status that scores the attempt as a failed `0.0`, separate from a
candidate timeout.

### Rehearsal-only execution decision

For the S rehearsal only, the owner accepted running model-generated pool tasks and the RL
reward channel in Docker Desktop on the owner's Mac, under stated mitigations applied for every
execution window: Docker Desktop file sharing emptied and verified (a pinned-image run that
mounts the home directory must fail), the executor environment carrying no credentials, the
image digest checked before each batch, these abuse tests re-run on the Mac against the image in
use, and the Docker virtual machine's data reset afterwards. Public benchmarks run instead on
disposable continuous-integration runners, which hold no secrets and are discarded after use.
This is a rehearsal boundary at a small model size, not the production host decision. The main
model's post-training will need the sandbox co-located with the GPU; that decision and its cost
are out of scope here.

### Known residual risks

- **Guessed outputs.** The channel defeats forged *passes*, not a candidate that computes or
  guesses the correct output by other means; weak tests remain a task-quality concern.
- **Same-UID `/proc` access inside the container.** All processes in the container run as the
  same unprivileged user, so a candidate can read peer `/proc` entries within the container.
  This does not reveal the expected values (they never enter the container) and does not cross
  the container boundary, but it is visibility the grading model does not rely on and the audit
  flags.
- **Docker Desktop's built-in `/var/folders` file share.** This share cannot be removed from
  Docker Desktop's configuration and is a known residual. A container cannot reach it unless
  code asks for a mount, and the sandbox never requests any mount; the exposure would require an
  escape from the container into the Docker virtual machine first.
- **Emulation on the Mac.** The pinned image is `linux/amd64` emulated on an arm64 laptop. This
  is a development environment with emulation in the loop, not a production host, so timing and
  any emulation-layer behavior are not a security guarantee.
