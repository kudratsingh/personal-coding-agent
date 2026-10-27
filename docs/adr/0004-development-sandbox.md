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
`assert CALL == LITERAL` (and the close variants `assert LITERAL == CALL`,
`assert CALL is None/True/False` and `assert not CALL`), the host parses `LITERAL` with
`ast.literal_eval` and keeps it; only `CALL`'s source enters the container with the candidate.
A `CALL` that itself compares anything, for example `(f(x) == 5) is True`, is not supported,
because the comparison would carry the expected value into the container and be judged there.
The container's trusted supervisor runs the candidate in a child process as the main script,
followed by the tests' imports, as the evaluation program does; it evaluates each `CALL` there
and reports a type-tagged encoding of its value. The host rebuilds that value from
built-in types only and decides the check as the hidden assertion itself would: Python `==`
against the literal it kept (so `2.0` passes `== 2` and `True` passes `== 1`, as the
evaluation runner's in-process assertion would), identity for `is None/True/False`, and
truthiness for `assert not CALL`. Because the rebuilt value holds only built-in types, no
candidate-defined `__eq__` ever runs on the host. The reward is `passed / total`, with the
denominator taken from the task file. For I/O tasks the cases run in turn under one deadline
shared by all of them; only the inputs enter the container, the container returns each case's
bounded stdout, and the host compares with expected outputs it never shared. Cases or calls
that do not run count as failed.

The attempt budgets are 8 s for a function task (4 s per call) and, for an I/O task, the task
pool verification's own deadline rule: 2 s plus 0.25 s per case, at most 60 s, kept in one
function (`scratchcoder.sandbox.budget.io_seconds`) that the pool verification also uses, with
the same container limits. As in the pool verification's I/O program there is no separate
per-case limit, so a reference that passed the pool verification in time is not cut short;
a case that never ends uses up the budget and leaves the later cases unrun. Each result
records the budget it ran under, and the host's stall deadline is that budget plus 15 s.

This closes the in-program grading hole: because the candidate never learns which value would
pass, forging the reported value — including flooding the result channel, printing fake output,
or faking a completion — earns no more than genuinely producing the value, and a value that
cannot be encoded (an object whose `__eq__` always returns `True`, a live handle, a function)
is a failure. Subclasses of the representable types are read through the base type's own
storage, so a `Counter` reports as a `dict` and an overridden `__eq__` changes nothing.
Assertions outside the supported forms make a task not parseable for this channel; those tasks
are screened out of RL rather than graded in-process. A static audit flags risky identifiers
and patterns on each attempt as a monitoring signal, not a verdict.

The host treats everything that comes out of the container as untrusted data: every reported
value is validated for shape, length and type before comparison, and any decoding error is a
failed check, never an exception in the scorer. Report values travel base64-encoded, so the
largest legitimate report is fixed by the payload before the run.

I/O cases run from a fork server inside the container: one clean interpreter starts per
attempt and each case runs in a fresh fork of it, which finishes through the interpreter's own
shutdown. No candidate state survives from one case to the next, and per-case verdicts match
the evaluation runner on the reference corpus in the tests; the difference is cost (a fork
instead of an interpreter start), so the cases spend less of the shared deadline than the pool
verification, which starts an interpreter per case.

**Limits.** This adds grading integrity, not a stronger isolation boundary: it reuses every
container flag unchanged. It is not a tamper-proof grader against a candidate that simply
guesses the right output, and it does not judge whether a task's tests are strong. Evaluation
scoring of record is unchanged and still runs the hidden tests in-process in the container.

### Distinct host-side statuses

The supervisor writes a `started` frame, tagged with a random per-invocation nonce, before any
candidate process exists, and a nonce-tagged `final` frame with the report at the end. The host
classifies each invocation from those frames:

- **Host failures** (retried up to a cap, never scored as a reward, not even zero): no docker
  binary, `docker run` exit 125, or an unreachable daemon before `started` is `unavailable`;
  any other failure before `started` is `host_error`; a docker CLI that has not finished
  15 s after the attempt budget is `host_timeout`. The host reads the CLI's output without
  blocking, so a CLI that never returns is killed at that deadline.
- **Candidate outcomes** (scored): after `started`, a valid `final` frame is `completed`; a
  container exit 137 without one is `resource_kill` and scores 0.0; any other lost, malformed
  or oversized report is `malformed_report` and scores 0.0. A complete report is labelled
  `resource_kill` when the container's out-of-memory kill count rose during the attempt; the
  killed calls or cases count as failed and the rest still count.

Because every problem after `started` is scored, a candidate cannot turn its own failure into
a skipped, retried attempt. The supervisor also removes the ways a same-user candidate could
reach it: PID 1 handles the catchable signals with a no-op (a pid-namespace init never receives
the signals it has no handler for), and it marks itself non-dumpable, so the candidate cannot
open `/proc/1/fd`, `/proc/1/mem`, `/proc/1/environ` or `/proc/1/oom_score_adj`. After the run,
and after every I/O case, it kills every other process in the container, so no straggler holds
the pid limit, memory or a pipe into the next case.

The evaluation runner keeps its status vocabulary; when its own host watchdog ends a call it
still reports `timeout` but marks the result `host_stalled`. Such a result is not a verdict on
the candidate: callers must retry it and never score it. The task pool's verification already
retries such stalls; the evaluation harness does not yet.

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
  guesses the correct output by other means. Many hidden tests share a few expected values
  (booleans, small integers, empty lists), so a constant answer can earn partial reward on a
  noticeable share of tasks; screening should compare tasks against a constant-output baseline,
  and weak tests remain a task-quality concern.
- **Same-UID processes inside the container.** All processes run as the same unprivileged
  user. PID 1 is out of the candidate's reach (non-dumpable, signal-deaf), but candidate
  processes can still see each other and the fork server, and can read PID 1's command line,
  which is the supervisor source and holds no secret. A candidate that disrupts the fork server
  only fails its own cases: the next case gets a new fork server. None of this reveals expected
  values, which never enter the container.
- **Late host-side kills are scored.** If the Docker host kills a container after the
  candidate started (for example the virtual machine runs out of memory) and before it
  reported, the attempt scores 0.0 rather than being retried. This can only lower rewards, and
  the alternative would let a candidate that provokes such a kill skip its failure.
- **Candidate-defined result types.** A call whose result is an object of a type the
  candidate defined (not a built-in or a subclass of one) cannot be encoded, so the check fails
  in reward mode even where the evaluation runner's in-process assertion could pass it through
  that type's own `__eq__`. This is the price of never running candidate comparison code on
  the host; for built-in values and their subclasses the two decide alike.
- **Call sources are visible to the candidate.** The call sources and the setup imports enter
  the container and share the candidate's process; only the compared literal is kept from it.
  A literal written inside `CALL` itself (rare forms such as `assert f(1) - 2.5 == 0` or
  `assert not isclose(f(1), 2.5)`) can therefore be read by a candidate that inspects its
  process, which the static audit flags. A bare `assert CALL`, whose passing outcome is known in
  advance, is not supported.
- **Docker Desktop's built-in `/var/folders` file share.** This share cannot be removed from
  Docker Desktop's configuration and is a known residual. A container cannot reach it unless
  code asks for a mount, and the sandbox never requests any mount; the exposure would require an
  escape from the container into the Docker virtual machine first.
- **Emulation on the Mac.** The pinned image is `linux/amd64` emulated on an arm64 laptop. This
  is a development environment with emulation in the loop, not a production host, so timing and
  any emulation-layer behavior are not a security guarantee. An I/O solution whose own compute
  exceeds its budget still loses the cases it did not reach (reported in each result's
  `unfinished` count). With the per-case rule an I/O attempt can take up to 27 s at 100 cases,
  so the reward time per RL update is a measured quantity, not a fixed bound.
