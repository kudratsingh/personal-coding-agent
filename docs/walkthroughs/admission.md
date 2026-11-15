# Task admission: static gate, negative controls and reference checks

`scratchcoder.data.admission` decides whether a task may enter a development set or an RL pool. It never trusts a single verification run, and it keeps three kinds of evidence apart.

## Worked example

Take a function task whose hidden tests are `assert f(1) is None` and `assert f(2) == None`. Every check compares with `None`, so a function that does nothing passes. That is a mechanical fact of the text, so the static gate excludes the task as `none_only_checks` without running anything.

Take a sound task, `assert f(1) == 2`, `assert f(2) == 3` and `assert f(5) == 6`. The checks run five programs, each in its own fresh container through the evaluation path of record:

- the reference, three times, which gives three interpreters with three different string-hash seeds;
- a stub that defines every name the tests call and returns `None`;
- a stub that returns its first argument.

The task is admitted only if all three reference runs pass and the `None` stub fails. The three reference runs of a task are scheduled far apart: every task's first run comes before any second run, and every second run before any third. That way one burst of host load cannot reach all three. If the echo stub passes, the task stays admitted with the warning `echo_passed`, because the task may really be an identity.

## Evidence

| Kind | Effect | Examples |
| --- | --- | --- |
| Established defect | excluded | `none_only_checks`, `no_candidate_checks`, `reference_only_global`, `reference_failed` (no reference run passes, and at least one failed outright), `nondeterministic_reference` (runs disagree other than by timeouts or resource limits), the test-adequacy defects `defective_tests:accepts_none_stub`, `defective_tests:accepts_empty_output` and `defective_tests:not_runnable`, and reviewed exclusions with their recorded reasons |
| Warning | admitted, labelled | scan heuristics (a raise named in the prompt, a set-order or order-free pattern, a prompt-named global, a rounding clause), `two_or_fewer_checks`, `echo_passed`, `timing_unstable_reference` (runs disagree only by timeouts or resource limits) |
| Held | not admitted, no defect claimed | `reference_unverified` (the reference never passed, and only timed out or hit a resource limit), `conflicting_outcomes` (two cache rows of one run disagree) |
| Pending | neither | a run that is missing, or unresolved after its host retries |

Legitimate prompt-required globals and two-check tasks are therefore kept, as warnings. A host failure is never a defect: the run stays unresolved and the task stays pending until a later run resolves it.

## The cache

Every outcome, unresolved ones included, is appended to one JSON line. The line holds:

- the task ID and the SHA-256 of the whole task record;
- the run kind and index;
- the SHA-256 of the program;
- the status, whether it was resolved, and its host retries;
- the evaluator identity, which is the execution-policy SHA-256 of the path of record, so it changes with the supervisor, the limits or the verdict code;
- the pinned image and its local image ID.

A run is satisfied only by a resolved row whose identities all match the current ones, and whose program hash matches the planned program. If two such rows of one run disagree, the run is a conflict, never a silent choice. `execute` holds a lock on the cache, and it cuts a torn last line before appending. Rows under another identity stay in the file but are never reused. This fixes the pool verification cache's limits: it keyed a verdict by program and tests alone, so a repeated run could not be recorded, and it kept only the last attempt of a retried call.

## Counting

Reasons overlap. The excluded total is the size of the union of task IDs, and the per-reason counts are reported as overlapping.

## Running it

`tools/admission_check.py` has three subcommands, and each takes every input path explicitly. Sealed split files are refused before anything is opened.

- `plan` counts the runs each set needs.
- `execute` refuses to start without a passing `exec` window receipt (`tools.mac_window_check exec`) written for this very command in the last 30 minutes for the pinned image, and it takes the runtime image ID from that receipt. It also refuses while an SSH agent socket is set.
- `decide` writes one decision per task and a summary.

## Limits

Passing these checks does not prove a task deterministic or its tests adequate. It means no defect was established.

- Three runs can miss a rare nondeterministic outcome.
- The two stubs are only two of many weak programs.
- A heuristic warning is not a verdict either way.

Questions: why three fresh processes (each container draws a new string-hash seed, so set and dict order can differ); why separate timeout-only variation (host load is not a task defect); why keep pending tasks out (an unresolved run is not evidence); why bind the cache to the evaluator (verdicts from an unprotected supervisor must not satisfy the hardened one).
