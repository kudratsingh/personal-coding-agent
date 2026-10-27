# Component: reinforcement-learning reward channel

The reward channel scores a model's code against hidden tests for reinforcement-learning
rewards and RL candidate screening. It exists because the evaluation path of record puts the
candidate, the hidden tests and a completion marker in one program, which a reward-seeking
policy can exploit: read the marker out of code-object constants, print it from an
`excepthook`, call `os._exit(0)` before the assertions run, or return an object whose
`__eq__` always returns `True`. The reward channel removes the thing being faked by keeping
the expected answers on the host and never sending them into the container.

Evaluation scoring of record is unchanged; it still uses `scratchcoder.sandbox.runner.run`.

## Worked example

Hidden tests for a task, `assert add(1, 2) == 3` and `assert add(5, 5) == 10`:

1. The host parses each assertion. The right-hand side (`3`, `10`) is a safe literal, so the
   host keeps it. Only the left-hand call source (`"add(1, 2)"`, `"add(5, 5)"`) goes into the
   container, next to the candidate code.
2. Inside the container the trusted supervisor writes a `started` frame tagged with the host's
   random nonce, runs the candidate in a child process, evaluates each call there, and collects
   a type-tagged encoding of each result from a private pipe: `["int", 3]`, `["int", 10]`.
3. The supervisor writes a nonce-tagged `final` frame carrying those result rows
   (base64-encoded). The host checks every row's shape, rebuilds each value from built-in
   types only, and decides each check as the assertion itself would: `3 == 3` and
   `10 == 10`. Two of two pass, so `reward = 2 / 2 = 1.0`. A candidate that returned `3.0`
   would pass too, because `3.0 == 3` is true in Python and the in-process assertion would
   pass it.

A candidate that returns `0` reports `["int", 0]` for both, matches neither `3` nor `10`, and
earns `0.0`. A candidate that returns an always-equal object cannot be encoded (its type is not
one of the representable types), so that call is a failure. A candidate that forges
`["int", 3]` on the pipe gains nothing, because it still has to produce the value `10` for the
second check, and it was never told what `10` is. A candidate that writes a malformed row such
as `["bytes", 10**20]` only fails that check: the host validates the row and never raises.

## Data flow and shapes

```
Task.hidden_tests ──► host parse ──►  (setup imports, [call sources], [expected literals])
                                             │                 │ kept on host
                                 call sources │ + candidate code + nonce
                                             ▼
                                   docker run (hardened, same flags as the
                                   evaluation runner) → reward_program.py (PID 1)
                                             │  writes {"frame": "started", "nonce": …}
                                   child process: run candidate, eval each call,
                                   write ["kind", payload] per call to a private pipe
                                             ▼
                                   supervisor → {"frame": "final", "nonce": …, "report": …}
                                             ▼
host: classify the invocation from the frames → validate and decode each row into a
      built-in value → decide each check as the assertion would (==, is, not) against the
      kept literals → passed / total (total from the task file)
```

- **Function mode.** `parse_function_checks` returns a `FunctionChecks` with the import
  `setup` and one `Check` per assertion. A `Check` is `(call, predicate, expected)` where
  `predicate` is `"eq"` (Python `==` between the rebuilt value and `expected`), `"is"` (the
  rebuilt value is the `None`, `True` or `False` singleton) or `"falsy"` (for
  `assert not CALL`, the rebuilt value is falsy). Supported assertion forms:
  `assert CALL == LITERAL`, `assert LITERAL == CALL`, `assert CALL is None/True/False` (by
  identity, so `is 0` is not accepted), and `assert not CALL`, where `LITERAL` is any
  `ast.literal_eval` value of a representable type. `CALL` must not contain a comparison or be
  an `and`/`or`/`not` expression: in `(f(x) == 5) is True` the literal `5` would enter the
  container and be compared there by candidate objects. A bare `assert CALL` is not supported
  at present (an open choice; see the design choices below). Any other top-level statement, or
  any assertion outside these forms, makes the whole task **not parseable**
  (`parse_function_checks` returns `None`, `is_parseable` is `False`), so a partial parse can
  never under-count the denominator and over-reward. A task whose expected value could not fit
  in one result row (64 KiB by default) is not parseable either, and neither is one whose
  call sources and import lines together exceed the sandbox input limit (128,000 bytes),
  because they enter the container with the candidate. In the container the child
  runs the candidate as the main script `/tmp/candidate.py`, where the supervisor writes its
  source: a fresh `__main__` module in `sys.modules` (so its `if __name__ == "__main__":`
  block runs and its classes pickle), `__file__` and `sys.argv[0]` are `/tmp/candidate.py`,
  `__builtins__` is the builtins module, and stdin is empty (the driver's configuration
  arrived there and cannot be re-read). Then come the hidden tests' imports and each call in
  order, as the evaluation program runs the candidate followed by the tests, and the driver
  ends by printing a random completion marker as long as the evaluation program's. As there,
  the candidate's own output beyond the 64 KiB cap (the marker counts toward it) fails every
  check, and so, once every call has reported, does a nonzero exit status (an `atexit`
  handler calling `os._exit(1)`, a thread that keeps the process alive), an output stream
  still open when the budget ends (a forked process that keeps the pipe), and output that
  does not end with the marker plus whitespace (an `atexit` print, a destructor's complaint
  on stderr, `sys.stdout.close()`, a replaced `sys.stdout`). A process the budget ends before
  every call reported keeps the calls that did, which the evaluation program does not do (it
  fails a timeout whole). The source is compiled as UTF-8 bytes, as the file is read, so a
  byte-order mark is dropped and an encoding line is honoured or rejected as it is there.
- **I/O mode.** `parse_io_cases` returns the inputs and the expected outputs. Only the inputs
  enter the container. The supervisor starts one clean interpreter per attempt as a fork
  server; each case runs in a fresh fork of it, in turn, under one deadline shared by all the
  cases (see the budgets below). The forked child sees what a fresh
  `python -I -S /tmp/candidate.py` would see
  (standard streams, `sys.argv`, a new `__main__`, the startup module set) and finishes through
  the interpreter's own shutdown, so exit statuses, `atexit` output and buffered streams behave
  as in the evaluation path. A case whose stdout exceeds the output cap fails, as in the
  evaluation path. Files a case writes to `/tmp` stay for the later cases, as in the pool
  verification's I/O program. Each case's stdout returns base64-encoded and the host compares it, after
  `strip()`, with the expected output it kept. Every case runs (no stop at first failure);
  cases that do not run count as failed. A task with malformed cases, more than 100 cases, inputs
  that together exceed the sandbox input limit (128,000 bytes), or an expected output larger
  than the output cap is not parseable.
- **The encoding and the comparison** (`reward_program.DRIVER` in the container,
  `_value_from_wire` and `check_passes` on the host). The encoding is type-tagged, so the host
  rebuilds exactly the built-in value the call returned. It then uses Python's own semantics,
  as the in-process assertion would on that value: `2.0 == 2` and `True == 1` pass, `[1, 2]`
  does not equal `(1, 2)`, sets compare order-independently and dicts by their items,
  `-0.0 == 0.0` passes, `nan` equals nothing, and `1e999` in a hidden test is infinity.
  `is None/True/False` stays identity, so `1` fails `is True`. Subclasses of the representable
  types are read through the base type's own storage, so a `Counter` reports as a `dict`, a
  named tuple as a `tuple`, and an `IntEnum` member as an `int`; a subclass is judged by that
  base value, never by its own `__eq__`. A `bytearray` reports as `bytes` and a dict keys view
  as a `frozenset`, which compare with any literal exactly as the originals do. A value whose
  type is not one of `None, bool, int, float, str, bytes, list, tuple, set, frozenset, dict`
  (or a subclass, recursively, or one of those two) cannot be encoded and counts as a failure,
  even a standard-library value such as `Decimal('1.5')` or `Fraction(1, 2)` that would pass
  the in-process assertion. Because the rebuilt value contains only built-in types, `==` on
  the host never runs a candidate-defined method.
- **Budgets.** A function task's attempt has 8 s (`FUNCTION_BUDGET_SECONDS`), and each call
  4 s within it (`PER_CALL_SECONDS`), more than the pool verification's 2 s for the same task,
  so a slower function solution can still earn reward. An I/O task's attempt has the task pool
  verification's
  deadline: 2 s plus 0.25 s per case, at most 60 s (`scratchcoder.sandbox.budget.io_seconds`,
  which `tools/task_pool.py` `limits_for` also uses), so 3 cases get 2.75 s and 100 cases
  27 s. As in the pool verification's I/O program, the cases share that one deadline and
  there is no separate per-case limit: a case may use whatever time the earlier cases left, so
  a reference that passed the pool verification is not cut short here. A case that never
  ends uses up the budget, and the cases after it count as failed and as `unfinished`. The
  container's CPU and memory limits are the pool verification's as well. `score_attempt`
  accepts an explicit `budget` and `per_case` for tests; RL uses the defaults. The host
  declares `host_timeout` 15 s after the budget.
- **Statuses.** `score_attempt` returns a `RewardResult(status, reward, passed, total,
  audit_flags, elapsed_seconds, detail, unfinished, outcomes, budget_seconds)`. `outcomes`
  holds one pass flag per hidden check; `unfinished` counts checks that produced no result in
  time; `budget_seconds` is the budget the attempt ran under (`None` for a task that is not
  parseable).

  | Status | When | Reward |
  | --- | --- | --- |
  | `completed` | the final report arrived | `passed / total` |
  | `resource_kill` | the out-of-memory killer fired during the attempt, or exit 137 after `started` without a report | `passed / total` (killed checks fail), or `0.0` |
  | `malformed_report` | after `started`, the report was lost, malformed or a supervisor error | `0.0` |
  | `oversized` | the attempt is larger than the sandbox input limit; no container started | `0.0` |
  | `invalid_source` | the code has an unpaired surrogate and cannot be written as UTF-8; no container started | `0.0` |
  | `not_parseable` | the hidden tests are outside the supported forms | `None` |
  | `unavailable` | no docker binary, exit 125, or an unreachable daemon, before `started` | `None` |
  | `host_error` | any other failure before `started`, or a `spawn_failed` report | `None` |
  | `host_timeout` | the docker CLI had not finished 15 s after the budget | `None` |

  The last three are host failures. `score_with_host_retries` retries them up to a cap and, if
  they persist, returns the host failure rather than scoring `0.0`. `detail` says why an
  attempt that ran scored as it did (`output limit`, `exit status -9`, `out-of-memory kill`).
- **Degenerate-answer baseline.** `degenerate_baseline(task)` returns the best reward that
  one answer needing no solution would earn: a fixed return value or printed line
  (`FUNCTION_DEGENERATE_ANSWERS`, `IO_DEGENERATE_ANSWERS`), a value copied from the public
  tests, or echoing an argument (the first, the second, the last), all the arguments (as a
  tuple or a list), or the input (all of it, or its first or last line or token). It runs no
  code: it applies the channel's own comparison to the parsed hidden tests. For a function
  task only checks whose call is a direct call of a candidate function (`f(1)`, not
  `len(f(1))`) take the answer, so the value is a lower bound. The answer is assumed to
  define the names the tests use (module data such as `f(products)`): a program that does not
  gets a `NameError` there and earns less. The list of answers is still finite, so a policy
  that reads the task can do better than the baseline says.

## Design choices and common mistakes

- **Why the expected values stay on the host.** Forgery resistance does not come from an
  unforgeable channel — the candidate shares the child process with the driver and can write
  anything to the result pipe. It comes from the candidate never learning which value would
  pass. The supervisor keeps the first row per call index and drops junk, over-long or
  out-of-range rows; the host fixes the denominator from the task file, so flooding the pipe
  cannot raise the reward above genuine correctness.
- **Why the `started` frame decides attribution.** A host failure is skipped and retried, so
  if a candidate could cause one, a reward-seeking policy would learn to escape its zero
  reward. Before `started`, no candidate process exists, so any failure there is the host's.
  After it, every lost or malformed report scores 0.0, with two exceptions that a candidate
  cannot reach: a `spawn_failed` report, which PID 1 writes only when the candidate's first
  process could not be started, and a CLI that is still running at the stall deadline, which
  the supervisor's own bound (`OVERRUN_SECONDS`, 7 s past the budget) keeps out of a
  candidate's reach. An attempt too large for the input limit is refused on the host, before
  any container, so it cannot exhaust the supervisor before `started`. The nonce lets the host
  ignore any other text on the container's output.
- **Why PID 1 is hardened.** All processes in the container share one user. Without hardening,
  a candidate could interrupt the supervisor with `SIGINT`, write junk into its stdout through
  `/proc/1/fd/1`, or raise its `oom_score_adj` so the out-of-memory killer picks it. PID 1
  therefore handles catchable signals with a no-op and marks itself non-dumpable. It also kills
  every other process after the run and after each I/O case.
- **Why a fork server for I/O.** Under amd64 emulation an interpreter start costs about 75 ms,
  so 99 cases of fresh interpreters spend about 7.4 s before any candidate work. A fork costs
  about 2 ms, so the reward channel spends less of the shared deadline than the pool
  verification, which starts an interpreter per case. Each case still gets its own process
  from a clean parent, and the parent never holds an input or an expected output.
- **Why bare `assert CALL` is left out, and what protects the predicate-only forms.** The
  forms `assert not CALL` and `is None/True/False` have a passing outcome known in advance
  (falsy, or one singleton), and a bare `assert CALL` would too (any truthy value). What
  protects them is not the channel but that the candidate never sees which form a check uses;
  a constant answer still passes every check of its form, which `degenerate_baseline` counts.
  Supporting bare asserts would add the same kind of exposure; whether to do so is open.
- **Why a degenerate-answer baseline.** The RL advantage is a reward minus its group's mean,
  so on a task where returning `None` or printing `YES` earns 0.3, that answer beats a typical
  failing sample and the policy is paid for not solving. Screening can record the baseline per
  task and drop tasks it rewards too well.
- **Why Python `==`, not strict types.** The reward should decide a check the way the hidden
  assertion does, or RL would push the policy toward a stricter target than the evaluation
  that scores it (a strict comparison fails a correct `n / 2` against `== 2`). The
  channel's protection never depended on strictness: it comes from the expected value staying
  on the host and from rebuilding the reported value from built-in types, so a
  candidate-defined `__eq__` has nothing to run on. A value that cannot be a safe literal on
  the right-hand side (for example `float('nan')` written as a call) makes the task not
  parseable, so it never reaches comparison.
- **Mistake to avoid.** Do not run `==` on the host on anything but the rebuilt built-in
  value. Comparing an object from the candidate's process, or rebuilding into a type with its
  own `__eq__`, would hand the comparison back to candidate code.
- **Static audit.** `audit_attempt(code)` is pure and flags risky identifiers and patterns
  (`_getframe`, `co_consts`, `inspect`, `excepthook`, `atexit`, `os._exit`, `os.kill`,
  `__eq__`/`__ne__`/`__bool__` overrides, builtins tampering, `/proc`, `/tmp`, `__file__`,
  `sys.modules`, `gc`, `ctypes`, `signal`, dynamic imports, frame access through tracebacks,
  generators or `sys._current_frames`, trace hooks, `traceback`, `pdb`, `marshal`, imports of
  `__main__`, and `exec`/`eval`/`compile` of code that is not a constant). It scans every
  string constant in its final form (so `'__e' 'q__'` is seen), resolves module aliases
  (`import sys as s`), constant `__import__`/`importlib.import_module` names and names
  assembled from constant pieces (`+`, f-strings, `''.join`), audits constant code handed to
  `exec`, and does not flag `__exit__`, `can_exit` or `re.compile`. It reads the program as the
  sandbox does (the source as UTF-8 bytes, so a leading byte-order mark does not hide the
  syntax-tree scan), folds each chain of `+` once whatever its length, and flags code too
  deeply nested for the parser as `unauditable` (only the text scan saw it). It is best-effort:
  a name computed at run time any other way (`chr`, `bytes.fromhex`) is not seen, and a
  constant expression nested deeper than the interpreter's recursion limit counts as not
  constant. A flag is a monitoring signal, not a verdict: audited code still runs in the sandbox and is still scored
  only against the secret expectations. `AuditTally` accumulates the counts; its
  `rewarded_rate` (flagged share of attempts with reward above zero) is the leakage stop's
  rate.
- **Mistake to avoid.** Do not reuse this channel for evaluation scoring of record. The reward
  denominator here is the number of parsed hidden checks, which for not-parseable tasks is
  undefined; those tasks are screened out of RL, not scored in-process.
- **Mistake to avoid.** Do not treat `resource_kill` or `malformed_report` as host failures to
  retry: both happen after the candidate started and are the candidate's to own.

## Verification

- Implementation: [`src/scratchcoder/sandbox/reward.py`](../../src/scratchcoder/sandbox/reward.py)
  (host) and [`src/scratchcoder/sandbox/reward_program.py`](../../src/scratchcoder/sandbox/reward_program.py)
  (container supervisor), using the shared hardened command in
  [`src/scratchcoder/sandbox/runner.py`](../../src/scratchcoder/sandbox/runner.py) and the
  shared I/O deadline rule in
  [`src/scratchcoder/sandbox/budget.py`](../../src/scratchcoder/sandbox/budget.py).
- Reference tests: [`tests/test_reward_channel.py`](../../tests/test_reward_channel.py). The unit
  tests drive parsing; the comparison against Python's own `==` on the same values (numeric
  types, `nan`, `-0.0`, infinity, containers, hash-equal keys), identity and truthiness forms;
  malformed rows and values; the audit; the retry cap; the budgets (an I/O task gets the same
  deadline and container limits from the pool verification's `limits_for` as from the reward
  channel, and the host stall deadline follows the budget); and the host-status classification
  through a fake `docker` binary (a hung CLI, exit 125, an unreachable daemon, exit 137 before
  and after `started`, junk and forged frames). The Docker-backed tests prove that marker
  forgery, always-equal objects,
  forged and malformed result lines, early exit, `excepthook`/`atexit`, signals to PID 1,
  writes to PID 1's `/proc` files, output floods and invalid UTF-8, sleeping past the budget,
  fork bombs, filling `/tmp` and killing the fork server neither raise the reward above what
  correct values earn nor turn into host failures; that a correct solution earns `1.0` and a
  partially correct one the right fraction; that an out-of-memory kill is a `resource_kill`
  distinct from a timeout; that a correct 99-case I/O solution earns `1.0` with eight attempts
  running at once; that I/O cases share the pool verification's deadline (a 5 s case among 40
  passes, 100 cases taking about 10 s in all earn `1.0`, a case that never ends leaves the rest
  unrun); that the fork server's per-case verdicts equal the evaluation runner's on a reference
  corpus, and its all-cases verdict the pool verification's; that function-mode checks agree
  with the evaluation runner's in-process assertion, except for the documented reported-type
  and subclass residuals; that the candidate runs as the main script with an empty stdin (its
  own globals through `sys.modules['__main__']`, `sys.argv`, `inspect.getsource` and pickling
  work, and a candidate that rewinds stdin to read the call sources finds nothing); that output
  past the cap (the marker included), an `atexit` exit status, a never-ending thread, output
  after the last call, a stream a forked process holds open, and a closed or replaced stdout
  fail every check, as in evaluation, and that source text (a byte-order mark, encoding
  lines) is read as the evaluation program reads it; that a candidate that stops the fork
  server is still scored well inside the stall deadline; and that `/tmp` files persist
  across I/O cases in both paths. Unit tests also cover the `oversized` refusal, the
  supervisor's `spawn_failed` report before any candidate process and its absence after one,
  and the degenerate-answer baseline.
- Commands: `uv run --locked pytest tests/test_reward_channel.py` for the unit tests, and
  `SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_reward_channel.py` on a
  disposable Docker worker for the Docker-backed tests.

## Questions and answers

1. **Where do the expected test values live?** On the host. Only the candidate code and the
   left-hand call sources (function mode) or the case inputs (I/O mode) enter the container.
2. **How can a candidate not just print the right answer?** It can print anything, but the host
   compares the reported value against a hidden expected value it never shared, so matching it
   requires computing it — that is, solving the task.
3. **What happens to a task whose hidden tests use an unsupported assertion?** It is not
   parseable for this channel; RL screening drops it rather than falling back to in-process
   assertions.
4. **Why is a host failure not a reward of 0?** A stall or an unavailable daemon is a problem
   with the grader, not the candidate. Scoring it 0 would teach the policy to avoid tasks the
   host happens to fail on. The caller retries, then skips. Failures a candidate could cause are
   scored instead, because they all happen after `started`.
5. **Can a check fail here that the evaluation runner would pass?** Yes, when the call returns
   a value outside the reported types: a class the candidate defined with its own `__eq__`,
   or a standard-library value such as `Decimal('1.5')`, `Fraction(1, 2)`, `complex(2, 0)` or an
   empty `deque`. It cannot be encoded, so it fails here, while the in-process assertion would
   compare it. A subclass of a reported type is judged by its base value, so its own `__eq__`
   can make the two disagree in either direction. For plain built-in values the two agree.
6. **Why does an I/O attempt's budget grow with its cases?** It is the deadline the task
   pool verification gave the same task (2 s plus 0.25 s per case), so a reference that passed
   the pool verification in time is not cut short by the budget. A flat budget would cut
   correct solutions on tasks with many cases.
7. **Is this a kernel security boundary?** No. It reuses the development container's isolation
   flags unchanged and adds grading integrity on top. See
   [ADR 0004](../adr/0004-development-sandbox.md) for the threat boundary and residual risks.
8. **Can an answer that solves nothing still earn reward?** Yes, on weak tests: returning
   `None` or printing `YES` passes every check that expects that value.
   `degenerate_baseline(task)` measures this per task so RL screening can drop such tasks.
