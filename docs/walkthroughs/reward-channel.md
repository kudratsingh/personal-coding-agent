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
2. Inside the container the trusted supervisor runs the candidate in a child process,
   evaluates each call there, and reports a type-tagged encoding of the result on a private
   pipe: `["int", 3]`, `["int", 10]`.
3. The host decodes each reported value and compares type and value against the literal it
   kept. Two of two match, so `reward = 2 / 2 = 1.0`.

A candidate that returns `0` reports `["int", 0]` for both, matches neither `3` nor `10`, and
earns `0.0`. A candidate that returns an always-equal object cannot be encoded (its type is not
one of the representable types), so that call is a failure. A candidate that forges
`["int", 3]` on the pipe gains nothing, because it still has to produce the value `10` for the
second check, and it was never told what `10` is.

## Data flow and shapes

```
Task.hidden_tests ──► host parse ──►  (setup imports, [call sources], [expected literals])
                                             │                 │ kept on host
                                 call sources │ + candidate code
                                             ▼
                                   docker run (hardened, same flags as the
                                   evaluation runner) → reward_program.py (PID 1)
                                             │
                                   child process: run candidate, eval each call,
                                   write ["kind", payload] per call to a private pipe
                                             ▼
                                   supervisor → one JSON object to the host stdout
                                             ▼
host: decode each reported value → strict type+value compare vs the kept literals
      → passed / total, with total taken from the task file
```

- **Function mode.** `parse_function_checks` returns a `FunctionChecks` with the import
  `setup` and one `Check` per assertion. A `Check` is `(call, predicate, expected)` where
  `predicate` is `"eq"` (compare the decoded value to `expected`) or `"falsy"` (for
  `assert not CALL`, the host checks the decoded value is falsy). Supported assertion forms:
  `assert CALL == LITERAL`, `assert LITERAL == CALL`, `assert CALL is None/True/False`, and
  `assert not CALL`, where `LITERAL` is any `ast.literal_eval` value of a representable type.
  Any other top-level statement, or any assertion outside these forms, makes the whole task
  **not parseable** (`parse_function_checks` returns `None`, `is_parseable` is `False`), so a
  partial parse can never under-count the denominator and over-reward.
- **I/O mode.** `parse_io_cases` returns the inputs and the expected outputs. Only the inputs
  enter the container; the supervisor runs the candidate once per case under a per-case
  deadline and the per-attempt budget, returns each case's bounded stdout, and the host
  compares with the expected outputs it kept. Every case runs (no stop at first failure);
  cases that do not run count as failed. I/O tasks are always parseable.
- **The encoding** (`reward_program._encode` in the container, `_canonical_from_wire` on the
  host) preserves exact type: `bool` is distinct from `int`, `tuple` from `list`, `set` is
  compared order-independently, and `dict` by its items. A value whose type is not one of
  `None, bool, int, float, str, bytes, list, tuple, set, frozenset, dict` (recursively) cannot
  be encoded and counts as a failure.
- **Statuses.** `score_attempt` returns a `RewardResult(status, reward, passed, total,
  audit_flags, elapsed_seconds, detail)`. `reward` is `None` for a host failure
  (`unavailable`, `host_timeout`, `host_error`) and for a `not_parseable` task, so a host
  problem is never mistaken for a zero reward. A resource kill (OOM) is a distinct status with
  reward `0.0` — a failed attempt, not a host failure. `score_with_host_retries` retries host
  failures up to a cap and, if they persist, returns the host failure rather than scoring
  `0.0`.

## Design choices and common mistakes

- **Why the expected values stay on the host.** Forgery resistance does not come from an
  unforgeable channel — the candidate shares the child process with the driver and can write
  anything to the result pipe. It comes from the candidate never learning which value would
  pass. The host fixes the denominator from the task file and caps `passed` at the total, so
  flooding the pipe with forged rows, duplicate indices or out-of-range indices cannot raise
  the reward above genuine correctness.
- **Why a separate supervisor.** The candidate runs in a child process, not in the supervisor
  (PID 1). The supervisor only frames the result: it keeps the candidate's stdout off the
  container's stdout, bounds every buffer, enforces a per-call limit and an overall
  per-attempt wall budget (default 8 s), and emits one clean JSON object. A candidate that
  defeats the per-call alarm can at most consume the budget, after which the child is killed
  and the remaining calls are unrun failures — the reward only falls.
- **Why `float` comparison is by value, not repr.** `-0.0 == 0.0` matches the original `==`
  semantics; `nan` never matches, as `== nan` never would. A value that cannot be a safe
  literal on the right-hand side (for example `float('inf')` written as a call) makes the task
  not parseable, so it never reaches comparison.
- **Static audit.** `audit_attempt(code)` is pure and flags risky identifiers and patterns
  (`_getframe`, `co_consts`, `inspect`, `excepthook`, `atexit`, `os._exit`, `__eq__` overrides,
  builtins tampering, `/proc`, `__file__`, `sys.modules`, `gc`, `ctypes`, `signal`, and
  `open` of a `/tmp` path). A flag is a monitoring signal, not a verdict: audited code still
  runs in the sandbox and is still scored only against the secret expectations. `AuditTally`
  accumulates the counts the RL loop watches for its leakage stop.
- **Mistake to avoid.** Do not reuse this channel for evaluation scoring of record. The reward
  denominator here is the number of parsed hidden checks, which for not-parseable tasks is
  undefined; those tasks are screened out of RL, not scored in-process.

## Verification

- Implementation: [`src/scratchcoder/sandbox/reward.py`](../../src/scratchcoder/sandbox/reward.py)
  (host) and [`src/scratchcoder/sandbox/reward_program.py`](../../src/scratchcoder/sandbox/reward_program.py)
  (container supervisor), using the shared hardened command in
  [`src/scratchcoder/sandbox/runner.py`](../../src/scratchcoder/sandbox/runner.py).
- Reference tests: [`tests/test_reward_channel.py`](../../tests/test_reward_channel.py). The unit
  tests drive parsing, the strict type-and-value comparison, the audit, the host statuses and the
  retry cap with fake containers. The Docker-backed abuse tests prove that marker forgery,
  always-equal objects, forged result lines, early exit, exceptions from `excepthook`/`atexit`,
  output flooding, sleeping past the budget, fork bombs and reading `/proc` cannot raise the
  reward above what correct values earn, that a correct solution earns `1.0` and a partially
  correct one the right fraction, and that a resource kill is distinct from a candidate timeout.
- Commands: `uv run --locked pytest tests/test_reward_channel.py` for the unit tests, and
  `SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_reward_channel.py` on a
  disposable Docker worker for the abuse tests.

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
   with the grader, not the candidate. Scoring it 0 would teach the policy to cause host
   failures. The caller retries, then skips.
5. **Is this a kernel security boundary?** No. It reuses the development container's isolation
   flags unchanged and adds grading integrity on top. See
   [ADR 0004](../adr/0004-development-sandbox.md) for the threat boundary and residual risks.
