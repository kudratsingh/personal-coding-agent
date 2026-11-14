# Component: evaluator conformance catalogue

The project scores generated code on two paths. The **scoring-of-record path**
(`scratchcoder.eval.harness.isolated_execution`, which runs `sandbox/runner.py` with the
`program.py` supervisor) runs the candidate, the hidden tests and a completion-marker `print`
as one script. It accepts the run when the process exits 0 and its output ends with the marker.
The **RL reward channel** (`scratchcoder.sandbox.reward.score_attempt`) keeps every expected
value on the host. It sends only the call expressions into the container and compares the
values the calls report.

The conformance catalogue (`scratchcoder.sandbox.conformance`) is a declared set of small
synthetic cases, both controls and attacks. Each case records the verdict that each path gives
**today**. It is a characterization, not a wish list. A weakness that exists now is written
down as current behaviour with a `known_weakness` label. A later hardening has to change that
expected verdict on purpose, in the same change that alters the evaluator.

None of the cases contains dataset text. All of them were written for this catalogue.

## Worked example

Take the task "write `add(a, b)`" with the hidden tests `assert add(2, 3) == 5` and
`assert add(-4, 10) == 6`. Here are four candidates:

| Case | Candidate | Path of record | Reward channel |
|---|---|---|---|
| `correct_program` | `return a + b` | `passed` (control) | `completed`, 1.0 (control) |
| `constant_return` | `return 5` | `failed`: the second assertion stops the script | `completed`, 0.5: one of two checks (intended divergence, partial credit) |
| `eq_override_object` | returns an object whose `__eq__` is always `True` | `passed` (**known weakness**: the assertion runs in the candidate's process) | `completed`, 0.0: the object cannot be encoded, so every check fails (control) |
| `marker_forgery_code_consts` | reads the marker from the module's code-object constants, prints it, then calls `os._exit(0)` | `passed` (**known weakness**) | `completed`, 0.0: no call reported a value (control) |

The reward channel resists the last two because the comparison happens on the host. A forged
marker does not help, because the host still needs reported values that equal expectations the
candidate never saw.

The reward channel has its own known weakness, `side_effect_stub`. The prompt asks the function
to print a greeting, but the tests only assert `greet(...) is None`. An empty function then
earns full credit on **both** paths, because no assertion checks the side effect.

## Data flow and shapes

```
Case(name, mode, prompt, code, tests, solves,
     record=RecordVerdict(statuses, label),
     reward=RewardVerdict({(status, reward), ...}, label),
     audit_flags)
        │
        ├── pure tests: parse code/tests as text (never run), reward.parse_task(case.task()),
        │               reward.audit_attempt(code), label/verdict consistency
        │
        └── Docker-gated tests (inside the isolated container only):
              isolated_execution(code, tests, mode).status        ∈ record.statuses
              score_attempt(case.task(), code) → (status, reward) ∈ reward.outcomes
```

- **Record verdicts** are sets of `runner.Result.status` values: `passed`, `failed`,
  `timeout`, `output_limit` or `resource_limit`. Host failures (`unavailable`,
  `infrastructure_error`, a host stall) are never an expected verdict. A gated test treats one
  as a test failure, not a skip.
- **Reward verdicts** are sets of `(RewardResult.status, reward)` pairs, for example
  `("completed", 0.5)`, or `("not_parseable", None)` for a task the channel refuses.
- **Labels** take the form `kind` or `kind: reason`. The kind is one of `control`,
  `known_weakness`, `intended_divergence`, `refused` or `nondeterministic`. The unit tests
  check that each label agrees with its verdict:
  - a `control` passes a solution and fails a non-solution;
  - a `known_weakness` gives a non-solution credit;
  - `refused` means exactly `not_parseable`;
  - only `nondeterministic` may allow more than one outcome.

The catalogue has 28 cases in eight groups:

- **Controls:** a correct program, a return-`None` stub, a `pass` body, a program that defines
  nothing, a constant return and an echo of the argument.
- **Zero-check tests:** `pass`, a comment only, and a `def check(...)` that is never called.
  Each is scored against a stub. The path of record passes all three, and the reward channel
  refuses all three.
- **Marker forgeries:** reading the marker from `/tmp/candidate.py`, from code-object
  constants or by walking frames, and printing it from an `excepthook` or an `atexit` callback
  before `os._exit(0)`.
- **Early exits without the marker:** `os._exit(0)` and `sys.exit(0)` before the tests, and
  an excepthook that prints the marker but leaves the exit status at 1. All of them fail on both
  paths.
- **Reading the expected values:** `expected_value_lookup` parses the assertions out of
  `/tmp/candidate.py` and returns the expected literal for each call. On the path of record
  every check then executes and passes. On the reward channel the file holds only the
  candidate.
- **Tricks in the shared process:** an always-equal object, an always-equal `int` subclass, a
  replaced `math.isclose`, and a module-level `isinstance` that shadows the builtin.
- **Contract gaps in the tests:** a side-effect task without a side-effect assertion, and a
  reference whose result order depends on the string hash seed (`nondeterministic` on both
  paths).
- **Limit and mode differences:** a call that sleeps 2.5 s (a timeout on the 2 s path of
  record, full credit within the reward channel's 4 s per-call deadline), plus a correct I/O
  program and a constant I/O program.

## Design choices and common mistakes

- **Characterize, do not aspire.** An expected verdict says what the current code does. When a
  hardening closes `marker_forgery_code_consts` on the path of record, the record verdict
  changes from `passed` to `failed` and the label from `known_weakness` to `control`. The
  pinned list in `test_known_weaknesses_are_pinned` changes in the same commit. A silent
  improvement fails the gated test just as a silent regression does.
- **A reported nonce and count is not a trustworthy verdict.** Code that runs in the
  candidate's own process can read anything that process holds, so it can forge both.
  `expected_value_lookup` goes further: it makes every check execute and pass, so a correct
  count would not expose it. The path of record keeps its weaknesses until the comparison
  moves outside the candidate's process, as the reward channel's host-side comparison does.
- **The static audit is monitoring, not a guard.** `audit_flags` records what
  `reward.audit_attempt` reports for each candidate. Two forgeries, a replaced `math.isclose`
  and a shadowed `isinstance`, carry no flag at all.
- **Refusal is not a pass.** The reward channel refuses a task with no supported assertion
  (`not_parseable`, reward `None`). The tests confirm that it never starts a container for one.
- **Predicted, not yet observed.** The expected verdicts were derived by reading the runner,
  the supervisors and the reward channel. They have not yet been run against the pinned
  container. The first gated run is the observation: any mismatch there is a finding to
  investigate, and it must not be edited away.
- **Out of scope:** NumPy-backed benchmark scoring (the EvalPlus image), benchmark suites
  (none of them is parseable on the reward channel), and the sealed test split.

## Verification

- Implementation: [`src/scratchcoder/sandbox/conformance.py`](../../src/scratchcoder/sandbox/conformance.py).
- Tests: [`tests/test_evaluator_conformance.py`](../../tests/test_evaluator_conformance.py).
- The pure tests run with `make test` and need no Docker.
- The gated tests run every case through both real paths in the pinned container. On the
  owner's Mac they run only through a passing execution-window check
  ([`tools/mac_window_check.py`](../../tools/mac_window_check.py)):

```sh
env -u SSH_AUTH_SOCK uv run --locked python -m tools.mac_window_check exec \
  --image-id <recorded stdlib image ID> --dotenv .env -- \
  env SCRATCHCODER_SANDBOX_TESTS=1 uv run --locked pytest tests/test_evaluator_conformance.py
```

The gated tests do not need the NumPy (EvalPlus) image.

## Questions and answers

1. *Why does `constant_return` fail on one path and earn 0.5 on the other?* The path of record
   gives a whole-task verdict: the first failing assertion stops the script. The reward
   channel gives partial credit, `passed / total`.
2. *Why can the reward channel's own marker be read (`marker_forgery_frame_walk`) without
   this being a weakness?* That marker only checks the process's end-of-output behaviour.
   Credit comes from reported values compared on the host, and forging the marker supplies
   none.
3. *What would stop `zero_check_pass` from passing on the path of record?* The task itself
   would be refused before any candidate runs: the host counts the declared checks in the
   test text statically and rejects tests with too few. No count reported from inside the
   candidate's process is needed or trusted, because that process can forge one.
4. *Why is `hash_order_reference` allowed two outcomes?* Under `python -I` the string hash
   seed changes in every process, so `list(set(words))` returns a different order from run to
   run.
5. *Does an empty audit-flag tuple mean a candidate is honest?* No.
   `shadow_builtin_isinstance` forges a pass with no flag.
