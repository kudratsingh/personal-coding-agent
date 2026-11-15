# Component: the agent loop, harness v0 (mode H)

The smallest complete coding-agent workflow: find the function a failing test points at, ask
the model for one function, apply it, run the public tests, show the model the failure, retry
within budget, then grade the result with hidden checks and hand back a diff. The model only
writes one function per turn (the one thing the chat-v1 checkpoint was trained to do); the
harness does everything else, so every result is tagged as harness-assisted ("mode H").

Workspace and model code never run on the host. The host parses code with `ast` and edits
text; every execution goes through the evaluation path of record,
`scratchcoder.eval.harness.isolated_execution`, in the hardened container.

## Worked example

A two-module repository with one bug (the test suite's generic fixture):

```python
# arith.py
from steps import clamp


def double(x):
    """Twice ``x``."""
    return x + x + 1  # the bug


def scaled(x, low, high):
    return clamp(double(x), low, high)


# steps.py
def clamp(value, low, high):
    return max(low, min(value, high))


# test_arith.py (public)
from arith import double, scaled


def test_double():
    got = double(2)
    assert got == 4, repr(got)
```

1. **Feedback 0.** The public tests run in the sandbox. The trusted driver prints one row per
   test; the host condenses the failure:

   ```text
   RESULT run_tests: failed (1 of 1 tests)
   FAIL test_double (test_arith.py:6)
       assert got == 4, repr(got)
   AssertionError: 5
   ```

2. **Locate.** From `test_double` the host follows calls statically: `double` resolves through
   `from arith import double` to `arith.py::double`. (A test calling `scaled` would reach
   `scaled`, then `clamp` and `double`, nearest first.)
3. **Prompt.** One chat-v1 user turn: the task, a map (`arith.py: double, scaled`), a
   `SNIPPET arith.py::double` block with the function's current source, the condensed failure
   and a one-line instruction. Each block is counted with the frozen tokenizer.
4. **Edit.** The model replies with a python block. The host takes the one top-level function
   whose name is a shown function, here `double`, and replaces exactly the lines of the old
   `double` (decorators included). A reply that defines `triple` is refused (`unknown_name`);
   an `import re` the function never uses is dropped and recorded.
5. **Feedback.** The public tests run again. If they pass, the loop stops; otherwise the next
   prompt shows the refusal or the new failure, plus a `CHANGES SO FAR` block: the removed and
   added lines of the workspace against the original repository (every applied edit, not only
   the last one).
6. **Grade.** Two more containers: the hidden correctness checks and the preservation checks,
   each a plain assertion script that imports the edited modules. Solved means the public run,
   the correctness run and the preservation run all have supervisor status `passed`.
7. **Deliver.** A review packet: the unified diff (`difflib`), touched files, the feedback before
   and after, every turn's outcome, and the static audit flags of each applied function.

## Data flow and shapes

| Stage | Input | Output |
| --- | --- | --- |
| Case load | `case.json`, `repo/*.py`, hidden checks, reference files, registry | `Case` (all files hash-checked) |
| Pack | file name -> source text | one program: a prelude whose import hook serves each module from its text |
| Feedback call | (packed workspace, public driver), mode `function` | `Result` -> `SandboxRun(status, resolved, stdout)` |
| Condense | driver rows (untrusted, bounded) + host copy of the files | observation text |
| Locate | parsed tests and modules, failing test names | functions, nearest first, allowed files only |
| Budget | task, map, snippets, changes so far, observation | prompt token IDs; generation cap = min(324, 1,024 − prompt) |
| Policy | prompt IDs, cap, decoding | first closed python block (or none) |
| Edit | reply code, shown functions | new file text, or a refusal reason |
| Grade call | (packed source modules, hidden check script) | supervisor status per check set |

Block budgets (tokens): task 120, map 80, snippets 250 (one function at most 180), changes
so far 100, observation 150; prompt at most 700. Over budget a block is truncated; then the
changes-so-far block, the map and the farthest snippets are dropped, in that order. If the prompt still
does not fit, the instance ends as a context failure.

Failure classes for an unsolved, scored instance: `context` (the prompt did not fit, or the
target was never shown), `formatting` (no usable code), `tool_interface` (code the host refused:
an unknown name, several functions, a needed helper or import), `action_selection` (edits applied
but never to a target function), `model_capability` (the target was edited, the checks still
fail). `grading` marks an instance whose sandbox calls stayed host failures after the retries: it
is unscored, never a pass or a fail.

## Configurations, recovery and the funnel

`tools/agent_dev_run.py run` takes a named configuration, recorded in the scorecard
identity. The defaults reproduce the first development configuration exactly (a test pins its
prompt hashes). Cases, seeds (`turn_seed(case, turn, sample index)`), turns, token budgets, the
generation cap rule and feedback budgets are the same in every configuration, so a comparison
changes one factor at a time.

- `--prompt harness|training`. `training` renders the turn like an SFT pool row: the task, the
  shown functions' code without headers, the failing public tests' source, and on repair turns
  only the condensed failure, then the instruction. No map and no changes block. Same block
  budgets; the tests get the observation's 150.
- `--sampled-temperature`, `--sampled-top-p` change the sampled rows only; greedy rows are
  unchanged. These are agent decoding settings, separate from the RL screening protocol, which
  the agent package never imports (a test checks this and the screening constants).
- `--recovery off|intact-function`. Only for a reply that opens a python block and never closes
  it. It keeps a **verbatim prefix** of the block text: the whole text when the model ended its
  turn (end-of-turn token, not the length limit) and the text parses (`eos_whole_parse`), or the
  longest prefix that parses and is followed by another column-0 line, so every function in it
  ended before the generation ran out (`followed_by_top_level`). Nothing is added, edited or
  re-indented, and test outcomes are never consulted. The prefix then meets the same reply rules
  and editor as any reply. The turn records `recovered`, the rule and the kept span; the trace
  keeps the full original completion.

The scorecard's **funnel** counts, per generation and per instance: closed block (recovered
apart), parseable, exactly one shown function, that function is a target, applicable edit,
public tests passed, solved. `unknown_name` refusals are split into model task-following
failures (`invented_name`, `public_test_name`, `unshown_repo_function`) and a harness defect
(`locate_omitted_target`). Every refusal is also re-derived by an independent audit
(`scratchcoder.agent.audit`); a contradiction (a closed block reported missing, parseable code
refused as unparsable, a single shown function refused as several, an import the target
already has) is flagged as a harness defect. `agent_dev_run compare --runs DIR... --out-md
--out-json` writes one table across runs.

## Design choices and common mistakes

- **One program per call, not a new sandbox.** The runner takes one program and one test string.
  The repository is packed into the program; the import hook compiles each module under its own
  file name, so tracebacks name `textkit.py`, not the container's script. No supervisor or runner
  change was needed.
- **The verdict is the supervisor's status.** Feedback rows are printed by code in the same
  process as the workspace, so they are untrusted: they become model context and the packet,
  never a grade. Grading reads only the typed, nonce-framed status. No `repr` text is parsed for
  a grade.
- **Hidden checks stay out of the model's view.** Their sources go only into grading calls; the
  trace records their statuses, never their text, and grading output is not kept.
- **Residual risk.** As for every function-mode task, the checks run in the candidate's process,
  with their literals in memory and in the container's script file. A candidate could inspect
  them or the completion marker and forge a pass. The harness records `audit_flags` (the reward
  channel's static audit) for every applied function so such attempts are visible; it does not
  claim to prevent them.
- **Only the function's lines are written.** Other top-level statements in a reply are dropped
  when the function does not use them, and the reply is refused when it does, so an edit never
  changes code outside one function and never touches a test file or a file outside the allowed
  set.
- **Fresh single-turn prompts.** The checkpoint was trained on one user turn answered by one
  python block, so each turn is a new one-turn conversation, not a growing chat.
- **Generations are reused across conditions.** The single condition's turn and the repair
  condition's first turn have the same prompt and seed; the second is a cache hit, recorded as
  `cached`.
- **Common mistake:** deleting the prelude's helper names after installing the hook. The hook's
  methods look them up when an import happens, later.

## Verification

- Implementation: [`src/scratchcoder/agent/`](../../src/scratchcoder/agent/) and
  [`tools/agent_dev_run.py`](../../tools/agent_dev_run.py).
- Tests: [`tests/test_agent_workspace.py`](../../tests/test_agent_workspace.py) and
  [`tests/test_agent_loop.py`](../../tests/test_agent_loop.py), run by `make test`. They use
  scripted executors and fake policies only; a reference-fix oracle solves the fixture through the
  full loop, and one test checks that `isolated_execution` receives exactly the packed program and
  check text.
- The development run needs a passing execution-window receipt for its own command:

```sh
env -u SSH_AUTH_SOCK uv run --locked python -m tools.mac_window_check exec \
  --image-id <recorded image ID> --dotenv <absolute path to .env> \
  --receipt <new run directory>-receipt.json -- \
  uv run --locked python -m tools.agent_dev_run run --cases <dev cases> \
  --tokenizer <tokenizer> --checkpoint <checkpoint> --checkpoint-sha256 <hex> \
  --out <new run directory> --receipt <new run directory>-receipt.json
```

It validates every case in the sandbox first (original fails, reference passes three times,
negative controls fail), runs the oracle, then the model rows.

## Questions and answers

1. *Why can a feedback row not decide "solved"?* Workspace code prints into the same stream; it
   could print a passing row. Only the supervisor's status counts.
2. *What happens when the model rewrites a function that was not shown?* The edit is refused as
   `unknown_name`, the next prompt says so, and the turn counts against the budget.
3. *Why is a host failure not a failure?* It says nothing about the candidate; scoring it would
   mix infrastructure noise into capability numbers.
4. *Which prompt block is dropped first when the prompt is too long?* The changes-so-far
   block, then the map, then the farthest snippets.
5. *What does the oracle prove?* That locate, prompt, edit, feedback and grading fit together:
   writing the reference function through the same editor must solve every valid case.
