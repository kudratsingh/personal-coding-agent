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
| Budget | task, map, snippets, changes so far, observation | prompt token IDs; generation cap = min(324, 1,024 − prompt − prefill) |
| Policy | prompt IDs, cap, decoding | first closed python block (or none) |
| Edit | reply code, shown functions | new file text, or a refusal reason |
| Grade call | (packed source modules, hidden check script) | supervisor status per check set |

Block budgets (tokens): task 120, map 80, snippets 250 (one function at most 180), changes
so far 100, observation 150; prompt at most 700. Over budget a block is truncated; then the
changes-so-far block, the map and the farthest snippets are dropped, in that order. If the prompt still
does not fit, the instance ends as a context failure.

`agent_dev_run plan` checks each case statically before any run (`prompt.fit_bound`). Its
worst-case prompt of any turn takes the turn-1 prompt, grows the snippet block to its full 250
(a later turn can show a longer function, such as an edited one grown up to the 180-token
function budget), adds two tokens of separator for each snippet beyond the turn-1 count that a
turn could show (each snippet costs at least its header, the headers of the case's showable
functions must share the 250, and an edit never adds a function), adds the observation and
changes-so-far blocks at their full budgets and a few tokens for their separators. Blocks are
counted alone, so this is an estimate within that per-join allowance. A case fits when that prompt is within 700 tokens, that prompt plus
1.2 times the whole reference fix fits the 1,024 context, and 1.2 times the largest reference
function plus its two block tokens fits the 324-token cap. Since 700 + 324 = 1,024, a prompt
within the budget always leaves the full cap. This is planning only; at run time the budgets
are enforced by truncation, drops and the context failure.

**Case IDs are identities.** Every output is keyed by case ID: trace and packet names
(`{case}--{policy}--{condition}--{decoding}`), scorecard cells and comparison rows. An ID is 1 to
32 letters, digits, `_` or `-`, starts with a letter or digit, does not end with `-` and never
contains `--`, so a file name splits back into one ID. Loading a registry refuses any malformed
ID and any two IDs that are equal ignoring case (they would share a file on a case-insensitive
file system), whichever cases a run selects. A scorecard refuses two results for the same
instance and case IDs that collide; a comparison refuses two runs with the same name and any
run whose case IDs collide. So two distinct cases can never merge into one count or overwrite
each other's files.

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
  longest prefix that parses and is followed by a new logical line at indentation 0 (found with
  the tokenizer, so a bracket or backslash continuation is not a boundary): every function in
  it ended before the generation ran out (`followed_by_top_level`). The dropped tail must be
  inert: after trailing blank and comment lines, it must parse completely into expression
  statements that bind no name and contain no comprehension or assignment expression. A
  truncated tail, or anything else, could change what the kept function means, so recovery
  refuses instead; so does any completion with a form feed, vertical tab or carriage return. Nothing is added, edited or
  re-indented, and test outcomes are never consulted. The prefix then meets the same reply rules
  and editor as any reply. The turn records `recovered`, the rule and the kept span; the trace
  keeps the full original completion.
- `--prefill-name` (off by default) is a **harness aid**, not a model capability. Before each
  model generation the harness writes the start of the reply: a python block opener (in chat-v1
  the `<|python_start|>` token, the counterpart of a markdown `` ```python `` line) and
  `def <name>(`, where `<name>` is the function of the nearest shown snippet (`prompt.shown[0]`,
  the selection the prompt already displays). It is computed from the prompt alone
  (`prompt.name_prefill`); the policy is never given the case's targets, reference or hidden
  checks, and a test runs it on a case whose target is not the nearest shown function. It is
  skipped, and the turn records `prefill_skipped: decorated`, when that function has decorators:
  the editor replaces a function's whole span, decorators included, and a reply opened with
  `def <name>(` cannot write them, so a prefilled edit would silently delete them. The prefill
  never copies other lines of the shown source. The reply that is parsed is prefill + generated
  tokens. Prefill tokens count against the 1,024 context (the cap is
  min(324, 1,024 − prompt − prefill)) but not against the generated-token budget, and they are
  recorded apart: `prefill_function`, `prefill_text`, `prefill_tokens` and `prefill_skipped` in
  the trace, the turn and the packet; generated tokens exclude them. The oracle rows never use
  it. The scorecard and the comparison label name-following under the prefill as
  harness-supplied: the `one_shown_function` and `target_function` stages and the name failures
  then count names the harness wrote, never model task-following. After the run, and for
  reporting only, the tool reads the traces back and reports how many instances and generations
  had a prefilled name equal to a target's. The comparison carries the same counts per row: a
  `Prefilled generations` column (`k of N generations`) and a name-prefill applicability table
  with the skipped generations, prefill tokens and the instances and generations with a prefilled
  target name, each with its denominator. Without the option, prompts, generation requests and
  traces are byte-for-byte those of earlier runs (a test pins their hash). Byte-pair merges can
  differ at the boundary between the prefilled `(` and what follows, compared with encoding the
  whole function at once.

The scorecard's **funnel** counts nested stages, per generation and per instance: closed or
recovered block, parseable, exactly one shown function, that function is a target, an applied
edit to it, public tests passed, solved. Each stage counts only what reached the previous one;
side columns hold fresh generations, closed and recovered blocks apart, applied edits to a
function that is not a target, and instances solved without a target edit. `unknown_name` refusals are split into model task-following
failures (`invented_name`, `public_test_name`, `unshown_repo_function`) and a harness defect
(`locate_omitted_target`). Every refusal is also re-derived by an independent audit
(`scratchcoder.agent.audit`); a contradiction (a closed block reported missing, parseable code
refused as unparsable, a single shown function refused as several, an import the target
already has) is flagged as a harness defect. `agent_dev_run compare --runs DIR... --out-md
--out-json` writes one table across runs. The first run is the baseline; every run must share
its code commit (clean tree), cases, case registry, checkpoint, tokenizer, seeds, decoding rows, conditions, budgets,
threads, batch size, supervisor and runtime image, or the comparison is refused (unless
`--allow-noncomparable`, which marks those rows). Each row names the configuration fields that
differ and is labelled baseline, single factor (prompt, sampled decoding as one factor,
recovery, or the name prefill) or combined. Sampled decoding is named by what changed from the
default T 1.0, top-p 1.0: T 0.6 with top-p 0.95 is "lower temperature plus nucleus sampling
(T 0.6, top-p 0.95)", never just a lower temperature. The name prefill may differ from the
baseline only as the one differing factor; combined with another factor the row is marked
non-comparable. `--vary-checkpoint` compares checkpoints (a trained model against its parent on
the same cases, seeds and code): the checkpoint then leaves the shared fields and may differ only
as the one differing factor, labelled with its hash prefix; every other field must still match. In both markdown tables every count carries its denominator: funnel stages out
of the row's generations or instances, failure classes out of scored instances, name failures
out of generations and harness defects out of instances.

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
- Tests: [`tests/test_agent_workspace.py`](../../tests/test_agent_workspace.py),
  [`tests/test_agent_loop.py`](../../tests/test_agent_loop.py),
  [`tests/test_agent_experiments.py`](../../tests/test_agent_experiments.py),
  [`tests/test_agent_prefill.py`](../../tests/test_agent_prefill.py) and
  [`tests/test_agent_case_ids.py`](../../tests/test_agent_case_ids.py), run by `make test`. They use
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
6. *Does a solve under `--prefill-name` show that the model follows the task?* No. The harness
   supplied the function name; only what the model generated after `def <name>(` is its own.
