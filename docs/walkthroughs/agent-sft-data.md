# Component: synthetic agent-training data (edit and repair rows)

The agent loop's model writes one function per turn. To teach it the loop's prompt layout,
the generator builds small fresh repositories with a known bug, runs them through the real
loop with a scripted policy, and turns the prompts the loop builds into supervised rows:

- an **edit row**: the turn-1 prompt (task, map, shown functions, the real failure of the
  original repository's public tests) answered by the reference function;
- a **repair row**: the prompt after a wrong attempt (the attempt's diff in the
  `CHANGES SO FAR` block and its real failure as the observation) answered by the reference
  function.

The host never runs generated code. It renders text, parses it with `ast` and computes expected
values with its own model of each function. Repositories, references, wrong attempts and the
independent oracle run only in the sandbox, through `isolated_execution`.

## Worked example

A toy template (the tests' fixture) has a target `scale(x)` that should return `x * k` for an
instance constant `k`:

```python
# toolbox.py (reference)
def scale(x):
    """x times 3."""
    return x * 3
```

The bug kind `factor_up` is a text substitution, `x * 3` to `x * (3 + 1)`. The template's
spec function, ordinary project code, models both versions:

```python
def _scale(p, x, bug=None):
    factor = p["k"] + 1 if bug == "factor_up" else p["k"]
    return x * factor
```

For the public inputs `2` and `0` the generator calls the spec function twice per input. For
`2` it gets 6 for the reference and 8 for the bug, so `test_scale_two` (written as
`assert got == 6`) is a designed failing test. For `0` both give 0, so the bug does not reach
that test. The hidden correctness checks hold every target input. Inputs whose outcome the bug
leaves unchanged go into the preservation checks as well, so the original passes them by
construction. The task is written from the first designed failing test:
`Bug: scale(2) returns 8 instead of 6. Fix the bug.`

The wrong attempts are AST mutants of the reference: `return x * 4` (off by one) and
`return x` (wrong return value). The row builder runs the loop with a policy that replies with
mutant 1, then mutant 2, then the reference. That gives three prompts and three rows: one edit
row and two repair rows.

## Data flow and shapes

| Stage | Input | Output |
| --- | --- | --- |
| Render | template, seed, bug kind | names, module names, constants, structural variant; canonical sources |
| Assemble | rendered functions | modules (docstring, imports, functions), test module, hidden checks, task |
| Mutate | reference target | mutants that parse, keep the name and signature, and differ (after a canonical rewrite) from the reference and the original |
| Static check | instance | case-loader validation, sizes (20 lines, 180 tokens), the `agent_dev_run plan` fit rule, target shown, blocklist |
| Specs | catalog, instances | `specs.json`: behaviour, signature, constants, input domain, oracle file format |
| Loop | case, mutants, reference, oracle text | prompts of every turn, the loop's statuses |
| Render rows | prompt, reference | chat-v1 `SFTRow` (`id`, `ids`, `source`, `spans`) plus a metadata line |
| Contamination | rows | rejected row IDs only, the input's SHA-256 and the case registries checked |
| Mix | agent rows, replay rows, both contamination results | rows at a target ratio of assistant-loss tokens, at most a few instances per distinct reply |

Sandbox calls per instance, with `k` mutants: the original's public tests (F0), one public run
per mutant, the reference's public run, one grading call and the original against the hidden
correctness checks. That is `k + 4` calls, six with the default two mutants. The grading call's
test string holds the hidden correctness checks, the preservation checks and the independent
oracle's checks, so one typed verdict decides acceptance. The loop's `grade_checks` option
makes the loop grade with that one call; its default is unchanged. The four negative controls
run once per template, at one call each, before the template's first instance. A template's
instances are built only when all four controls failed its grading script, in this run or in an
earlier build against the same oracle (`--controls-from`). Otherwise they are recorded as
rejected and spend no calls. `controls.all_failed` is empty (`null`), not true, when no
control ran.

Every call is counted in one ledger file before it starts. The build refuses a call that would
take the total over the cap, and it refuses to start when the planned calls exceed what the cap
leaves. It also refuses a ledger file that does not exist or has no prior entries (a mistyped
path would restart the count at zero), a `--max-calls` above the approved cap, and a ledger
that another run holds: the run keeps an exclusive lock on the ledger and reads its entries only
after taking it. Earlier run records are counted once each, keyed by resolved path and file
hash; a copy of a counted record is skipped and reported.

## Design choices and common mistakes

- **Expected values come from project code, never from emitted code.** The spec function is a
  second implementation written next to the template. A mistake shared by the spec function
  and the reference text would validate itself, so a separate author writes an independent
  oracle from the natural-language spec alone. Its checks run in the same grading call.
- **The oracle is text.** The host reads the oracle file, checks it with `ast` (required
  functions, standard-library imports) and embeds it as a string literal in the test string.
  It never imports it.
- **The oracle set is pinned.** The oracle directory carries a `SHA256SUMS` manifest of every
  other file in it. `validate-and-build` takes the manifest's own SHA-256 as
  `--independent-sha256`, verifies every listed hash and refuses an unlisted, missing or
  changed file. It reads every oracle once, before the first call. Each validation record,
  control row and row-metadata line carries the manifest hash and the oracle file's hash.
- **Ambiguity is settled in the spec text.** When the oracle author reads a spec in a way the
  template did not intend, the spec text is clarified and `specs.json` is exported again. The
  author's files are never edited. A parametrised test checks that the spec text states each
  settled edge case and that the spec function gives the stated answer.
- **Sources are canonical.** Every template source must equal what `ast.unparse` prints. A
  mutant printed by `ast.unparse` then differs from the reference only on its mutated lines,
  so the `CHANGES SO FAR` block shows the real change rather than formatting noise.
- **Bug flags are unique within a family.** A caller's spec passes the bug kind to the
  functions it calls, so two functions that react to the same flag name would corrupt
  expected values. The catalog refuses that.
- **Tracebacks are modelled, then observed.** A traceback task names the frames the spec
  function's own exception passed through (each spec mirrors its function's calls). The build
  accepts the instance only if the original's real sandbox observation shows the same error
  line and raising function, both in the failure block of the test the task names.
- **Task claims are observed too.** A failing-test task (W1, W3) says which test fails and what
  the call returns or raises under the bug; those values come from the spec model. The
  instance records that claim, and the build accepts it only if the original's observation
  has a block for that test showing `AssertionError: <the claimed value>` (the public tests'
  assertion message) or the claimed exception. Otherwise the rows would pair a task statement
  with an observation that contradicts it.
- **A mutant may be equivalent on the public tests.** The host cannot always tell without running
  it. When a mutant passes the public tests, the loop stops and the instance is rejected. Some
  equivalent or never-ending mutants can be predicted from the code alone, and those sites are
  skipped: swapping the arguments of `max`, `min`, a helper marked symmetric or a `zip` whose
  pair is used symmetrically; `len(x[0])` to `len(x[-1])`; a default of `d.get` when `d` was
  built by a dict comprehension; `-0`; and inside a `while` loop, its test, the `if` that
  guards adding to its worklist, and the statements that update a name the test reads. A
  negation that only re-expresses a comparison (`not n % 2 == 1` for `n % 2 == 0`) counts as
  the same tree. Mutation kinds that usually change outcomes are tried first.
- **Diversity is counted by template, family and reply, not by row.** Thousands of rows from a
  few templates are variations, not independent problems. Every row of an instance has the
  same reply (the reference target), and a template renders only a limited number of distinct
  targets. So `generate` caps the instances that share one reply (`--max-per-reply`, three by
  default) and reports, per template, the distinct replies kept and the distinct replies its
  seeds can reach. Size an expansion by that reach, not by the call budget. An expansion uses
  a new `--seed-base`, which becomes part of each instance ID (`<template>-<bug>-s<base>-<n>`),
  and names the earlier sets with `--prior-instances`: their replies count toward the cap, and
  an ID they already hold is refused. `mix` applies the
  same cap and reports distinct replies, distinct (prompt, reply) pairs and the most rows on one
  reply, beside distinct templates, families, workflow types and mutation kinds.
- **Contamination results are bound to their inputs.** `mix` needs both results (agent rows
  and replay rows). Each must have been computed on exactly the file being mixed (by SHA-256)
  and against exactly the evaluation-case registries named with `--case-registry-sha256`. The
  development-task check on these rows is an n-gram check only, and every result and manifest
  says so.
- **Training instances are not evaluation cases.** Generated instances keep the case format,
  but the case loader refuses them unless asked for them (`synthetic=True`), so they cannot be
  scored as development cases. A path with any part starting with `heldout` is refused.

## Verification

- `tests/test_agent_synth.py`: catalog breadth and uniqueness rules; every template and bug kind
  builds, parses and passes the case loader's static validation; determinism; expected values
  and designed failing tests from the spec model; cross-module, stub and traceback layouts;
  mutant and control properties; the mutation skip rules and the canonical comparison; static
  checks and the blocklist; spec-text edge cases; task claims.
- `tests/test_agent_synth_rows.py`: rows from the real loop with a scripted executor (edit and
  repair rows, user-turn IDs equal to the harness prompt IDs, loss mask on the reply only);
  rejection when a mutant passes, when the original passes correctness, when a traceback is
  not observed in one failure block, or when a task claim is contradicted; negative controls;
  the loop's `grade_checks`; oracle text handling and the `SHA256SUMS` pin; the ledger's cap,
  seeding, lock and record keys.
- `tests/test_agent_synth_tool.py`: contamination signals and status-only output, mixing by
  loss tokens and the reply cap, and the `tools/agent_sft_data.py` subcommands, including the
  receipt, cap, ledger, oracle-pin, control-gate, contamination-binding, held-out and
  synthetic-case refusals.

Run them with `uv run --locked pytest tests/test_agent_synth*.py`.

## Questions and answers

1. *Why does the generator not just run the reference to get expected values?* Generated code
   is untrusted and runs only in the sandbox. The spec function gives the values on the host,
   and the sandbox confirms them.
2. *What stops a wrong spec function from producing wrong training targets?* The reference must
   also agree with an oracle written independently from the natural-language spec, in the
   same grading call.
3. *Why are repair rows built inside the loop rather than by pasting a diff into a prompt?* So
   the prompt is byte-for-byte what the harness shows after a real failed attempt: the same
   locate, budgets, truncations and condensed sandbox output.
4. *What does a 1:1 mixture mean here?* Equal assistant-loss tokens on the agent and replay
   sides. Row counts are reported beside it.
