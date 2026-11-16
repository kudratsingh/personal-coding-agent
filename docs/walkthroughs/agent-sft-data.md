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
| Dev family | pool dev and train, instance sets and builds, specs, rows metadata, replay index | one status-only result: dropped instances and rows with reason classes, newly excluded replay rows, both inputs' SHA-256 |
| Mix | agent rows, replay rows, both contamination results, the dev-family result | rows at a target ratio of assistant-loss tokens, at most a few instances per distinct reply |

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
  an ID they already hold is refused. A passing grading call does not show that the
  independent checks have any power of their own, so `--oracle-controls-only` runs the four
  negative controls against the independent oracle alone, once per template, and builds
  nothing. It first runs the reference against the same checks: a template's rejections count
  only when that positive control passed, since a script that fails for every candidate would
  otherwise read as a perfect rejection. Its rows go to `oracle-controls.jsonl` and never stand in for the gating
  controls. `mix` applies the
  same cap and reports distinct replies, distinct (prompt, reply) pairs and the most rows on one
  reply, beside distinct templates, families, workflow types and mutation kinds.
- **Contamination results are bound to their inputs.** `mix` needs both results (agent rows
  and replay rows). Each must have been computed on exactly the file being mixed (by SHA-256)
  and against exactly the evaluation-case registries named with `--case-registry-sha256`.
  Their development-task signal is an n-gram check; the development task-family rule is a
  separate result that `mix` also requires (next section).
- **Training instances are not evaluation cases.** Generated instances keep the case format,
  but the case loader refuses them unless asked for them (`synthetic=True`), so they cannot be
  scored as development cases. A path with any part starting with `heldout` is refused.

## Development task families (`dev-family`)

An n-gram check misses a row that asks for the same behaviour as a development task in other
words. The pool's development split is protected by a task-family rule, `task-clusters-v1`
([walkthrough](task-clusters.md)), and `dev-family` applies that same rule to the
synthetic rows and to the replay rows (`scratchcoder.agent.synth.dev_family`). It runs
nothing: it reads text and parses it with `ast`.

**Nodes.** The rule needs a prompt and a solution per task. A pool task has one of each. A
synthetic instance gets several, so that no reading of "its prompt" or "its solution" is
missed:

- prompts: the task statement, the template's specification text (from every `specs.json`
  given), the target function's docstring and the user text of each of its rows;
- solutions: the reference target (every row's reply), the original buggy or stub target,
  each module of the fixed repository, and each top-level function or class of those modules
  on its own (test modules are left out; tests are never a signal).

The instance stands for every (prompt, solution) pair of these, joined into one family, so a
link through any pair marks the whole instance. Its node ID is `synth:<instance>`, which can
never equal a pool task's upstream ID. Two nodes under one instance ID (two instance sets that
reuse an ID) are joined, as the pilot intake joins repeated IDs.

**Links and families.** Two nodes are linked by a union signal: equal docstring-stripped
solution AST, equal AST up to renaming, equal upstream ID, or 3-word prompt shingle Jaccard at
least 0.5. A family is a connected component. The graph holds every pool development task,
every pool training task and every synthetic instance: accepted ones, the builds' rejected
ones, and superseded sets (`--bridge-instances`). Rejected and superseded instances are never
judged, but their links count, so a kept instance that reaches a development task only through
a rejected one is still caught ("bridge protection"). Excluded instances stay in the graph.

**Judging an accepted instance**, as the holdout integrity scan judges a subject:

| Status | When | Reason class |
| --- | --- | --- |
| excluded | its family holds any development task | `dev_family` |
| excluded | it shares a defined function or class name with a development task at prompt Jaccard ≥ 0.25 | `dev_named_paraphrase` |
| flagged | otherwise, a shared name at prompt Jaccard ≥ 0.10 | `dev_named_flag` |
| clear | otherwise | |

Excluded and flagged instances are both dropped: a flag cannot be adjudicated without
behaviour runs, so it is treated as unresolved and kept out. Then every instance of a template
that holds a dropped instance is dropped too (`template`), since a template is one task family
with renamed variants. Bridges are not judged, but their own shared-name edges to development
tasks are scored: a template that holds a bridge at the flag level or above is dropped the
same way (`templates_dropped_through_bridge_named_edges`). A dropped instance loses all its
rows (the edit row and both repair rows). The result reports instance-level and
template-expanded counts, and which drops only bridges reach (`only_through_bridges`: the same
judging with no rejected or superseded instance in the graph); the expanded set is the one
`mix` uses.

**Replay rows** are pool training tasks, judged by the family rule when the replay set was
built. A replay row is dropped when synthetic nodes add a development task to its family:
either its family held none without them (`rejected_new_chain`), or it held some and the
synthetic nodes join it to one more (`rejected_new_dev_link`; that link was never
adjudicated). Rows whose family held a development task without synthetic nodes and gains none
keep their adjudicated outcome and are counted. So are rows with a direct shared-name edge to
a development task: pending flag edges stay as the replay stage reported them. With
`--replay-record`, the replay stage's record path, its SHA-256 and the pending-flag counts it
reports are written beside the in-graph figure, so a reader can see that those rows were kept
as reported, not re-judged.

**Why the efficient form is the same rule.** The cross product of an instance's prompts and
solutions can hold hundreds of items, and pairwise edges among identical keys are quadratic.
The port keeps only what decides the outcome: nodes sharing a key are chained instead of
linked pairwise (same components); identical shingle sets are chained and every other prompt
pair is found once by the exact prefix-filtered search; and an instance's strongest
shared-name edge to a development task is its highest prompt Jaccard over the targets that
define one of its names. The reference tests run copies of the original `dev_family_ids` and
`scan` on the cross-product items of small fixtures and check that the port decides the same.

### Worked example

Development task `d0` sums `values` times three. Accepted instance `A` ships a helper
`shared_helper(z)`. Rejected instance `R` ships the same helper, and its task statement
paraphrases `d0`'s prompt (Jaccard ≥ 0.5). Without `R`, `A` is clear. With `R` in the graph,
`A`—`R` share a solution AST and `R`—`d0` share a paraphrase, so `A` is in `d0`'s family and
is excluded. Instance `F` defines a function with `d0`'s name and its task shares a few
3-word shingles with `d0`'s prompt (Jaccard 0.13): it is flagged and dropped. `F`'s template
sibling `G` is clear on its own; it is dropped by template expansion. A pool training task
`t` shares only its program with a superseded instance whose module also holds `d1`'s
program renamed: `t` is a replay row newly excluded.

### Results and binding

`dev-family` writes one result for both sides, status only (instance and row IDs, reason
classes, counts): `dev_family: "task-clusters-v1"`, the thresholds, the pool development and
training files' SHA-256, every instance set's registry SHA-256 and build file SHA-256, the
specification files' SHA-256, and per side `input_kind` and `input_sha256` (the rows metadata
and the decoded replay index). It never writes a prompt, solution, signal or development
task ID.

Each `--judged INSTANCES BUILD` pair must belong together. The build's `record.json` must name
the instance set's registry (`instances_registry_sha256`), and every row's `spec_sha256` must
equal the registered `case.json` hash of its instance. A superseded set that reuses the
instance IDs therefore cannot be judged in place of the set the rows came from. Leaving out
`--bridge-instances` prints a warning, and `mix --require-instance-registry SHA...` refuses a
result whose graph did not hold each named set (judged or bridge).

`mix --dev-family` refuses a result for another input, another rule or other thresholds, one
that did not judge every mixed replay kind, or one whose development file differs from the
file a contamination result checked; the agent-case and benchmark contamination results are
still required. The mix record states both development checks and the registry of every
instance set the result held. `manifest` describes builds, which come before the rule, so it
states the rule as required before mixing; with `--dev-family` it also records that result's
SHA-256, dropped counts and survivors.

```bash
uv run --locked python -m tools.agent_sft_data dev-family \
  --dev POOL/dev.jsonl --train POOL/train.jsonl \
  --judged INSTANCES_A BUILD_A --judged INSTANCES_B BUILD_B \
  --bridge-instances SUPERSEDED --specs SPECS_1 SPECS_2 \
  --rows-meta ROWS_META --decoded REPLAY_INDEX --replay-record REPLAY_RECORD \
  --out DEV_FAMILY.json
```

Limits: the rule is static, so two programs that behave alike but share no signal are not
linked. Because every instance of a template shares its specification text, one link usually
puts the whole template in a development family. The graph holds the pool's development and
training tasks only, never a sealed split, so a chain that passes through a held-out task is
not seen.

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
  receipt, cap, ledger, oracle-pin, control-gate, contamination-binding, dev-family-binding,
  held-out and synthetic-case refusals.
- `tests/test_agent_dev_family.py`: the reference tests against copies of `dev_family_ids` and
  `scan` (`tests/dev_family_reference.py`) on cross-product items, the link construction
  against `find_edges` components, each status and reason, the bridge regressions (kept
  instance and replay row reaching a development task through a rejected instance), a replay
  row joined to a further development task, template expansion (also through a bridge's
  shared-name edge), flagged instances dropped, the subcommand's result and refusals (including
  a swapped instance set and build), and the reader on `generate` output.

Run them with `uv run --locked pytest tests/test_agent_synth*.py tests/test_agent_dev_family.py`.

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
5. *Why drop a whole template when one instance matches a development task?* A template's
   instances differ only in names, constants and bug kinds; they are one task family. A
   flagged or named-paraphrase match cannot be resolved without behaviour runs, so the whole
   family stays out.
