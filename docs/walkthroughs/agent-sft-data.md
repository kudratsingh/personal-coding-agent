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

### Family items and exposure items

The rule needs a prompt and a solution per task. A pool task has one of each. A synthetic
instance's rows show much more than the task they train: buggy code, wrong attempts, helper
functions, harness boilerplate. Buggy input code is not a correct solution of the repair task,
but the model still sees it. So an instance carries two kinds of items (item mapping
`synthetic-items-v3`):

- **Family items** join families, as a pool task's prompt and solution do. They are what the
  rows train: the prompts are the task statement with the generator's fixed phrasing removed,
  the template's specification text (from every `specs.json` given) and the target
  function's docstring; the one solution is the reference target, which is every row's whole
  reply. Every case has one `module::function` target, so no whole module is a family item.
  The instance stands for every (prompt, solution) pair of these, joined into one family.
- **Exposure items** are everything else a row shows: the original (buggy or stub) target,
  the wrong attempts, each module of the original and the fixed repository and each of its
  top-level functions and classes, code inside each row's user text, and the full task
  statement and user texts as text. They never join a family. They are checked directly, so
  buggy and helper code still meets the AST and overlap checks (next section).

**Scaffolding** is what the generator emits identically for more than one template. It is
defined from the generator source, not from results, and it is never a family item:

- the task-statement phrasings (`instances.TASK_PHRASES`) and the traceback framing
  (`TRACEBACK_HEADER`, `FRAME_LINE`). `strip_task` removes them and keeps the values the
  generator filled in: the call, the values, test, exception, function and file names, and
  the traceback's frames and error line. A statement that matches no phrasing is refused,
  never stripped by guesswork;
- the harness prompt's fixed blocks (map, snippet headers, changes so far, results,
  instruction), which is why row user texts are exposure text only;
- the catalog's non-target functions. `unit_kinds` reads each template's roles from the
  catalog: the target, a *shared* role that another template also holds, or one only this
  template holds. Most templates of a family share their helpers.

Synthetic node IDs are `synth:<instance>`, which can never equal a pool task's upstream ID.
Two nodes under one instance ID (two instance sets that reuse an ID) are joined, as the pilot
intake joins repeated IDs.

**Links and families.** Two nodes are linked by a union signal on their family items: equal
docstring-stripped solution AST, equal AST up to renaming, equal upstream ID, or 3-word prompt
shingle Jaccard at least 0.5. A family is a connected component. The graph holds every pool
development task, every pool training task and every synthetic instance: accepted ones, the
builds' rejected ones, and superseded sets (`--bridge-instances`). Rejected and superseded
instances use the same mapping and are never judged, but their links count. A kept instance
that reaches a development task only through a rejected one is still caught ("bridge
protection"). Excluded instances stay in the graph.

### Judging an accepted instance

As the holdout integrity scan judges a subject, with the exposure checks added. The *exposure
targets* are the development tasks and every pool task in a family that holds one (on the
graph with the synthetic nodes).

| Status | When | Reason class |
| --- | --- | --- |
| excluded | its family holds any development task | `dev_family` |
| excluded | a defined name of its reference target is shared with a development task at prompt Jaccard ≥ 0.25 | `dev_named_paraphrase` |
| excluded | an exposure code item has the solution AST or renamed AST of an exposure target | `prompt_code_exposure` |
| excluded | an exposure text has prompt Jaccard ≥ 0.5 with an exposure target's prompt | `prompt_text_exposure` |
| excluded | a name defined in its family or exposure code is shared with a development task at Jaccard ≥ 0.25 over its family and exposure texts | `prompt_named_exposure` |
| flagged | otherwise, the first shared-name signal at ≥ 0.10 | `dev_named_flag` |
| flagged | otherwise, the second shared-name signal at ≥ 0.10 | `prompt_named_exposure` |
| clear | otherwise | |

Excluded and flagged instances are both dropped: a flag cannot be adjudicated without
behaviour runs, so it is treated as unresolved and kept out. Then every instance of a template
that holds a dropped instance is dropped too (`template`), since a template is one task family
with renamed variants. Bridges are not judged, but their own direct signals (shared-name edges
and every exposure check) are scored. A template that holds a bridge with one is dropped the
same way (`templates_dropped_through_bridge_hits`). `--drop-template` drops every instance of
the named templates whatever their judgement (`owner_ruling`); it needs `--drop-ruling`, the
record of the ruling, whose path and SHA-256 the result keeps. A dropped instance loses all
its rows (the edit row and both repair rows). The result reports instance-level and
template-expanded counts, each exposure signal's subject count, and which drops only bridges
reach (`only_through_bridges`: the same judging with no rejected or superseded instance in the
graph). The expanded set is the one `mix` uses.

### Replay rows

Replay rows are pool training tasks, judged by the family rule when the replay set was built.
A replay row is dropped when synthetic nodes add a development task to its family. Either its
family held none without them (`dev_family_new_chain`), or it held some and the synthetic
nodes join it to one more (`dev_family_new_link`; that link was never adjudicated). Rows whose
family held a development task without synthetic nodes and gains none keep their adjudicated
outcome and are counted.

A shared-name flag edge to a development task in the cluster manifest
(`--pending-flag-manifest`) is pending until someone reviews it. A replay row on such an edge
is kept only when an existing record resolves every one of its edges as not equivalent: the
edge's own review, or a decision (`--flag-decisions`) that is final `not_equivalent` for
exactly that (training task, development task) pair. Every other such row is dropped
(`pending_flag_unresolved`). The result lists the candidates, the decision lines that keep a
row and the manifest and decision files' SHA-256. With `--replay-record`, the replay stage's
own count of rows on pending flag edges must equal the candidates found here, or the run is
refused.

### Evidence of the mapping change

The earlier mapping, `synthetic-items-v2`, made every prompt and every solution a family item.
A buggy target, a helper that many templates share, or a formulaic harness text then joined
unrelated templates into one family. Every link that mapping made and this one does not must
be explained. `dev-family --evidence PATH` writes a private file with every such link
(pairwise, any union signal, pool-only pairs left out). Each link carries the category of the
items that made it:

| Category | The item that made the link |
| --- | --- |
| `original_target_as_solution` | the original (buggy or stub) target |
| `repo_module_as_solution` | a whole module of the fixed repository |
| `shared_scaffolding_code` | a non-target function that the catalog gives more than one template |
| `non_target_function` | a non-target function only this template has |
| `non_target_code_unresolved` | a non-target function whose role the recorded names do not give |
| `harness_text` | a row's user text |
| `task_phrase_scaffolding` | the task statement with its fixed phrasing |

A link counts against the side that holds no retained item (reference target, specification,
docstring, pool items), because a retained item on both sides would make the link in both
mappings. A removed link with no category is a bug, and the run fails. The file also gives
every link this mapping adds (`stripped_task_statement`), each template pair with the signals,
item kinds and defined names involved, the earlier mapping's judgement and the transitions
between the two. The result carries the summary counts and the file's SHA-256.

### Why the efficient form is the same rule

The cross product of an instance's prompts and solutions can hold many items, and pairwise
edges among identical keys are quadratic. The port keeps only what decides the outcome. Nodes
sharing a key are chained instead of linked pairwise, which gives the same components.
Identical shingle sets are chained, and every other prompt pair is found once by the exact
prefix-filtered search. An instance's strongest shared-name edge to a development task is its
highest prompt Jaccard over the targets that define one of its names. The reference tests run
copies of the original `dev_family_ids` and `scan` on the family items of small fixtures and
check that the port decides the same.

### Worked example

Development task `d0` sums `values` times three. Template `A`'s target is that program with
other names, so `A`'s instances are in `d0`'s family (`dev_family`). Template `B`'s target
reverses a string, but its repository holds the same generic `bounded` helper as `A`'s. Under
the earlier mapping the helper was a family item and joined `B` to `A`, so `B` was excluded.
Now the helper is shown code: it is no development solution, so `B` is clear. The evidence
records the removed `A`–`B` link as `shared_scaffolding_code`, naming `bounded`.

Template `C`'s buggy original target is `d0`'s program renamed. The fix differs, so no family
item links `C`. But every row of `C` shows that code, so `C` is excluded
(`prompt_code_exposure`), and its sibling `C2` is dropped by template expansion. A rejected
instance `R` that shares `A`'s helper only stays out of any family. A rejected instance whose
task statement paraphrases `d0` and whose target is a training task's program still bridges
that training task into `d0`'s family. A replay row `t` on a pending flag edge to `d1`, with
no decision on that pair, is dropped.

### Results and binding

`dev-family` writes one result for both sides, status only (instance and row IDs, reason
classes, counts). It records `dev_family: "task-clusters-v1"`, the thresholds, the item
mapping, the pool development and training files' SHA-256, every instance set's registry
SHA-256 and build file SHA-256, the specification files' SHA-256, the owner drops, the pending
flag block, the link-change summary and the evidence file's SHA-256. Per side it records
`input_kind` and `input_sha256` (the rows metadata and the decoded replay index). It never
writes a prompt, solution or development task ID.

Each `--judged INSTANCES BUILD` pair must belong together. The build's `record.json` must name
the instance set's registry (`instances_registry_sha256`), and every row's `spec_sha256` must
equal the registered `case.json` hash of its instance. A superseded set that reuses the
instance IDs therefore cannot be judged in place of the set the rows came from. Leaving out
`--bridge-instances` prints a warning, and `mix --require-instance-registry SHA...` refuses a
result whose graph did not hold each named set (judged or bridge).

`mix --dev-family` refuses a result of the earlier schema or another item mapping, and one for
another input, another rule or other thresholds. It also refuses one that did not judge every
mixed replay kind, did not apply the pending flag edges, or used a development file other than
the one a contamination result checked. With `--require-dropped-template` it refuses a result
that did not drop those templates by owner ruling. The agent-case and benchmark contamination
results are still required. The mix record states both development checks, the registry of
every instance set the result held, and how many pending-flag candidates entered the mix.
`manifest` describes builds, which come before the rule, so it states the rule as required
before mixing. With `--dev-family` it also records that result's SHA-256, item mapping, owner
drops, dropped counts and survivors.

```bash
uv run --locked python -m tools.agent_sft_data dev-family \
  --dev POOL/dev.jsonl --train POOL/train.jsonl \
  --judged INSTANCES_A BUILD_A --judged INSTANCES_B BUILD_B \
  --bridge-instances SUPERSEDED --specs SPECS_1 SPECS_2 \
  --rows-meta ROWS_META --decoded REPLAY_INDEX --replay-record REPLAY_RECORD \
  --pending-flag-manifest CLUSTER_MANIFEST --flag-decisions DECISIONS \
  --drop-template TEMPLATE --drop-ruling RULING \
  --evidence PRIVATE/EVIDENCE.json --out DEV_FAMILY.json
```

Limits:

- The rule is static, so two programs that behave alike but share no signal are not linked.
- Every instance of a template shares its specification text, so one link usually puts the
  whole template in a development family.
- Exposure code is compared with development tasks and with pool tasks in their families, not
  with the family items of other synthetic instances. Shown code that equals another template's
  reference target is excluded only when that code meets a development family's pool program
  itself.
- `clusters.py` filters no trivial programs beyond those that do not parse. A short helper that
  equals a short development solution up to renaming is therefore an exposure hit.
- A stripped task statement is short and can link two instances that name the same call and
  values. The evidence file lists every such added link.
- The graph holds the pool's development and training tasks only, never a sealed split, so a
  chain that passes through a held-out task is not seen.

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
  `scan` (`tests/dev_family_reference.py`) on family items, the link construction against
  `find_edges` components, each status and reason, the bridge regressions (kept instance and
  replay row reaching a development task through a rejected instance), a replay row joined to
  a further development task, template expansion (also through a bridge's direct signal),
  flagged instances dropped, the two mapping regressions (templates joined only by a shared
  helper stay apart; a development solution shown in a prompt is excluded), text and
  shared-name exposure, the task-phrasing stripper, pending flag edges and their citations,
  the owner drop, evidence categories (on `generate` output too, and a removed link without
  one refused), the subcommand's result and refusals (including a swapped instance set and
  build), and the reader on `generate` output.

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
5. *Why are buggy code and helpers not family items?* Each row trains one reply, the
   reference target. A buggy original or a helper that many templates share says nothing
   about which task the row teaches, so as a family link it joins unrelated templates.
   The model still sees that code, so it is checked directly against development
   solutions.
6. *Why drop a whole template when one instance matches a development task?* A template's
   instances differ only in names, constants and bug kinds; they are one task family. A
   flagged or named-paraphrase match cannot be resolved without behaviour runs, so the whole
   family stays out.
