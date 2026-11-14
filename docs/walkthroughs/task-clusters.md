# Component: task-family clusters and adjudication

A held-out dev task should not have an equivalent copy in training. The pool build already
drops benchmark IDs, near-duplicate prompts and loose prompt paraphrases of holdouts, but it
compares *prompts*. Two tasks can ask for the same function in quite different words, so
the same behaviour can sit on both sides of the split. This component looks for such
*families* in reference solutions and prompts, records why each pair was linked, and keeps
the decision about equivalence for a behavioural test and a reviewer.

It finds candidates. It never confirms that two tasks are equivalent on its own.

## Worked example

Three tasks, reference code only:

```python
# dev/count                         # train/count                       # train/changed
def count_matching_ends(text):      def count_matching_ends(text):      def count_matching_ends(text):
    """Count matching ends."""          # same thing, a comment             """Count matching ends."""
    total = 0                           total = 0                           total = 0
    for i in range(len(text)):          for i in range(len(text)):          for i in range(len(text)):
        for j in range(i, len(text)):       for j in range(i, len(text)):       for j in range(i, len(text)):
            if text[i] == text[j]:              if text[i] == text[j]:              if text[i] != text[j]:
                total += 1                          total += 1                          total += 1
    return total                        return total                        return total
```

1. **Solution AST key.** Each reference is parsed, module/class/function docstrings are
   removed, and the AST is dumped without line numbers and hashed. `dev/count` and
   `train/count` get the same key: comments, blank lines and docstrings are not part of
   the AST. `train/changed` differs by one operator (`==` against `!=`), so its key differs.
2. **Canonical AST key.** Every name the program binds is renamed in order of first
   appearance (`count_matching_ends` becomes `_v0`, `text` becomes `_v1`, and so on), and
   annotations are dropped. `def total(values)` and `def summed(xs)` with the same body
   shape match here but not on the first key. This is a *candidate* signal: an identity
   function `def f(x): return x` matches every other identity function, whatever its task.
3. **Prompt shingles.** Prompts are cut into 3-word shingles and compared by Jaccard
   similarity. At 0.5 or more the edge joins a cluster. With a shared defined name, 0.25
   or more and 0.10 or more are recorded as *flag-only* edges for a reviewer.
4. **Upstream identity.** The same upstream record under two IDs joins a cluster.

The edge `dev/count`–`train/count` carries `solution_ast` and `canonical_ast`, plus
evidence: shared defined names `["count_matching_ends"]`, prompt Jaccard, and whether
source, mode and module-level names agree. Union-find over the joining edges gives the
cluster `{dev/count, train/count}`; `train/changed` stays alone.

## Data flow and shapes

```
pool train.jsonl, dev.jsonl ──► Item(id, split, source, prompt, solution, mode, roles)
SFT and RL-candidate IDs    ──► roles: "sft_train", "rl_candidate"
                                 │
          find_edges ◄───────────┘  N items → E edges, each {signal: value, evidence}
                │
          components (union-find over union signals) → clusters (lists of item indices)
                │
          cluster_record → {cluster_id, members, categories, edges, behavior[], adjudication,
                            uncertainty[]}
                │
          tools.task_clusters candidates → manifest.json + record.json (new directory)
                │
          tools.task_equivalence run (sandbox, inside a Mac window) → behavior.jsonl
                │
          tools.task_equivalence merge → manifest with behaviour outcomes
                │
          tools.task_clusters adjudicate → reviewer decision + adjudication-log.jsonl
```

Pair search is not quadratic over the whole pool. Exact keys group items by hash. The 0.5
prompt rule uses prefix filtering: shingles are ordered rarest first, and two sets that
reach Jaccard *t* must share an element among the first `len − ceil(t·len) + 1` of each.
Named rules compare only items that share a defined name.

**Manifest** (`task-cluster-manifest-v1`). Each cluster lists its members (split, source,
ID, mode, module-level names, roles), every edge with its signals and evidence, a pending
behaviour record per cross-split joining edge, an adjudication record (status, reviewer,
date, rationale) and uncertainty notes. Statuses are `pending`, `confirmed_equivalent`,
`not_equivalent` and `unclear_quarantined`.

**Behaviour harness.** For a pair (A, B), A's reference runs against B's hidden and public
tests, B's against A's, and each against its own hidden tests as a control. Every call
repeats three times. Tasks record no entry point, so the harness finds the names B's tests
call that B's reference defines; when A defines one different module-level name of the same
kind, one alias line `b_name = a_name` is appended and recorded. Outcomes:

| Directions (A on B, B on A) | Controls | Outcome |
| --- | --- | --- |
| both pass every repeat, partner tests have ≥ 2 checks | pass | `equivalent` |
| both fail every repeat | pass | `not_equivalent` |
| one passes, one fails | pass | `unclear` (partial agreement) |
| public passes, hidden fails (or the reverse) | any | `unclear` |
| repeats disagree | any | `unclear` (nondeterministic) |
| both pass, fewer than 2 checks | pass | `unclear` (weak tests) |
| any | a control fails | `unclear` |
| modes differ, or names cannot be mapped | — | `unclear`, decided before running |

**Future builds.** `splits.family_grouping` in a pool spec (absent or `null` by default)
switches `tools/task_pool.py` to assign whole families: a TACO test task's family goes to
its holdout whole, MBPP families top up the human minimum, and a model-written family fills
dev or test only if every member comes from a named holdout source and it fits; any other
family stays in training. No family spans two splits. With the option absent, the
assignment is byte-for-byte the historical rule.

## Design choices and common mistakes

- **Static matches are evidence, not proof.** Two prompts with identical reference code
  can still specify different contracts or tests. Equivalence is decided by behaviour and
  a reviewer. The validator refuses `confirmed_equivalent` unless every behaviour record
  of the cluster is `equivalent`, and refuses `not_equivalent` unless every one is
  `not_equivalent`.
- **Unclear is kept, not resolved by default.** An `unclear_quarantined` decision must add
  a written uncertainty note, and no decision removes one. A cluster that joins a function
  task to an I/O task starts quarantined, because the harness cannot run one against the
  other's tests.
- **Named prompt rules do not join clusters.** On the pool, shared names such as `Book`,
  `add` or `area` with a 0.25 prompt overlap chained hundreds of unrelated tasks together
  through training-internal links. They are recorded as flags.
- **The canonical key joins more than the exact key.** On the pool it multiplies the
  number of dev tasks linked to training. Those extra links are candidates; the summary
  keeps the exact-key view separate so the two are never mixed.
- **The harness runs only inside the isolated sandbox.** `run` refuses unless a fresh,
  passing window receipt written for that command names the expected image, and no
  credential variable is set. An undecided call (infrastructure error, unavailable
  backend, host stall) is retried and then stops the run; it is never scored.
- **Sealed files are refused by name.** The tools never open the sealed test, the holdout
  union or the RL-only file.

## Verification

- Implementation: [`src/scratchcoder/data/clusters.py`](../../src/scratchcoder/data/clusters.py),
  [`tools/task_clusters.py`](../../tools/task_clusters.py),
  [`tools/task_equivalence.py`](../../tools/task_equivalence.py) and `assign_families` in
  [`tools/task_pool.py`](../../tools/task_pool.py).
- Reference tests: [`tests/test_clusters.py`](../../tests/test_clusters.py) checks the pair
  search against brute force and union-find against breadth-first search, and checks
  synthetic stand-ins for known overlapping pairs: the build's paraphrase rule misses them, the
  cluster scan joins them, and a one-operator change stays apart.
  [`tests/test_task_clusters.py`](../../tests/test_task_clusters.py),
  [`tests/test_task_equivalence.py`](../../tests/test_task_equivalence.py) and
  [`tests/test_task_pool_families.py`](../../tests/test_task_pool_families.py) cover the
  manifest, the harness with a fake sandbox, and the split assigner. A local-only test
  reproduces the recorded exact-key overlap counts when the ignored pool files are present.
  The real-sandbox harness test runs only with `SCRATCHCODER_SANDBOX_TESTS=1` inside a
  passing window.
- Command: `uv run --locked pytest tests/test_clusters.py tests/test_task_clusters.py tests/test_task_equivalence.py tests/test_task_pool_families.py`.

## Questions and answers

1. *Why does a whitespace-only difference still match, but a one-operator change not?*
   The key hashes the parsed AST, which has no whitespace or comments, while `==` and `!=`
   are different AST nodes.
2. *Why is the canonical key only a candidate signal?* Renaming collapses every program of
   the same shape, including unrelated ones such as identity functions.
3. *Can the static scan confirm a training-exposed dev task?* No. It can list one as
   `pending`; confirmation needs behaviour results and a reviewer's decision.
4. *What happens when references agree on public tests but not hidden ones?* The pair is
   `unclear` and the reason is recorded; a reviewer may quarantine it with the uncertainty
   kept.
5. *Does turning on family grouping change the historical pool?* No. The option is off
   unless a spec sets it, and an off option runs the unchanged historical assignment.
