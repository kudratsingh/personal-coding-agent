# Rehearsal task pool

`tools/task_pool.py` builds the small-model rehearsal's task pool, used for SFT, RL and the held-out dev and test sets. It runs on the owner's laptop under the capped verification the owner approved: the first 20,000 OpenCodeInstruct rows, the first 10,000 AceCode-87K rows, all MBPP tasks except the MBPP+ IDs, and TACO's easy level. The spec is [configs/task-pool-rehearsal.json](../configs/task-pool-rehearsal.json). Task files stay in the ignored `runs/` tree, because they hold hidden tests and reference solutions.

## How a task gets in

1. **Download.** Each pinned source file is checked against its listed size and SHA-256. A file the earlier audits already hold is hard-linked instead of fetched again.
2. **Adapt.** The reviewed adapters turn each row into the shared task schema. TACO's adapter takes standalone easy stdin/stdout problems only:
   - Problems from HackerRank are excluded, because TACO's card says their rights are unknown (D-33).
   - A problem whose statement still points at a removed figure is excluded.
   - So is one that accepts several valid answers or a numeric tolerance, since exact output comparison cannot grade it.
   - A problem with more than 100 test cases keeps its first 100 distinct cases, the sandbox's per-call maximum.
3. **Screen.**
   - A task must fit the S model's context: the public prompt plus the reference solution within 960 tokens.
   - Its reference and tests must fit the sandbox input limit.
   - It must not contain ARC-Easy text.
4. **Verify.** Every reference runs against its public tests and, separately, its hidden tests in the disposable [sandbox](sandbox.md). The deadline for an I/O call grows with its case count: 2 s plus 0.25 s per case. A stalled container is retried, never counted as a verdict. For AceCode, up to three recorded attempts with a perfect upstream score are tried in order.
5. **Admit.** `build_pool` drops benchmark IDs, near-duplicates (5-word-shingle Jaccard of 0.85 or more) and benchmark n-gram matches. Those n-grams must contain at least three words, so runs of numbers in test data do not match. Only tasks whose reference passes both test sets are admitted.
6. **Hold out.**
   - TACO test-split tasks are divided evenly between dev and test. Because there are fewer than 300 of them, MBPP tasks top the human-written share up to 300.
   - OpenCodeInstruct tasks, chosen by ID hash, fill the rest of each split to 2,000.
   - A training task that loosely paraphrases a holdout is then dropped. That means 3-word-shingle Jaccard of 0.5 or more, or 0.25 or more when both define the same function name.
7. **RL-only pool.** AceCode tasks with no attempt that passed here go to a separate RL-only file (D-34). They are screened against the benchmarks and the holdouts and never used for SFT or as holdouts.

Every task has one public test; the rest are hidden.

## The 2026-10-25 pool

| Source | Rows selected | Adapted | Verified and kept | Train | Dev | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCodeInstruct (model-written) | 20,000 | 17,284 | 7,828 | 2,091 | 1,850 | 1,850 |
| AceCode-87K (model-written) | 10,000 | 9,792 | 4,614 | 4,588 | 0 | 0 |
| TACO easy (human-written) | 9,104 | 1,916 | 1,567 | 1,514 | 26 | 26 |
| MBPP minus MBPP+ (human-written) | 974 | 593 | 545 | 266 | 124 | 124 |
| **Total** | | | | **8,459** | **2,000** | **2,000** |

**Authorship by split:**

| Split | Human-written | Model-written |
| --- | ---: | ---: |
| Train | 1,780 | 6,679 |
| Dev | 150 | 1,850 |
| Test | 150 | 1,850 |

**RL-only pool:** 4,660 AceCode tasks from 5,157 candidates. 493 were dropped as benchmark or holdout matches and 4 for prompts over 480 tokens.

**Verification, in sandbox calls:**
- 48,041 cached;
- 66 executed in the final rebuild;
- 4 re-run because the container had stalled.

**Why rows fell out:**
- **OpenCodeInstruct:**
  - 2,716 had tests that were not assertions;
  - 227 did not fit the context;
  - 8,928 references failed their tests here, 941 of those because they import modules the sandbox does not provide;
  - 2,037 training tasks were dropped as holdout paraphrases.
- **AceCode:** 5,157 tasks had no attempt that passed here and became RL-only candidates. 14 were rescued by a second attempt, and 26 training tasks were dropped as holdout paraphrases.
- **TACO easy:** 6,928 were not standalone stdin/stdout problems, and 178 references failed.
- **MBPP:** 47 were near-duplicates or benchmark matches, and 31 were holdout paraphrases.

## Task families (optional, for future builds)

A pool spec may set `splits.family_grouping` to `"task-clusters-v1"`. The holdout assignment then treats each task family, found from reference-solution ASTs and prompt paraphrases, as one unit, so equivalent tasks never sit on both sides of the split. Only TACO, MBPP and the named model sources are ever held out; a held-out family's members from any other source, such as AceCode, are dropped from the pool and counted. The option is absent from [configs/task-pool-rehearsal.json](../configs/task-pool-rehearsal.json); absent or `null`, the build runs the historical per-task assignment unchanged. The 2026-10-25 pool was built without it and has not been rebuilt. See the [task-cluster walkthrough](walkthroughs/task-clusters.md).

## Limitations

- Licenses are as declared on each dataset card. The seed provenance of the model-written sources is not independently cleared.
- A reference that times out under the sandbox limits counts as invalid.
- Rewordings looser than the paraphrase rule remain. Tasks with the same reference solution can sit in both training and dev; candidate families are listed by `tools/task_clusters.py` for behavioural adjudication.
- RL-only tasks have model-synthesized tests that no reference has passed here.
- The pool is sized for the S rehearsal's 1,024-token context; a larger model needs its own build.

## Pilot intake from a row range

`python -m tools.pilot_intake` prepares further training candidates from a row range of a pool source file the pool build already pinned, without rebuilding the pool.
- **Reading.** It reads rows `[--start-row, --start-row + --rows)` of the receipted OpenCodeInstruct file, in file order. The file's size and SHA-256 must match the pool's completed download receipt. Rows the pool build itself read are refused, because they include the pool's holdouts.
- **Rules.** Each row goes through the reviewed adapter and then these rules, in this order; the first rule that fails is the drop reason.
  1. The build's intake screens: the context fit and the sandbox input limit.
  2. The RL prompt limit, which is the build's RL-only rule.
  3. ARC-Easy and benchmark protection.
  4. Pool and range ID checks.
  5. n-gram contamination against dev, checked over all four task fields. This is the build's RL-only rule, made stricter.
  6. Near-duplicates of pool train, dev or earlier rows, keeping the first in file order.
  7. The build's holdout-paraphrase rule against dev.
  8. The frozen family rules against dev. Families are computed over dev, pool train and every adapted row of the range, so a row dropped for any other reason still links its family-mates. This is stricter than the build, which put only verified tasks in its family graph: it can drop a candidate the build would keep, never the reverse.
- **What it does not do.** It runs no code and opens no sealed split.
- **Outputs.** It writes the kept candidates, every row's outcome, the scan bridges, and a record with:
  - the inputs and rules;
  - counts, including the scan bridges';
  - software, lockfile and tokenizer identities.
- **Scan bridges.** `scan-bridges.jsonl` holds every adapted row the intake dropped, for any reason including the family rule, as train-split tasks in row order. These rows are rejected and never train. They exist so that the sealed-test scan uses the same family graph as the dev check: pass them to the scan as a bridge set, so a candidate linked to a test task only through a dropped row is still excluded.
  - A dropped row whose family fields (source, prompt, reference, mode, entry point) equal a task already held under its ID adds no link, and is left out.
  - Any other repeated ID is kept and counted. The scan's request refuses repeated bridge IDs, so a repeat can never drop out of the family graph unnoticed; such a range needs a reviewed change before it is scanned.
  - Rows the adapter refused have no task and cannot link anything.
  - When the record's `scan_bridges.tasks` is 0, the file is empty: leave it out of the scan request, which refuses an empty set. An empty set links nothing.
- **What comes next.** Protection against the sealed test split is a separate, approved step: the [holdout integrity scan](walkthroughs/holdout-integrity-scan.md) runs on this output, with pool train, pool dev and the scan bridges as bridge sets. Reference verification follows.
- **Row offset.** `tools.task_pool._parquet_rows` takes the 0-based offset `start`. The pool build itself still reads from row 0.
