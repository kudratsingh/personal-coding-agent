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
- **What comes next.** Protection against the sealed test split is a separate, approved step: the [holdout integrity scan](walkthroughs/holdout-integrity-scan.md) runs on this output, with pool train, pool dev and the scan bridges as bridge sets. Reference verification follows ([pilot stages](#pilot-stages-after-the-integrity-scan)).
- **Row offset.** `tools.task_pool._parquet_rows` takes the 0-based offset `start`. The pool build itself still reads from row 0.

## Pilot stages after the integrity scan

`python -m tools.pilot_stages` takes a pilot's clear candidates from the [holdout integrity scan](walkthroughs/holdout-integrity-scan.md) to a screening report, in one run directory, with every limit enforced in code. It reuses the reviewed pieces and loosens none of them.

- **`init --out RUN`.** Creates the run directory, which must not exist, and pins every input by SHA-256:
  - the clear pilot candidates and the surviving RL remainder;
  - the scan's `summary.json` and the intake `record.json`;
  - the pool spec, for its verification policy;
  - the SFT checkpoint, with its record, and the tokenizer.

  The checkpoint must hash to its record's selected SHA-256. An input listed in a `SHA256SUMS` beside it must match that line. Every later stage checks the inputs it reads against these hashes again.
- **Reruns.** Every run is entered in one registry, `runs/pilot-stages-registry.jsonl` in the checkout, whatever `--out` is. Entries are keyed by the content of the pilot tasks, and an entry is written before its run.
  - A second run on the same tasks needs `--supersedes PRIOR`, naming the latest run on them. That run must have stopped on a sandbox host failure, with no containment failure.
  - The new run starts with the old run's sandbox calls and generation time. Its 12 h wall time runs from the first run's init, gaps between runs included. A rerun resets no cap.
  - Every stage refuses a run that is not the latest registered run on its tasks.
- **Stages, in order.** Each needs the previous one completed and refuses to run twice.

  | Stage | What it does | Runs code |
  | --- | --- | --- |
  | `verify` | Each reference against its public and hidden tests, once per distinct call, through `tools.task_pool.verify` (its resumable cache and retry rules). The evaluation path's deadline must equal the pool's rule under the spec's policy for every call. | yes, at most 20,000 sandbox calls |
  | `admit` | [Admission](walkthroughs/admission.md) through `tools.admission_check`'s planning and recording: the static gate, three breadth-first reference runs and the negative controls. An admitted task must also be parseable under the reward grammar, and its best constant answer must earn less than the keep rule's threshold. | yes, at most 25,000 |
  | `select` | The first 1,000 admitted candidates in `rl-candidate-v1` hash order, with their positions. The rest is a reserve. | no |
  | `generate` | Needs a passing `scratchcoder sft-format-gate` record in the current gate format, measured on the CPU after `init`, of the same checkpoint and tokenizer; its path and SHA-256 are pinned on first use. The existing seeded sampler draws the screening protocol's 8 samples per candidate on the CPU, in `float32`, at batch 64, in chunks of 64 candidates, at torch's default thread count. Each chunk's start and time are logged durably, so the 4 h cap counts interrupted invocations: one killed inside a chunk is charged the longer of its longest chunk and the time since that chunk started. No chunk starts unless the longest one measured still fits. | no |
  | `score` | Rewards through the local reward channel and the unchanged keep rule (`scratchcoder.rl.command.screen_attempts`, header required), under reward grammar v2 only. | yes, at most 9,000 reward calls |
  | `report` | The measured outcome and one recommendation by the pilot's fixed decision rule. It also runs after a stop, and then draws no decision. | no |

- **Execution window.** `verify`, `admit` and `score` refuse to start without a passing receipt from `tools.mac_window_check exec` for that exact command line, under 30 minutes old (`tools.admission_check`'s rule). The receipt's image ID is bound to the run on first use, and a later stage under another image or evaluator identity is refused.
- **Budget.** A ledger in the run directory counts every sandbox try, retries included. Each try is charged before it is made, against its stage's cap and 60,000 for the whole run, and a try that would pass either cap is not made.
- **Hard stops.** Each is recorded with its reason, after which only `report` runs:
  - a containment failure: a supervisor that could not establish a required protection, on its first occurrence. No queued call starts after it;
  - a sandbox host failure that outlasted its retries;
  - any cap reached: calls, the run budget, 4 h of generation, or 12 h of wall time from the first `init` of a chain of reruns (checked at every stage, call and chunk);
  - under 20% of candidates verified;
  - more than 5% of admission runs unresolved after retries;
  - a failing format gate;
  - a reward grammar or facts identity other than v2;
  - a first chunk of generation that projects past 4 h.

  A stopping stage refuses further calls and waits for those in flight before it records the stop. An interrupted stage with no stop recorded resumes from its cache. A run with no cleared candidates completes every stage and reports that no candidates remained.
- **The attempts header.** Besides the usual fields, the header carries an optional `environment` record:
  - the torch thread count actually in use;
  - the torch, Python and NumPy versions;
  - the candidate order's SHA-256 and the seeds' SHA-256;
  - the first chunk's timing and its projection.

  A header without it reads and writes exactly as before. `scratchcoder.eval.checkpoint.load_checkpoint` sets torch to one thread, so the generator restores the default count after loading.
- **Report.** It gives:
  - the combined pool and the shortfall to the floor;
  - the yield, the all-zero share and the formatting, execution and genuine failure shares, with Wilson intervals;
  - per-task pass@8;
  - each step's drops by reason;
  - the remainder before and after the scan;
  - every stopping criterion's status.

  It states that CPU and CUDA screening are different numerical executions of the same intended protocol, and that the decision thresholds are planning heuristics.
- **Outputs.** Every output is a new file, and each stage writes a `SHA256SUMS`. The report adds one for the whole run.
- **What it does not do.** Nothing runs automatically after the report. Any further intake or screening is a proposal for the owner.
