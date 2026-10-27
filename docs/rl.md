# Rehearsal RL: screening, stops and selection

The rehearsal's short reinforcement-learning run trains the SFT checkpoint on code tasks, rewarding each sampled program by the fraction of hidden tests it passes. A compute-matched "more SFT" control trains the same checkpoint on the same tasks' reference solutions. This page covers which tasks RL may train on, when a run stops, and which checkpoint each arm reports. The [RL update walkthrough](walkthroughs/rl-update.md) explains one policy-gradient step, and the [more-SFT control walkthrough](walkthroughs/more-sft-control.md) explains the control. The code is in `src/scratchcoder/rl/`. Nothing here has run on a GPU yet: the run itself needs its own approval.

## Screening

Screening runs after SFT. It samples 8 programs per RL candidate from the SFT checkpoint, scores them through the [reward channel](walkthroughs/reward-channel.md), and keeps a candidate only if:

1. **Parseable:** every hidden test is in a form the reward channel can score (`is_parseable`);
2. **No cheap answer:** the best constant answer earns less than **0.5** (a setting). A constant answer is a fixed return value or printed line, a value copied from the task's public tests, or echoing the first argument or the input (`degenerate_baseline`; no code runs). An advantage is a reward minus its group's mean, so a task that pays a constant answer well pays the policy for answering without solving. The owner set this threshold;
3. **Something to learn from:** its 8 rewards are not all equal. An all-equal group has zero advantage and adds no gradient;
4. **Within the cap:** it is among the first 1,000 eligible tasks in ID-hash order, `sha256("rl-candidate-v1|" + task ID)`.

If fewer than **320** tasks are kept (enough for 10 updates of 32 without reuse), screening returns a floor result for the owner and no task list. The record lists, per candidate, its 8 rewards and statuses, its constant-answer value and answer, and its drop reason, with counts by source and reason.

**Measured before generation (2026-10-26).** The 1,000 RL candidates are 540 OpenCodeInstruct, 391 TACO and 69 MBPP tasks from the pool's train split. Rules 1 and 2 need no model, and `scratchcoder rl-prescreen` counts them:

| Source | Candidates | Parseable | Constant answer ≥ 0.5 | Eligible before rule 3 |
| --- | ---: | ---: | ---: | ---: |
| OpenCodeInstruct | 540 | 500 | 80 | 420 |
| TACO | 391 | 391 | 81 | 310 |
| MBPP | 69 | 69 | 28 | 41 |
| **All** | **1,000** | **960** | **189** | **771** |

Under the rehearsal tokenizer, all 771 chat-v1 prompts fit the 1,024-token context. 764 of them leave room for the full 256 new tokens. Rule 3 needs the SFT checkpoint's samples, so the kept count is not yet known.

## The run

100 updates of 32 tasks × 8 samples: temperature 1.0, top-p 1.0, at most min(256, 1,024 − prompt) new tokens, the chat-v1 prompt with the public test, one optimizer step per batch. Tasks are drawn by a seeded order that reshuffles each pass. The learning rate is 3e-6 after a 5-update warmup. Every setting is in `RLConfig`.

**Dev checks** run before the first update and after updates 12, 25, 50, 75 and 100. Each check samples on a fixed 200-task subset of the pool's dev split: the 200 tasks with the smallest `sha256("rl-dev-subset-v1|" + task ID)`. It takes one greedy program and 8 samples at temperature 0.8 and top-p 0.95, seeded from the task ID, at most min(512, 1,024 − prompt) new tokens. It reports:

- greedy pass@1;
- the 8-sample mean pass rate and pass@8;
- beside them, what a constant answer earns on the subset, in both units, so a score near it is visible:
  - the **constant pass rate**, the share of tasks on which the best constant answer passes every hidden test. It is in the same unit as greedy pass@1, the 8-sample mean and pass@8, so compare those with it;
  - the **constant baseline**, the mean fraction of hidden tests that answer passes. It is in the unit of a mean reward (the per-update monitoring), not of a pass rate. On the real 200-task subset it is 0.279, but a constant answer passes every hidden test on only 11 of the 200 tasks (5.5%), so a dev pass@1 of a few percent is at the constant answer's level, not far below 0.279.

  A dev task the reward channel cannot parse counts as 0 in both, and the count of such tasks is recorded.

Each check saves the weights. Every update's log also carries the mean constant-answer reward of its tasks beside its mean reward.

## Stops

| Reason | When |
| --- | --- |
| `non_finite` | a reward, the loss or the gradient norm is not finite; no step is taken |
| `collapse` | the mean number of distinct completions per group is below 2 for 5 consecutive updates |
| `dev_patience` | greedy pass@1 is not above the best so far for 3 consecutive checks |
| `leakage` | more than 1% of rewarded attempts (reward > 0) carry static-audit flags, counted from the first update, once at least 100 rewarded attempts exist |
| `reward_wait` | the reward source (or a dev check's execution) takes longer than its timeout; the weights are saved and the run returns cleanly |
| `reward_host_failure` | the reward source cannot resolve a host failure after its retries; the weights are saved and the run returns cleanly, as for `reward_wait` |
| `external` | the caller's `should_stop`, read between updates, asks to stop (cost, wall or idle); its text is the stop detail |
| `completed` | all 100 updates ran |

**The stop-time check.** A stop whose weights changed since the last check takes one more check, so the last weights are scored, except:

- a stop from the execution side (`reward_wait`, `reward_host_failure`), because the check needs that side;
- a `should_stop` stop, unless `check_at_external_stop` is set, because a cost, wall or idle stop exists to stop spending and the check is about 1,800 generations and a Mac round trip;
- either kind of stop when its flag is off (`check_at_stop` for the run's own stops, `check_at_external_stop` for `should_stop`).

Weights that no check covers are saved and listed as the result's `stop_checkpoint`. The check taken at a `leakage` stop is recorded (`selectable: false`) but never selected: those weights were trained into a reward exploit. Dev patience reached at the final update is a `completed` run with the patience noted in the stop detail, since no update is left to cut short.

**Failures.** An unexpected error anywhere (a lost bridge, an out-of-memory error, an `on_check` that raises), including in the stop-time check, saves the latest weights, logs `rl_failed` and propagates, with no result file. The caller can pass `on_check`, called with each check's record once its checkpoint file exists, to upload the weights at once. `run_rl` and `run_more_sft` refuse a task that is not a train-split task or whose hidden tests the reward channel cannot parse, and the control also a task without a reference solution.

An update whose groups are all skipped takes no optimizer step and is listed in the result.

**The reward timeout** has no built-in value. The caller derives it from the per-attempt budgets (8 s for a function task; 2 s plus 0.25 s per case, at most 60 s, for an I/O task), the 15 s of host slack the channel adds to a try, the number of execution workers, and the tries. `LocalRewardSource` retries a host failure 3 times after the first try (4 tries, waiting 1, 2 and 4 s), where the channel's own helper `score_with_host_retries` counts 3 tries in all. A worst case without retries is `ceil(attempts / workers)` times the longest budget: 256 I/O attempts at the 26.75 s budget of a 99-case task on 8 workers is about 856 s before slack, so a fixed value such as 420 s would stop a run that is merely slow.

## Selection

Each arm reports the check with the highest dev-subset greedy pass@1 among its selectable checks. Ties go to the higher 8-sample mean, then to the later check.

**The control's checks:** the start (RL's update-0 check of the same weights), after 1, 2, 4 and 8 passes over the references, and at the end of its budget.

**The control's budget:** GPU-busy time equal to RL's measured rollout-generation plus training-step seconds, within ±5%. Compile, dev checks and execution waits are excluded in both arms. The measured mismatch is reported. Neither arm compiles, so the control has no `compile` setting (its compile time could not be separated from the budget). Its attention kernel and optimizer path are settings (`attention`, `optimizer_foreach`): they should match the RL model (`RLResult.attention`) and the SFT stage, or the same seconds buy fewer control updates than they would on that path.

## Commands

- `scratchcoder rl-prescreen --tasks <pool train.jsonl> --candidates <rl_candidates.jsonl>`: rules 1 and 2 by source. It runs no code.
- `scratchcoder rl-screen --tasks ... --candidates ... --attempts <attempts.jsonl> --output <record.json> --timeout <seconds>`: scores screening attempts in the local Docker sandbox and writes the screening record. It refuses a candidate that is not a train task.

The run itself is driven from Python (`scratchcoder.rl.loop.run_rl`, `scratchcoder.rl.control.run_more_sft`) by the post-training worker, which supplies the reward source, the dev evaluator's executor and `should_stop`.

## Limitations

- **Untested at scale:** everything above is tested on CPU with tiny models, fake rewards and authored tasks, plus a Docker test of the local reward source on authored and real train tasks. GPU speed, the kept count and any learning are unmeasured.
- **Constant baseline:** it is a lower bound. A function-task check whose call is not a direct call of the candidate's function counts as failed for every constant answer.
- **Audit flags:** a flag is a signal, not a verdict, and a missing flag proves nothing (see the reward channel's limits).
