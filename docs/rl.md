# Rehearsal RL: screening, stops and selection

The rehearsal's short reinforcement-learning run trains the SFT checkpoint on code tasks, rewarding each sampled program by the fraction of hidden tests it passes. A compute-matched "more SFT" control trains the same checkpoint on the same tasks' reference solutions. This page covers which tasks RL may train on, when a run stops, and which checkpoint each arm reports. The [RL update walkthrough](walkthroughs/rl-update.md) explains one policy-gradient step, and the [more-SFT control walkthrough](walkthroughs/more-sft-control.md) explains the control. The code is in `src/scratchcoder/rl/`. Nothing here has run on a GPU yet: the run itself needs its own approval.

## Screening

Screening runs after SFT. It samples 8 programs per RL candidate from the SFT checkpoint, scores them through the [reward channel](walkthroughs/reward-channel.md), and keeps a candidate only if:

1. **Parseable:** every hidden test is in a form the reward channel can score (`is_parseable`);
2. **No cheap answer:** the best constant answer earns less than **0.5** (a setting). A constant answer is a fixed return value or printed line (singletons, the integers −1 to 10, empty values and `[0]`; for I/O also lines such as `YES` and `NO`), a value copied from the task's public tests, or an echo: of the first, second or last argument, of all arguments, or of the input or its first or last line or token (`degenerate_baseline`; no code runs). An advantage is a reward minus its group's mean, so a task that pays a constant answer well pays the policy for answering without solving. The owner set this threshold;
3. **Something to learn from:** its 8 rewards are not all equal. An all-equal group has zero advantage and adds no gradient;
4. **Within the cap:** it is among the first 1,000 eligible tasks in ID-hash order, `sha256("rl-candidate-v1|" + task ID)`.

If fewer than **320** tasks are kept (enough for 10 updates of 32 without reuse), screening returns a floor result for the owner and no task list. The record lists, per candidate, its 8 rewards and statuses, its constant-answer value and answer, and its drop reason, with counts by source and reason.

**Measured before generation (2026-10-26).** The 1,000 RL candidates are 540 OpenCodeInstruct, 391 TACO and 69 MBPP tasks from the pool's train split. Rules 1 and 2 need no model, and `scratchcoder rl-prescreen` counts them:

| Source | Candidates | Parseable | Constant answer ≥ 0.5 | Eligible before rule 3 |
| --- | ---: | ---: | ---: | ---: |
| OpenCodeInstruct | 540 | 500 | 84 | 416 |
| TACO | 391 | 391 | 89 | 302 |
| MBPP | 69 | 69 | 43 | 26 |
| **All** | **1,000** | **960** | **216** | **744** |

Under the rehearsal tokenizer, every eligible chat-v1 prompt fits the 1,024-token context, and at most 7 leave less room than the full 256 new tokens. Rule 3 needs the SFT checkpoint's samples, so the kept count is not yet known.

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

**Failures.** An unexpected error anywhere (a lost bridge, an out-of-memory error, an `on_check` that raises), including in the stop-time check, saves the latest weights, logs `rl_failed` and propagates, with no result file. The caller can pass `on_check`, called with each check's record once its checkpoint file exists, to upload the weights at once. `run_rl` and `run_more_sft` refuse a task that is not a train-split task or whose hidden tests the reward channel cannot parse, and the control also a task without a reference solution. On a Pod that holds no hidden tests, both take `facts` computed on the Mac in place of the tests ([below](#a-pod-without-hidden-tests)).

An update whose groups are all skipped takes no optimizer step and is listed in the result.

**The reward timeout** has no built-in value. The caller derives it from the per-attempt budgets (8 s for a function task; 2 s plus 0.25 s per case, at most 60 s, for an I/O task), the 15 s of host slack the channel adds to a try, the number of execution workers, and the tries. `LocalRewardSource` retries a host failure 3 times after the first try (4 tries, waiting 1, 2 and 4 s), where the channel's own helper `score_with_host_retries` counts 3 tries in all. A worst case without retries is `ceil(attempts / workers)` times the longest budget: 256 I/O attempts at the 26.75 s budget of a 99-case task on 8 workers is about 856 s before slack, so a fixed value such as 420 s would stop a run that is merely slow. When the rewards come through the [Mac bridge](#the-mac-bridge-run-2), the Pod's timeout also covers the bridge's own time.

## Selection

Each arm reports the check with the highest dev-subset greedy pass@1 among its selectable checks. Ties go to the higher 8-sample mean, then to the later check.

**The control's checks:** the start (RL's update-0 check of the same weights), after 1, 2, 4 and 8 passes over the references, and at the end of its budget.

**The control's budget:** GPU-busy time equal to RL's measured rollout-generation plus training-step seconds, within ±5%. Compile, dev checks and execution waits are excluded in both arms. The measured mismatch is reported. Neither arm compiles, so the control has no `compile` setting (its compile time could not be separated from the budget). Its attention kernel and optimizer path are settings (`attention`, `optimizer_foreach`): they should match the RL model (`RLResult.attention`) and the SFT stage, or the same seconds buy fewer control updates than they would on that path.

## The Mac bridge (run 2)

In run 2 the RL loop and the more-SFT control run on the Pod, but rewards and dev verdicts need the hidden tests and the Docker sandbox, which stay on the Mac. The Mac reaches the Pod over SSH; the Pod cannot reach the Mac. The bridge carries each request and its reply as files written by atomic rename. The [bridge walkthrough](walkthroughs/rl-bridge.md) has the schemas, a worked example and every failure mode.

**On the Pod** (`scratchcoder.rl.bridge`), one `BridgeClient` per bridge directory serves both of these:

- `BridgeRewardSource`, the reward source;
- `BridgeDevEvaluator`, the dev evaluator: the pool evaluator with a `BridgeDevExecutor`, each request labelled with its check index.

Each call writes a request into `outbox/`. The request holds the kind (`reward` or `dev`), the update or check index, the attempt count, the SHA-256 of the canonical attempts payload, the SHA-256 of the Pod's task file and a request ID. The call then waits for the matching reply in `inbox/`:

- **A wrong reply** (wrong ID, kind, index, count, SHA-256 or schema, or over 4 MiB) is set aside and the wait goes on. More than 3 raise `BridgeReplyError`.
- **No acceptable reply within the timeout** withdraws the request and raises `RewardTimeout` (`reward_wait`).
- **A `host_failure` reply** raises `RewardHostFailure`; a host failure is never scored.
- **A refusal** raises `BridgeRefused`, which ends the run as an unexpected error, with the latest weights saved.

While a call waits, `client.heartbeat()` returns `{"status": "awaiting_rewards", ...}`: the request ID, kind, index and age.

**On the Mac**, `scratchcoder rl-bridge` (`scratchcoder.rl.bridge_mac`) polls the Pod's outbox over SSH. It uses the rehearsal's SSH options and a child environment that holds no `R2_` variable, no Runpod key, no SSH agent and no `.env` name. Each request is pulled with a size bound and checked against the schema. Then:

- **Repeats:** a request answered before gets its stored reply, keyed by request ID and payload SHA-256, and is never scored again.
- **Refusals:** the Mac refuses a request older than one already answered, a reused ID, a task-file SHA-256 other than the configured one, and any task outside the phase's allowed list. For rewards that list is the kept screening list; for dev checks, the 200-task dev subset.
- **Scoring:** the rest is scored against the Mac's full task files. Rewards go through `LocalRewardSource`; dev verdicts through the evaluation runner, in parallel, with host failures retried. A dev program that cannot be written as UTF-8 (an unpaired surrogate) fails without running, as the reward channel scores it `invalid_source`.
- **Mac-side bounds:** `--reward-timeout` and `--dev-timeout` are required. The server scores one request at a time, so a request the Pod has already withdrawn would keep the next one waiting. Set each to the Pod's timeout for that kind less the bridge's allowance (below); a request that runs past its bound gets a `timeout` reply, and no new program or retry starts after it.
- **One server per Pod:** the server takes a lease on the Pod, `server.lease` in the bridge directory, in the same SSH call as each listing, and renews it every few seconds, also while a long request is scored. A second server is refused (`BridgeLeaseHeld`, exit code 3) until the first has been silent for 60 s. Its ID is kept in `server.id` in `--state`, so a restart over the same state directory takes its own lease back at once.
- **Replies:** each reply is stored on the Mac before it is pushed. It carries, per attempt, only `AttemptReward.to_json()` or a pass flag; no hidden test or expected value reaches the Pod.
- **SSH failures** are retried 4 times with backoff (1, 2, 4, 8 s). `--remote-root` must be a normalised absolute path (no trailing or doubled slash, no `.` or `..`); the remote program exits with a distinct code on input it refuses, and the Mac raises that at once instead of retrying it as an outage.
- **The image check** (`--image-id`) retries a failed or stalled `docker image inspect` 3 times (1, 2, 4 s) before it stores a `host_failure`; an inspect that succeeds and names another image is final.
- **Progress:** `progress.json` records the request being scored and the last one seen and answered, with times, so a watchdog can count a reward wait as progress.

**The Pod's timeout** is the Mac's worst-case scoring time (above, for rewards; for a dev check, its programs over the workers at the evaluation runner's deadlines) plus the bridge's allowance:

- up to one Mac poll interval (2 s by default) before the Mac sees the request;
- one listing, one read and one write over SSH, each normally about a second and bounded by `--ssh-timeout` (120 s; a listing, 30 s);
- each of those retried up to 4 times, waiting 1 + 2 + 4 + 8 = 15 s, to ride out a short SSH drop;
- up to one Pod poll interval (0.5 s) before the Pod sees the reply.

The Mac's own bound is the Pod's timeout less this allowance, so a slow request ends in a `timeout` reply the Pod can still read. A Mac-side bound stops only work that has not started: a program already running in the sandbox finishes at the runner's own deadline (26.75 s for a 99-case I/O task, plus 15 s of host slack), so the Pod's dev timeout should leave that much slack after a reward timeout. A Pod-side wait is not interrupted by the run's `should_stop` (cost or wall time), so the stop margin must cover one full timeout.

**Wired for run 2.** The run-2 worker ([post-training runs](posttrain-runs.md#run-2)) does each of these. `posttrain_ops.bridge_timeouts` derives every bound in one place from the channel's longest budget (27 s for a 100-case I/O task), its 15 s slack, 4 tries, 8 workers and the bridge's polls, SSH timeouts and retries: the Mac's `--reward-timeout` 5,600 s and `--dev-timeout` 9,492 s (225 rounds of 42 s plus 42 s for a program still running), a bridge allowance of 1,397.5 s, so the Pod waits 6,997.5 s and 10,889.5 s; the specification, the runbook's `rl-bridge` command and the watchdog policy use these values and a test keeps them equal. The worker keeps one full wait (10,889.5 s) free before every update and every request (`wait_margin`). The facts' and the task files' SHA-256s are computed from the bytes `scratchcoder rl-pod-export` wrote, pinned by the binding and checked on the Pod. While a request waits, the worker heartbeats `awaiting_rewards` and counts a fresh `server.lease` as progress for the in-Pod idle rule; a lease stale for 600 s ends the wait as `bridge_lost`. The Mac watchdog does not read `progress.json`: its run-2 policy bounds the idle extension of an `awaiting_rewards` heartbeat by the Pod's own timeout for that wait plus 60 s.

## A Pod without hidden tests

The plan keeps hidden tests on the Mac: the Pod holds prompts, public tests and reference solutions. But `run_rl`, `run_more_sft` and `PoolDevEvaluator` read each task's hidden tests where they run, for two things: whether the reward channel can parse them, and the best constant answer's reward (the monitoring baseline and the dev check's constant baseline). So the Mac computes both and the Pod takes them as inputs:

- `scratchcoder.rl.withheld.withheld_task` is a task as the Pod may hold it: the hidden tests replaced by a fixed placeholder (a `Task` cannot be empty there), and the reference solution kept for a train task only.
- `scratchcoder.rl.facts.task_facts` computes each task's fact on the Mac from the real tests: parseable or not, and the constant answer's reward. The answer itself is left out, since it can be a hidden test's expected value. `facts_bytes` writes them as a canonical file; `load_facts` on the Pod checks its SHA-256 against a configured value and its exact schema.
- `run_rl(facts=...)`, `run_more_sft(facts=...)` and `PoolDevEvaluator(baseline=...)` (also through `BridgeDevEvaluator`) use them in place of the tests. `constant_baseline_from_facts` rebuilds the dev baseline and equals `constant_baseline` of the real tasks.
- Without them a withheld task is refused with an error that says so. It would otherwise train against a baseline computed from the placeholder, or record a dev baseline of 0 with every task unparseable.

Tasks that carry their real hidden tests behave exactly as before, so the Mac-side and local runs are unchanged. The [bridge walkthrough](walkthroughs/rl-bridge.md) shows the flow.

## Commands

- `scratchcoder rl-prescreen --tasks <pool train.jsonl> --candidates <rl_candidates.jsonl>`: rules 1 and 2 by source. It runs no code.
- `scratchcoder rl-screen --tasks ... --candidates ... --attempts <attempts.jsonl> --output <record.json> --timeout <seconds>`: scores screening attempts in the local Docker sandbox and writes the screening record. It refuses a candidate that is not a train task. An attempts file the post-training worker wrote starts with a header (`scratchcoder.rl.attempts`) naming the candidates export it was generated from, the candidate list's SHA-256, the checkpoint and the sampling settings; `rl-screen` checks that the header's sampling is the screening protocol's (8 samples, temperature 1.0, top-p 1.0, at most 256 new tokens, the screening seed base), checks the list, derives the export from the full train file again and compares SHA-256s, and requires every candidate's samples exactly once. `--require-header` refuses a file without one ([post-training runs](posttrain-runs.md)).
- `scratchcoder rl-bridge --host <user@ip> --port <port> --identity-file ... --known-hosts ... --remote-root <Pod bridge dir> --train-tasks <pool train.jsonl> --kept <screening record> --dev-tasks <pool dev.jsonl> --pod-reward-sha256 <hex> --pod-dev-sha256 <hex> --state <Mac dir> --reward-timeout <seconds> --dev-timeout <seconds>`: serves one Pod's rewards and dev verdicts until interrupted. The two timeouts are required Mac-side bounds (above). `--image-id` checks the pinned sandbox image's local ID before each request. It never opens the sealed test or holdout split (it refuses the name a path resolves to, so a link to one is refused too), and it refuses to start while a secret is in its environment. It also reads the `.env` files' variable names to refuse those: by default `.env` in the current directory and in the checkout, at least one of which must exist; pass `--dotenv PATH`, or `--no-dotenv` if there is none. It exits 2 on any of these refusals and 3 when another server holds the Pod.

- `scratchcoder rl-pod-export --train-tasks <pool train.jsonl> --kept <screening record> --dev-tasks <pool dev.jsonl> --output <new dir>`: writes run 2's Pod files on the Mac: the withheld kept tasks (`rl-tasks.jsonl`, references kept), the withheld 200-task dev subset (`rl-dev.jsonl`), the facts file (`rl-facts.json`) and a record of the exact bytes' SHA-256s. It is deterministic, refuses the sealed splits by name and through links, and refuses a kept task without a reference or with hidden tests the channel cannot parse.

The run itself is driven from Python (`scratchcoder.rl.loop.run_rl`, `scratchcoder.rl.control.run_more_sft`) by the post-training worker, which supplies the reward source, the dev evaluator's executor and `should_stop`. In run 2 (`tools/posttrain_run2.py`) those are the bridge's `BridgeRewardSource` and `BridgeDevEvaluator`, wrapped so each wait heartbeats `awaiting_rewards` and keeps the margin.

## Limitations

- **Untested at scale:** everything above is tested on CPU with tiny models, fake rewards and authored tasks, plus a Docker test of the local reward source on authored and real train tasks. GPU speed, the kept count and any learning are unmeasured.
- **The bridge has not crossed a real link:** it is tested over a local directory pair and over its remote program run by a local interpreter in place of `ssh`. SSH to a Pod, its latency and its failures are unmeasured.
- **Hidden tests on the Pod:** the bridge sends none, and with the Mac's facts (above) neither do the Pod's task files. The facts are computed from the real tests on the Mac, so the Pod trusts the file whose SHA-256 it is configured with; for run 2 that value, like the task-file SHA-256s, is computed from the bytes `rl-pod-export` wrote and pinned at the owner's gate.
- **One request can stall the next:** a program that stalls the sandbox host on all 4 tries makes the whole dev check a host failure, and stops RL. That is the evaluation runner's rule (a host failure is never scored), not the bridge's.
- **Constant baseline:** it is a lower bound. A function-task check whose call is not a direct call of the candidate's function counts as failed for every constant answer.
- **Audit flags:** a flag is a signal, not a verdict, and a missing flag proves nothing (see the reward channel's limits).
