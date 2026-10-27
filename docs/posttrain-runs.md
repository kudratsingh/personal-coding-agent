# Post-training run operations

The rest of the rehearsal continues from run B's `mixed-final` checkpoint in two paid runs on one Runpod A40 Pod each, under limits the owner approved on 2026-10-26:

- **Run 1:** the generation probe, ARC-Easy on run A's two finals and run B, base-B evaluation generation, SFT, ARC and evaluation generation of the SFT weights, and RL screening generation. Hard ceiling **$3.00**. Commands: the [run 1 runbook](../POSTTRAIN_RUNBOOK.md).
- **Run 2** ([below](#run-2)): RL with the Mac computing rewards and dev verdicts through the reward bridge, the compute-matched more-SFT control, ARC-Easy and evaluation generation of both selected checkpoints. Its ceiling, about $5.80, is fixed at the owner's gate from run 1's measured rates, and it stops at the lower of that ceiling and 120% of the gate's estimate. Commands: the [run 2 runbook](../POSTTRAIN_RUN2_RUNBOOK.md).

**Status (2026-10-27): pre-launch.** Both runs' workers, their safeguards, the Mac-side tools and the runbooks are written and CPU-tested, each with an end-to-end smoke replayed into the real supervisor (run 2's with the Mac's bridge server in a thread, and once with the Mac scoring in the Docker sandbox). Nothing has run on a GPU, no input has been staged to R2, and nothing has been spent. Every figure below is a projection.

## Stages

`tools/posttrain_run.py` runs the stages of [configs/rehearsal-a40-s-post-1.json](../configs/rehearsal-a40-s-post-1.json) in order. Each stage's record in `report.json` holds its seconds and every output's SHA-256 and size. Every artifact is uploaded as soon as it exists, as one label under `posttrain-runs/1/<pod id>/<label>/`, each label with a `record.json` snapshot of the report.

| Stage | Reads | Writes (R2 label: files) |
| --- | --- | --- |
| `probe` | B `mixed-final`; the pool dev export (prompt lengths) | `probe`: `generation-benchmark.json` |
| `arc-base` | A `stage1-final`, A `stage2-final`, B `mixed-final`; the ARC-Easy test questions | `arc-base-<checkpoint>`: `arc.json`, one per checkpoint |
| `eval-base` | B `mixed-final`; the pool dev and test exports; the pinned HumanEval, HumanEval+ and MBPP+ releases | `eval-base-<file>`: `completions.jsonl`, seven files |
| `sft` | B `mixed-final`; the SFT rows; the code and language dev sets | `sft`: `sft.json`, `updates.jsonl`; `sft-weights`: `checkpoint.pt` (weights only) |
| `arc-sft` | the selected SFT weights; the ARC-Easy questions | `arc-sft-sft`: `arc.json` |
| `eval-sft` | the selected SFT weights; the same task files | `eval-sft-<file>`: `completions.jsonl`, seven files |
| `screen` | the selected SFT weights; the RL candidates export and list | `screen`: `attempts.jsonl` |

Then `report.json` at the prefix root. The seven evaluation files of a stage are pool dev and pool test, each greedy and n = 8 (temperature 0.8, top-p 0.95, seed 0), and HumanEval, HumanEval+ (task 32 excluded) and MBPP+ greedy. Base B uses `base-pool-completion-v1` on the pool and `base-completion-rstrip-v1` on the benchmarks; the SFT weights use `chat-v1` everywhere ([evaluation](evaluation.md)).

**Probe.** `benchmark_generation` at batch 1, 64, 128 and 256 on the 256 chat-v1 prompt lengths around the pool dev median (about 190 tokens), 256 new tokens each, temperature 0.8 and top-p 0.95, in bf16. Below the planning speeds, 5,000 generated tokens a second at batch 64 or 10,000 at batch 256, the run stops with `generation_below_planning_speed`, its rates in `report.json`, at most about $0.39 in (the setup allowance, the probe and the whole finalization reserve). Otherwise the batch-64 rate replaces the planning speed in every later projection.

**SFT.** One pass over the 3,390 rows of record (2,871 code, 519 language) in 16 x 2 rows per update: 106 updates, 3.47M token slots. Peak 6e-5, 3% warmup (3 updates), cosine to 10%, a fresh AdamW (0.9 / 0.95, weight decay 0.1, clip 1.0), bf16 with fused attention and `torch.compile`, assistant-token loss divided by each update's assistant tokens. The token-weighted assistant-token dev loss over the 2,000 code and 300 language dev conversations is measured at updates 26, 53, 80 and 106; each new lowest is saved as a weights-only checkpoint (about 541 MB), and only the lowest is kept and uploaded. After 50 updates the slot throughput over the step seconds of updates 3 to 50 must reach 90% of run B's 56,765 slots a second, or the run stops with `sft_throughput_below_limit`.

**Screening.** `screening_rollouts` from the selected SFT weights on the 1,000 candidates (OpenCodeInstruct 540, TACO 391, MBPP 69), 8 samples at temperature 1.0 and top-p 1.0, at most min(256, 1,024 - prompt) new tokens, with the screening seeds; `attempts.jsonl` is what `scratchcoder rl-screen` reads, with a header binding it to the candidates export ([RL](rl.md)).

## Hidden tests never leave the Mac

The Pod holds prompts, public tests and reference solutions only. `scratchcoder pod-task-export` writes a pool split, the RL candidates or the ARC-Easy questions for the Pod: every hidden test is replaced by a placeholder that names only its salted SHA-256, keeps the file a valid task file, and fails any program run against it. The export is deterministic, and its record binds it to the source file by SHA-256. A pin in the specification protects the dev split and the candidates, but the test split's export is pinned only at the go, so every pool and candidates input is checked to be an export wherever it is handled: `posttrain_binding check-exports` at the go, the binding before it writes or stages anything, the Pod's `verify` and the worker at its start all refuse a file in which any task carries hidden tests, naming task IDs only.

`eval-execute` and `rl-screen` take the full task file. When a header names a task file other than the one given, they derive the export from the given file again and compare SHA-256s; the record is never read. Because each placeholder hashes the test it replaced, a full file with other hidden tests derives another export and is refused, and an export given as the full file is refused too. The pool dev baseline records its file's export hash, so the scorecard matches it to completions generated on the Pod.

## Limits

Every time and cost limit is derived from the ceiling and the quoted rate in one place, `posttrain_ops.ceiling_limits`, never from the pretraining rehearsal's fixed 60,000 s. At $0.60 an hour (the $0.59 GPU quote and the 40 GB disk):

| Limit | Run 1 |
| --- | ---: |
| Worker cost stop | $3.00 |
| Mac watchdog cost stop | $2.99 |
| Pod wall | 18,000 s |
| In-Pod timer | 17,820 s |
| In-Pod idle self-destruct | 900 s without activity |
| Mac watchdog idle limit | 600 s |
| Transfer window after the last GPU work | 900 s |

The **launch rule** (worker and readiness receipt): the lifetime left must hold every stage at the planning rates, generation priced at every attempt's token cap, plus the 900 s finalization reserve: 10,003 s, which leaves about 130 minutes for setup. The **projected-finish rule**: at the start of every stage the remaining stages at the measured rates plus the reserve must end before the lifetime does, or the run stops with `projected_finish_after_deadline` before starting the stage. The generation rate is the probe's, or the completed evaluation files' observed rate (priced at their token caps, as the projections price them) once that is lower. The **finalization reserve**: a stage still running 900 s before the end of the lifetime stops there with `finalization_reserve`; unlike the wall and cost stops it finalizes in full, so the drain, `report.json` and `r2_final` fit before the in-Pod timer fires. Before any stage the worker checks that every evaluation file and the screening hold the tasks the specification prices.

## Safeguards

| Safeguard | Enforced in | Tested in |
| --- | --- | --- |
| Ceiling-derived lifetime: $3.00, $2.99, 17,820 s of 18,000 s | `posttrain_ops.ceiling_limits`; the worker (`RunLimits`), the receipt (`check_policy`; the deadline no later than the billing origin plus 17,820 s; no dry-run timer), the policy file (`posttrain_ops limits`), the bootstrap's `--lifetime-seconds` (the 17,820 s timer lifetime) | `test_posttrain_guards.py`, `test_posttrain_readiness.py`, `test_posttrain_runbook.py`, `test_posttrain_bootstrap.py` |
| In-Pod idle self-destruct after 900 s | `pod_self_destruct.py --idle-seconds --activity` (the worker's heartbeat, the bootstrap's progress stamps); `progress_unix` is the hook for run 2's reward waits | `test_posttrain_guards.py`, `test_posttrain_integration.py` |
| Probe stop below the planning speeds | `PosttrainWorker.run_probe` | `test_posttrain_run.py`, `test_posttrain_integration.py` |
| SFT speed check after 50 updates | `PosttrainWorker.sft_speed_check` | `test_posttrain_run.py` |
| Projected finish before every stage, re-based on slow evaluation files | `PosttrainWorker.projected_finish`, `observe_generation` | `test_posttrain_run.py` |
| Finalization reserve: no stage runs into the last 900 s | `PosttrainWorker.tick`, `stopped` (`finalization_reserve`) | `test_posttrain_run.py` |
| Wall, cost and idle stops during GPU work | `PosttrainWorker.tick`, `stopped` | `test_posttrain_run.py` |
| R2 prefix guard | `posttrain_ops.GuardedStore`, the worker's only store | `test_posttrain_run.py`, `test_posttrain_guards.py` |
| Weights-only upload rate, 3.06 MB/s | `posttrain_ops.REQUIRED_UPLOAD_BYTES_PER_SECOND`; the receipt's `r2_round_trip_tested` | `test_posttrain_guards.py`, `test_posttrain_readiness.py` |
| One bucket-scoped Pod token per run, over SSH stdin | `rehearsal_ops pod-env`, the bootstrap's 0600 key check; revocation and `.env` removal are owner runbook steps | `test_rehearsal_bootstrap.py`, `test_posttrain_bootstrap.py` |
| Mac execution windows: none of Docker Desktop's four removable default shares mountable (the built-in `/var/folders` share is recorded as a known residual), no credentials in the executor's environment, no identity in the shell's or the login session's SSH agent, the pinned image ID | `tools/mac_window_check.py` before every batch; emptying Docker Desktop's file sharing and `ssh-add -D` are owner steps | `test_mac_window_check.py` |
| Hidden tests stay on the Mac | `scratchcoder.eval.export`; `posttrain_binding.require_export` (check-exports, build, verify, the worker); `validate_against_tasks`, `rl.command.check_attempts_header` | `test_pod_task_export.py`, `test_posttrain_binding.py`, `test_posttrain_run.py` |
| Verified teardown | `pilot_supervisor --mode posttrain` (`PosttrainSupervisor`) and `posttrain_ops.verify_final` | `test_posttrain_integration.py` |

## Projected cost

`python -m tools.pilot_feasibility posttrain` at the planning speeds and run B's measured training rate, with the smaller SFT set:

| Stage | Hours | Dollars |
| --- | ---: | ---: |
| Setup (bootstrap, fetch, tests, preflight) | 0.30 | $0.18 |
| Generation probe | 0.10 | $0.06 |
| ARC-Easy on three parents | 0.06 | $0.04 |
| Base-B evaluation generation (18.6M tokens at cap) | 1.07 | $0.64 |
| SFT (106 updates, 4 dev-loss checks, compile) | 0.07 | $0.04 |
| ARC-Easy on the SFT weights | 0.02 | $0.01 |
| SFT evaluation generation (18.8M tokens at cap) | 1.08 | $0.65 |
| RL screening generation (2.05M tokens at cap) | 0.12 | $0.07 |
| Finalization | 0.25 | $0.15 |
| **Total** | **3.08** | **$1.85** |

At twice the planning speed the run takes about 1.98 hours and $1.19. At half the planning speed the probe stops the run after at most about $0.39. Every pool attempt is priced at its token cap, as if none stopped early.

## After run 1

On the Mac, in an execution window: `eval-execute` of the four dev completions files and the dev constant-answer baseline; `eval-seal` of the four test completions files (they execute only after all four endpoints are frozen, after run 2); `rl-screen` of the screening attempts; and `tools.benchmark_dispatch` of the six benchmark files on GitHub Actions. The screening record and the measured rates re-price run 2 at the owner's gate.

## Run 2

`tools/posttrain_run.py run` runs a run-2 specification with `Run2Worker` (`tools/posttrain_run2.py`): run 1's worker with two more stage kinds and the reward bridge. Every artifact goes up as soon as it exists, as one label under `posttrain-runs/2/<pod id>/<label>/`, recorded in `report.json` with its seconds and SHA-256s.

| Stage | Reads | Writes (R2 label: files) |
| --- | --- | --- |
| `rl` | run 1's selected SFT weights; the withheld RL task file, the withheld dev subset and the facts file | `rl-check-<update>`: `checkpoint.pt` at each dev check (updates 0, 12, 25, 50, 75, 100); `rl`: `result.json`, `events.jsonl`; `rl-stop-weights` when weights no check covers |
| `more-sft` | the same SFT weights and files; RL's busy seconds, update-0 check and attention kernel | `more-sft-check-<step>`: `checkpoint.pt` after 1, 2, 4 and 8 passes and at the budget's end; `more-sft`: `control-result.json`, `control-events.jsonl` |
| `arc-post` | each arm's selected checkpoint evaluated here; the ARC-Easy questions | `arc-post-rl`, `arc-post-more-sft`: `arc.json` |
| `eval-rl`, `eval-more-sft` | each selected checkpoint evaluated here; run 1's pool exports and benchmark releases | `eval-<arm>-<file>`: `completions.jsonl`, seven files each (`chat-v1`) |

Then run 1's finalization, after the bridge directories (`outbox/`, `inbox/`, `done/`, `withdrawn/`, `rejected/`, `server.lease`, `server.lease.lock`, `client.lock`) are uploaded as one label, `bridge` (`bridge.tar`), so a late reply is kept; a file the Mac replaces while it is archived is kept whole, and one it removes is named in `report.json` and left out.

**RL** runs `run_rl` on the frozen protocol: 100 updates of 32 kept tasks x G = 8, temperature 1.0, top-p 1.0, at most min(256, 1,024 - prompt) new tokens, tasks shuffled with seed 0 and reshuffled each pass, a fresh AdamW (0.9 / 0.95, weight decay 0.1) at 3e-6 after 5 linear warmup updates, clip 1.0, no KL, and the RL loop's stops (non-finite, collapse, dev patience 3, leakage above 1% of rewarded attempts, reward wait, external). Rewards come through `BridgeRewardSource` and dev checks through `BridgeDevEvaluator`, one `BridgeClient` for the run, with the Mac's facts in place of the hidden tests and the cached encoder for every prompt; the model trains with fused attention and the foreach AdamW, as SFT did. Every dev check, check 0 included, generates at batch 256. `report.json` records RL's GPU-busy seconds (rollout generation plus training steps) and, separately, the dev checks' generation, the Mac waits and the checkpoint saves (RL does not compile). A stop that is a result (completed, collapse, dev patience, leakage, non-finite) lets the control run; the others are [arm stops](#an-arms-stop).

**The control** runs `run_more_sft` from the same SFT weights on the kept tasks' reference solutions (chat-v1, assistant-only loss), SFT's optimizer at a constant 6e-5 after 10 warmup updates, 16 x 2 rows an update, its GPU-busy budget RL's measured busy seconds within 5%, RL's attention kernel, its start RL's update-0 check, its dev checks at batch 256 as RL's (both arms read `rl.dev_batch_size`). `report.json` records the measured mismatch.

**Selection.** Each arm reports the selectable check with the highest dev-subset greedy pass@1, ties to the higher 8-sample mean, then to the later check; `report.json`'s `selection` lists every check's scores and the winner's label, and the ARC and evaluation stages read the two winners.

**Same weights** (the owner's run 2 gate rulings). Once an arm has selected, the worker compares its checkpoint with the SFT weights as it loaded them, every tensor's name, dtype, shape and bytes (`scratchcoder.rl.checkpoint.compare_to_loaded`: the SFT file loaded as RL loads it, so an SFT file with mixed dtypes, which loads into float32, still matches RL's update-0 check; two saves of one model are two files with two SHA-256s, so a file hash cannot decide it). An arm with the SFT weights is not evaluated again: its `arc-post` entry and its `eval-<arm>` stage are skipped, and `selection.endpoints.<arm>` records the comparison, `evaluation: reuses-run-1-sft`, `independent: false`, and for each artifact the arm would have produced the run 1 artifact that stands for it, with its pin. The control starts from RL's update-0 check, so this is what happens when neither arm beats its start; it replaces the earlier flag for two arms on one file. Run 1's eight SFT evaluation artifacts (`arc-sft-sft/arc.json` and the seven `eval-sft-<file>/completions.jsonl`) are pinned in the specification's `sft_results` by R2 key, size and SHA-256 at the gate, so the reuse is verifiable on the Mac (`tools/posttrain_endpoints.py plan`). An arm whose weights differ is `evaluation: pod`.

### An arm's stop

A wall or cost stop ends the run, as in run 1. A stop of the Mac's side in either arm (`reward_wait`, a timeout on either side; `reward_host_failure`; `bridge_lost`) and a margin stop (`wait_margin`) end only that arm's training, after its latest weights reach R2 (the owner's run 2 gate rulings):

| Stop | In RL | In the control |
| --- | --- | --- |
| Mac's side | RL's endpoint selected from its scored checks and evaluated; the control skipped | the control not evaluated; RL's endpoint evaluated |
| `wait_margin` | as for the Mac's side (the control could not run its dev checks) | the control's endpoint selected from its scored checks; each arm evaluated as the projected-finish rule admits it: when both do not fit but RL's alone does, the control is recorded `not-evaluated` (`arm_stop.control_not_admitted`) and RL's is evaluated |
| wall or cost | the run ends | the run ends |

The evaluation that follows needs no Mac wait and runs under the projected-finish rule, the finalization reserve and every limit; the projection prices only the stages and ARC entries that will run. An arm with the SFT weights is not evaluated (above). With no RL dev check scored there is nothing to select and the run finalizes at once. The run then ends `stopped` with the arm's reason, recorded in `report.json`'s `arm_stop` and the archived summary. While the evaluation runs the heartbeat reads `running`, never `awaiting_rewards`, so the watchdog's 600 s idle rule and the in-Pod 900 s rule apply as to any stage.

**Can a margin stop happen?** Yes, in the control, and it grows more likely the faster run 1 measures. The margin is fixed (one full wait at its worst, 11,746.5 s), while the time the 120% stop leaves after the control is 20% of the estimate plus the evaluation, ARC and reserve that follow it. Roughly, the control's last check keeps `0.2 x estimate (in seconds at $0.60/h) + the evaluation and ARC after the control + its dev wait - 180 - 11,746.5` seconds free; when that is negative the control margin-stops, at its last check or earlier, before its budget is spent. A faster Mac lowers the estimate; a faster GPU lowers both the estimate and the evaluation after the control, so it bites hardest. RL keeps at least 1,392 s free in every case below, so RL never margin-stops at these rates. The cases, each with the estimate projected at the given rates (the planning time scaled) and the run simulated at those same rates, from a 900 s setup (and a 1,500 s one):

| Mac time, GPU time (x planning) | Estimate, stop | First margin stop, 900 s setup | 1,500 s setup |
| --- | --- | --- | --- |
| 1.0, 1.0 | $3.53, $4.23 | none (515 s free at the control's last check) | the control's last check |
| 1.1, 1.0 | $3.67, $4.39 | none (695 s) | none (95 s) |
| 0.9, 1.0 | $3.40, $4.07 | none (335 s) | the control at 85% of its budget |
| 0.75, 1.0 | $3.19, $3.83 | none (65 s) | the control at 80% |
| 0.7, 1.0 | $3.13, $3.75 | the control's last check | the control's fourth check, at 80% |
| 0.6, 1.0 | $2.99, $3.58 | the control at 85% | the control at 60% |
| 1.0, 0.9 | $3.35, $4.02 | the control at 80% | the control at 60% |
| 1.0, 0.8 | $3.17, $3.80 | the control at 40% | the control at 20% |
| 0.5, 1.0 or 0.8, 0.8 | $2.86 or $2.90 | refused at the gate (below) | refused at the gate |

A margin stop in the control is safe (the worker selects its endpoint from the checks it scored and evaluates both arms as admitted, `test_no_control_update_starts_without_one_full_wait_free`), but the control then has not spent RL's GPU-busy budget, so the comparison it makes is not compute-matched; `more_sft.within_tolerance` says so. Whether to base the margin on run 1's measured wait or to raise the stop fraction for a low estimate is the owner's decision; the code keeps the fixed margin and the 120% stop.

### Inputs and the gate

`configs/rehearsal-a40-s-post-2.json` is the approved template (`posttrain_approved.run2_specification`). The values fixed at the owner's gate are `null` there: the ceiling; the gate's estimate (`limits.gate_estimate_usd`, the run 2 projection's total at run 1's measured rates and the gate's quoted price) and the cost limit it gives (`limits.cost_limit_usd`); run 1's selected SFT weights (key, size and SHA-256 of the `sft-weights` label in run 1's verified report); run 1's SFT evaluation artifacts (`sft_results`); the three files `scratchcoder rl-pod-export` writes; and the pool test export (run 1's: the sealed split is not read again). `python -m tools.posttrain_approved run2-gate` fills them from the gate's own files, the estimate read from the projection's JSON output (`--feasibility`) and never typed, and writes the launch specification. It refuses a projection of another template, of another run 1 report, at another quoted rate or above the $0.59 GPU quote. The worker, `check_run2` and the readiness receipt refuse a `null`, a non-finite or non-positive estimate, and a written cost limit other than its derivation.

`scratchcoder rl-pod-export` reads the pool train split, the screening record's kept list and the pool dev split on the Mac and writes, deterministically:

- `rl-tasks.jsonl`: each kept task with its hidden tests replaced by the fixed placeholder and its reference kept;
- `rl-dev.jsonl`: the fixed 200-task dev subset, withheld, without references;
- `rl-facts.json`: each task's parseability and best constant answer's reward, computed from the real tests (the answer itself is left out);
- `rl-pod-export.json`: the SHA-256 and size of the sources and of the exact bytes written.

The binding pins the three files; the binding, the Pod's `verify` and the worker refuse a file in which any task carries its hidden tests, and the worker checks the facts file with `load_facts` against its pinned SHA-256 before any stage starts.

### The reward bridge's bounds

Every bound on one wait is derived in one place, `posttrain_ops.bridge_timeouts`, and the specification writes them down (it is refused if one differs):

| Bound | Rehearsal value | Derivation |
| --- | ---: | --- |
| One program's longest try | 42 s | the reward channel's longest budget, 27 s (2 s + 0.25 s x 100 cases), + 15 s host slack |
| One reward attempt | 175 s | 4 tries x 42 s + 1 + 2 + 4 s of backoff |
| Mac `--reward-timeout` | 5,600 s | ceil(256 / 8) = 32 rounds x 175 s |
| Mac `--dev-timeout` | 9,492 s | ceil(1,800 / 8) = 225 rounds x 42 s, + 42 s for a program still running |
| Bridge allowance | 1,397.5 s | 2 s Mac poll + 5 tries x (30 + 120 + 120 s) SSH timeouts + 3 x 15 s backoff + 0.5 s Pod poll |
| The Mac's checks before a request | 857 s | one execution-window pass at its worst, 11 `docker` calls x 60 s + 2 `ssh-add` x 30 s + 1 `launchctl` x 10 s = 730 s, + the image check, 4 tries x 30 s + 1 + 2 + 4 s = 127 s |
| Pod reward / dev timeout | 7,854.5 s / 11,746.5 s | the Mac's bound plus the allowance and the checks |
| Watchdog idle extension | 7,914.5 s / 11,806.5 s | the Pod's timeout for the wait + 60 s |
| Wait margin | 11,746.5 s | one full wait, the longer kind ($1.96 at $0.60/h) |
| Lost bridge | 600 s | the in-Pod idle limit less 300 s |

The proposal's flat 420 s is not used: 256 slow I/O attempts on 8 workers can take far longer, and a fixed value would stop a run that is merely slow.

**The Mac's window, repeated.** The bridge (`scratchcoder rl-bridge`, started through `mac_window_check exec`) runs the execution-window checks on its own process (`scratchcoder.sandbox.window`): its environment holds no `.env` name, secret or SSH agent variable; no SSH agent the Mac names (the shell's the bridge was started from, its path handed down by `mac_window_check exec`, and the login session's) holds an identity; the pinned image has its recorded ID; none of Docker Desktop's four removable default shares mounts, each refused with Docker's `mounts denied` answer, and the `/var/folders` residual mounts (the known residual, and the pass's positive control). It runs them at its start and again before scoring a request once the last pass began 900 s ago or more (`--window-check-interval`, which accepts nothing longer); each pass leaves a receipt in its state's `window-checks/`, and a restarted bridge numbers its passes after the highest receipt there. A pass takes seconds; the Pod's timeouts hold one at its worst (above). At the planning rates RL sends a reward request about every 80 s, so a pass runs before about one request in eleven and costs the Mac about 1% of its time. A failed or errored pass stops the bridge (exit 4) before anything more is scored: its lease lapses, the Pod sees a lost bridge, and the [arm-stop rule](#an-arms-stop) applies. The Pod's R2 token stays in `.env` during the bridge window under this child-environment rule, the accepted rule: the bridge's environment holds no `.env` variable, and its checks refuse to go on once one appears.

### Liveness while the Pod waits

- **Heartbeat:** while the bridge client waits, the worker's sleep between looks at its inbox heartbeats `awaiting_rewards` with the client's `heartbeat()` fields (request ID, kind, index, count, start, age) every 5 s.
- **Dev-check generation:** between two waits a dev check generates 1,800 programs on the GPU at batch 256 (about 92 s at the planning speed); every generation batch stamps the GPU activity and refreshes the heartbeat (`Run2Worker.dev_tick`), so neither idle rule nor the heartbeat's staleness reads it as a hung worker.
- **Mac watchdog:** under run 2's policy an `awaiting_rewards` heartbeat naming a reward or dev request lets the GPU idle for that kind's extension, and never more, so a worker that hangs while claiming to wait is still deleted. Run 1's policy grants none. The receipt requires the policy's extensions to be exactly the derived ones.
- **In-Pod idle rule:** the Mac's server renews `server.lease` in the Pod's bridge root, dated by the Pod's own clock, with every poll and, while it reads, scores or answers a request, every few seconds (so a slow SSH call never lets it lapse while the Mac is alive). During a wait the worker stamps `progress_unix` from a lease renewed within its 60 s; the in-Pod timer reads it through its existing `--activity` heartbeat, and a stale lease adds nothing. A dead Mac therefore lets the Pod go within 900 s of its last renewal.
- **Lost bridge:** a wait with no lease renewal for 600 s (from the later of the wait's start and the last renewal) is given up as `bridge_lost`: the latest weights are uploaded, the arm's training ends and the evaluation that needs no Mac runs before the finalization ([an arm's stop](#an-arms-stop)), all before the in-Pod timer would fire.
- **Margin:** `should_stop` is not read during a wait, so before every update and every request the worker ends the arm's training (`wait_margin`) unless one full wait fits before the cost stop, the wall and the finalization reserve. The run goes on with what needs no Mac wait and finalizes in full.

### Limits of run 2

Every limit derives from the cost limit, the lower of the ceiling and 120% of the gate's estimate rounded down to the cent, in one place (`ceiling_limits` with the estimate; `posttrain_ops limits` prints them). At the planned $5.80 ceiling and an estimate of $3.53 (the projection at the planning speeds) the cost limit is $4.23: the worker stops at $4.23, the watchdog at $4.22, the Pod wall is 25,380 s, the in-Pod timer 25,200 s; the in-Pod idle limit 900 s, the watchdog's 600 s outside waits, the transfer window 900 s. Only an estimate of $4.84 or more lets the $5.80 ceiling bind (34,800 s and 34,620 s). The policy file must carry exactly the derived wall and cost stop. The launch rule needs 20,288 s at the planning rates, which leaves 4,912 s of the 25,200 s for setup; a plan that does not fit is refused. The gate itself (`posttrain_approved.gate_launch_fit`) refuses limits that cannot hold the planned 900 s setup plus that need, so the refusal comes before any billing and not at the readiness receipt after the Pod has billed its setup: at a $5.80 ceiling that is an estimate below about $2.98 (a stop below $3.57), which a run 1 about twice as fast as planned on the Mac would give. The gate prints what its limits leave for setup, and the projection reports it for each scenario (`launch_at_this_estimate`). Run 2 has no probe: the projected-finish rule prices RL's later stages at RL's measured rates, and the evaluation at its batch's planning speed, lowered by slow evaluation files; a fast measurement never loosens it, as in run 1. (RL's dev checks generate at batch 256 and the evaluation at 64, so the checks' rate does not stand in for the evaluation's.)

### Safeguards of run 2

| Safeguard | Enforced in | Tested in |
| --- | --- | --- |
| Gate values filled, the approved protocol unchanged | `Specification.from_dict`, `posttrain_approved.check_run2`, the receipt | `test_posttrain_run2.py`, `test_posttrain_run2_readiness.py` |
| Cost limit at min(ceiling, 120% of the gate's estimate), every limit from it; the estimate from the projection's output | `posttrain_ops.estimate_limit_usd`, `ceiling_limits`, `specification_limits`; `posttrain_approved.gate_estimate`; `check_policy` | `test_posttrain_run2.py`, `test_posttrain_run2_readiness.py`, `test_posttrain_run2_guards.py`, `test_posttrain_run2_feasibility.py`, `test_posttrain_run2_runbook.py` |
| Dev checks of both arms at one batch, 256 | `posttrain_approved.RUN2_RL`, `Run2Worker.dev_evaluator` | `test_posttrain_run2.py` |
| Same weights decided bit for bit; such an arm reuses run 1's pinned SFT results | `scratchcoder.rl.checkpoint.compare_to_loaded`, `Run2Worker.choose_endpoint`; `sft_results` (`check_sft_results`); `tools/posttrain_endpoints.py`; the unseal record; the scorecard | `test_rl_checkpoint.py`, `test_posttrain_run2.py`, `test_run2_endpoints.py`, `test_eval_executor.py`, `test_scorecard.py` |
| An arm's stop of the Mac's side or margin stop ends that arm, not the run | `Run2Worker.stop_kind`, `stop_arm`, `skip_reason`, `remaining_stages`, `after_stages` | `test_posttrain_run2.py` |
| The window checks repeated while the bridge runs, failing closed | `scratchcoder.sandbox.window`, `BridgeServer.window_check`, `bridge_command.window_pass` | `test_rl_bridge_window.py`, `test_mac_window_check.py` |
| Derived wait bounds, equal in the specification, the policy and the runbook | `posttrain_ops.bridge_timeouts`, `specification_bridge_timeouts` | `test_posttrain_run2_guards.py`, `test_posttrain_run2_runbook.py` |
| Watchdog extension while `awaiting_rewards`, bounded by kind | `runpod_watchdog._awaiting_within_extension`; `check_policy` | `test_posttrain_run2_guards.py`, `test_posttrain_run2_integration.py` |
| In-Pod idle: a fresh lease is progress, a stale one is not | `Run2Worker.observe_wait`, `lease_renewed`; `pod_self_destruct --activity` | `test_posttrain_run2_guards.py`, `test_posttrain_run2.py` |
| Lost bridge before the in-Pod timer | `Run2Worker.observe_wait` (`bridge_lost`) | `test_posttrain_run2.py` |
| Dev-check generation stamps the GPU and the heartbeat after every batch | `Run2Worker.dev_tick`, `scratchcoder.rl.dev.ticking_generations` | `test_posttrain_run2.py`, `test_rl_dev.py` |
| Margin for one full wait | `Run2Worker.should_stop`, `before_wait` (`wait_margin`) | `test_posttrain_run2.py`, `test_posttrain_run2_guards.py` |
| A wait ended by an error withdraws its request | `BridgeClient._exchange` | `test_rl_bridge.py` |
| Each check's weights in R2 at once; the latest weights and the arm's event log before any end, in either arm | `Run2Worker.upload_weights`, `salvage` | `test_posttrain_run2.py` |
| No hidden test on the Pod | `scratchcoder.rl.pod_export`; `posttrain_binding.require_withheld`, `require_facts`; `load_facts` | `test_rl_pod_export.py`, `test_posttrain_run2.py` |
| Run 1's safeguards (ceiling-derived lifetime, projected finish, reserve, prefix guard, verified teardown) | as run 1 | the run-2 tests above and run 1's |

### Projected cost of run 2

`python -m tools.pilot_feasibility posttrain --specification configs/rehearsal-a40-s-post-2.json --ceiling-usd 5.80` at the planning speeds, in the proposal's table form (with `--run1-report`, `--screening` and `--dev-results` it adds a table at run 1's measured rates):

| Stage | Hours | Dollars |
| --- | ---: | ---: |
| Setup (bootstrap, fetch the SFT weights, tests, preflight) | 0.25 | $0.15 |
| RL rollouts (6.55M tokens at batch 256) | 0.18 | $0.11 |
| RL training steps (26.2M slots) | 0.13 | $0.08 |
| RL waiting on Mac rewards (100 x 48 s) | 1.33 | $0.80 |
| RL dev checks at batch 256: generation / Mac wait (6) | 0.15 / 0.50 | $0.39 |
| More-SFT training (equal to RL's busy time) | 0.31 | $0.19 |
| More-SFT dev checks at batch 256: generation / Mac wait (5) | 0.13 / 0.42 | $0.33 |
| Evaluation generation, RL and more-SFT (37.6M tokens at cap) | 2.17 | $1.30 |
| ARC-Easy on both selected checkpoints | 0.04 | $0.02 |
| Checkpoint loads and weights saves | 0.03 | $0.02 |
| Finalization | 0.25 | $0.15 |
| **Total** | **5.89** | **$3.53** |

At this estimate run 2 would stop at $4.23 (120%), a 25,380 s wall. With the Mac twice as slow on model output it is about 8.14 hours and $4.88, above that stop but under $5.80: the stop follows the estimate at run 1's measured rates, which the gate reads from the projection's output (`gate_estimate_usd`), never from this table. The ceiling is fixed at the gate.

### After run 2

On the Mac, `python -m tools.posttrain_endpoints plan` records each endpoint's source from the verified reports: base and SFT are run 1's; each arm is `pod`, `reuses-run-1-sft` (checked against the `sft_results` pins, run 1's report and the Mac's copies) or `not-evaluated`. Then, in an execution window and only for the arms evaluated on the Pod: `eval-execute` of the dev completions, `eval-seal` of the test completions in run 1's ledger, and `tools.benchmark_dispatch` of the benchmark files on GitHub Actions. Only after the owner freezes all four endpoints (base B, SFT, RL, more-SFT) is the unseal record written from the plan: each distinct checkpoint executes its two test files once, one logged read each, and a reusing arm is mapped to SFT's reads (`{"reuses": "sft"}`; the ledger's `opened` event names it), an absent arm recorded as not evaluated. The scorecard's inputs file records the reuses; a reusing arm's rows are SFT's, marked not independent, and it is never compared with SFT. The [run 2 runbook](../POSTTRAIN_RUN2_RUNBOOK.md) gives every command.

## Limitations

- Generation speed has never been measured; the probe measures it first.
- Run 2's bridge has not crossed a real SSH link, and its bounds are worst cases of the channel's rules: a real wait is expected to take minutes, not hours. A live but stuck Mac can hold the Pod idle up to the extension before the watchdog deletes it, at most about $1.97 of a reward or dev wait.
- The SSH rescue carries only the weights label (`checkpoint.pt` and `record.json`). Any other label R2 never verified is listed for a rescue by hand and teardown is not approved, but the watchdog and the in-Pod idle rule still delete the Pod 600 to 900 s after its last GPU work: the runbook's rescue block has about 2 to 7 minutes after the handover.
- Termination is layered best effort: there is no provider-enforced Pod lifetime.
- One seed per arm; the results are integration evidence at 135M parameters, not capability claims.
