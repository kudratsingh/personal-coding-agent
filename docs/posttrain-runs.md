# Post-training run operations

The rest of the rehearsal continues from run B's `mixed-final` checkpoint in two paid runs on one Runpod A40 Pod each, under limits the owner approved on 2026-10-26:

- **Run 1** (this page): the generation probe, ARC-Easy on run A's two finals and run B, base-B evaluation generation, SFT, ARC and evaluation generation of the SFT weights, and RL screening generation. Hard ceiling **$3.00**.
- **Run 2** (a later change): RL with the Mac computing rewards, the compute-matched more-SFT control, and their evaluation generation. Its ceiling, about $5.80, is fixed at the owner's gate from run 1's measured rates.

**Status (2026-10-27): pre-launch.** The worker, its safeguards, the Mac-side tools and the [runbook](../POSTTRAIN_RUNBOOK.md) are written and CPU-tested, including an end-to-end smoke replayed into the real supervisor. Nothing has run on a GPU, no input has been staged to R2, and nothing has been spent. Every figure below is a projection.

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
| Mac execution windows: none of Docker Desktop's five default shares mountable, no credentials in the executor's environment, no identity in the shell's or the login session's SSH agent, the pinned image ID | `tools/mac_window_check.py` before every batch; emptying Docker Desktop's file sharing and `ssh-add -D` are owner steps | `test_mac_window_check.py` |
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

## Run 2 hooks

The stage machinery is a registry of kinds (`STAGE_KINDS`): run 2 adds RL, the more-SFT control and their evaluation stages on the same worker. The watchdog's `posttrain` mode already accepts an `awaiting_rewards` heartbeat, with the ordinary idle rule until run 2 adds its Mac-stamped reward progress, and the in-Pod timer counts a heartbeat's `progress_unix` as activity.

## Limitations

- Generation speed has never been measured; the probe measures it first.
- The SSH rescue carries only the weights label (`checkpoint.pt` and `record.json`). Any other label R2 never verified is listed for a rescue by hand and teardown is not approved, but the watchdog and the in-Pod idle rule still delete the Pod 600 to 900 s after its last GPU work: the runbook's rescue block has about 2 to 7 minutes after the handover.
- Termination is layered best effort: there is no provider-enforced Pod lifetime.
- One seed per arm; the results are integration evidence at 135M parameters, not capability claims.
