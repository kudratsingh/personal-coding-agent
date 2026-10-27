# Post-training run worker

`tools/posttrain_run.py` runs post-training run 1 on one A40 Pod that already exists: a generation probe, ARC-Easy, evaluation generation of the base and SFT checkpoints, one SFT pass and RL screening generation. It never creates, rents or deletes a machine. Every artifact leaves through R2 as soon as it exists; the Mac's supervisor verifies all of them before the Pod may be deleted. Hidden tests never reach the Pod. Operations are in [post-training runs](../posttrain-runs.md) and the [runbook](../../POSTTRAIN_RUNBOOK.md).

## Worked example

**The smoke.** `python -m tools.posttrain_run smoke --output runs/posttrain-smoke` writes tiny fixtures and runs all seven stages on CPU into an in-memory bucket in about three seconds. The model has 1 layer, width 32 and a 192-token context; the tokenizer is a byte-level BPE with 96 merges trained on text like the fixtures' (`smoke_tokenizer`), so a prompt like `Write add_11(x) returning x + 11.` plus its public test is 24 chat-v1 tokens (61 with no merges). With 3 pool tasks per split and the protocol cap of 512, each pool attempt may generate `min(512, 192 - 24) = 168` tokens. A greedy file has 3 attempts and an n = 8 file 24.

**The limits of run 1.** The ceiling is $3.00 at $0.60 an hour. `ceiling_limits(3.00, 0.60)` gives the wall `floor(3.00 x 3600 / 0.60) = 18,000 s`, the timer `18,000 - 180 = 17,820 s`, the worker's cost stop $3.00 and the watchdog's $2.99. A Pod whose create request went out at `ORIGIN` has its timer armed for `ORIGIN + 17,820`.

**The launch plan.** At the planning rates: the probe 360 s; ARC on three parents `3 x (20 + 54) = 222 s`; base evaluation `7 x 20` s of loads plus `(2 x 2,000 x 9 x 512 + 705 x 256) / 5,000 = 3,722.5 s` of generation, 3,862.5 s; SFT `20 + 120 + 106 x 32,768 / 56,765 + 4 x 827,850 / 94,862 + 4 x 5 = 256 s`; ARC on the SFT weights 74 s; SFT evaluation 3,898.6 s (the chat benchmarks cap at 512); screening `20 + 1,000 x 8 x 256 / 5,000 = 429.6 s`; and the 900 s reserve. That is 10,003 s, so the worker launches only if that much of the 17,820 s is left.

**The SFT plan.** 3,390 rows at 32 per update is `ceil(3,390 / 32) = 106` updates, the last with 30 real rows and 2 rows of padding. Warmup is `round(0.03 x 106) = 3` updates. Dev loss is measured at `round(k x 106 / 4)` for k = 1, 2, 3, that is updates 26, 53 and 80 (Python rounds 26.5 to 26 and 79.5 to 80), and at 106.

**The projected finish.** Say the probe measures 7,000 tok/s at batch 64 and eval-base is about to start at 1,800 s. The work remaining is eval-base `140 + 18,612,480 / 7,000 = 2,799 s`, SFT 256 s, ARC 74 s, eval-sft 2,825 s, screening 313 s and the 900 s reserve: 7,167 s, ending at 8,967 s, inside the 17,820 s. The stage starts. Had the probe measured 2,500 tok/s, it would already have stopped the run.

## Data flow and shapes

1. **Launch gates (`run`).** The receipt must be `pilot_readiness_receipt posttrain`'s, for run 1, bound to this specification, binding and preflight by SHA-256; every gate `true`; the policy no weaker than the ceiling-derived limits (`check_policy`); the armed deadline within the derived lifetime; the timer's idle rule watching this worker's heartbeat (`check_self_destruct`); the R2 prefix this Pod's. The binding must name exactly the specification's inputs (`posttrain_binding.require_specification`), every input is re-hashed (`verify`), and the shipped revision must equal the receipt's.
2. **The worker's store.** `GuardedStore` wraps the R2 store: any request but GET and HEAD whose key is not under `posttrain-runs/1/<pod id>/` raises `PrefixViolation` before a socket opens. The uploader, the round trip and the final checks all use it.
3. **Each stage.** `run_stage` checks the wall, cost and idle limits, applies the projected-finish rule to a generation stage, runs `run_<kind>` and records its seconds and outputs. During GPU work `tick` refreshes the heartbeat's `last_gpu_activity_unix` and progress counter and stops at the wall or the cost stop.
   - `probe`: `[B, 256]` prompts of random tokens at the chosen lengths through `benchmark_generation`, one call per batch size.
   - `arc`: `evaluate_choices` with the project's label-likelihood scorer, float32, a tick every 64 questions.
   - `eval`: `generate_completions` with `ticking_generate`, which is `generate_many` batch for batch (prompts sorted longest first, cut into batches of 64) with a tick after each `[64, <=1024]` batch.
   - `sft`: an `SFTLoader` of `[16, 1024]` inputs and targets, two microbatches per update, through the `Trainer` with `loss_normalization: update_targets`; `assistant_loss` over each dev set at the quarter checks; `save_weights` for each new lowest.
   - `screen`: `screening_rollouts` on chunks of 64 candidates (`64 x 8 = 512` rows in batches of 64), then `write_attempts` with the header.
4. **Artifacts.** `export_label` publishes `exports/<label>/` with its files and `record.json`, lists every file in `report["artifacts"]` with its key, SHA-256 and size, and submits the label to the uploader as a milestone. Nothing is rolled or pruned.
5. **Finalization** is the pretraining rehearsal worker's (`Worker.finalize`), reused through two hooks: `transfer_window_seconds` and `closing_record`. The heartbeat's `r2_final` is `{report_key, report_sha256, artifact_count}`. The supervisor downloads the report pinned to that SHA-256 and HEADs every listed artifact.

## Design choices and common mistakes

- **Why a ceiling, not a wall.** A dead Mac must not let a Pod run to the pretraining rehearsal's $10: the timer, the watchdog and the worker all stop within $3.00 of spend. Deriving all three from one function keeps them from drifting apart.
- **Why an in-Pod idle rule.** The Mac watchdog's idle rule needs the Mac. The in-Pod timer's needs only the Pod: 900 s with no GPU work and no progress stamp deletes it. The bootstrap stamps progress after every step, so a slow but live setup is not idle; a worker's rewritten heartbeat is not activity unless its GPU timestamp or `progress_unix` moves.
- **Why every stage stops instead of skipping.** A stop leaves every artifact made so far in R2 and a report naming the reason; the owner re-prices. Skipping a stage would silently change what run 2 starts from.
- **Why the cached encoder.** The reference `ByteBPE.encode` applies all 65,000 merges to every chunk: about a second per pool prompt and a third of a second per ARC question with the real tokenizer, so encoding 2,000 prompts before the first batch would idle the GPU past the 600 s limit. Every prompt path (the probe, ARC, `generate_completions`, screening) uses `FastEncoder`, whose IDs are the reference's exactly, in about a millisecond a prompt; the tests disable the reference encoder while the worker runs.
- **Why weights only.** SFT's selected checkpoint feeds evaluation, screening and run 2, all of which start fresh optimizers, so the optimizer state (two thirds of a full checkpoint) is never needed.
- **A common mistake** is running `eval-execute` on the Pod export: its placeholders would fail every attempt. The executor refuses an export, and refuses a full file whose export differs from the one the completions name.

## Verification

- Worker, stops and gates: [tests/test_posttrain_run.py](../../tests/test_posttrain_run.py).
- End to end into the supervisor and the watchdog, the probe stop, refused labels and the idle rule: [tests/test_posttrain_integration.py](../../tests/test_posttrain_integration.py).
- Limits, the idle timer, the watchdog mode and the prefix guard: [tests/test_posttrain_guards.py](../../tests/test_posttrain_guards.py).
- Binding, receipt, bootstrap, projection and runbook: [test_posttrain_binding.py](../../tests/test_posttrain_binding.py), [test_posttrain_readiness.py](../../tests/test_posttrain_readiness.py), [test_posttrain_bootstrap.py](../../tests/test_posttrain_bootstrap.py), [test_posttrain_feasibility.py](../../tests/test_posttrain_feasibility.py), [test_posttrain_runbook.py](../../tests/test_posttrain_runbook.py).
- Exports and their re-derivation: [tests/test_pod_task_export.py](../../tests/test_pod_task_export.py).

```sh
uv run --locked pytest tests/test_posttrain_run.py tests/test_posttrain_integration.py tests/test_posttrain_guards.py tests/test_pod_task_export.py -q
uv run --locked python -m tools.posttrain_run smoke --output runs/posttrain-smoke
```

## Questions and answers

1. *Why does the probe-stopped run still publish `r2_final`?* Every label it made (the probe) verified in R2, so the supervisor can verify the stopped run's report and artifacts and approve teardown without a grace period.
2. *What happens if the SFT weights never reach R2?* The run fails `r2_milestones_unverified`; the supervisor pulls `exports/sft-weights/` over SSH with the verified pull, checks its SHA-256 and only then approves teardown.
3. *Why is the projection priced at the token caps?* Pool attempts that stop early only make the run shorter; pricing at the cap makes the rule an upper bound.
4. *Can an export be scored by mistake?* No: its placeholders are recognized and refused, and each one hashes the hidden test it replaced, so only the matching full file is accepted.
5. *What does run 2 add?* RL and more-SFT stage kinds, the reward bridge, and Mac-stamped reward progress for the `awaiting_rewards` heartbeat.
