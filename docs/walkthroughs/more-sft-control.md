# Component: the more-SFT control

RL starts from the SFT checkpoint and spends GPU time. If the RL checkpoint scores higher on dev, that could be RL, or just more training on the same tasks. The control answers this. It starts from the same SFT checkpoint and spends the same GPU-busy time on supervised training with the same tasks' reference solutions. A claim that RL helped beyond supervision needs RL to beat this control. The code is `src/scratchcoder/rl/control.py`, which drives the existing SFT loader and trainer unchanged ([SFT batching](sft-batching.md)).

## Worked example

**The budget.** The RL run measured 4.2 s of rollout generation and 0.8 s of training steps, so the control's budget is 5.0 s of GPU-busy time. Say every control update takes 0.5 s. Before each update the control asks whether `busy + mean step / 2 ≥ budget`:

| Updates done | Busy | Busy + 0.25 | Next |
| ---: | ---: | ---: | --- |
| 9 | 4.5 | 4.75 | 4.75 < 5: train one more |
| 10 | 5.0 | 5.25 | 5.25 ≥ 5: stop |

It stops at exactly 5.0 s, a mismatch of 0. With 1 s updates against a 2.6 s budget it stops after 3 updates. At 2 updates, 2 + 0.5 < 2.6, so it trains a third and ends at 3.0 s. That is 0.4 s or 15.4% over, outside the ±5% tolerance, and the result says so (`within_tolerance: false`). The rule takes one more update only when it would land closer to the budget than stopping now, if the next update takes the mean time.

**The checks.** With 771 kept tasks and 16 × 2 = 32 rows per update, pass `p` is complete after update `ceil(p × 771 / 32)`:

| Passes | Update |
| ---: | ---: |
| 1 | 25 |
| 2 | 49 |
| 4 | 97 |
| 8 | 193 |

A check follows each of those updates that the budget reaches, and one more at the budget end. The start check is RL's update-0 check, the same SFT weights, reused rather than repeated.

## Data flow and shapes

1. **Rows.** Each kept RL task becomes one chat-v1 row with `render_code_task(encoder, task)`: the prompt and public test, then `<|python_start|>{reference}<|python_end|><|assistant_end|>` with loss on the assistant tokens only. RL samples are never targets.
2. **Loader.** `SFTLoader(rows, sequence_length = context, batch_size = 16, accumulation_steps = 2, seed = 0, passes = 400)`. Rows longer than the context are dropped and counted. 400 passes is only the plan's upper bound; the budget ends the control long before.
3. **Trainer.** The existing `Trainer`, from the SFT parent (`parent_checkpoint`), with `loss_normalization: update_targets`. Each update's loss is the mean over its assistant targets. The optimizer is a fresh AdamW (betas 0.9/0.95, weight decay 0.1, clip 1.0). The cosine schedule with `min_lr_ratio = 1` warms up over 10 updates to 6e-5 and then stays there. Precision is bfloat16 autocast on CUDA and float32 on the CPU, as in SFT.
4. **Busy time.** The trainer's per-update `step_seconds` is synchronized wall time from reading the batch to the optimizer step. The control sums it. Dev checks, checkpoints and execution waits run in the step callback, outside that interval, and are recorded separately.
5. **Checks.** At each pass boundary the callback runs the dev evaluator on the trainer's model and saves a weights-only checkpoint (`checkpoints/step-NNNNN.pt`). Selection is RL's rule: the highest greedy pass@1, ties to the higher 8-sample mean, then to the later check.
6. **Result.** `control-result.json` records the budget, the busy time, the mismatch in seconds and as a fraction, whether it is within tolerance, updates, passes, drops, every check and the selection, beside the trainer's own `record.json` and `events-rank0.jsonl`.

## Design choices and common mistakes

- **Reuse, not a fork.** The control is the SFT path with different rows and a constant learning rate. A separate training loop could differ in a detail (normalization, clipping, precision) that would then be confounded with RL.
- **Match GPU-busy time, not updates or tokens.** RL and SFT do different work per update. The question is what the same GPU budget buys, so compile, dev checks and Mac waits are excluded from both arms. Neither arm compiles, so compile time is 0 in both.
- **Report the mismatch; do not hide it.** The stop rule keeps the mismatch within half an update in the usual case, but the result always carries the measured difference and its fraction.
- **Geometric check spacing.** Training on a few hundred references overfits within a few passes. Checks evenly spaced over about 60 passes would see only the overfit end and make the control look worse than it is. Checks after 1, 2, 4 and 8 passes catch its best point. This is the one place the control's selection differs from RL's, whose checks are evenly spaced.
- **The start is shared.** RL's update-0 check is the same weights and the same dev subset, so it is reused. Both arms then select from a set that includes the SFT starting point.
- **External stops.** `should_stop` (cost, wall or idle) is read once per update, as in RL. A dev check that times out stops the control with `reward_wait` and saves its unchecked weights. After a `should_stop` stop no end check is taken (unless `check_at_external_stop`), so a stop meant to cap spend spends nothing more; the unchecked weights are saved as the stop checkpoint. An unexpected error, in the end check too, saves the latest weights before it propagates.
- **Same kernels as the arm it is compared with.** `attention` and `optimizer_foreach` are settings so the control can run the same attention kernel and optimizer path as the RL model and the SFT stage. A control on a slower path would fit fewer updates into the same seconds, which would favour RL. There is no `compile` setting: neither arm compiles.

## Verification

`tests/test_rl_control.py` covers:

- the rows matching `render_code_task`;
- the pass boundaries, including an update that completes two passes;
- the budget stop under a scripted clock (10 updates of 0.5 s against 5 s, and a reported 15.4% mismatch);
- a zero budget selecting the start check;
- the external, exhausted-passes and dev-wait stops;
- the trainer's own record and events: `update_targets` normalization, a warmup of 1.5e-5, 3e-5, 4.5e-5, then 6e-5, and token counts below the update's capacity;
- an end-to-end CPU run: three RL updates on a tiny model with fake rewards, then the control on the RL run's measured busy time.

```sh
uv run --locked pytest tests/test_rl_control.py
```

These are CPU tests. They do not measure GPU timing or the control's dev score on real tasks.

## Questions and answers

1. *RL measured 300 s of generation, 60 s of training steps and 900 s of reward waits. What is the control's budget?* 360 s. Reward waits and dev checks are excluded in both arms.
2. *Why does the control not train on RL's sampled programs that scored 1.0?* Then it would be a form of RL. The control is supervised training on the reference solutions.
3. *The budget allows 5.3 passes. Which checks run?* The start (shared), after 1, 2 and 4 passes, and at the budget end. The 8-pass check is never reached.
4. *Why stop when `busy + mean / 2 ≥ budget` and not when `busy ≥ budget`?* Stopping at the first update past the budget would overshoot by up to one update every time. The half-step rule picks whichever side lands closer.
5. *How does the constant learning rate come from a cosine schedule?* With `min_lr_ratio = 1` the cosine decays from the peak to the peak, so after warmup the rate stays at 6e-5.
