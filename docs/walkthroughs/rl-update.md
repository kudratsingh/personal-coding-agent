# Component: the RL update (policy gradient with group-mean advantage)

One RL update samples several programs per task from the current policy, scores them against hidden tests, and nudges the policy toward the programs that scored above their task's average. The code is in `src/scratchcoder/rl/`: `policy.py` (advantages and the loss), `rollout.py` (prompts, seeds, generation), `rewards.py` (attempts, rewards, the reward source) and `loop.py` (the run, its stops and its records). [RL screening, stops and selection](../rl.md) covers the run around the update.

## Worked example

Take a vocabulary of three tokens and a toy policy whose next-token logits are a learnable vector `θ = (0, ln 2, ln 3)` at every position, so `π = (1/6, 1/3, 1/2)` whatever the input. Three tasks, every prompt the single token `0`:

| Group | Rewards | Mean | Advantages `A = r − mean` | Attempts (tokens) | Skipped |
| --- | --- | ---: | --- | --- | --- |
| A | 1, 0 | 0.5 | +0.5, −0.5 | `[0, 2]`, `[1]` | no |
| B | 0.5, 0.5 | 0.5 | 0, 0 | `[2, 2, 2]`, `[1]` | yes: all rewards equal |
| C | 1, 0, 0.5 | 0.5 | +0.5, −0.5, 0 | `[2]`, `[0]`, `[1, 1, 1]` | no |

**The denominator.** `N` counts the attempt tokens of every non-skipped group: `2 + 1` for A and `1 + 1 + 3` for C, so `N = 8`. Group B's four tokens are not counted. C's third attempt has zero advantage: it counts in `N` but contributes no gradient, so it is never run through the model.

**The loss.** `L = −(1/N) Σᵢ Aᵢ Σₜ log π(yᵢₜ)`:

```
L = −(1/8) [ 0.5 (ln 1/6 + ln 1/2) − 0.5 ln 1/3 + 0.5 ln 1/2 − 0.5 ln 1/6 ]
  = (ln 2 − 0.5 ln 3) / 8
```

**The gradient.** For this policy `∂ log π(y)/∂θ = e_y − π`. The advantage-weighted token count is `Σᵢ Aᵢ nᵢ = 0.5·2 − 0.5·1 + 0.5·1 − 0.5·1 = 0.5`, so

```
Σᵢ Aᵢ Σₜ (e_y − π) = 0.5(e₀ + e₂) − 0.5 e₁ + 0.5 e₂ − 0.5 e₀ − 0.5 π
                    = (−1/12, −2/3, 3/4)
∂L/∂θ = −(1/8)(−1/12, −2/3, 3/4) = (1/96, 1/12, −3/32)
```

A descent step therefore raises `θ₂` (token 2 appeared in the above-average attempts) and lowers `θ₁` (token 1 appeared in a below-average one). `tests/test_rl_policy.py` checks both numbers exactly.

## Data flow and shapes

1. **Tasks.** `task_schedule` orders the kept tasks for pass `p` by `pass_key(seed, p, id)`, the first eight bytes of `sha256(f"{seed}|{p}|{id}")` (the SFT loader's order rule), with seed 0. Update `u` (numbered from 1) takes the next block of 32. A pass's last block shorter than 32 is not used; the next pass reshuffles. So an update never repeats a task, and no task repeats within a pass.
2. **Prompts.** Each task's chat-v1 prompt: `<|bos|><|user_start|>{prompt}\n\n{public test}<|user_end|><|assistant_start|>`.
3. **Rollouts.** `generate_rollouts` sends 32 × G = 256 rows to `generate_many`: temperature 1.0, top-p 1.0, stops `<|python_end|>` and `<|assistant_end|>`, at most `min(256, 1,024 − prompt)` new tokens. Row `(task, sample)` of update `u` has seed `attempt_seed(2^41, task, (u − 1)·G + sample)`. For one task these indices differ for every `(u, sample)`, so its seeds are distinct by construction. Dev checks (base 0) and screening (base 2^40) use other bases, so the three families never meet for a task. The loop also checks every seed it uses and raises on a repeat. Output: 256 token lists of length ≤ 256, the stop token included when one was produced.
4. **Attempts.** `extract_program` takes the first closed python block; no block gives `code = None`. An `Attempt` holds `(task_id, sample, code, completion_tokens)`. Only the code travels to the machine that runs it.
5. **Rewards.** `RewardSource.rewards(u, attempts)` returns one `AttemptReward(reward, status, audit_flags, seconds, unfinished)` per attempt, in order. A `None` program scores 0 with status `no_program` and never runs. A host failure is retried by the source, and raises `RewardHostFailure` if it persists; it is never a 0. `LocalRewardSource` runs `scratchcoder.sandbox.reward.score_attempt` in 8 threads with the channel's default budgets.
6. **Groups.** Rewards reshape to `[32, G]`. `group_advantages` gives `A = r − mean(r)` per group, or exact zeros and a skip flag when all G rewards are equal.
7. **Sequences.** `policy_sequences` keeps attempts of non-skipped groups with nonzero advantage and at least one token, as `prompt + attempt` with the prompt length, and returns `N`.
8. **Micro-batches.** Sequences are sorted longest first and cut into micro-batches of `micro_batch` (16 by default). Each becomes right-padded inputs `tokens[:-1]` and targets `tokens[1:]`, `[m, L]`, plus a mask that is true where the target is an attempt token (`t ≥ prompt_length − 1`) and the advantage on those positions.
9. **Forward.** `model(inputs)` under bfloat16 autocast on CUDA (float32 on the CPU) gives logits `[m, L, V]`. `log_softmax` in float32 and a gather give `log π` at each target, `[m, L]`. These come from the trainer's own forward pass on the policy being updated, never from the sampler.
10. **Backward.** Each micro-batch backpropagates `−Σ(mask · A · log π) / N`. Because every micro-batch divides by the same `N`, the accumulated gradient is the full-batch gradient.
11. **Step.** Clip the global norm to 1.0, set the learning rate (3e-6 × (k + 1)/5 for the first five optimizer steps `k = 0..4`, then 3e-6 constant), take one AdamW step (betas 0.9/0.95, weight decay 0.1 on matrices except the embedding), and discard the batch. An update whose groups are all skipped takes no step and is recorded in `skipped_updates`.
12. **Monitoring.** One `rl_update` event per update: mean reward beside the mean constant-answer baseline of the update's tasks, skipped share, mean attempt length, timeouts, distinct completions per group, audit counts, loss, gradient norm, learning rate, and seconds split into rollout generation, training step and reward wait.

## Design choices and common mistakes

- **Mean only, no standard deviation.** Dividing by a group's standard deviation would give a group whose rewards barely differ the same push as one whose rewards differ a lot. The advantage stays in reward units.
- **Skip all-equal groups, and leave them out of `N`.** An all-equal group has zero advantage everywhere, so it adds nothing to the sum. Counting its tokens in `N` would only shrink every other group's step by an amount that depends on how many groups happened to tie.
- **One denominator for the whole update.** Dividing each micro-batch by its own token count would make a token's weight depend on which micro-batch it landed in. Dividing each sequence by its own length would give a short attempt's tokens more weight than a long one's. Here every attempt token of the update has weight `1/N`.
- **Log probabilities from the trainer's forward.** The gradient needs an autograd graph through the current weights, which the sampler never builds, and the sampler's cached bfloat16 decoding can round differently from a full forward. Sampling and training differ only in floating-point rounding: top-p is fixed at 1.0, and the loss scores tokens under `softmax(logits / T)` with the sampling temperature, so the update is on-policy. A top-p below 1 is rejected, because the sampler's truncated distribution is not the softmax the loss scores.
- **Zero-advantage sequences are counted but not run.** Their gradient is exactly zero, so skipping their forward pass saves compute without changing anything.
- **Right padding is safe.** Every padded position comes after the real ones, so causal attention never lets it change a scored logit. The tests check that the pad ID does not change the gradient.
- **Strictly on-policy.** One optimizer step per batch, then the batch is discarded. There is no KL term and no reference model.
- **Mistakes the tests catch:** counting prompt tokens as attempt tokens, counting skipped groups in `N`, normalizing by the group's standard deviation, a different gradient when the micro-batch size changes, and a repeated seed.

## Verification

`tests/test_rl_policy.py` covers:

- the worked example's gradient `(1/96, 1/12, −3/32)` and loss `(ln 2 − 0.5 ln 3)/8` to 1e-15, and the same gradient from autograd on the loss written term by term;
- advantages and the skip rule, and the denominator over non-skipped groups;
- micro-batches of 1, 2, 3, 7 and 64 against each sequence run alone and unpadded through a real float64 model, and the pad ID's irrelevance;
- a non-finite loss raising before backward, and CPU bfloat16 staying close to float32.

`tests/test_rl_loop.py` covers:

- the schedule;
- the 25,600 rehearsal rollout seeds being distinct and disjoint from screening and dev seeds;
- the rollout protocol (prompts, temperature, top-p, token cap, stops, seeds, no sampler log probabilities);
- the warmup schedule, skipped updates, every stop and the monitoring fields;
- the stop-time check (skipped after a `should_stop` stop by default, never selected after a leakage stop), patience at the final update, the saved weights on an unexpected error, `on_check` and the refusal of non-train or unparseable tasks.

`tests/test_rl_control.py` runs three RL updates end to end on a tiny CPU model with a fake reward source.

```sh
uv run --locked pytest tests/test_rl_policy.py tests/test_rl_loop.py tests/test_rl_rewards.py
```

These are CPU tests with authored inputs and fake rewards. They do not measure GPU throughput or learning on real tasks.

## Questions and answers

1. *A group's rewards are 0.25, 0.25, 0.25, 1. What are the advantages?* −0.1875, −0.1875, −0.1875, +0.5625: the rewards minus their mean of 0.4375, not divided by anything.
2. *Why does group B in the example not change `N`?* It is skipped: all its rewards are equal, so it is left out of both the sum and its denominator.
3. *If the micro-batch size changes from 16 to 4, does the step change?* No. Each micro-batch divides by the same `N`, so the accumulated gradient is the same full-batch gradient up to floating-point rounding.
4. *Why not use the log probabilities the sampler can return?* They carry no autograd graph, and cached bfloat16 decoding can round differently from the full forward pass that the gradient is taken through.
5. *An attempt produced no python block. Does it affect the update?* Yes. It scores 0, which lowers its group's mean, and its tokens get a negative advantage when other attempts in the group scored above 0.
