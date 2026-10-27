# Generation and the key-value cache

Generation continues a prompt one token at a time. Every transformer layer computes keys and values `[B,H,T,D]` for the prompt; each later token needs new queries, keys and values only for its own position, so the earlier keys and values are cached. The code is in `src/scratchcoder/model/gpt.py` (`KVCache`, `GPT.decode`) and `src/scratchcoder/engine/generation.py` (the sampler, `generate_batch`, `generate_many` and the older `generate`). `src/scratchcoder/engine/benchmark.py` measures its speed.

`generate_batch` decodes prompts of different lengths together. Each row keeps its own positions, budget, stop tokens and seed, so a row's output does not depend on which other rows share its batch, apart from float rounding (see Uniforms below). `generate_many` cuts a long list into length-sorted batches. The older `generate(model, prompts, ...)` for equal-length prompts keeps its contract and runs through the same loop.

## Worked example

Take two prompts, A = `[5 6 7]` and B = `[9]`, with `max_new_tokens = 2` on a model of context 8.

**Left padding.** The batch is as wide as the longest prompt, `W = 3`. Shorter rows get padding on the left, so `pad = [0, 2]` and the ids are `[[5 6 7], [0 0 9]]`. Every row's last prompt token is now in slot `W − 1 = 2`, and every new token goes into the same slot for all rows. The rows move in lockstep, and one cache length serves the whole batch.

**Positions.** Row `b` keeps its token for slot `s` at position `s − pad[b]`:

```text
slot           0    1    2  |  3
row A token    5    6    7  |  a0
row A position 0    1    2  |  3
row B token    0    0    9  |  b0
row B position 0*   0*   0  |  1        * padding: clamped, never seen by real tokens
```

Rotary embeddings use these positions, not the slot numbers. B's prompt token is rotated as position 0, exactly as it would be alone.

**Budgets.** Each row may add `min(max_new_tokens, context − its own prompt length)` tokens: `min(2, 8 − 3) = 2` for A and `min(2, 8 − 1) = 2` for B. A row's budget depends only on its own prompt, so batching never changes it. The last chosen token is never fed back, so the cache needs `W + max(budget) − 1 = 4` slots.

**Masks.** A real query sees real keys at or before its own slot. In the prefill, B's query at slot 2 sees only slot 2. A padding query sees its own causal past (slots up to itself); nothing real ever reads its output, but its softmax row is never empty. When decoding, the query in slot 3 sees slots 0–3 for row A and slots 2–3 for row B.

**Logits.** Because of left padding, the prefill only projects slot 2 to the vocabulary. The model returns `[B, V]` logits, not `[B, W, V]`. At batch 256 with 600-token prompts and 65,536 entries, full prefill logits would need about 38 GiB in float32.

**Sampling.** Say row A's filtered probabilities are `[0.2, 0.3, 0.5]` at step 0. Their running sum is `[0.2, 0.5, 1.0]`. A's uniform for step 0 comes from its seed: say `u = 0.42`. The first running sum above `0.42 × 1.0` is at index 1, so A draws token 1. With `u = 0.5` it would draw token 2, because the running sum at index 1 is not above 0.5.

**Stops.** If row A's stop set contains `a0`, A finishes after one token with `finish_reason = "stop"` and tokens `[a0]`. B continues to its budget (`"length"`). A's later slots are still computed with filler tokens, but they are discarded.

## Data flow and shapes

**Request.** `generate_batch(model, prompts, max_new_tokens=..., seeds=..., stops=..., ...)` takes token lists. It checks every ID against the vocabulary, every prompt against the context and the counts of seeds (one per prompt when sampling) and stop sets (one per prompt). It builds the left-padded `ids [B, W]` and `pad [B]` on the model's device, and a stop table `[B, M]` padded with `-1`.

**Cache.** `KVCache(config, pad, width=W, capacity=C)` builds, once per call:

- cosine and sine `[B, C, D/2]`: rows of a float32 `[context, D/2]` rotary table, gathered at each row's positions (clamped to `context − 1`; only already-finished rows reach the clamp), and cast once to bfloat16 when autocast needs it;
- `padding [B, C]`: true for slot `s < pad[b]`;
- `future [C, C]`: true for key slot after query slot;
- per layer, keys and values `[B, H, C, D]`, allocated on first write and never regrown. Both use the dtype of `v`, the activation dtype. Under bfloat16 autocast, query/key normalization's float32 gain makes `k` float32; attention would round it to bfloat16 at every step, so `store` rounds it once on write instead. The result is the same bit for bit, and the key cache is half the size.

**One forward.** `GPT.decode(ids [B, T], cache)` runs the next `T` slots. In each layer, `q`, `k` and `v` are `[B, H, T, D]`; the rotation is the slice `[B, 1, T, D/2]`; `store` writes `k`, `v` into slots `length .. length+T−1` and returns views `[B, H, length+T, D]`; the mask is `[B, 1, T, length+T]`; attention scores are `[B, H, T, length+T]`. For a decoding step (`T = 1` after the prompt) the mask is just a slice of `padding`. The final norm and vocabulary projection use only the last slot, in float32 even under bfloat16 autocast, and return `[B, V]`. `cache.length` then advances by `T`.

**Prefill.** The prompt slots run in chunks of `c` slots, with `c` chosen so that `B × H × c × C` stays at most 2^27 score elements (`prefill_chunk` overrides it). The automatic `c` depends on the batch size and on the capacity, which the other rows' budgets set: at `B = 256` and `H = 12` it is 98 slots for capacity 445, while `B = 1` takes a whole prompt in one chunk. Chunks do not change the result beyond float rounding; tests compare chunk sizes 1, 3 and the whole prompt within a float32 tolerance.

**Loop.** The uniforms for all steps are computed before the loop as float64 `[B, N]`. Each step updates device tensors only: a per-row non-finite flag, the token `[B]` (filler 0 for inactive rows), its log probability, the counts, the stop flags and `active = not stopped and count < budget`. The host reads `active.any()` once every `stop_check_interval` steps (default 8) to exit early, then once at the end reads the non-finite flags, the longest count and the results. A non-finite logit in an active row raises `ValueError` after the loop.

**Result.** Each `Completion` holds the new `tokens` (a stop token, when produced, is the last), `finish_reason` (`"stop"` or `"length"`), its `budget` and, with `log_probs=True`, one float per token. `generate_many` sorts prompts longest first (ties by input index), calls `generate_batch` per chunk of `batch_size` and returns results in input order.

## The sampler

**Filtering.** Temperature 0 is greedy: the first maximal logit, with log probability 0 (a point mass). For a positive temperature `T`:

- logits are scaled in float32, or in float64 for float64 logits or `T < 2^-100`;
- for `T ≥ 1` they are divided first (`|logit/T| ≤ |logit|`, so finite extremes cannot overflow); for `T < 1` they are centred on the row maximum first, so the maximum stays an exact zero and overflow can only reach `−inf`, the limit of a negligible probability;
- top-k keeps the `k` largest scaled logits (`topk`, then scatter);
- top-p keeps each token whose higher-ranked probability mass is still below `p`. This needs a stable descending sort, and the mask is scattered back to vocabulary order. With `top_p = 1.0` nothing is sorted.

`sampling_log_probs` returns this distribution. `tests/test_sampling_integrity.py` checks it against the previous sampler's filtering for float16, bfloat16, float32 and float64 logits at several temperatures and filters, and against hand-computed probabilities at extreme temperatures (down to the smallest subnormal).

**Draw.** `inverse_cdf` takes the softmax in vocabulary order, accumulates it in float64, and returns the first token whose running sum exceeds `u × total`. The target is held strictly below the total, so rounding cannot step past the last token. When top-k or top-p removed tokens, zero-probability entries are pinned to the previous running maximum. A GPU's parallel scan can round a removed token's running sum slightly above its predecessor's; with the pin, such a token can never be selected.

**Uniforms.** Row `b` at step `t` uses Philox4x32-10 (Salmon et al. 2011) with the row's 64-bit seed as key and `(t, 0, 0, 0)` as counter. Two output words give 53 random bits, so `u = bits / 2^53` lies in `[0, 1)`. The 32-bit multiplications are done in 16-bit halves on int64 tensors, so nothing overflows on CPU or CUDA. There is no generator state: a row draws the same numbers whatever batch it is in. `torch.multinomial` takes one generator for the whole call, so a row's draw would depend on its neighbours and on the batch size. A row's logits can still differ by float rounding between batch shapes: the batch size and the prefill chunk size change the order of floating-point operations, and on a GPU matrix-multiply kernels also depend on the shape. A token could change only if its uniform fell within that rounding of a running-sum boundary. So a row's tokens are the same alone or in any batch up to float rounding, not exactly. The CPU tests see identical tokens; this has not been verified on a GPU or in bfloat16 there.

**Seeds.** Evaluation attempts use the harness formula, `attempt_seed` in `src/scratchcoder/eval/harness.py`: the configured seed, plus the integer value of the first 16 hex digits of the task ID's SHA-256, plus the attempt index, modulo `2^63 − 1`. A batched caller passes one such seed per row. `generate(..., seed=s)` gives row `r` the seed `s + r`, so a one-prompt call with an attempt's seed is that attempt.

**Version.** The sampler is named `philox4x32-10-inverse-cdf-v1` (`SAMPLER_VERSION`). Reports of sampled runs (temperature above 0) record it as `sampler_version`, and `generation-benchmark` prints it. Seeded draws from the earlier `torch.multinomial` sampler are not reproduced. No result of record used sampled decoding: the reference, EvalPlus and ARC-Easy results are greedy or likelihood-scored. One edge case changed by design: at `top_p = 1.0` the earlier sampler could drop a far tail whose preceding mass rounded to 1.0, and this one keeps every token.

**Log probabilities.** With `log_probs=True`, each token's value is `weights[token] − logsumexp(weights)` over the scaled, filtered log weights, which is the distribution actually sampled. It is 0 for greedy decoding. These are not the model's temperature-1 log probabilities unless the sampler ran at temperature 1 with no filters. Nor will they equal what the trainer recomputes: bfloat16 training runs the whole forward pass under autocast, so its vocabulary projection is a bfloat16 multiply, while generation projects in float32 (and the cached path orders its arithmetic differently even in float32). They suit strictly on-policy use; do not use them as the old log probabilities of a ratio or a KL term, or compare them exactly with training values.

## Design choices and common mistakes

- **Left, not right, padding.** With right padding, rows would write their new tokens into different slots and need per-row cache lengths.
- **Positions per row.** Rotary attention depends only on the distance between positions, so shifting all of a row's slots by `pad` would leave its scores unchanged apart from rounding. Per-row positions matter because the cache can hold more slots than the context: `W + max(budget) − 1` exceeds it when a long prompt shares a batch with a short prompt and a long budget. Slot numbers would then reach the table's last position, `context − 1`, and the short row's newest tokens would all get that position, while its own positions `slot − pad` stay inside the context. A test runs such a batch and compares each short row's logits with the row run alone; the padded-logits test cannot detect slot-number positions.
- **Padding queries.** If a padding query hid every key, its softmax would be `NaN`. That `NaN` reaches the padding slot's values, and `0 × NaN` then poisons real rows. Letting padding queries see their causal past keeps everything finite; the padding tokens themselves cannot change real logits (tested by swapping the filler token).
- **No cache growth.** Appending with `torch.cat` copies the whole cache at every step. The preallocated buffer writes one slot per step and slices views. The older `forward_cached` path with growing tuples remains for existing callers and reference tests.
- **Fixed tables.** The rotary angles and masks are built once per call. The previous loop rebuilt the rotary table and causal mask in every layer at every step.
- **Host reads.** Reading a device flag with `bool()` blocks the host until the GPU has finished. The loop does it only at the stop-check interval and once at the end; a test counts these reads.
- **Precision.** `precision="bfloat16"` runs the transformer body under bfloat16 autocast. The vocabulary projection, the sampler and the log probabilities stay in float32/float64, so ties and probabilities are not rounded to bfloat16. The autocast region spans the whole loop (sampling runs in a nested disabled region), so weight casts are cached across steps. Callers record the precision they used.
- **Static batching.** Finished rows keep computing filler tokens until the batch ends; the batch is not compacted. Sorting by length (`generate_many`) keeps rows of similar length and budget together.
- **Device.** `load_checkpoint(..., device="cuda")` resolves the device the same way training does, without fallback. CPU tests establish equivalence only; GPU speed, memory and bfloat16 numerics are measured by the probe.

## Verification

Run:

```sh
uv run --locked pytest tests/test_generation.py tests/test_sampling_integrity.py tests/test_generation_benchmark.py
```

- `tests/test_generation.py` keeps a frozen copy of the previous single-prompt greedy loop. In float32, batched greedy output over prompts of lengths 1–31 equals it token for token, as does the older `generate` with and without a stop token. It also checks that:
  - a seeded row equals its single-row run and its `generate_many` runs at batch sizes 1, 2 and 7;
  - padded rows match their unpadded logits, for several prefill chunk sizes;
  - when the slots pass the context, short rows still match their unpadded logits, which fails if positions are slot numbers;
  - cache prefill plus decoding equals the full forward pass, with and without query/key normalization;
  - budgets are per row and unchanged by sorted batching;
  - stop sets end each row, and early exit happens at the first check after every row stopped;
  - log probabilities match the filtered distribution recomputed from a full forward pass;
  - bfloat16 autocast runs and keeps float32 logits;
  - with query/key normalization under bfloat16 autocast, both caches are bfloat16 and the logits equal those of a float32 key cache bit for bit;
  - no sort runs at `top_p = 1.0` and no `torch.cat` runs in the loop;
  - host reads are a fixed handful, whatever the number of steps;
  - non-finite logits raise and invalid requests are rejected.
- `tests/test_sampling_integrity.py` checks Philox against three published known-answer vectors and a scalar reference, uniforms against the bit formula, `inverse_cdf` against hand-computed cases (including a simulated scan rounding), seeded draw frequencies against the filtered distribution, and the filtering rules.
- `tests/test_generation_benchmark.py` runs the probe command on CPU with tiny models.

As a local check, the previous code and this code decoded a synthetic-only nano checkpoint greedily with a 256-token budget. The checkpoint was prepared with `tools/prepare_eval_nano.py`, the recipe of the [EvalPlus](../evalplus.md) integration runs, but it is not the same file. All 425 HumanEval+ and MBPP+ prompts that fit its 256-token context produced identical tokens, both one prompt at a time and as a single batch.

`scratchcoder generation-benchmark` prints throughput as JSON; see [evaluation](../evaluation.md). For example:

```sh
uv run --locked scratchcoder generation-benchmark --config configs/smoke.yaml \
  --batch-sizes 1 4 --prompt-lengths 8 20 --max-new-tokens 16 --top-p 0.95
```

## Scoring and the base protocol

Answer scoring sums only continuation-token log probabilities. A padded batch is safe because causal attention prevents later padding from changing earlier scores. A slow per-prefix reference verifies this indexing.

For base-model code evaluation, prompt whitespace is part of the protocol: `rstrip()` removes trailing whitespace before encoding `[1,T]` prompt IDs. The completion retains its initial newline/indentation, and the scored source is stripped prefix plus completion up to the first original stop string. The reference model and project checkpoint generator share this rule; no extra public-test instruction is appended in base mode. Authored tests observe the actual tokenizer/generation prefix and persisted report/scorecard metadata. Legacy instruction-style generation remains explicitly selectable and separately recorded. See the [evaluation policy](../evaluation.md) for comparability limits.

## Questions and answers

- Why cache keys and values but not queries? Later tokens attend to earlier keys and values; no later step uses an earlier query.
- Why left padding with per-row positions? Every row's newest token is in the same slot, so one cache length serves all rows. Rotary angles follow each row's own positions, which stay inside the context while the row is active even when the slot numbers pass it.
- Why may a padding query see padding keys? So its softmax row is never empty; a `NaN` there would reach real rows through `0 × NaN`.
- Why a counter-based sampler? A row's uniform depends only on its seed and step, so a seeded row's random numbers do not change with batch size or neighbours (its logits can still differ by rounding), which `torch.multinomial` with one generator cannot give.
- Why pin zero-probability entries in the running sum? A parallel scan can round a removed token's running sum upward; pinning keeps that token unselectable.
