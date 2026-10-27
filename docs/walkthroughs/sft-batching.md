# SFT batching, assistant-token normalization and dev loss

Supervised fine-tuning trains on whole chat-v1 conversations, one per row, and only on the assistant's tokens. `SFTLoader` turns rendered rows into fixed `[B, L]` microbatches. The trainer's `update_targets` normalization divides each update by its number of assistant targets. `assistant_loss` measures the same quantity on a dev set. The code is in `src/scratchcoder/data/sft.py`, the `_target_normalized` branch of `src/scratchcoder/train/trainer.py`, and `src/scratchcoder/train/validation.py`. The [chat-v1 walkthrough](chat-format.md) explains how a row and its mask are built.

## Worked example

Take three rows, written as `(token, carries loss)`:

| Row | Tokens | Assistant targets |
| --- | --- | --- |
| `a` | `256 1 [2] [3]` | 2 |
| `b` | `256 4 [5]` | 1 |
| `c` | `256 [6] 7 [8] [9]` | 3 |

Brackets mark tokens that carry loss. With `L = 4` and `B = 2`, seed 0 reads them in the order `b, a, c`. Each row's inputs are `ids[:-1]` and its targets `ids[1:]`, padded to `L`. Pad inputs are BOS (256) and pad targets are `IGNORE` (−100):

| Microbatch | Row | Inputs | Targets | Source |
| --- | --- | --- | --- | --- |
| 1 | `b` | 256 4 256 256 | −100 5 −100 −100 | language |
| 1 | `a` | 256 1 2 256 | −100 2 3 −100 | code |
| 2 | `c` | 256 6 7 8 | 6 −100 8 9 | code |
| 2 | padding | 256 256 256 256 | −100 −100 −100 −100 | (empty) |

Row `c` has five tokens, so four input/target pairs: it fits `L = 4` exactly. A row of six tokens would be dropped and counted as `over_length`. A row with no target carrying loss would be counted as `no_assistant_targets`.

**Normalization.** Now put each row in its own microbatch: `B = 1`, three microbatches per update (`A = 3`), so the capacity is `1 × 4 × 3 = 12` target slots. The microbatches hold 1, 2 and 3 assistant targets, 6 in all.

| Normalization | Weight of each assistant target |
| --- | --- |
| `update_targets` (SFT): sum ÷ 6 | 1/6 each: the gradient of the mean over the update's 6 targets |
| `capacity` (pretraining default): sum ÷ 12 | 1/12 each: the same direction, half the size |
| Mean per microbatch ÷ 3 (not used) | 1/3 for `b`'s target, 1/6 for `a`'s, 1/9 for `c`'s |

The last row shows the mistake the update-wide divisor avoids: averaging each microbatch first makes a target's weight depend on how full its microbatch happens to be.

## Data flow and shapes

**Where the rows come from.** `tools/sft_data.py` writes the training rows; see [SFT data](../sft-data.md). Every code task with a verified reference whose row fits becomes a row, so code is never subsampled. Language is sized to the code by a cap on its share of the training rows' loss tokens. A row's loss tokens are the targets it contributes that are not `IGNORE`, which the trainer counts under `update_targets` (row `c` above has 3). `loss_tokens(row)` in `src/scratchcoder/data/sft.py` counts them, and the build's cap, its record and the loader's `total_targets` and tickets all use it. If all the language conversations keep that share at or below 30%, all are used. Otherwise they are taken in a fixed hash order until the next one would pass 30%, and the rest are dropped.

For example, take code rows of 2,000 tokens in all, 700 of them loss tokens. In hash order the conversations have 160, 240 and 130 tokens, of which 100, 150 and 80 are loss tokens:

| Conversation | Language loss tokens if taken | Loss-token share if taken | Taken |
| --- | ---: | ---: | --- |
| 1 | 100 | 100 / 800 = 12.5% | yes |
| 2 | 250 | 250 / 950 = 26.3% | yes |
| 3 | 330 | 330 / 1,030 = 32.0% | no: past 30%, so it and every later one are dropped |

The measured loss-token share is 26.3%. Measured in all tokens, the same two conversations are 400 / 2,400 = 16.7%, because the code rows' long prompts carry no loss. The build record reports both shares with their code and language counts, the cap and its basis, and how many conversations the cap dropped (here one).

**Rows.** An `SFTRow` holds an `id`, a `source` label, `ids` and `mask` (tuples of length `n`). On disk each row is one JSON line with `id`, `source`, `ids` and `spans`. `spans` is the list of half-open `[start, end)` ranges where the mask is true. `read_rows` rejects duplicate IDs and malformed spans.

**Order.** Pass `p` sorts the kept rows by the first eight bytes of `sha256(f"{seed}|{p}|{row id}")`, read as a big-endian integer, with the ID breaking ties. Anyone can recompute it without a library shuffle. The rehearsal uses seed 0 and one pass. With several passes each pass has its own order.

**Microbatches.** `next_batch()` returns `B` input rows and `B` target rows of length `L`, plus each row's source (`""` for padding). After the last real row, whole padding rows fill the final update, so every update has `A` microbatches of the same shape. `planned_updates` is `ceil(rows × passes / (B × A))`, and `next_batch` raises once they are all emitted.

**State.** `state_dict()` is `{signature, rows}`, where `rows` counts the rows emitted. The signature binds every input row's content hash, the drops, the order, `L`, `B`, `A`, the seed, the passes and the pad ID. `load_state_dict` rejects another signature, or a count that is not a multiple of `B` or lies beyond the plan, without changing the loader. `ticket()` reports rows, assistant targets, updates and per-source counts read so far.

**Trainer.** With `train.loss_normalization: update_targets` the trainer first reads all `A` microbatches of an update. It then counts their targets that are not `IGNORE`: that count is `N`. Each microbatch backpropagates `sum of −log p over its real targets ÷ N`. The accumulated gradient is therefore the gradient of the mean cross-entropy over the update's `N` assistant targets. The logged `loss` is that mean, and `tokens` is `N`. A stop before the last read restores the loader and RNG exactly, as in every other update. The default, `capacity`, is unchanged: unmasked microbatches take the original path and masked ones divide by `B × L × A`.

**Dev loss.** `assistant_loss(model, rows, batch_size=16)` right-pads each batch to its longest row and sums `−log p` over every assistant target in float64. It divides by their number. `token_weighted_loss` combines several results, for example the code and language dev sets, into one token-weighted mean.

## Design choices and common mistakes

- **One conversation per row, no packing.** A row never sees another conversation, so no attention mask across documents is needed. Right padding follows every real position, so causal attention keeps it from changing a scored logit.
- **Rows over `L` pairs are dropped, not truncated.** A truncated row would teach the model an answer cut off mid-way, or with no `<|assistant_end|>`.
- **Why read the whole update first.** The divisor must be known before the first backward pass, because the gradients of all microbatches are summed. Reading `A` small microbatches of token lists costs nothing next to the forward passes.
- **Why a new mode instead of changing the masked branch.** Pretraining divides a padded final update by capacity, so each pretraining target keeps one fixed weight. Changing that would change pretraining. A test checks that a default-configuration pretraining run equals the written-out update bit for bit, and that the new branch never runs there.
- **The cap bounds language, not code.** The code rows are the point of the rehearsal's SFT, so all of them are kept and the language share is fitted around them. Taking a hash-order prefix, rather than the conversations that pack closest to 30%, keeps the choice independent of conversation lengths, so a reader can recompute it from the IDs alone. `tests/test_sft_data.py` checks the prefix, the cap at its exact boundary, and that nothing is dropped when all the language fits.
- **Why loss tokens.** Under `update_targets` every loss token of an update has the same weight, so the loss-token share is language's share of what the gradient sees. Context tokens carry no loss, so a cap measured in all tokens would let code's long prompts stand in for loss tokens, and language would take a larger share of the gradient than the cap states.
- **Unsupported combinations raise.** z-loss is rejected with `update_targets` at configuration time. More than one process is rejected by the masked branch. An update with no assistant targets raises instead of dividing by zero.

## Verification

`tests/test_sft_loader.py` covers:

- the hand-built padding and targets above, the drops and exhaustion;
- the seeded order recomputed by hand for two seeds and two passes;
- resume from every microbatch boundary after a JSON round trip;
- rejected states leaving the loader unchanged;
- the signature, tickets and span serialization.

`tests/test_sft_training.py` covers:

- the update's gradient against `torch.nn.functional.cross_entropy` on unpadded rows, divided by the assistant-target count, across two accumulated microbatches;
- that it differs from a mean of microbatch means;
- the capacity gradient as the same gradient times `N / (B × L × A)`;
- a stop inside an update, then checkpoint and resume, equal to an uninterrupted run;
- the default pretraining update, bit for bit, against the written-out reference;
- `assistant_loss` against the direct per-row mean for batch sizes 1, 2 and 5, in float32 and bf16.

`tests/test_sft_data.py` covers the rows' source: the hash-order prefix under the 30% cap in both bases, the cap at its exact boundary in both bases, the record's shares recounted from the written rows (loss tokens as the loader's targets that are not `IGNORE`), language kept whole when it fits, and the default of no count cap. `tests/test_sft_loader.py` checks that `loss_tokens` equals the loader's non-`IGNORE` targets for each authored row.

```sh
uv run --locked pytest tests/test_sft_loader.py tests/test_sft_training.py tests/test_sft_data.py
```

These are CPU tests on authored rows. They do not measure GPU throughput.

## Questions and answers

1. *An update holds 40 assistant targets in one microbatch and 10 in the other. What weight does each target get?* 1/50 under `update_targets`, whichever microbatch it is in.
2. *Why is the logged loss comparable with the dev loss?* Both are the mean `−log p` over assistant targets only, weighted by token.
3. *Why does a fully padded final microbatch still run?* It keeps every update at `A` microbatches of one shape. Its loss is an exact zero, so it adds nothing to the gradient.
4. *Why hash the row ID rather than shuffle with a random generator?* The order then depends only on the IDs and the seed, which anyone can recompute, and not on a library version.
5. *What does `rows` in the saved state mean after a stop inside an update?* The trainer restores the state from before the update, so `rows` is always at an update boundary in a checkpoint.
