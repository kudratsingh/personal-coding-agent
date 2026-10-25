# One-pass loader and masked final update

A rehearsal run reads every document of its streams exactly once, in the shared order from `pass_order`. `OnePassLoader` packs that order into fixed `[B, L]` rows so that every shard token is a next-token target exactly once. The trainer then gives the padded final update the same weight per target as every other update. The code is in `src/scratchcoder/data/onepass.py`, `cross_entropy_sum` in `src/scratchcoder/optim/adamw.py`, and the masked branch in `src/scratchcoder/train/trainer.py`.

## Worked example

Take two shards. With their BOS (256), `web` holds `[256 1 2 3]` and `[256 4 5]`, and `py` holds `[256 6]`, `[256 7 8 9 10]` and `[256 11 12 13]`: 18 shard tokens. Read them in the order py[1], web[0], py[0], web[1], py[2], with `L = 4`, `B = 2` rows per microbatch and `A = 2` microbatches per update.

The token stream is one sentinel BOS followed by the documents' own tokens:

```text
position  0         1   2 3 4 5    6   7 8 9    10  11   12  13 14   15  16 17 18
token     256       256 7 8 9 10   256 1 2 3    256 6    256 4  5    256 11 12 13
          sentinel  py[1]          web[0]       py[0]    web[1]      py[2]
```

Window `k` takes inputs `stream[4k : 4k+4]` and targets `stream[4k+1 : 4k+5]`:

| Window | Inputs | Targets | Source |
| --- | --- | --- | --- |
| 0 | 256 256 7 8 | 256 7 8 9 | py |
| 1 | 9 10 256 1 | 10 256 1 2 | py |
| 2 | 2 3 256 6 | 3 256 6 256 | web |
| 3 | 256 4 5 256 | 4 5 256 11 | web |
| 4 | 11 12 256 256 | 12 13 -100 -100 | py |
| padding | 256 256 256 256 | -100 -100 -100 -100 | (empty) |

Reading the targets top to bottom gives positions 1 to 18, each once. The sentinel at position 0 is only an input. Each document's own BOS is the target that follows the previous document's last token.

There are `ceil(18 / 4) = 5` windows and an update holds `B × A = 4` rows, so the pass plans `ceil(5 / 4) = 2` updates. Update 1 is windows 0 to 3. Update 2 is window 4 and one padded row, then a microbatch of two padded rows. After those four microbatches `next_batch` raises a `RuntimeError` naming the planned updates.

The final update has 2 real targets. Its microbatches are summed over those targets and divided by `B × L × A = 16`, the update's full capacity. Update 1 already gives each target the weight `1/16`, since its unmasked microbatches use `mean / A`. The update logs the mean loss over its 2 real targets and `tokens = 2`.

## Data flow and shapes

**Construction.** `OnePassLoader(shards, order, *, sequence_length, batch_size, accumulation_steps)` takes `(stream name, TokenShard, shard SHA-256)` triples in the same order as the `pass_order` input, so ordinal `i` names the same shard in both. It checks that:

- the shards are distinct and share one declared BOS and vocabulary;
- each file's SHA-256 matches the supplied hash (one extra read of every shard);
- `order` lists every document exactly once.

Per-document data is four NumPy arrays of `N` entries, about 26 bytes a document (52 MB for 2 million documents):

- `starts` and `lengths` (`int64`);
- the reading order as global ids (`int64`);
- shard ordinals (`uint16`).

Each shard file is opened once as a `<u2` memory map. The hot path reads document slices from those maps and never calls `TokenShard.__getitem__`, which reopens the file and builds a Python list.

**Totals.** `total_targets` is `T`, the sum of document lengths. `total_windows` is `W = ceil(T / L)`, and `planned_updates` is `U = ceil(W / (B × A))`. A pass emits `U × A` microbatches, `U × B × A − W` of its rows fully padded. At S shape (`L = 1,024`, `B × A = 128`), a 10^9-token stage gives `W = 976,563` and `U = 7,630`. Its final update holds 51 windows, 51,712 real targets of 131,072 slots. The 2 × 10^9-token mixed pass gives `W = 1,953,125` and `U = 15,259`, with 103,424 real targets in the final update.

**One microbatch.** `next_batch` fills an `int64` buffer of shape `[B, L + 1]`. Column 0 is the *carry*: the previous window's last target, or the sentinel for window 0. Columns `1..count` are the next targets, copied from memory-map slices. Inputs are `buffer[:, :L]` and targets `buffer[:, 1:]`; positions at or after `count` become BOS and `IGNORE = -100`. The result is returned as lists of lists to match `BatchSource`. Each row's source is the stream of its first target; padded rows have the source `""`.

**Masked loss.** `cross_entropy_sum(logits [B, L, V], targets [B, L])` returns the summed `-log p` over targets that are not `-100`, plus their count. Ignored positions gather class 0 and are then zeroed. That keeps the gather in range and gives them no gradient. With every target ignored, the sum is an exact zero that still backpropagates. Autograd keeps one fp32 `[B, L, V]` tensor, as `cross_entropy` does.

**Trainer.** The trainer checks each microbatch's target lists for `-100`. With no `-100` it runs the existing code unchanged. Otherwise it calls `_masked_microbatch`: the configured shape is required, z-loss and multiple processes are rejected, and it backpropagates `sum / (B × L × A)`. If any microbatch of an update was masked, the logged loss is the mean over the update's real targets. Unmasked means are rescaled to sums for this. The `tokens` metric is the real target count. An update with no real targets raises. A stop inside a masked update restores the loader exactly as before.

## Tickets, state and resume

`ticket()` reports what the loader has read:

- `documents` and `targets`, `windows` and `updates`;
- `multiset_sha256` and `order_chain_sha256`;
- `order_sha256`, which is `order_digest(order)`;
- `per_stream`, with each stream's `documents` and `tokens`;
- `complete`, true only once the loader is exhausted.

Each document's hash is the SHA-256 of its uint16 little-endian bytes with its BOS. That is the `tokens_sha256` field of the S build's `documents.jsonl`, so `multiset_digest` over those fields predicts the multiset. The multiset adds the hashes modulo 2^256, so the stage 1 and stage 2 tickets combine with `combine_digests` into the mixed run's value. The order chain starts at 64 zeros and folds each document as `sha256(f"{chain}|{shard_sha256}|{index}|{document_sha256}")`; `chain_digest` computes one link. Before exhaustion, the documents count and both digests cover only documents whose every token has been a target. `targets` and per-stream `tokens` also count a partly read document.

`state_dict()` holds only small values, because the trainer copies it before every update:

- the configuration signature, which covers the shards, the order digest and the shape;
- `rows`, `windows`, `cursor` (position in the order) and `offset` (within the current document);
- `sentinel`, true until the first window;
- both digests as hex strings.

`load_state_dict` checks these fields against each other before it changes anything:

- `rows` must be a multiple of `B`, and `windows` must equal `min(rows, W)`;
- `cursor` and `offset` must match the position recomputed from `windows`;
- `sentinel` must be true exactly when `windows` is zero;
- both digests must be hex strings, and the zero values before the first document.

It then reads the carry token again from the shard bytes. A rejected state leaves the loader unchanged. A saved digest is not rehashed against the shards; it is trusted as the loader's own record. A document that does not start with BOS, or holds a token outside the vocabulary, raises a `ValueError` from `next_batch`, and the loader stays at the start of that microbatch.

## Design choices and common mistakes

- **The sentinel.** Without it, the first document's BOS would only ever be an input. With it, the pass has exactly one target per shard token, so the target count equals the build's token quota and the digests cover every document whole.
- **The ragged last window is kept.** Dropping it would leave up to `L − 1` tokens untrained, so the pass would no longer be one complete reading of each document.
- **Padded rows instead of a short last microbatch.** A different shape could make `torch.compile` recompile and would change memory use on the last update. Padding keeps every microbatch at `B × L`.
- **Dividing by capacity, not by the real count.** The unmasked path's `mean / A` is `sum / (B × L × A)`. Using the same constant gives each target in the final update the weight it would have in any other update. Dividing by the real count would weigh the example's 2 final targets 8 times more than the others, and a 10^9-token S stage's final 51,712 targets about 2.5 times more. The final update still applies a normal AdamW step, with momentum and weight decay, even though its gradient is smaller.
- **No change to existing callers.** The masked branch runs only when a target is `-100`, which no other loader produces. A one-off local comparison, not a committed test, found bitwise-equal losses, metrics and weights from five existing configurations before and after the change. The test below checks the unmasked path against the earlier update, written out, with zero tolerance.
- **Unsupported combinations raise.** z-loss with `IGNORE` raises, because its mean of `log Z²` would include padded positions. More than one process also raises: the one-pass loader has no rank partition. In that case the other ranks would wait at their next collective until it times out.

## Verification

`tests/test_onepass_loader.py` covers:

- the hand-computed packing above, including the ragged window, the padded rows and the exhaustion error;
- every token a target once and the sentinel only an input, under three `pass_order` seeds;
- the `planned_updates` formula, including a document that is only its BOS;
- resume from every microbatch boundary, including cuts inside a document, after a JSON round trip;
- malformed states rejected with the live loader unchanged, and mismatched hashes or orders rejected;
- documents without BOS or with an out-of-vocabulary token rejected without advancing;
- tickets checked against hashes of the authored tokens: the stage 1 and stage 2 multisets combine to the mixed one, with matching per-stream counts;
- `cross_entropy_sum` against PyTorch's summed cross-entropy with `ignore_index`, the all-ignored zero and the saved-memory bound;
- one padded final update: gradients equal a summed cross-entropy over the real targets divided by 32, with loss and `tokens` checked;
- unpadded one-pass training equal, bit for bit, to the earlier update written out, with the masked loss shown not to run;
- a stop inside the masked update, then checkpoint and resume, equal to an uninterrupted run;
- rejection of z-loss and of steps beyond the planned updates.

```sh
uv run --locked pytest tests/test_onepass_loader.py tests/test_onepass_order.py
```

These are CPU tests on authored shards. They do not measure GPU throughput or check the S build's real shards.

## Questions and answers

1. *Why are there `ceil(T / L)` windows and not `ceil((T + 1) / L)`?* The stream has `T + 1` tokens but only `T` targets; the sentinel adds an input, not a target. Each window has `L` target slots.
2. *Why is the final update's sum divided by `B × L × A` and not by its real target count?* So each real target has the same gradient weight as in every other update. The logged loss still uses the real count, so it stays comparable with validation.
3. *What does the order chain add to the multiset?* The multiset ignores order, so the stage tickets can be combined. The chain changes if any two documents swap places, so it pins the exact reading order.
4. *Why can the saved state be so small?* The order and the shard bytes are fixed, so a window count is enough to recompute the position. The digests are the only history that cannot be recomputed cheaply.
5. *Why does a fully padded microbatch still run?* It keeps every update at `A` microbatches of the same shape. Its loss is an exact zero, so it adds nothing to the gradient.
