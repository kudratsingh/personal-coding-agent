# Component: whole-document validation windows and batched validation

The [periodic validation](training-loop.md#periodic-held-out-validation) in the training loop scores only the first window of each held-out document. The [rehearsal](../rehearsal-data.md) validation shards call for two larger measurements: a fixed subset of an exact number of target tokens per domain, cheap enough to measure often, and every token of the shards, measured a few times per run. `document_windows` cuts every document into windows, `selection_sha256` names the selection, and `validate_batched` scores many windows per forward pass.

## Worked example

Take a tokenizer with one merge, so ID 272 is the two raw bytes `hi`; IDs 256 to 271 are special tokens with no raw bytes. A shard holds four documents; each starts with BOS 256:

| Document | Tokens |
| --- | --- |
| 0 | `256 272 32 97 98 99` |
| 1 | `256` |
| 2 | `256 257 258` |
| 3 | `256 100 257 101 102` |

With `sequence_length=2` a window holds at most three tokens and windows start at positions 0, 2, 4 of each document. Its first two tokens are the input and its last two are the targets:

| Window | Targets | Raw target bytes |
| --- | --- | ---: |
| `256 272 32` | `272 32` | 2 + 1 = 3 |
| `32 97 98` | `97 98` | 2 |
| `98 99` | `99` | 1 |
| `256 100 257` | `100 257` | 1 + 0 = 1 |
| `257 101 102` | `101 102` | 2 |

Document 1 has no token after BOS, so it has no target. Document 2's only window predicts two special tokens. It has no raw bytes and bits per byte cannot divide by zero, so that window is skipped and its two targets are not counted. Token 257 in document 3 is still scored: its window has text bytes, and the special token adds zero bytes, as in `validate`. Each token after a BOS is a target exactly once: five from document 0 and four from document 3.

With `target_tokens=6`, the first three windows give five targets. The skipped window adds none, and the next window is cut to one target: `256 100`, one byte. The selection then has exactly six targets. With `target_tokens=10` the shard is too short (it has nine countable targets) and the call raises instead of returning fewer.

If every logit is zero, every target has probability `1/273`, so the loss is `ln 273` nats per token whatever the padding. The tests score the windows `256 97 98 99` (three targets, three bytes) and `256 272` (one target, two bytes) in one padded batch: bits per byte is `4 × log2(273) / 5`.

## Data flow and shapes

1. A document is read from the shard as `n` token IDs, BOS first. Window starts are `0, L, 2L, ...` below `n - 1`, and window `s` is `tokens[s : s + L + 1]`. A one-token remainder has no target and is never produced.
2. The window's raw byte count is the sum of the tokenizer's byte lengths of its targets (`tokens[1:]`). Special tokens count zero. A whole-window count of zero skips the window.
3. `validate_batched` takes up to `B` windows. The longest has `W` tokens; shorter rows are right-padded with token 0 into `ids [B, W]`. Inputs are `ids[:, :-1]` and targets `ids[:, 1:]`, both `[B, W-1]`. The mask `[B, W-1]` is true at positions below each row's target count.
4. The model gives logits `[B, W-1, V]`, in bfloat16 when `autocast_bf16=True`. A float32 log-softmax over `V` and a gather of each target give log probabilities `[B, W-1]`. Padded positions are set to zero, and the batch is summed in float64 to one negative log likelihood.
5. Across batches the totals are NLL, target tokens and raw bytes. The loss is `NLL / tokens`, and bits per byte is `NLL / ln 2 / bytes`. The returned dictionary has the same keys as `validate`. `on_batch` runs after each batch.

For the S model, a full batch of 16 rows of 1,024 positions gives `16 × 1,024 × 65,536` logits. That is 2 GiB in bfloat16, plus 4 GiB each for the float32 copy and the log-softmax: about 10 GiB transient, by arithmetic. A training microbatch of the same shape holds those same tensors plus activations for backward, so validation at batch 16 needs less memory than a training step that already fits.

## Design choices and common mistakes

**Stride `L`, windows of `L + 1`.** Consecutive windows of a document share one token: the last target of a window is the first input of the next. Every token after BOS is a target once and no target is scored twice. Each window restarts its attention context, so a target late in a long document sees at most `L` earlier tokens, not the whole document. This is not sliding-window evaluation, and a document longer than `L` scores slightly worse than it would with full context. The rule is fixed and the same for both runs, so it does not bias a comparison between runs.

**No window crosses a document.** Training packs documents across sequence boundaries, but a validation target never conditions on another document's text.

**An exact target count, not a window count.** Two runs that use the same shards and the same `target_tokens` score the same targets. The subset is the first `target_tokens` targets in shard order, not a random sample, so the shard order fixed at build time decides which documents it covers.

**Zero-byte windows are skipped.** `ValidationWindow` requires a positive byte count, the bits-per-byte denominator. Pretraining documents are encoded without special tokens, so this case arises only if a document contains specials after its BOS.

**Name the selection.** `selection_sha256` streams the same SHA-256 that the pilot records for its held-out windows, `sha256(json.dumps([asdict(w) for w in windows], sort_keys=True))`. Changing a token, a byte count or the window order changes it. Record it with every validation result so that two numbers can be shown to measure the same targets.

**Memory: materialize the subset, stream the full set.** `document_windows` returns a tuple, which `ValidationWindow` requires to hold Python integers. That costs about 40 bytes per token in CPython: on a synthetic shard the size of S validation-web (7,621 documents, 6,940,770 targets, random IDs from a 65,536-entry vocabulary), the tuple held 266 MiB and took 1.6 s to build on this Mac. The S validation streams have 6,948,391 (web) and 5,334,710 (Python) shard tokens, so both together would hold about 470 MiB. That fits, but it is not needed: `iter_document_windows` yields the same windows lazily with an 8 MiB peak, and `validate_batched` accepts any iterable. For the full sets, create a fresh stream for each pass and for its hash; it re-reads the shard each time. A one-million-target subset is about 40 MiB and can stay in memory. A stream checks its limits when it is created, but finds a shard too short for `target_tokens` only at its end, which is another reason to materialize the subsets.

**Right padding is safe.** Causal attention lets a position see only earlier positions, and padding comes after every real token of its row. Rows start at position 0, so rotary positions are unchanged. Padded targets are replaced by zero with `masked_fill`, not multiplied by the mask, because `NaN × 0` is still `NaN`.

**Agreement, not bitwise identity.** `validate` multiplies a float32 mean by the window's token count. `validate_batched` sums float32 per-token values in float64. On the test model the two losses differ by 2e-8 relative. Batched and unbatched forward passes use matrix multiplications of different shapes, so they agree to float32 rounding (4e-9 relative on the test model), not bit for bit. Bfloat16 autocast moved the test model's loss by under 1e-5 relative, but that untrained model has nearly uniform logits, so the figure says nothing about a trained S model. Compare runs only at the same precision.

**Padding costs compute.** Batches are formed in window order, and the last window of each document is usually short. Sorting windows by length would cut padding, but it would require holding every window in memory and would change the summation order. It is not done.

As in `validate`: no gradients, no random numbers, and the model's previous train/eval mode is restored even when a window is rejected. A window longer than the model context, or with an ID outside the vocabulary, raises before its batch runs. `on_batch` is called after each completed batch, the last one included, and is how a long full pass keeps a heartbeat fresh.

## Verification

Implementation: [`src/scratchcoder/train/validation.py`](../../src/scratchcoder/train/validation.py). Reference tests: [`tests/test_validation_windows.py`](../../tests/test_validation_windows.py).

```sh
uv run --locked pytest -q tests/test_validation_windows.py tests/test_validation.py
```

The tests check the hand-computed windows and byte counts above at three window lengths. They check the exact `target_tokens` cut at every count from 1 to 9 and the too-short error. They rebuild 30 random documents from their windows at six lengths, to show that every target appears once and in order. They check lazy reading (one document read for the first window), eager limit checks, the uniform-logit loss with padding, and `validate_batched` against `validate` at batch size 1 (relative 1e-6). They compare batch sizes 2, 3, 16 and larger than the input with batch size 1 (relative 1e-5), with reference and fused attention. They also cover lazy iterables, `on_batch` order, bfloat16 closeness, restored model mode and weights, input rejection, and the selection hash against the JSON identity. All checks run on CPU with a tiny float32 model. They are not GPU or real-shard measurements. The memory and time figures above come from a one-off local script, not from CI.

## Questions and answers

1. *Why does the second window of a document start at position `L` and not `L + 1`?* Its first token is only context. Starting at `L` makes the target right after the first window's last target the first target of the second window, so no token is skipped.
2. *Why skip a window with zero raw bytes instead of keeping it?* Bits per byte divides by raw bytes, and `ValidationWindow` rejects zero. Skipping it also leaves its targets out of the token count, so the loss and bits per byte cover the same targets.
3. *Why can padding at the end of a row not change the score of a real token?* Causal attention lets a position see only itself and earlier positions, and every padded position comes after every real one.
4. *Why stream the full validation set instead of building a tuple?* The tuple costs about 40 bytes per token in Python integers, about 470 MiB for both S validation streams. The stream holds one document and one batch at a time.
5. *What should be recorded with each validation result?* The `selection_sha256` of the windows, their target token and byte counts, and the precision setting. Two numbers are comparable only when all of these match.
