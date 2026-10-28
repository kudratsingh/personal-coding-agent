# Component: special-token rows (a fresh draw and their own learning rate)

The model ties its input embedding to its output layer: the logit of token `i` is `h · e_i`, the hidden state times row `i` of the embedding. Pretraining never shows the chat special tokens (`<|user_start|>` through `<|python_end|>`, the output markers and the seven reserved tokens, IDs 257 to 271). As outputs they only ever get the softmax's push-down, which is the same for every token that is not the target, so their rows drift together toward one shared direction. In run B's `mixed-final` checkpoint, the parent of SFT, those fifteen rows have pairwise cosines of about 0.997 and a cosine of about 0.994 with the mean of the ordinary rows.

SFT must then learn that `<|python_start|>` follows `<|assistant_start|>`, not `<|assistant_end|>`. The logit gap between the two is `h · (e_python_start − e_assistant_end)`, and that difference vector is nearly zero, so a modest learning rate barely moves it. Run 1's SFT weights ended their turn before opening a python block on every prompt ([format gate](sft-format-gate.md)).

Run 1b does two things before and during SFT. The code is `src/scratchcoder/train/special_rows.py`, the optimizer change is in `src/scratchcoder/optim/adamw.py`, and the settings are `TrainConfig.special_rows_init`, `special_rows_seed` and `special_rows_lr_scale`.

## Worked example

Take a tiny embedding of 4 ordinary rows in 2 dimensions, `[1, 0]`, `[3, 0]`, `[1, 2]`, `[3, 2]`, and two special rows collapsed onto `[2, 1]`.

1. **Statistics of the ordinary rows.** The mean is `m = [2, 1]`. The spread per dimension is the population standard deviation around that mean: `s = [1, 1]`.
2. **The draw.** A seeded CPU generator draws `z ~ N(0, 1)`, one row of `D` values per special row, in ascending ID order. Each new row is `m + s ⊙ z`. With `z = [0.5, −1.2]` and `[−0.3, 0.8]` the rows become `[2.5, −0.2]` and `[1.7, 1.8]`. Their cosine is about 0.63; before, it was 1.
3. **The learning rate.** The optimizer gets a `[V, 1]` multiplier: 10 on rows 257 to 271, 1 on every other row. AdamW's step for row `r` is `−lr × scale_r × m̂ / (√v̂ + ε)`, so a special row moves ten times as far as it would at the base rate, and every other row exactly as before.

For run B's real embedding (65,536 rows of width 768), `|m|² ≈ 0.74` and `Σ s_d² ≈ 0.74`, so two fresh rows have an expected cosine of about `|m|² / (|m|² + Σ s_d²) ≈ 0.50`; the seed-0 draw of run 1b's fifteen rows from run B's weights gives a mean pairwise cosine of 0.50 and a largest of 0.55, and 0.53 among the three output specials. The format gate's bar for those three is 0.9, checked after training.

## Data flow and shapes

- `reinitialize_special_rows(weight [V, D], seed, rows=SPECIAL_ROWS)` reads the ordinary rows (every row that is not a special token) in float32 on the CPU, forms `m [D]` and `s [D]`, draws `[15, D]` noise from `torch.Generator().manual_seed(seed)` and writes the new rows in place, in the weight's dtype and device. It returns a record: the method, the seed, the rows, the ordinary row count, `|m|`, the mean spread, the SHA-256 of the fifteen rows before and after, and their largest and mean pairwise cosine after.
- `row_scale(V, scale=10.0)` is the `[V, 1]` float32 multiplier.
- `Trainer` redraws the rows after loading the parent checkpoint when `special_rows_init` is `ordinary-mean-noise`, and passes the multiplier for `embedding.weight` when `special_rows_lr_scale` is not 1. The SFT stage records the redraw in `report.json`'s `sft.special_rows`.
- `AdamW(..., lr_scales=[...])` takes one entry per parameter: a finite, nonnegative float tensor that broadcasts to the parameter, or `None`. The step multiplies the first moment by the scale before the division (`_scaled`). Both the per-tensor loop and the list kernels (`foreach`) do this.

## Design choices and common mistakes

- **`<|bos|>` keeps its row.** Pretraining shards begin every document with it, so its row is trained as an input and as a target; in run B it is clearly apart from the others (cosine about −0.06). Chat-v1 never makes it an SFT target: it is the first token of every row and the padding. Redrawing it would throw away a learned row for nothing.
- **Why the ordinary rows' own spread.** A row at the mean plus small noise would start nearly as collinear as before, because the specials already sit at the mean. Noise with the ordinary rows' per-dimension spread makes a new row look like a typical ordinary row in scale, and two new rows no more alike than two ordinary rows sharing that mean.
- **Why the CPU generator.** The draw does not depend on the device, the CUDA generator's state or the trainer's own seed, so the same seed gives the same rows on the Mac and on the Pod. The record's SHA-256s make that checkable.
- **Why 10x, and only there.** Fifteen fresh rows must learn their meaning in 212 updates at a peak of 6e-5, while the other 65,521 rows should move as in any SFT run. A per-row scale does both. Adam normalizes each element's step, so the scale is the ratio of step sizes, exactly.
- **No weight decay on scaled parameters.** The embedding is already exempt from decay. The optimizer refuses a scale on a decayed parameter, so decoupled decay never needs rescaling and cannot silently go wrong.
- **A scale of 1 is the old update, bit for bit.** Multiplying by 1.0 is exact in floating point, and a scale of `None` skips the multiply. The trainer passes `None` unless the scale differs from 1, so run 1, the pretraining runs and every older configuration train exactly as before. Older checkpoints resume with the defaults (`ADDED_TRAIN_DEFAULTS`).
- **The scale is not optimizer state.** Like `foreach`, it is rebuilt from the configuration; a resumed trainer builds it again, and the configuration is compared on resume.

## Verification

`tests/test_special_rows.py` covers:

- the rows (257 to 271, never `<|bos|>`) and the 10x constant;
- the draw recomputed by hand in float64 from the formula, every other row untouched;
- determinism for one seed and different rows for another;
- collapsed rows redrawn to a largest cosine well below 0.9;
- refusals: a non-matrix, a negative seed, unsorted rows, rows outside the vocabulary;
- a scale of ones reproducing the plain update exactly, in the loop and the list kernels, in float32 and float64;
- the list kernels matching the loop with a scale;
- three scaled float64 updates recomputed from the AdamW formula, a scale of 0 never moving its row;
- ten times the scale giving ten times the step;
- refusals of a scale on a decayed parameter, a mismatched shape, a NaN or a negative scale; the optimizer owning its copy;
- the trainer's defaults changing nothing, and the trainer redrawing and scaling the rows when configured.

```sh
uv run --locked pytest tests/test_special_rows.py
```

## Questions and answers

1. *Why do never-used rows end up collinear?* As outputs they only receive the gradient of the softmax's normalizer, `p_i × h`, which points the same way for every such row. They all move along the same average hidden direction.
2. *Why not just raise the whole learning rate?* That would move the 65,521 trained rows ten times as far too, which SFT at this scale does not need and which risks forgetting.
3. *Does the draw leak information from the evaluation?* No. It reads only the embedding's own ordinary rows and a seed.
4. *What checks that it worked?* The format gate's separation check (largest cosine among the three output specials below 0.9) and its behavioural checks, on the selected SFT weights, before anything else reads them.
