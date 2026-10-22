# Decoder-only transformer

Token IDs `[B,T]` become embeddings `[B,T,C]`. A block normalizes each position by its root-mean-square feature magnitude, projects queries/keys/values, splits heads to `[B,H,T,D]`, and applies adjacent-pair rotary positions to queries and keys.

Attention scores `[B,H,T,T]` divide query-key dot products by `sqrt(D)`. A causal mask hides future positions before softmax. Probability-weighted values return to `[B,T,C]`, pass through an output projection, and join a residual connection. A second normalization feeds a width-4C exact-GELU MLP and another residual.

After all blocks, a final normalization and the shared embedding matrix produce logits `[B,T,V]`. All matrix projections are bias-free. Total parameters are `V*C + L*(12*C*C + 2*C) + C`, including normalization gains.

For example, width 32 and four heads gives head width 8. A length-eight sequence uses attention scores `[B,4,8,8]`; changing tokens at position five or later cannot change logits at positions zero through four.

`tests/test_model.py` checks attention outputs and gradients against PyTorch scaled-dot-product attention, normalization against a reference, rotary norm preservation, exact parameter accounting, and causality. These are correctness checks, not trained capability measurements.

## Initialization and optional query/key normalization

All matrices start from a normal distribution with standard deviation 0.02; the two projections that write into the residual stream (the attention output and the MLP down projection) start at `0.02 / sqrt(2L)`, the GPT-2 depth scaling, so the residual stream's variance does not grow with the number of blocks at step zero. With `model.qk_norm: true`, queries and keys are RMS-normalized per head (gains shared across heads) before the rotary rotation, which bounds attention logits at high learning rates (Wortsman et al. 2023); this adds `2 × head_dim` parameters per layer and is off in the agreed pilot shape. See the [training recipe](../training-recipe.md).

## Fused attention path

The reference path above materializes the `[B,H,T,T]` scores in bfloat16 and the softmax in float32, which at context 1,024 dominates activation memory. Setting `train.attention: fused` routes the uncached training and validation forward through the scaled-dot-product primitive, which applies the scale, the causal mask and the softmax inside one kernel and keeps no score matrices for backward. The mathematics is unchanged; the model, its parameters and its checkpoints are identical, and cached generation always uses the reference path. `tests/test_model.py` compares the fused kernel, the fused model and a short fused training run against the reference path in float32 and under bfloat16 autocast, and `tools/pilot_preflight.py` repeats the comparison on the actual device before a pilot. One caveat: fused CUDA backward kernels may not be bitwise deterministic across runs, so exact resume is a property of saved state rather than of a repeated trajectory when the fused path is active. On this CPU box the S model's activation growth per 1,024-token sequence under bfloat16 autocast fell from 2,617 MiB (reference attention) to 1,760 MiB (fused); these are allocator proxies, not CUDA measurements.

Questions: why scale dot products (stable magnitudes); why mask before softmax (zero probability on future tokens); why residuals (preserve signal and gradients); why shared weights (fewer parameters); why normalize in float32 (reduce low-precision error).
