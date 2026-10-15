# Decoder-only transformer

Token IDs `[B,T]` become embeddings `[B,T,C]`. A block normalizes each position by its root-mean-square feature magnitude, projects queries/keys/values, splits heads to `[B,H,T,D]`, and applies adjacent-pair rotary positions to queries and keys.

Attention scores `[B,H,T,T]` divide query-key dot products by `sqrt(D)`. A causal mask hides future positions before softmax. Probability-weighted values return to `[B,T,C]`, pass through an output projection, and join a residual connection. A second normalization feeds a width-4C exact-GELU MLP and another residual.

After all blocks, a final normalization and the shared embedding matrix produce logits `[B,T,V]`. All matrix projections are bias-free. Total parameters are `V*C + L*(12*C*C + 2*C) + C`, including normalization gains.

For example, width 32 and four heads gives head width 8. A length-eight sequence uses attention scores `[B,4,8,8]`; changing tokens at position five or later cannot change logits at positions zero through four.

`tests/test_model.py` checks attention outputs and gradients against PyTorch scaled-dot-product attention, normalization against a reference, rotary norm preservation, exact parameter accounting, and causality. These are correctness checks, not trained capability measurements.

Questions: why scale dot products (stable magnitudes); why mask before softmax (zero probability on future tokens); why residuals (preserve signal and gradients); why shared weights (fewer parameters); why normalize in float32 (reduce low-precision error).
