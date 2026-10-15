# Loss and optimization

For logits `[B,T,V]`, subtract the largest logit per position, compute log of the summed exponentials, and subtract the selected target logit. Averaging `[B,T]` yields stable next-token cross-entropy. Tests compare the scalar and its gradients to a trusted reference, including large logits.

AdamW maintains first and second moment tensors with the same shape as each parameter. Each step updates exponential averages, corrects their initial zero bias, divides the corrected first moment by the square-root second moment plus epsilon, and takes a learning-rate-scaled step. Decoupled weight decay multiplies selected weights before the update. Parameters without gradients do not advance their step counters.

Training excludes embeddings and normalization gains from decay. The optimizer itself accepts a per-parameter mask. Warmup raises the rate linearly; cosine decay then reaches a configured minimum. Gradient clipping measures the global norm and rescales all gradients together, rejecting nonfinite norms.

For a gradient `[3,4]`, the norm is five. Clipping to one multiplies both coordinates by approximately one fifth, preserving direction.

`tests/test_optimizer.py` compares 100 float64 updates with PyTorch AdamW, checks loss/gradient equivalence, and verifies schedule points and clipping by hand. Optimizer state is serializable for exact-resume testing in the training component.

Questions: why bias correction (zero-state startup); why decouple decay (separate regularization from gradient moments); why clip globally (preserve direction); why stable log-sum-exp (avoid overflow); why save moment state (weights alone do not resume the same optimization path).
