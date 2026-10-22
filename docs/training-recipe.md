# Training recipe: evidence and options

Prepared 2026-10-22 alongside the pilot readiness audit. This page records what comparable
small-model pretraining runs do, which of those practices this repository now implements,
and which specification enables what. Everything here was verified on CPU only; no GPU run
has produced a number yet.

## What comparable runs look like

| Run | Size | Recipe points relevant here |
| --- | --- | --- |
| GPT-2 small ([Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [nanoGPT](https://github.com/karpathy/nanoGPT)) | 124M, 12 x 768, context 1,024, tied embeddings | Normal init std 0.02 with the two residual projections per block scaled by 1/sqrt(2L); AdamW betas (0.9, 0.95), weight decay 0.1 on matrices only; warmup then cosine; batch about 0.5M tokens; gradient clip 1.0 |
| Pythia-160M ([Biderman et al. 2023](https://arxiv.org/abs/2304.01373)) | 162M, 12 x 768, rotary | Peak learning rate 6e-4, batch 2M tokens, cosine to 10% of peak, no dropout |
| SmolLM2-135M ([Allal et al. 2025](https://arxiv.org/abs/2502.02737)) | 135M, deep-and-narrow (30 x 576), GQA, tied embeddings, RMSNorm, rotary | Warmup-stable-decay (WSD) schedule so stable-phase checkpoints can be continued, trillions of tokens, large batches |
| Small-scale proxies for instabilities ([Wortsman et al. 2023](https://arxiv.org/abs/2309.14322)) | 10M to 1B sweeps | Attention-logit growth and output-logit divergence appear at high learning rates even at small scale; QK layer-normalization and a z-loss of 1e-4 x log Z^2 remove both and flatten learning-rate sensitivity |
| PaLM ([Chowdhery et al. 2022](https://arxiv.org/abs/2204.02311)), OLMo 2 ([OLMo team 2024](https://arxiv.org/abs/2501.00656)) | 8B to 540B | z-loss 1e-4; OLMo 2 adds QK-norm and reordered norms for stability |
| modded-nanoGPT speedruns ([Jordan et al.](https://github.com/KellerJordan/modded-nanogpt)) | 124M, 8 x H100 | Rotary, QK-norm, ReLU^2 MLP, zero-initialized projections, untied head, logit softcap, Muon optimizer, flash attention with sliding windows, batch and sequence-length schedules; validation loss 3.28 on FineWeb reached in minutes |

## What this repository implements now

All of the following are reference-tested on CPU (`tests/test_model.py`, `tests/test_optimizer.py`,
`tests/test_training.py`) and documented in the walkthroughs. Defaults preserve the previously
agreed S model exactly; the options are selected per specification.

| Practice | Where | Default | Why |
| --- | --- | --- | --- |
| Depth-scaled residual init (0.02 / sqrt(2L) on the attention output and MLP down projections) | `GPT.__init__` | always on | GPT-2 standard; keeps residual-stream variance from growing with depth at step 0; does not change the architecture or parameter count |
| Fused attention through the scaled-dot-product primitive | `train.attention: fused` | off | removes the `[B,H,T,T]` score matrices from memory; CPU proxy per 1,024-token sequence fell from 2,617 MiB to 1,760 MiB; equivalence-tested in float32 and bf16; A40 preflight repeats the check |
| Log-softmax cross-entropy (one saved fp32 copy of the logits) | `cross_entropy` | always on | same arithmetic as the explicit max/exp/sum/log chain; 4 instead of 12 bytes per logit saved |
| z-loss `1e-4 x mean(log Z^2)` | `train.z_loss` | 0 (off) | Wortsman et al. and PaLM stabilizer for output logits; the logged `loss` stays the plain next-token loss so arms and validation remain comparable; the z term is logged separately |
| QK-normalization (RMSNorm on per-head queries and keys before rotary, gains shared across heads) | `model.qk_norm` | off | removes attention-logit growth at high learning rates; adds 2 x head_dim gains per layer (1,536 parameters for the S shape, so 135,287,040 instead of 135,285,504); `expected_parameters` accounts for it |
| Warmup-stable-decay schedule with a linear tail | `train.schedule: wsd`, `train.decay_fraction` | cosine | lets a stable-phase checkpoint continue without re-warming; the 500-step pilot keeps cosine so the step-100 learning-rate screen compares arms on one schedule |
| Single-synchronization gradient clipping | `clip_gradients` | always on | one device-to-host read instead of one per parameter tensor (74 for the S model) |
| List-kernel AdamW (`torch._foreach_*`) | `train.optimizer_foreach` | off | the identical update issued as one kernel per operation instead of per tensor; tested bitwise equal to the per-tensor loop |
| `torch.compile` of the training forward | `train.compile` | off | typical 1.3x to 2x speedup for an eager model on a GPU; compile time is paid once per distinct shape (`dynamic=True` covers the probe sizes); compiled training is tested against eager within 1e-4 |

## Which specification enables what

| Specification | Attention | Optimizer | z-loss | QK-norm | Compile | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/pilot-a40-s.json` (agreed pilot of record) | fused | foreach | 0 | off | off | 135,285,504 |
| `configs/pilot-a40-s-improved.json` (recommended if the owner accepts the recipe change) | fused | foreach | 1e-4 | on | on | 135,287,040 |

The pilot of record keeps the agreed objective and architecture and only takes the changes that
are numerically equivalent or memory-only. The improved variant changes the objective (z-loss)
and the architecture (QK-norm gains), which is a scientific-protocol change: it must be an
explicit owner decision, and its parameter count differs by 1,536. Both pass the same launch gate.

## Options considered and deferred

- Untied input/output embeddings, ReLU^2 or SwiGLU feed-forward, logit softcap and Muon: each
  changes the parameter count or the optimizer family materially; they are the natural next
  experiments once the pilot has produced measured A40 numbers.
- Document-boundary attention masking (used by Llama 3 and the speedruns): needs block-sparse
  masks (FlexAttention) to stay fused; the current packing lets attention cross document
  boundaries, which is the GPT-2 and Pythia behaviour.
- Larger validation sets: the bound data holds 32 held-out documents (at most 32 x 1,024 target
  tokens). A rebuilt binding with 128 or more validation documents would reduce selection noise
  but changes the bound dataset identity and is therefore an owner decision.
- Larger effective batches (0.5M to 2M tokens) as in GPT-2 and Pythia: the agreed pilot batch is
  131,072 tokens to fit the 60-minute budget.
