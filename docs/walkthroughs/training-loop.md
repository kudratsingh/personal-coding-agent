# Training, continuation, and resume

The loader supplies input and one-position-shifted target arrays `[B,T]`. The model produces `[B,T,V]` logits; loss averages the target negative log probabilities. Accumulated minibatch losses are scaled before backpropagation so their sum equals one effective-batch mean. Distributed ranks average gradients before clipping and stepping.

A run configuration supplies the learning rate, Adam betas/epsilon, decay, clipping, accumulation, and precision. AdamW excludes embedding and normalization parameters from decay. Bfloat16 autocast is explicit in configuration; reference checks use float32. Each step records loss, rate, gradient norm, and throughput to local JSONL.

`tests/test_bfloat16.py` additionally exercises the existing CPU autocast path on authored synthetic tokens. At three fixed seeds it compares logits, float32 loss and gradients against the same float32 model: logits use absolute tolerance 0.005/relative 0.03; loss uses 0.001/0.001; each parameter's gradient L2 error must be at most 3% of the reference norm (exact zero for a zero reference). These are bounded tiny-fixture checks, not universal numerical-error guarantees. Autocast logits must actually be bfloat16 while master parameters, loss and gradients remain float32 and finite.

The CPU precision tests also compare uninterrupted and interrupted/resumed losses, final weights, optimizer moments/counters and loader state exactly, with accumulation factors one and two. A 100-step synthetic one-batch memorization test requires loss below 0.05 and a reduction by more than 50 times. No tolerances or core arithmetic are changed to make these tests pass. They do not validate CUDA kernels, GPU precision/throughput, distributed mixed precision, production stability or coding capability; the separately approved GPU pilot remains open.

A resume checkpoint includes weights, optimizer moments/counters, step, loader buffers/positions/RNG, CPU/CUDA RNG state per rank, configuration, and input identities. Writes are atomic. Exact resume requires the same configuration, data/tokenizer identity, and world size.

Starting a new stage is different: a parent checkpoint supplies only weights and lineage. Its optimizer, schedule, loader state, and step counter are fresh. This lets the same trainer continue on a new data mix without pretending to resume the previous optimization path.

Tests stop a tiny run halfway, serialize it, resume it, and compare every later loss and final parameter exactly with an uninterrupted CPU run. Stage-two tests compare step-zero validation and fresh counters. Actual rented GPU throughput, durable uploads, and watchdog behavior remain separate gates.

Questions: why save loader buffers (avoid skipped/repeated tokens); why save RNG (reproduce stochastic state); why reject changed data (same counters can mean different examples); why reset stage-two moments (new optimization phase); why gather per-rank state (each rank consumed different data).

## Periodic held-out validation

Set `train.validation_interval` to a positive step count and supply `train --validation path/to/domains.json`, a map from domain names to local held-out shard paths relative to that JSON file. `train.validation_documents` limits the number of fixed document-prefix windows per domain; each window has at most `sequence_length` targets. This is a bounded diagnostic sample, not a full-corpus estimate. The caller must supply genuinely held-out shards.

Rank zero records per-domain validation at step zero, each interval, and the final step. The token-weighted mean loss is total negative log likelihood divided by target tokens. Bits per byte divides the same NLL by `log(2)` and the raw target byte count. Byte counts come directly from the tokenizer's byte table: decoding a partial UTF-8 sequence to a replacement character would give an incorrect denominator. Special tokens contribute no raw text bytes.

Validation uses no gradients, restores the previous model mode, and never advances the training loader or optimizer. Fixed windows and byte counts have a checkpoint identity; exact resume rejects changed validation inputs. `tests/test_validation.py` checks uniform-logit reference arithmetic, partial UTF-8 bytes, unchanged weights/RNG/loader/training losses, configured logging steps, and changed-input rejection. The explicit training CLI test supplies separate synthetic web and Python validation shards.
