# Training, continuation, and resume

The loader supplies input and one-position-shifted target arrays `[B,T]`. The model produces `[B,T,V]` logits; loss averages the target negative log probabilities. Accumulated minibatch losses are scaled before backpropagation so their sum equals one effective-batch mean. Distributed ranks average gradients before clipping and stepping.

A run configuration supplies the learning rate, Adam betas/epsilon, decay, clipping, accumulation, and precision. AdamW excludes embedding and normalization parameters from decay. Bfloat16 autocast is explicit in configuration; reference checks use float32. Each step records loss, rate, gradient norm, and throughput to local JSONL.

A resume checkpoint includes weights, optimizer moments/counters, step, loader buffers/positions/RNG, CPU/CUDA RNG state per rank, configuration, and input identities. Writes are atomic. Exact resume requires the same configuration, data/tokenizer identity, and world size.

Starting a new stage is different: a parent checkpoint supplies only weights and lineage. Its optimizer, schedule, loader state, and step counter are fresh. This lets the same trainer continue on a new data mix without pretending to resume the previous optimization path.

Tests stop a tiny run halfway, serialize it, resume it, and compare every later loss and final parameter exactly with an uninterrupted CPU run. Stage-two tests compare step-zero validation and fresh counters. Actual rented GPU throughput, durable uploads, and watchdog behavior remain separate gates.

Questions: why save loader buffers (avoid skipped/repeated tokens); why save RNG (reproduce stochastic state); why reject changed data (same counters can mean different examples); why reset stage-two moments (new optimization phase); why gather per-rank state (each rank consumed different data).
