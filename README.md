# Personal Coding Agent

A small language model and Python coding agent built from scratch: byte-level BPE, a decoder-only transformer, AdamW, resumable training, cached generation, supervised fine-tuning, on-policy RL and isolated code evaluation.

The aim is to follow the whole path from training data to an agent that writes and repairs code, with core learning components implemented here and checked against reference implementations. **Real-corpus base-model pretraining has run at rehearsal scale. Trained coding-agent capability and post-training acceptance remain open.**

## Measured so far

Two **135M-parameter** rehearsal models each trained on the same **2B-token multiset**, from a shared initialization, on one A40 per run. The comparison tested a two-stage English→Python recipe against a mixed recipe.

| Final checkpoint | English validation bits/byte ↓ | Python validation bits/byte ↓ |
|---|---:|---:|
| Two-stage | 1.1328 | 0.7747 |
| Mixed | 1.0637 | 0.7827 |

The mixed recipe retained English better; the Python difference was small, with one seed per arm. This compares recipes rather than order alone: the two-stage run also reset its optimizer and schedule. The main-model decision is mixed pretraining. These are validation-loss results, not code-generation scores. [Protocol, runs and limits](docs/rehearsal-runs.md#measured-results-2026-10-26).

The isolated HumanEval harness was also checked with a **third-party reference model**: 14/164 tasks passed under the project's stripped-prompt protocol, with no empty completions, and 164/164 canonical solutions passed. That result validates the reference evaluation path; it is not this project's model performance. [Reference evaluation](docs/reference-evaluation.md).

## What is implemented

- **Data and tokenization:** provenance-aware manifests, deterministic filtering/splits, byte-level BPE, token shards and held-out contamination checks.
- **Model and optimizer:** a decoder-only transformer built from tensor primitives, rotary positions, RMSNorm and a local AdamW implementation, with reference comparisons.
- **Training:** domain validation, exact checkpoint/resume, gradient accumulation, CPU distributed continuation checks and opt-in optimized paths with equivalence tests.
- **Generation and post-training code:** cached autoregressive sampling, SFT and an on-policy group-advantage RL loop, plus development checks and bounded stopping rules.
- **Evaluation:** isolated Python execution, task admission and canonical-solution preflight, code and multiple-choice metrics, and explicit context-limit accounting.

Implementation, synthetic smoke runs and measured model capability are tracked separately in the [public status](docs/status.md). SFT/RL rehearsal results, full-suite acceptance, released weights and a hosted coding service are not claimed.

## Run locally

Install [uv](https://docs.astral.sh/uv/); the project pins its Python and dependencies.

```bash
make setup
make check
make smoke
```

`make check` runs publication/workflow checks, lint, types, tests and documentation checks. `make smoke` runs a small synthetic pipeline. Neither command downloads a training corpus or starts paid compute. Code-execution evaluation needs the deliberately isolated runner described in the [sandbox guide](docs/sandbox.md); generated code is untrusted.

Training data, weights, run artifacts and private planning stay outside Git. Public clones can follow the roadmap and development guide without owner-only files.

## Code and documentation

| Area | Entry point |
|---|---|
| Tokenizer and data pipeline | [tokenizer](src/scratchcoder/tokenizer/), [data](src/scratchcoder/data/) |
| Transformer and generation | [model](src/scratchcoder/model/), [engine](src/scratchcoder/engine/) |
| On-policy post-training | [rl](src/scratchcoder/rl/) |
| Design and implementation walkthroughs | [Architecture](docs/architecture.md), [walkthroughs](docs/walkthroughs/) |
| Reproduction and current milestone | [Development](docs/development.md), [status](docs/status.md), [roadmap](docs/roadmap.md) |
| Contributing | [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), [security policy](SECURITY.md) |

## License

A license has not yet been selected. The repository is public for inspection; no reuse license is granted.
