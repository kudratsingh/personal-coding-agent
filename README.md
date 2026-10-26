# Personal Coding Agent

An open, from-scratch learning project for building a small coding agent: tokenizer and model foundations, training and evaluation tooling, safe code execution, and an iterative repair loop.

This repository contains a tested from-scratch Python implementation and a reproducible synthetic development pipeline. It is not yet a trained coding agent or a hosted service.

## Current status

Data/tokenizer tooling, a compact transformer, optimizer, resumable two-stage training with domain validation, cached generation, isolated code evaluation, and multiple-choice scoring are implemented. Real-corpus pretraining has run at rehearsal scale: two 2B-token runs of the 135M-parameter S model on A40 GPUs, comparing two-stage and mixed pretraining ([results](docs/rehearsal-runs.md#measured-results-2026-10-26)). Post-training and full evaluation-suite acceptance remain pending.

The reference benchmark gate G3 is closed with owner acceptance: the pinned external model passed 14/164 HumanEval tasks (8.54%) using the project's stripped-prompt protocol, with zero empty completions. All 164 canonical solutions passed the same isolated harness. This is not evidence of our own model's coding capability. See [reference evaluation](docs/reference-evaluation.md) for the protocol and limitations.

See [the public status page](docs/status.md), [roadmap](docs/roadmap.md), and [architecture overview](docs/architecture.md).

## Project principles

- Build core learning components transparently and test them against trusted references.
- Keep generated code isolated and treat it as untrusted.
- Keep evaluation data separate from training data and report limitations honestly.
- Start with small, reproducible local experiments before using paid compute.
- Keep credentials, data, model weights, run artifacts, and private planning out of version control.

## Getting started

1. Read [the agent guide](AGENTS.md) and [development guide](docs/development.md).
2. Create a local environment and copy `.env.example` to `.env` only when a local secret is required.
3. Choose a small, well-scoped issue or the next milestone in [the roadmap](docs/roadmap.md).
4. Open a pull request with tests and documentation for the change.

Install uv, then run the local checks:

```sh
make setup
make check
make smoke
```

The owner's local planning entry point is `private/START_HERE.md`. It is intentionally absent from public clones; contributors can work from the public roadmap and issues.

## Documentation

- [Public roadmap](docs/roadmap.md)
- [Architecture overview](docs/architecture.md)
- [Project status](docs/status.md)
- [Development guide](docs/development.md)
- [Documentation index](docs/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

A license has not yet been selected. Until one is added, this repository is viewable but no permission is granted to reuse its contents.
