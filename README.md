# Personal Coding Agent

An open, from-scratch learning project for building a small coding agent: tokenizer and model foundations, training and evaluation tooling, safe code execution, and an iterative repair loop.

This repository currently contains the public project charter and contribution scaffold. Implementation and benchmark claims will be added only as they are reproducible.

## Current status

Pre-implementation. The first public milestone is a minimal, locally reproducible development scaffold.

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

To check the starter documentation with Python 3.11 or newer:

```sh
python3 tools/check_docs.py
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
