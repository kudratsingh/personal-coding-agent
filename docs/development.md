# Development guide

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
make setup
make check
```

`.python-version` pins Python 3.12.13. `uv.lock` locks developer dependencies; `make setup` installs the editable package and its environment using `uv sync --locked`. Use `uv run --locked` for commands so another project's activated environment is not used.

`make lint`, `make typecheck`, `make test`, and `make docs-check` run independently; `make check` runs all four. CPU CI uses the same commands on Linux and macOS.

`make smoke` runs a complete offline synthetic fixture pipeline: tokenizer/data preparation, two tiny pretraining stages, checkpoints, sampling, and held-out token validation for both domains. Artifacts stay in ignored `runs/smoke-*`. The supplied `configs/smoke.yaml` is intentionally tiny; this check does not establish benchmark capability or security of generated-code execution. It never executes generated code.

For individual local commands, use `uv run --locked scratchcoder --help` or `make <target> ARGS='...'`. Tokenizer training accepts explicit local UTF-8 corpus files. `data-build --synthetic --output runs/fixture` builds only synthetic material, never downloads gated data. Training requires a config, tokenizer, source-shard JSON map, and run ID. Evaluation, serving, and paid commands remain unavailable until implemented.

For now, use `.env.example` only as a template for local configuration; never commit the resulting `.env` file.

## Project layout

```text
configs/       versioned experiment configuration
src/scratchcoder/  shared utilities and component namespaces
tests/         automated tests
tools/         developer utilities
docs/          public documentation
private/       ignored owner-only material
```

Core component namespaces are placeholders; their presence does not imply implemented models, tokenizers, or code isolation.

## Reproducibility expectations

Record configuration, source revision, seed, environment, and input provenance for every meaningful run. Keep large artifacts out of Git and document where they are stored only when they are safe to share.

## Safety expectations

Do not use `eval`, `exec`, or unrestricted subprocess execution for model output. Treat every generated program and external dataset as untrusted input.
