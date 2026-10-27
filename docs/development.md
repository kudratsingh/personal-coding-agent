# Development guide

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
make setup
make check
```

`.python-version` pins Python 3.12.13. `uv.lock` locks developer dependencies; `make setup` installs the editable package and its environment using `uv sync --locked`. Use `uv run --locked` for commands so another project's activated environment is not used.

`make workflow-check`, `make privacy-check`, `make lint`, `make typecheck`, `make test`, and `make docs-check` run independently; `make check` runs all six. CPU CI uses the same commands on Linux and macOS. The [workflow check](compute.md) limits runner declarations to the reviewed subset; it is not a billing guarantee. The publication check rejects tracked paths matching repository `.gitignore` rules, including force-added files; it does not scan contents for secrets. Documentation checks include new nonignored public Markdown before staging, exclude ignored owner-only docs, and reject symlink documents without reading them. See [private-material guidance](private-material.md).

`make smoke` runs a complete offline synthetic fixture pipeline: tokenizer/data preparation, two tiny pretraining stages, checkpoints, sampling, and held-out token validation for both domains. Artifacts stay in ignored `runs/smoke-*`. The supplied `configs/smoke.yaml` is intentionally tiny; this check does not establish benchmark capability or security of generated-code execution. It never executes generated code.

For individual local commands, use `uv run --locked scratchcoder --help` or `make <target> ARGS='...'`. Tokenizer training accepts explicit local UTF-8 corpus files. `data-build --synthetic --output runs/fixture` builds only synthetic material, never downloads gated data. Training requires a config, tokenizer, source-shard JSON map, and run ID. Optional `--validation` supplies a separate held-out domain shard map when `train.validation_interval` is enabled; see the [training walkthrough](walkthroughs/training-loop.md).

Training accepts `--device cpu`, `--device cuda`, or `--device cuda:N`; the default is CPU. Unsupported devices, unavailable CUDA, and invalid CUDA ordinals are rejected before a run is created. There is no fallback to another device and no resource provisioning. The run record stores the resolved device in `hardware.training_device`, with CUDA represented by an explicit ordinal. The Linux `uv.lock` environment installs CPU PyTorch; installing a CUDA-enabled PyTorch build is a separate environment setup step. CPU and mocked-device tests check selection and timing behavior, but do not establish GPU correctness, performance, or launch readiness.

`scratchcoder rl-prescreen` counts RL candidates by the screening rules that need no model and runs no code; `scratchcoder rl-screen` scores screening attempts in the local Docker sandbox and writes the keep-rule record ([rehearsal RL](rl.md)). `scratchcoder eval` evaluates code-task JSONL through isolation; `scratchcoder eval-choice` scores local choice-task JSONL without executing code; `scratchcoder eval-validation` reports held-out domain loss/bits per byte from local shards. All require explicit checkpoint/tokenizer/input/output paths and record local reports/scorecards. `scratchcoder generation-benchmark` prints generation throughput as JSON for a checkpoint or a configuration with random weights. The post-training pipeline adds `eval-generate`, `eval-execute`, `eval-seal`, `eval-baseline` and `scorecard`. See [evaluation](evaluation.md). Serving and paid commands remain explicit unavailable placeholders; no paid compute is authorized without permission for the specific run.

The optional independent tokenizer reference runs with `uv run --locked --extra baseline pytest tests/test_tokenizer_external.py`. It trains only on synthetic fixtures, never downloads a pretrained model or corpus, and does not become part of the core path. A dedicated CI job runs it; the base environment explicitly skips it when the extra is absent.

For now, use `.env.example` only as a template for local configuration; never commit the resulting `.env` file.

The optional `evaluation` extra supplies locked PyArrow file-format parsing for [ARC-Easy](arc-easy.md), separate from core learning/scoring logic. `uv run --locked --extra evaluation pytest tests/test_arc.py` tests authored Parquet fixtures without downloading a benchmark. Base checks explicitly skip parser-dependent cases when the extra is absent; dedicated CI runs them. Full real-suite workflows are manual and do not authorize paid resources.

## Project layout

```text
configs/       versioned experiment configuration
src/scratchcoder/  shared utilities and component namespaces
tests/         automated tests
tools/         developer utilities
docs/          public documentation
private/       ignored owner-only material
```

Tokenizer, data, model, optimizer, training, engine, evaluation, development sandbox and RL modules are implemented; the RL run and its more-SFT control are CPU-tested but have not run on a GPU ([rehearsal RL](rl.md)). Agent repair-loop and serving namespaces remain future work. Consult [status](status.md) for acceptance gaps rather than inferring completion from a package name.

## Reproducibility expectations

Record configuration, source revision, seed, environment, and input provenance for every meaningful run. Keep large artifacts out of Git and document where they are stored only when they are safe to share.

## Safety expectations

Do not use `eval`, `exec`, or unrestricted subprocess execution for model output. Treat every generated program and external dataset as untrusted input.
