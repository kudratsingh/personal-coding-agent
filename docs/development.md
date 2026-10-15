# Development guide

## Local setup

The starter documentation checker uses only Python's standard library and requires Python 3.11 or newer:

```sh
python3 tools/check_docs.py
# or
make docs-check
```

The first implementation milestone will pin the training runtime, dependency lockfile, package layout, test harness, and install command. The documentation check does not claim that the training pipeline is implemented.

For now, use `.env.example` only as a template for local configuration; never commit the resulting `.env` file.

## Planned project layout

```text
configs/       versioned experiment configuration
src/           implementation packages
tests/         automated tests
tools/         developer utilities
docs/          public documentation
private/       ignored owner-only material
```

## Reproducibility expectations

Record configuration, source revision, seed, environment, and input provenance for every meaningful run. Keep large artifacts out of Git and document where they are stored only when they are safe to share.

## Safety expectations

Do not use `eval`, `exec`, or unrestricted subprocess execution for model output. Treat every generated program and external dataset as untrusted input.
