# Public project status

**Stage:** Foundation

The Python package scaffold, locked development environment, checks, and CPU CI are implemented. There is no model, training result, benchmark score, hosted service, or released weight to report yet.

## Completed starter work

- Public roadmap, architecture, development, data, evaluation, and sandbox guides.
- Shared coding-agent entry points and contributor templates.
- Documentation link check and a matching CI workflow.
- Python 3.12.13, uv dependency lock, component namespaces, lint, types, tests, and Linux/macOS CPU CI.
- Validated YAML configuration and atomic, schema-tested run records with provenance and parent lineage.
- Local JSONL metrics, an opt-in tracker adapter, and structured debug logs with tracebacks.
- Manifest validation, deterministic filtering/splits, and a byte-level BPE implementation with reference tests.

**Verification:** `make check` passes locally. Smoke deliberately exits nonzero while stages are missing.

## Next public milestone

Build licensed-data tooling and the tokenizer, then model/trainer and isolated evaluation components.

## Reporting policy

Public results will identify the configuration, evaluation method, and known limitations. Incomplete, single-run, or non-reproducible results will be labeled accordingly.
