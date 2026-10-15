# Public project status

**Stage:** Foundation

Local foundation, synthetic data/tokenization, model/trainer, and generation components are implemented and reference-tested. No external-corpus training, benchmark capability, hosted service, or released weight is claimed.

## Completed starter work

- Public roadmap, architecture, development, data, evaluation, and sandbox guides.
- Shared coding-agent entry points and contributor templates.
- Documentation link check and a matching CI workflow.
- Python 3.12.13, uv dependency lock, component namespaces, lint, types, tests, and Linux/macOS CPU CI.
- Validated YAML configuration and atomic, schema-tested run records with provenance and parent lineage.
- Local JSONL metrics, an opt-in tracker adapter, and structured debug logs with tracebacks.
- Manifest validation, deterministic filtering/splits, and a byte-level BPE implementation with reference tests.
- Shards, exact-resume training, CPU distributed checks, cached sampling, and a synthetic two-stage smoke path.

**Verification:** `make check` and `make smoke` pass locally. CPU CI runs the same checks and synthetic smoke on Linux and macOS.

## Next public milestone

Build isolated code evaluation and task/metric tooling. Real-source ingestion, GPU pilot, and reference benchmark validation remain separately gated.

## Reporting policy

Public results will identify the configuration, evaluation method, and known limitations. Incomplete, single-run, or non-reproducible results will be labeled accordingly.
