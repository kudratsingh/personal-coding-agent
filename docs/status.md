# Public project status

**Stage:** Foundation

The Python package scaffold, locked development environment, checks, and CPU CI are implemented. There is no model, training result, benchmark score, hosted service, or released weight to report yet.

## Completed starter work

- Public roadmap, architecture, development, data, evaluation, and sandbox guides.
- Shared coding-agent entry points and contributor templates.
- Documentation link check and a matching CI workflow.
- Python 3.12.13, uv dependency lock, component namespaces, lint, types, tests, and Linux/macOS CPU CI.

**Verification:** `make check` passes locally. Smoke deliberately exits nonzero while stages are missing.

## Next public milestone

Add typed configuration and reproducible run records, then logging infrastructure before the data and tokenizer milestone.

## Reporting policy

Public results will identify the configuration, evaluation method, and known limitations. Incomplete, single-run, or non-reproducible results will be labeled accordingly.
