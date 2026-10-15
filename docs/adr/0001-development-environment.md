# ADR 0001: reproducible development environment

**Status:** Accepted for the development scaffold
**Date:** 2026-10-15

## Context

Local and CI commands must install the same tools, detect incomplete pipeline work, and run without rented hardware.

## Decision

Use Python 3.12.13, uv with a committed lockfile, an editable `src/scratchcoder` package, Ruff, strict mypy, and pytest. Run matching Make commands in Linux and macOS CPU CI. Core model dependencies are added with the components that need them.

`make smoke` fails with a list of missing stages until actual end-to-end work replaces it. Unimplemented commands must not report success or provision resources.

## Consequences

Development setup is reproducible without GPU packages. Package namespaces communicate intended structure but do not claim implementation. Changes to dependency inputs require regenerating and reviewing the lockfile.

## Verification

`make setup && make check` validates installation, formatting, lint, types, tests, and documentation. CLI tests cover incomplete smoke execution from outside the checkout and reject unsupported commands.
