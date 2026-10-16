# ADR 0002: typed configuration and atomic run records

**Status:** Accepted for local infrastructure
**Date:** 2026-10-15

## Decision

Use validated dataclasses loaded with YAML's safe loader. Reject unknown fields, invalid model shapes, impossible sequence lengths, and invalid source shares before execution. Serialize the resolved configuration into each run record.

Write JSON run records atomically with source revision, dirty state, seed, input hashes, parent lineage, hardware, timings, artifact location, and cost fields. Keep records outside version control. A missing Git checkout is recorded explicitly rather than fabricated.

## Consequences and verification

Interrupted writes preserve the previous valid file. A committed JSON Schema describes the record shape. Tests round-trip YAML, validate a completed dummy record against the schema, reject unsafe run identifiers, and inject a failed JSON write.

The schema establishes record structure, not proof of data licensing, artifact durability, or successful training.

[ADR 0009](0009-resume-provenance.md) introduces explicit schema-2 resume lineage and failure recording while preserving legacy schema-1 record serialization. Numerical checkpoint format remains separate.
