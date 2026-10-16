# ADR 0009: explicit resume provenance in run-record schema 2

**Status:** Accepted for local infrastructure
**Date:** 2026-10-16

## Context

An exact resume restores optimizer/loader/RNG state; starting another stage loads weights with fresh optimization state. `parent_run_id` describes the latter and cannot identify an exact-resume checkpoint. Previously a failed resume could also leave its instantiated run marked `running`.

The original JSON schema rejects extra fields. Adding resume metadata silently under version 1 would break strict consumers without an explicit version change.

## Decision

New run records use schema 2 and require a nullable `resume_checkpoint`. Successful resume records the checkpoint's SHA-256, source run ID and saved step before subsequent training. The checkpoint is hashed and read through the same open handle; checkpoint files must remain immutable during reading. Raw paths and credentials are not stored in this metadata. Stage-parent lineage stays separate.

The current schema validates both versions. Existing schema-1 records remain readable and save without the new field, preserving their original strict shape. New schema-2 records require updated validators; old strict validators are not forward-compatible. Numerical training checkpoints retain their existing version-1 format.

A failed resume marks the instantiated run failed, records a safe failure event (including attempted file hash when available), and writes the existing sanitized debug trace. Source IDs and step bounds are checked before applying weights. This does not make an entire trainer restore transactional; discard the trainer after a failed restore.

## Consequences and verification

Records identify a resume without confusing it with a new stage or storing an owner-specific file path. Hashes provide content identity, not publisher authentication, legal-rights proof or durable remote backup.

Tests freeze the original v1 schema, validate both formats, preserve legacy round trips, reject malformed resume metadata, reproduce missing identity/stale status, and verify failure events/redaction. Existing float32/bfloat16 exact-resume and distributed tests remain unchanged. No paid compute or phase acceptance follows from this infrastructure change.
