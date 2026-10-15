# ADR 0003: local-first structured diagnostics

**Status:** Accepted for local infrastructure
**Date:** 2026-10-15

## Decision

Always write structured local JSONL events before optional hosted metric delivery. Keep the tracker behind a small protocol, with W&B available only as an explicit optional adapter. Do not initialize online clients during import, tests, or default local runs.

Use the standard logging interface for debugging, with isolated handlers and JSON-formatted tracebacks. Restrict file permissions and redact known credential forms. Avoid logging raw data or secrets in the first place.

## Verification and limitations

Tests cover ten-step local/tracker parity, offline operation, nested redaction, traceback formatting, tracker failure, and the adapter with a mocked client. Live hosted delivery remains an account-dependent integration check. Pattern-based redaction cannot identify every possible secret.
