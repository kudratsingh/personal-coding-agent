# ADR 0007: explicit, fingerprint-bound find-zero oracle correction

Status: accepted, opt-in only. Date: 2026-10-16.

## Context

The pinned HumanEval+ OriginFmt release passes 163/164 canonical solutions. Its task-32 assertion tries to unpack a scalar root as polynomial coefficients, producing `TypeError`. The [pinned official evaluator](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/eval/__init__.py) instead checks the absolute polynomial residual. Untouched-release results must remain distinguishable from any correction.

## Decision

Keep unmodified tests as the default. An explicit `--correct-find-zero-oracle` option enables only `find-zero-residual-v1`: replace the single malformed scalar-unpacking assertion with an absolute polynomial-residual check using the original input and unchanged `0.0001` tolerance. Require the exact pinned release specification, task ID, provenance, entry point, and original normalized hidden-script SHA-256. Unexpected bytes, absent/duplicate tasks, repeated expressions, and other suites fail before execution. Do not change inputs, canonical code, prompts, or other tasks.

Record the correction version, task ID, original release hash, original/effective hidden-script hashes, effective individual task hash, and reference evaluator revision in both canonical and checkpoint reports. The report's existing aggregate task hash identifies the effective complete suite. A corrected canonical preflight must pass every task before checkpoint scoring.

## Consequences

Corrected results are not untouched OriginFmt results or a reproduction of the official evaluator's timing/resource policy. Earlier failed records remain valid evidence. This decision does not explain the separate original-HumanEval external-model score discrepancy, authorize benchmark training, or approve paid compute. Additional oracle changes require separate review.

## Verification

Synthetic fixtures verify exact preservation of all bytes outside the replaced expression, fingerprint/provenance rejection, default behavior, and identical correction metadata in both report kinds. Disposable-runtime tests accept two distinct valid roots, reject a negative-residual wrong root, and reproduce the original scalar-unpacking failure. A separate manual workflow validates the complete pinned release; unit tests alone cannot establish its result. See the [walkthrough and run evidence](../evalplus.md).
