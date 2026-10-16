# ADR 0005: explicit label-likelihood choice evaluation

**Status:** Accepted for local development evaluation
**Date:** 2026-10-16

## Decision

Score each space-prefixed answer label after a fixed question/options/`Answer:` prompt, using the sum of continuation-token log probabilities. Do not score the prompt, expose the correct label to the scorer, normalize by token count, or silently truncate oversized questions. Break ties by listed order. Record this protocol and task/checkpoint/tokenizer identities in every report.

## Consequences

This reuses the reference-tested from-scratch answer scorer without sampling noise or generated-code execution. Label length and ordering may affect results; comparisons require identical formatting and suites. This protocol is not claimed equivalent to a published ARC evaluation. Source rights, provenance, and real-suite validation remain separate gates.

## Verification

The [walkthrough](../walkthroughs/multiple-choice.md) identifies the slow scoring reference, hand-computed metric cases, schema/key-isolation tests, and synthetic checkpoint CLI test.
