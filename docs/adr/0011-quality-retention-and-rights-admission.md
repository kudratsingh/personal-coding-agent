# ADR 0011: distinguish quality retention from rights admission

Status: accepted for audit tooling. Date: 2026-10-20.

## Context

The real Python sample first required a recognized original-file license header.
That gate discarded files before the quality filters, conflating missing inline
evidence with bad content. A permissive repository can contain files without a
license header. Conversely, a dataset's license tag does not prove downstream
rights for every file.

## Decision

Keep the existing default admission rule. Add an explicit
`--review-unresolved-licenses` mode that measures content quality independently.
Retain missing license evidence as JSON null, never invent a license. Both modes
are review outputs, not training approval. Repository/component license evidence
and required notices must be resolved separately before admission.

Attribute each file and its complete UTF-8 bytes to one primary decision in
filter order. License observations in quality-review mode overlap with quality
decisions and must not be summed as additional drops. Do not change thresholds
to meet a desired retention percentage. Historical sample and frozen-tokenizer
artifacts remain immutable.

## Consequences and verification

Capacity reports must label which policy their retention describes. A
quality-only forecast is conditional on separate rights review, not a license
clearance. Authored tests check byte/file conservation, unchanged default
behavior, and null licenses plus false final-training admission in review mode.
The [sample evidence](../real-samples.md) records the actual offline runs.
