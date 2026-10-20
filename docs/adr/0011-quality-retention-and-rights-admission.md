# ADR 0011: classify license headers separately from quality

Status: accepted for sample review. Date: 2026-10-20.

## Context

The real Python sample first required a recognized original-file license header.
That gate discarded files before the quality filters, conflating missing inline
evidence with bad content. A permissive repository can contain files without a
license header. Conversely, a dataset's license tag does not prove downstream
rights for every file.

## Decision

Replace the original blanket check with a header classifier. Recognize Apache-2.0,
MIT, BSD-2-Clause, BSD-3-Clause, ISC, PSF-2.0, Unlicense and CC0-1.0. Preserve the
license ID and recognition evidence per retained file. GPL/LGPL/AGPL/MPL markers
are a separate copyleft rejection; non-commercial/proprietary restrictions have
their own rejection class. A missing or unsupported grant is unrecognized, never
silently approved. "All rights reserved" alone is not a restrictive-license
trigger: a recognized BSD grant that follows can be retained.

Inspect leading comments/module strings within the existing 16,384-character
window. Skip only known leading StarCoderData metadata lines (`<reponame>`,
`<filename>`, `<gh_stars>`) for classification, without changing stored text.
Recognize single supported SPDX IDs, explicit named declarations/grants and
specific grant patterns. Unsupported/compound declarations remain unrecognized;
restrictive markers take precedence. This conservative heuristic is not a general
license parser or legal warranty. The Unlicense's affirmative permission for
commercial or non-commercial purposes is not an NC restriction.

The earlier quality-only bypass was diagnostic only and is removed from the
current tool. Keep unrecognized examples (first five raw lines of ten files)
in ignored run storage, not Git. No source or full build is approved by a review.

Attribute each file and its complete UTF-8 bytes to one primary decision in
filter order. The pre-quality permissive class includes files later rejected by
quality rules and must not be added to final-kept counts. Do not change thresholds
to meet a desired retention percentage. Historical sample and frozen-tokenizer
artifacts remain immutable.

## Consequences and verification

Capacity reports must label which policy their retention describes. The historical
93.4% quality-only result is not the classifier's admitted-byte forecast. Authored
tests cover each class, per-file IDs, byte/file conservation, private examples,
BSD rights-reserved wording, metadata prefixes and no source execution. Real
corpus rights, final holdouts and build authorization remain separate conditions.
The [sample evidence](../real-samples.md) records the actual offline runs.
